"""Resource-bounded Group Relative Policy Optimization for PTCG.

This is value-free GRPO/PPO fine tuning of the deck-aware replay policy.  A
group consists of games with the same deck, opponent archetype, and seat.  The
terminal rewards are standardized within that group, so only strategies that
do better than their matched siblings receive a positive advantage.

The learned actor is intended for Track 1 root priors.  It is not submitted as
a raw policy: pure BC without search scored 591.1, while the same family of
models used as search priors reached 967.1.
"""
from __future__ import annotations

import argparse
import bisect
import copy
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AGENT = os.path.join(ROOT, "track1_search", "agent")
TRAIN = os.path.join(ROOT, "track1_search", "train")
sys.path.insert(0, AGENT)
sys.path.insert(0, TRAIN)

import nn_features_rich as NF  # noqa: E402
import model as model_module  # noqa: E402


@dataclass
class Decision:
    kind: np.ndarray
    card: np.ndarray
    scal: np.ndarray
    mask: np.ndarray
    ctx: int
    stype: int
    chosen_pos: int
    old_logp: float
    advantage: float = 0.0


def load_agent_module():
    spec = importlib.util.spec_from_file_location(
        "grpo_agent_main", os.path.join(AGENT, "main.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._load_engine()
    module._load_card_db()
    module._ENGINE_TRIED = True
    return module


def read_deck(path):
    with open(path, encoding="utf-8") as source:
        deck = [int(line) for line in source if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"{path} contains {len(deck)} cards, expected 60")
    return deck


def configure_model_from_npz(path):
    with np.load(path) as weights:
        if "_meta" not in weights:
            raise ValueError(f"{path} has no architecture metadata")
        dim, layers, heads, dff = map(int, weights["_meta"])
    model_module.NF = NF
    model_module.D_MODEL = dim
    model_module.N_LAYERS = layers
    model_module.N_HEADS = heads
    model_module.D_FF = dff
    model_module.DROPOUT = 0.0
    return dim, layers, heads, dff


def load_model(path, device="cpu"):
    configure_model_from_npz(path)
    net = model_module.TCGNet()
    with np.load(path) as weights:
        state = net.state_dict()
        missing = []
        for key, value in state.items():
            if key not in weights or tuple(weights[key].shape) != tuple(value.shape):
                missing.append(key)
            else:
                state[key] = torch.as_tensor(weights[key])
        if missing:
            raise ValueError(f"checkpoint does not match model: {missing[:5]}")
    net.load_state_dict(state)
    return net.to(device)


def resource_guard(max_decisions, out_path):
    """Reject obviously unsafe configurations before allocating rollout data."""
    if max_decisions < 1 or max_decisions > 50_000:
        raise ValueError("--max-decisions must be in [1, 50000]")
    # A stored decision is about 8 KiB including Python/array overhead.  Keep a
    # conservative 4x allowance for batching, the models, and the simulator.
    need = max_decisions * 8 * 1024 * 4 + 2 * 1024**3
    available = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as source:
            fields = {line.split(":", 1)[0]: int(line.split()[1]) * 1024
                      for line in source if ":" in line}
        available = fields.get("MemAvailable")
    except OSError:
        pass
    if available is not None and need > 0.75 * available:
        raise RuntimeError(
            f"rollout cap needs a conservative {need / 2**30:.1f} GiB, but "
            f"only {available / 2**30:.1f} GiB is available")
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    free_disk = shutil.disk_usage(out_dir).free
    if free_disk < 5 * 1024**3:
        raise RuntimeError("less than 5 GiB free on the checkpoint filesystem")


def group_advantages(rewards):
    values = np.asarray(rewards, dtype=np.float32)
    if len(values) < 2:
        return np.zeros_like(values)
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(values)
    return (values - values.mean()) / (std + 1e-6)


def weighted_opponent_schedule(opponents, weights, slots=40, power=0.5):
    """Deterministic popularity-tempered opponent cycle.

    Raw replay popularity is too concentrated to use directly (one exact list
    can occupy roughly half the field), while a uniform top-N schedule badly
    underweights that matchup. Square-root weighting is the middle ground.
    """
    if not opponents:
        raise ValueError("opponent schedule cannot be empty")
    if slots < len(opponents):
        slots = len(opponents)
    if power <= 0:
        return list(opponents)
    mass = [max(float(weights.get(name, 1)), 1.0) ** power
            for name, _deck in opponents]
    cumulative = np.cumsum(mass).tolist()
    total = cumulative[-1]
    schedule = []
    for slot in range(slots):
        point = (slot + 0.5) * total / slots
        schedule.append(opponents[min(
            bisect.bisect_left(cumulative, point), len(opponents) - 1)])
    return schedule


def encode(M, obs, me, deck):
    sel = obs["select"]
    state = {"current": obs["current"], "select": sel, "decklist": deck}
    kind, card, scal, mask, slots = NF.encode(
        state, me, M.CARD, M.ATTACK, None)
    return kind, card, scal, mask, slots


def choose(model, M, obs, me, deck, rng, temperature, sample):
    sel = obs["select"]
    opts = sel.get("option") or []
    kind, card, scal, mask, slots = encode(M, obs, me, deck)
    valid = [(i, slots[i]) for i in range(len(opts)) if slots[i] >= 0]
    if not valid:
        return None, None
    with torch.inference_mode():
        logits, _value = model(
            torch.as_tensor(kind[None].astype(np.int64)),
            torch.as_tensor(card[None].astype(np.int64)),
            torch.as_tensor(scal[None]), torch.as_tensor(mask[None]),
            torch.as_tensor([int(sel.get("context") or 0)]),
            torch.as_tensor([int(sel.get("type") or 0)]))
    pos = torch.as_tensor([p for _i, p in valid])
    logp = torch.log_softmax(logits[0, pos] / temperature, dim=-1)
    j = int(torch.multinomial(logp.exp(), 1).item()) if sample \
        else int(logp.argmax().item())
    action = [valid[j][0]]
    kmax = max(1, min(int(sel.get("maxCount", 1)), len(opts)))
    # The engine exposes factorized options but expects a combination.  Match
    # the deployed agent: sample the first item, then greedily fill the rest.
    if kmax > 1:
        for jj in torch.argsort(logp, descending=True).tolist():
            option_i = valid[jj][0]
            if option_i not in action:
                action.append(option_i)
            if len(action) >= kmax:
                break
    rec = Decision(kind, card, scal, mask,
                   int(sel.get("context") or 0),
                   int(sel.get("type") or 0), valid[j][1],
                   float(logp[j].item()))
    return action, rec


def play_game(policy, reference, M, our_deck, opponent_deck, our_seat,
              rng, temperature, decision_budget):
    from kaggle_environments import make

    records = []

    def ours(obs):
        if obs.get("select") is None:
            return list(our_deck)
        sel = obs["select"]
        opts = sel.get("option") or []
        if not opts:
            return []
        if len(opts) == 1:
            return [0]
        me = (obs.get("current") or {}).get("yourIndex", our_seat)
        action, rec = choose(
            policy, M, obs, me, our_deck, rng, temperature, sample=True)
        if rec is not None and len(records) < decision_budget:
            records.append(rec)
        return M._validate(action, sel) or [0]

    def opponent(obs):
        if obs.get("select") is None:
            return list(opponent_deck)
        sel = obs["select"]
        opts = sel.get("option") or []
        if not opts:
            return []
        if len(opts) == 1:
            return [0]
        me = (obs.get("current") or {}).get("yourIndex", 1 - our_seat)
        action, _rec = choose(
            reference, M, obs, me, opponent_deck, rng, 1.0, sample=False)
        return M._validate(action, sel) or [0]

    env = make("cabt")
    agents = [ours, opponent] if our_seat == 0 else [opponent, ours]
    try:
        env.run(agents)
    except Exception as exc:
        print(f"  rollout crashed: {exc!r}", flush=True)
        return [], None
    reward = float(env.state[our_seat].reward)
    return records, reward


def collect(policy, reference, M, our_deck, opponent_decks, groups,
            group_size, max_decisions, temperature, rng, schedule_offset=0):
    all_records = []
    group_stats = []
    policy.eval(); reference.eval()
    for group_i in range(groups):
        scheduled = group_i + schedule_offset
        opponent_name, opponent_deck = opponent_decks[
            scheduled % len(opponent_decks)]
        # Across iterations, cover both seats against each archetype instead
        # of permanently coupling an opponent to one seat.
        our_seat = (scheduled // len(opponent_decks) + group_i) % 2
        games = []
        rewards = []
        for _ in range(group_size):
            remaining = max_decisions - len(all_records) - sum(len(x) for x in games)
            if remaining <= 0:
                break
            recs, reward = play_game(
                policy, reference, M, our_deck, opponent_deck, our_seat,
                rng, temperature, remaining)
            if reward is not None:
                games.append(recs); rewards.append(reward)
        advantages = group_advantages(rewards)
        active = bool(np.any(np.abs(advantages) > 0))
        for recs, advantage in zip(games, advantages):
            for rec in recs:
                rec.advantage = float(advantage)
                all_records.append(rec)
                if len(all_records) >= max_decisions:
                    break
        group_stats.append({"opponent": opponent_name, "seat": our_seat,
                            "rewards": rewards, "active": active})
        print(f"  group {group_i + 1}/{groups} vs {opponent_name}, seat "
              f"{our_seat}: rewards={rewards}, decisions={len(all_records)}",
              flush=True)
        if len(all_records) >= max_decisions:
            break
    return all_records, group_stats


def batch_loss(policy, reference, batch, device, clip, kl_beta, entropy_beta,
               temperature):
    kind = torch.as_tensor(np.stack([r.kind for r in batch]).astype(np.int64), device=device)
    card = torch.as_tensor(np.stack([r.card for r in batch]).astype(np.int64), device=device)
    scal = torch.as_tensor(np.stack([r.scal for r in batch]), device=device)
    mask = torch.as_tensor(np.stack([r.mask for r in batch]), device=device)
    ctx = torch.as_tensor([r.ctx for r in batch], device=device)
    stype = torch.as_tensor([r.stype for r in batch], device=device)
    chosen = torch.as_tensor([r.chosen_pos for r in batch], device=device)
    old_logp = torch.as_tensor([r.old_logp for r in batch], device=device)
    advantage = torch.as_tensor([r.advantage for r in batch], device=device)

    logits, _value = policy(kind, card, scal, mask, ctx, stype)
    options = (kind == 3) & (mask > 0.5)
    masked = (logits / temperature).masked_fill(~options, -torch.inf)
    logp = torch.log_softmax(masked, dim=-1)
    arange = torch.arange(len(batch), device=device)
    new_logp = logp[arange, chosen]
    ratio = torch.exp(new_logp - old_logp)
    surrogate = torch.minimum(
        ratio * advantage,
        torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage)

    with torch.no_grad():
        ref_logits, _ref_value = reference(kind, card, scal, mask, ctx, stype)
        ref_logp = torch.log_softmax(ref_logits.masked_fill(~options, -torch.inf), dim=-1)
    # PPO ratios use the tempered behavior policy that generated the rollouts.
    # KL uses the untempered distributions that deployment actually consumes.
    deploy_logp = torch.log_softmax(logits.masked_fill(~options, -torch.inf), dim=-1)
    deploy_probs = torch.where(options, deploy_logp.exp(), torch.zeros_like(deploy_logp))
    safe_deploy = torch.where(options, deploy_logp, torch.zeros_like(deploy_logp))
    safe_ref = torch.where(options, ref_logp, torch.zeros_like(ref_logp))
    kl = (deploy_probs * (safe_deploy - safe_ref)).sum(-1).mean()
    behavior_probs = torch.where(options, logp.exp(), torch.zeros_like(logp))
    safe_logp = torch.where(options, logp, torch.zeros_like(logp))
    entropy = -(behavior_probs * safe_logp).sum(-1).mean()
    loss = -surrogate.mean() + kl_beta * kl - entropy_beta * entropy
    return loss, float(kl.detach()), float(entropy.detach()), float(ratio.mean().detach())


def checkpoint(model, path, metrics):
    model_module.export_npz(model, path)
    with open(path + ".metrics.json", "w", encoding="utf-8") as target:
        json.dump(metrics, target, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=os.path.join(
        ROOT, "track1_search", "variants", "richbc_search_21k", "model.npz"))
    ap.add_argument("--deck", default=os.path.join(HERE, "decks", "grimmsnarl.csv"))
    ap.add_argument("--opponent-meta", default=None,
                    help="optional Python file emitted by tools/mine_decks.py; "
                         "used for rollouts only, never copied into submission")
    ap.add_argument("--out", default=os.path.join(HERE, "model_grpo.npz"))
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-decisions", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--clip", type=float, default=0.15)
    ap.add_argument("--kl-beta", type=float, default=0.04)
    ap.add_argument("--entropy-beta", type=float, default=0.002)
    ap.add_argument("--temperature", type=float, default=1.05)
    ap.add_argument("--meta-weight-power", type=float, default=0.5,
                    help="temper replay popularity in the opponent schedule; "
                         "0 selects each exact list uniformly")
    ap.add_argument("--meta-schedule-slots", type=int, default=40)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--max-wall-minutes", type=float, default=120.0)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--seed", type=int, default=917)
    args = ap.parse_args()

    if args.group_size < 2 or args.group_size > 8:
        ap.error("--group-size must be in [2, 8]")
    if args.groups < 1 or args.groups > 32:
        ap.error("--groups must be in [1, 32]")
    if args.threads < 1 or args.threads > 10:
        ap.error("--threads must be in [1, 10]")
    if args.meta_weight_power < 0 or args.meta_weight_power > 1:
        ap.error("--meta-weight-power must be in [0, 1]")
    if args.meta_schedule_slots < 1 or args.meta_schedule_slots > 400:
        ap.error("--meta-schedule-slots must be in [1, 400]")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA was requested but torch cannot access it")
    resource_guard(args.max_decisions, args.out)
    torch.set_num_threads(args.threads)
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.75)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    our_deck = read_deck(args.deck)
    M = load_agent_module()
    policy = load_model(args.init, "cpu")
    reference = copy.deepcopy(policy).eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    # Include the two current deck recommendations even if an older main.py
    # does not yet contain them in its mined archetype library.
    opponent_decks = list(M.META_DECKS.items())
    opponent_weights = dict(getattr(M, "META_WEIGHT", {}) or {})
    if args.opponent_meta:
        spec = importlib.util.spec_from_file_location(
            "grpo_opponent_meta", args.opponent_meta)
        if spec is None or spec.loader is None:
            ap.error(f"cannot import --opponent-meta {args.opponent_meta}")
        meta_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(meta_module)
        candidates = getattr(meta_module, "META_DECKS", None)
        if not isinstance(candidates, dict) or not candidates:
            ap.error("--opponent-meta must define a non-empty META_DECKS dict")
        opponent_decks = []
        opponent_weights = dict(getattr(meta_module, "META_WEIGHT", {}) or {})
        for name, deck in candidates.items():
            if (not isinstance(deck, list) or len(deck) != 60 or
                    not all(isinstance(card, int) for card in deck)):
                ap.error(f"--opponent-meta deck {name!r} is not 60 integer IDs")
            opponent_decks.append((str(name), deck))
    for name in ("grimmsnarl", "garchomp"):
        deck = read_deck(os.path.join(HERE, "decks", name + ".csv"))
        if tuple(deck) not in {tuple(x[1]) for x in opponent_decks}:
            opponent_decks.append((name, deck))
            opponent_weights[name] = 1
    unique_opponents = len(opponent_decks)
    opponent_decks = weighted_opponent_schedule(
        opponent_decks, opponent_weights, args.meta_schedule_slots,
        args.meta_weight_power)

    print(f"device={device}; rollout=cpu; torch_threads={args.threads}; "
          f"RAM decision cap={args.max_decisions}; model={configure_model_from_npz(args.init)}")
    print(f"deck={os.path.basename(args.deck)}; opponents={unique_opponents}; "
          f"schedule_slots={len(opponent_decks)}; "
          f"meta_weight_power={args.meta_weight_power}; "
          f"games/iter<={args.groups * args.group_size}")

    started = time.monotonic()
    history = []
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr,
                                  weight_decay=1e-5)
    run_config = {
        key: (os.path.abspath(value)
              if value and key in ("init", "deck", "out", "opponent_meta")
              else value)
        for key, value in vars(args).items()
    }
    for iteration in range(1, args.iters + 1):
        if (time.monotonic() - started) / 60.0 >= args.max_wall_minutes:
            print("wall-time cap reached before next iteration")
            break
        policy.to("cpu")
        reference.to("cpu")
        records, groups = collect(
            policy, reference, M, our_deck, opponent_decks,
            args.groups, args.group_size, args.max_decisions,
            args.temperature, rng,
            schedule_offset=(iteration - 1) * args.groups)
        active = sum(g["active"] for g in groups)
        train_records = [r for r in records if abs(r.advantage) > 0]
        if not train_records:
            print("no within-group reward variance; checkpointing unchanged policy")
            metrics = {"iteration": iteration, "decisions": len(records),
                       "active_groups": active, "groups": groups}
            history.append(metrics)
            checkpoint(policy, args.out, {
                "config": run_config, "history": history})
            continue

        policy.to(device).train()
        reference.to(device)
        aggregate = {"loss": 0.0, "kl": 0.0, "entropy": 0.0, "ratio": 0.0}
        steps = 0
        order = list(range(len(train_records)))
        for _epoch in range(args.epochs):
            rng.shuffle(order)
            for start in range(0, len(order), args.batch):
                batch = [train_records[i] for i in order[start:start + args.batch]]
                loss, kl, entropy, ratio = batch_loss(
                    policy, reference, batch, device, args.clip,
                    args.kl_beta, args.entropy_beta, args.temperature)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.8)
                optimizer.step()
                aggregate["loss"] += float(loss.detach())
                aggregate["kl"] += kl
                aggregate["entropy"] += entropy
                aggregate["ratio"] += ratio
                steps += 1
        averages = {key: value / max(steps, 1)
                    for key, value in aggregate.items()}
        metrics = {"iteration": iteration, "decisions": len(records),
                   "trained_decisions": len(train_records),
                   "active_groups": active, "optimizer_steps": steps,
                   **averages, "groups": groups}
        history.append(metrics)
        checkpoint(policy, args.out, {
            "config": run_config, "history": history})
        stem, ext = os.path.splitext(args.out)
        checkpoint(policy, f"{stem}_iter{iteration:03d}{ext}", metrics)
        print(f"iter {iteration}: {metrics}", flush=True)

    checkpoint(policy.to("cpu"), args.out, {
        "config": run_config, "history": history})
    print(f"exported {args.out}")


if __name__ == "__main__":
    main()

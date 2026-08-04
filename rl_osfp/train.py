"""Resource-bounded PPO best responses wrapped in optimistic fictitious play.

This is deliberately replay-free: weights start random and every gradient is
computed from a trajectory produced by the current frozen behaviour policy.
Fresh public episodes choose the deck population only; they never provide an
action label.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import json
import math
import os
import random
import re
import shutil
import time

import numpy as np
import torch

from .network import ActorCritic, NetworkConfig, config_dict, export_npz
from .policy import Decision, batch_logprob_entropy
from .rollout import GameTask, RolloutPool


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class LeagueEntry:
    """A historical opponent, referenced by file so workers can load it.

    Holding paths rather than modules keeps the parent's memory flat as the
    league grows and lets a resumed run rebuild the league from disk.
    """
    period: int
    path: str
    candidate_win_rate: float = 0.5


def load_pool(path: str) -> tuple[list[dict], list[dict], dict]:
    with open(path, encoding="utf-8") as source:
        payload = json.load(source)
    learner = payload.get("learner_decks") or []
    field = payload.get("field_decks") or []
    for group_name, group in (("learner_decks", learner), ("field_decks", field)):
        if not group:
            raise ValueError(f"{group_name} is empty")
        for item in group:
            cards = item.get("cards")
            if not isinstance(cards, list) or len(cards) != 60 or not all(
                isinstance(card, int) for card in cards
            ):
                raise ValueError(f"invalid deck in {group_name}: {item.get('name')}")
    return learner, field, payload


def weighted_deck(rng: random.Random, decks: list[dict]) -> dict:
    weights = [math.sqrt(max(1.0, float(deck.get("appearances", 1)))) for deck in decks]
    return rng.choices(decks, weights=weights, k=1)[0]


def choose_league_opponent(rng: random.Random, league: list[LeagueEntry]) -> LeagueEntry:
    # Prioritized fictitious play: spend more games on snapshots the learner
    # has not yet beaten, but retain a floor so cyclic older policies survive.
    weights = [max(0.08, 1.0 - entry.candidate_win_rate) for entry in league]
    return rng.choices(league, weights=weights, k=1)[0]


def resource_guard(args: argparse.Namespace) -> None:
    if not (1 <= args.periods <= 2000):
        raise ValueError("--periods must be in [1, 2000]")
    if not (4 <= args.games_per_period <= 512):
        raise ValueError("--games-per-period must be in [4, 512]")
    if not (500 <= args.max_decisions <= 200_000):
        raise ValueError("--max-decisions must be in [500, 200000]")
    if not (1 <= args.threads <= 16):
        raise ValueError("--threads must be in [1, 16]")
    cores = os.cpu_count() or 1
    if not (1 <= args.workers <= max(1, cores)):
        raise ValueError(f"--workers must be in [1, {cores}]")
    # Oversubscription measurably slows rollouts: the pool contends with itself
    # and with the parent, so refuse a configuration that cannot fit.
    if args.workers * args.threads_per_worker > cores:
        raise ValueError(
            f"--workers x --threads-per-worker ({args.workers * args.threads_per_worker}) "
            f"exceeds {cores} cores"
        )
    available = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as source:
            fields = {
                line.split(":", 1)[0]: int(line.split()[1]) * 1024
                for line in source if ":" in line
            }
        available = fields.get("MemAvailable")
    except OSError:
        pass
    # Stored observations are ~18 KiB after the 64-option correction. Keep a
    # 4x margin for tensors, optimizer state, arena, and historical policies.
    required = args.max_decisions * 18 * 1024 * 4 + 3 * 1024**3
    if available is not None and required > available * 0.70:
        raise RuntimeError(
            f"configuration needs a conservative {required / 2**30:.1f} GiB; "
            f"only {available / 2**30:.1f} GiB is available"
        )
    os.makedirs(args.out_dir, exist_ok=True)
    if shutil.disk_usage(args.out_dir).free < 10 * 1024**3:
        raise RuntimeError("less than 10 GiB free on the checkpoint filesystem")


def gpu_telemetry(device: torch.device) -> dict[str, float] | None:
    """Peak VRAM and thermals for the period, so a long run stays auditable.

    The update is a short burst against a long CPU rollout, so the card should
    show a low duty cycle; a climbing peak here means the batch is too large.
    """
    if device.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info()
    telemetry = {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "device_used_mib": (total - free) / 2**20,
        "device_total_mib": total / 2**20,
    }
    for name, key in (("temperature", "temperature_c"), ("power", "power_w")):
        try:
            if name == "temperature":
                telemetry[key] = float(torch.cuda.temperature())
            else:
                telemetry[key] = float(torch.cuda.power_draw()) / 1000.0
        except Exception:
            pass  # not every driver/build exposes these
    torch.cuda.reset_peak_memory_stats()
    return telemetry


def tensor_batch(batch: list[Decision], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "kind": torch.as_tensor(
            np.stack([x.kind for x in batch]).astype(np.int64), device=device
        ),
        "card": torch.as_tensor(
            np.stack([x.card for x in batch]).astype(np.int64), device=device
        ),
        "scal": torch.as_tensor(np.stack([x.scal for x in batch]), device=device),
        "mask": torch.as_tensor(np.stack([x.mask for x in batch]), device=device),
        "ctx": torch.as_tensor([x.ctx for x in batch], device=device),
        "stype": torch.as_tensor([x.stype for x in batch], device=device),
        "selected_slots": torch.as_tensor(
            np.stack([x.selected_slots for x in batch]).astype(np.int64), device=device
        ),
        "selected_count": torch.as_tensor(
            [x.selected_count for x in batch], device=device
        ),
        "low_count": torch.as_tensor([x.low_count for x in batch], device=device),
        "high_count": torch.as_tensor([x.high_count for x in batch], device=device),
        "old_logp": torch.as_tensor([x.old_logp for x in batch], device=device),
        "returns": torch.as_tensor([x.outcome for x in batch], device=device),
        "advantage": torch.as_tensor([x.advantage for x in batch], device=device),
    }


def ppo_update(
    model: ActorCritic,
    decisions: list[Decision],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rng: random.Random,
    args: argparse.Namespace,
) -> dict[str, float]:
    raw_advantage = np.asarray(
        [decision.outcome - decision.old_value for decision in decisions],
        dtype=np.float32,
    )
    mean, std = float(raw_advantage.mean()), float(raw_advantage.std())
    normalized = (raw_advantage - mean) / (std + 1e-6)
    for decision, advantage in zip(decisions, normalized):
        decision.advantage = float(advantage)

    model.to(device).train()
    order = list(range(len(decisions)))
    totals = {"loss": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0,
              "approx_kl": 0.0, "clip_fraction": 0.0}
    steps = 0
    stopped_early = False
    for _epoch in range(args.epochs):
        rng.shuffle(order)
        epoch_kl = []
        for start in range(0, len(order), args.batch):
            batch = [decisions[i] for i in order[start : start + args.batch]]
            data = tensor_batch(batch, device)
            option, count, value = model(
                data["kind"], data["card"], data["scal"], data["mask"],
                data["ctx"], data["stype"],
            )
            logp, entropy = batch_logprob_entropy(
                option, count, data["kind"], data["mask"],
                data["selected_slots"], data["selected_count"],
                data["low_count"], data["high_count"],
            )
            log_ratio = (logp - data["old_logp"]).clamp(-20.0, 20.0)
            ratio = log_ratio.exp()
            unclipped = ratio * data["advantage"]
            clipped = ratio.clamp(1.0 - args.clip, 1.0 + args.clip) * data["advantage"]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (value - data["returns"]).square().mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_mean

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            approx_kl = float((data["old_logp"] - logp).mean().detach())
            clip_fraction = float(((ratio - 1.0).abs() > args.clip).float().mean().detach())
            values = {
                "loss": float(loss.detach()), "policy": float(policy_loss.detach()),
                "value": float(value_loss.detach()), "entropy": float(entropy_mean.detach()),
                "approx_kl": approx_kl, "clip_fraction": clip_fraction,
            }
            for key, item in values.items():
                totals[key] += item
            epoch_kl.append(approx_kl)
            steps += 1
            # Checking only at epoch end is too coarse: at a large decision
            # budget an epoch is dozens of minibatches, so the policy can move
            # far past the trust region before the guard ever looks. Stop on
            # the first minibatch that exceeds it.
            if approx_kl > args.target_kl:
                stopped_early = True
                break
        if stopped_early:
            break
        if epoch_kl and sum(epoch_kl) / len(epoch_kl) > args.target_kl:
            stopped_early = True
            break
    model.to("cpu").eval()
    return {
        **{key: value / max(steps, 1) for key, value in totals.items()},
        "optimizer_steps": steps,
        "advantage_mean_raw": mean,
        "advantage_std_raw": std,
        "early_stop": float(stopped_early),
    }


def evaluation_tasks(
    candidate_path: str,
    league: list[LeagueEntry],
    learner_decks: list[dict],
    field_decks: list[dict],
    games: int,
) -> list[GameTask]:
    """Every league member's evaluation games, flattened for one parallel pass."""
    tasks = []
    for entry in league:
        for game_index in range(games):
            candidate_seat = game_index % 2
            own = learner_decks[game_index % len(learner_decks)]["cards"]
            other = field_decks[game_index % len(field_decks)]["cards"]
            if candidate_seat == 0:
                models, decks = (candidate_path, entry.path), (own, other)
            else:
                models, decks = (entry.path, candidate_path), (other, own)
            tasks.append(GameTask(
                left_model=models[0], right_model=models[1],
                left_deck=decks[0], right_deck=decks[1],
                record_seats=(), sample=False,
                seed=hash((entry.period, game_index)) & 0xFFFFFFFF,
                meta={"history_period": entry.period, "candidate_seat": candidate_seat},
            ))
    return tasks


def fold_evaluations(outcomes, league: list[LeagueEntry]) -> list[dict]:
    """Reduce flat evaluation outcomes back into one row per league member."""
    totals = {
        entry.period: {"wins": 0, "losses": 0, "draws": 0, "errors": 0, "invalid_actions": 0}
        for entry in league
    }
    for outcome in outcomes:
        record = totals[outcome.meta["history_period"]]
        seat = outcome.meta["candidate_seat"]
        record["invalid_actions"] += outcome.invalid_actions
        if outcome.error:
            record["errors"] += 1
        elif outcome.winner == seat:
            record["wins"] += 1
        elif outcome.winner == 1 - seat:
            record["losses"] += 1
        else:
            record["draws"] += 1
    rows = []
    for entry in league:
        record = totals[entry.period]
        decided = record["wins"] + record["losses"]
        record["win_rate"] = record["wins"] / decided if decided else 0.0
        entry.candidate_win_rate = float(record["win_rate"])
        rows.append({"history_period": entry.period, **record})
    return rows


def rebuild_league(out_dir: str, max_history: int) -> list[LeagueEntry]:
    """Reconstruct the league from archived snapshots when resuming."""
    entries = []
    for path in sorted(glob.glob(os.path.join(out_dir, "league_*.npz"))):
        match = re.search(r"league_(\d+)\.npz$", path)
        if match:
            entries.append(LeagueEntry(int(match.group(1)), path))
    while len(entries) > max_history:
        del entries[1]  # keep the initial anchor, drop the oldest after it
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=os.path.join(ROOT, "data", "fresh", "deck_pool.json"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "rl_osfp", "run"))
    parser.add_argument("--periods", type=int, default=12)
    parser.add_argument("--games-per-period", type=int, default=24)
    parser.add_argument("--eval-games", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=14_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=384)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=0.8)
    parser.add_argument("--target-kl", type=float, default=0.04)
    parser.add_argument("--self-play-prob", type=float, default=0.60)
    parser.add_argument("--archive-threshold", type=float, default=0.55)
    parser.add_argument("--archive-max-wait", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=8,
                        help="torch threads in the parent, used for the PPO update")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="rollout worker processes; the engine is process-global "
                             "so parallelism cannot be threads")
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--max-recorded-per-seat", type=int, default=1200,
                        help="per-game, per-seat cap on recorded decisions")
    parser.add_argument("--resume", action="store_true",
                        help="continue from training_state.pt in --out-dir")
    parser.add_argument("--max-wall-minutes", type=float, default=480.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    resource_guard(args)
    if not (0.0 < args.self_play_prob <= 1.0):
        parser.error("--self-play-prob must be in (0, 1]")

    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device_name = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.72)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    learner_decks, field_decks, pool_payload = load_pool(args.pool)
    model = ActorCritic(NetworkConfig()).to("cpu").eval()  # random initialization
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    league: list[LeagueEntry] = []
    metrics: list[dict] = []
    since_archive = 0
    first_period = 1
    work_dir = os.path.join(args.out_dir, "_work")
    os.makedirs(work_dir, exist_ok=True)

    if args.resume:
        state_path = os.path.join(args.out_dir, "training_state.pt")
        if not os.path.isfile(state_path):
            raise SystemExit(f"--resume needs {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first_period = int(state["period"]) + 1
        league = rebuild_league(args.out_dir, args.max_history)
        metrics_path = os.path.join(args.out_dir, "metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path, encoding="utf-8") as source:
                metrics = json.load(source).get("periods") or []
        print(
            f"resumed at period {first_period} with league "
            f"{[entry.period for entry in league]}",
            flush=True,
        )
    started = time.monotonic()

    run_config = dict(vars(args))
    run_config["pool"] = os.path.abspath(args.pool)
    run_config["out_dir"] = os.path.abspath(args.out_dir)
    run_config["network"] = config_dict(model)
    run_config["discount"] = 1.0
    run_config["initialization"] = "random"
    run_config["replay_action_labels"] = False
    print(
        f"device={device}; rollout_device=cpu; threads={args.threads}; "
        f"network={run_config['network']}; learner_decks={len(learner_decks)}; "
        f"field_decks={len(field_decks)}",
        flush=True,
    )

    # One pool for the whole run: a spawn start plus an Arena build costs
    # seconds, and the context manager guarantees workers die with the parent.
    with RolloutPool(args.workers, args.threads_per_worker) as rollout_pool:
        for period in range(1, args.periods + 1):
            elapsed_minutes = (time.monotonic() - started) / 60.0
            if elapsed_minutes >= args.max_wall_minutes:
                print("wall-time cap reached before next learning period", flush=True)
                break
            # Workers load the behaviour policy from disk; a period-unique name
            # keeps their (path, mtime) cache honest across the whole run.
            behaviour_path = os.path.join(work_dir, f"behaviour_{period:03d}.npz")
            export_npz(model, behaviour_path)
            rollout_started = time.monotonic()
            tasks = []
            for game_index in range(args.games_per_period):
                use_self = not league or rng.random() < args.self_play_prob
                if use_self:
                    opponent_path = behaviour_path
                    opponent_label = "current"
                    record_seats = (0, 1)
                else:
                    entry = choose_league_opponent(rng, league)
                    opponent_path = entry.path
                    opponent_label = f"history_{entry.period:03d}"
                    record_seats = (game_index % 2,)
                learner_seat = game_index % 2
                own_deck = weighted_deck(rng, learner_decks)
                opponent_deck = weighted_deck(rng, field_decks)
                if learner_seat == 0:
                    models = (behaviour_path, opponent_path)
                    decks = (own_deck["cards"], opponent_deck["cards"])
                else:
                    models = (opponent_path, behaviour_path)
                    decks = (opponent_deck["cards"], own_deck["cards"])
                tasks.append(GameTask(
                    left_model=models[0], right_model=models[1],
                    left_deck=decks[0], right_deck=decks[1],
                    record_seats=record_seats, sample=True,
                    temperature=args.temperature,
                    max_recorded=args.max_recorded_per_seat,
                    seed=rng.randrange(2**31),
                    meta={"game": game_index, "learner_seat": learner_seat,
                          "opponent": opponent_label, "learner_deck": own_deck["name"],
                          "opponent_deck": opponent_deck["name"]},
                ))

            decisions: list[Decision] = []
            game_rows = []
            for outcome in rollout_pool.run(tasks, chunksize=1):
                seats = tuple(outcome.decisions)
                if outcome.winner in (0, 1):
                    for seat in seats:
                        reward = 1.0 if outcome.winner == seat else -1.0
                        for decision in outcome.decisions[seat]:
                            decision.outcome = reward
                            decisions.append(decision)
                game_rows.append({
                    **outcome.meta, "winner": outcome.winner,
                    "steps": outcome.engine_steps,
                    "invalid_actions": outcome.invalid_actions, "error": outcome.error,
                })
            # Tasks are dispatched as a batch, so trim rather than stop early.
            if len(decisions) > args.max_decisions:
                rng.shuffle(decisions)
                decisions = decisions[: args.max_decisions]
            rollout_minutes = (time.monotonic() - rollout_started) / 60.0
            print(
                f"period {period:03d} rollout: {len(game_rows)} games, "
                f"{len(decisions)} decisions, {rollout_minutes:.2f} min, "
                f"invalid={sum(row['invalid_actions'] for row in game_rows)}",
                flush=True,
            )

            if not decisions:
                raise RuntimeError("no valid on-policy decisions were collected")
            update = ppo_update(model, decisions, optimizer, device, rng, args)

            candidate_path = os.path.join(work_dir, f"candidate_{period:03d}.npz")
            export_npz(model, candidate_path)
            evaluations = []
            all_pass = bool(league)
            if league:
                outcomes = list(rollout_pool.run(
                    evaluation_tasks(candidate_path, league, learner_decks,
                                     field_decks, args.eval_games),
                    chunksize=1,
                ))
                evaluations = fold_evaluations(outcomes, league)
                all_pass = all(
                    entry.candidate_win_rate > args.archive_threshold for entry in league
                )
            since_archive += 1
            archived = not league or all_pass or since_archive >= args.archive_max_wait
            if archived:
                league_path = os.path.join(args.out_dir, f"league_{period:03d}.npz")
                export_npz(model, league_path)
                league.append(LeagueEntry(period, league_path))
                if len(league) > args.max_history:
                    # Preserve the initial anchor and the most recent opponents.
                    del league[1]
                since_archive = 0

            period_metrics = {
                "period": period,
                "elapsed_minutes": (time.monotonic() - started) / 60.0,
                "games": game_rows,
                "decisions": len(decisions),
                "wins": sum(row["winner"] == row["learner_seat"] for row in game_rows),
                "losses": sum(row["winner"] == 1 - row["learner_seat"] for row in game_rows),
                "errors": sum(bool(row["error"]) for row in game_rows),
                "invalid_actions": sum(row["invalid_actions"] for row in game_rows),
                "update": update,
                "evaluations": evaluations,
                "archived": archived,
                "history_periods": [entry.period for entry in league],
                "gpu": gpu_telemetry(device),
            }
            metrics.append(period_metrics)
            export_npz(model, os.path.join(args.out_dir, "model_latest.npz"))
            export_npz(model, os.path.join(args.out_dir, f"model_period_{period:03d}.npz"))
            torch.save(
                {"period": period, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "config": run_config},
                os.path.join(args.out_dir, "training_state.pt"),
            )
            with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as target:
                json.dump({"config": run_config, "pool_source": pool_payload.get("source"),
                           "periods": metrics}, target, indent=2)
            print(
                f"period {period:02d} update={update} archived={archived} "
                f"history={[entry.period for entry in league]}",
                flush=True,
            )

    if not metrics:
        raise RuntimeError("training ended before completing a learning period")
    export_npz(model, os.path.join(args.out_dir, "model_final.npz"))
    print(f"finished {len(metrics)} periods; exported model_final.npz", flush=True)


if __name__ == "__main__":
    main()

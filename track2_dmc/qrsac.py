"""QR-SAC for PTCG, built to track2_dmc/QRSAC_SPEC.md.

Every design choice below traces to a measured failure from 2026-07-21; see the
spec for the evidence. Summary of what this does differently from the earlier
attempts:

  * Q-learning core, not policy gradient      (DG collapsed, DMC did not)
  * SEPARATE quantile head, freshly initialised, never warm started from the
    behaviour cloned policy logits            (warm start hurt: 10.8 vs 1.5 loss)
  * entropy TARGET with auto tuned alpha      (a fixed 0.01 bonus did not work)
  * distributional critic over quantiles      (returns are bimodal at +/- 1)
  * epsilon greedy self play for exploration  (logged data cannot rank actions)
  * anchor regulariser on untaken options     (they are otherwise unsupervised)
  * deployed as ROOT PRIORS only              (any other placement burns sims)

The trunk IS warm started from the behaviour cloned model, because the encoder
and the policy head keep their semantics. Only the new quantile head starts
fresh, which is the distinction the hybrid run got wrong.

Usage:
  py track2_dmc/qrsac.py --games 10
  py track2_dmc/qrsac.py --games 10 --scratch     # no trunk warm start
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AGENT = os.path.join(ROOT, "track1_search", "agent")
sys.path.insert(0, AGENT)
sys.path.insert(0, os.path.join(ROOT, "track1_search", "train"))
sys.path.insert(0, os.path.join(ROOT, "track4_policygrad"))

import nn_features as NF                                      # noqa: E402
import model as M1                                            # noqa: E402
from model import TCGNet, export_npz                          # noqa: E402
from train_dg import load_agent_module, import_weights        # noqa: E402

N_QUANTILES = 16


class QRSACNet(nn.Module):
    """Shared trunk, policy head, and a separate quantile Q head.

    The quantile head is a NEW module: the spec forbids reusing the policy head
    for regression targets, because cross entropy logits and returns in [-1, 1]
    live on different scales and warm starting across them measurably hurt.
    """

    def __init__(self, base: TCGNet):
        super().__init__()
        self.base = base
        self.q_head = nn.Linear(M1.D_MODEL, N_QUANTILES)
        nn.init.zeros_(self.q_head.bias)
        nn.init.normal_(self.q_head.weight, std=0.01)

    def trunk(self, kind, card, scal, mask, ctx, styp):
        b = self.base
        x = (b.card_emb(card) + b.kind_emb(kind) + b.scal_proj(scal))
        g = b.ctx_emb(ctx) + b.stype_emb(styp)
        x = torch.cat([x[:, :1, :] + g[:, None, :], x[:, 1:, :]], dim=1)
        x = x * mask[:, :, None]
        attn = (1.0 - mask)[:, None, None, :] * -1e9
        for blk in b.blocks:
            x = blk(x, attn)
        return b.ln_f(x)

    def forward(self, kind, card, scal, mask, ctx, styp):
        h = self.trunk(kind, card, scal, mask, ctx, styp)
        logits = self.base.pol_head(h).squeeze(-1)
        logits = logits.masked_fill(mask < 0.5, -1e9)
        quant = torch.tanh(self.q_head(h))            # [B, SEQ, N_QUANTILES]
        v = torch.tanh(self.base.val_fc2(
            torch.nn.functional.gelu(self.base.val_fc1(h[:, 0, :]),
                                     approximate="tanh"))).squeeze(-1)
        return logits, quant, v


def quantile_huber(pred, target, taus, kappa=1.0):
    """QR loss. pred [B, Nq], target [B], taus [Nq]."""
    d = target[:, None] - pred                                   # [B, Nq]
    absd = d.abs()
    huber = torch.where(absd <= kappa, 0.5 * d ** 2, kappa * (absd - 0.5 * kappa))
    return (torch.abs(taus[None, :] - (d.detach() < 0).float()) * huber).mean()


def option_token_mask(kind, mask):
    """Return the selectable-token mask used by collection and deployment.

    ``mask`` is an attention mask: it also contains the global, board, and hand
    tokens.  Treating it as an action mask makes the actor solve a different
    categorical problem from the one used by :func:`act` and ``_net_scores``.
    """
    return (kind == 3) & (mask > 0.5)


def option_policy_stats(logits, qmean, option_mask):
    """Policy, entropy, and expected Q over selectable options only."""
    if not bool(option_mask.any(dim=-1).all()):
        raise ValueError("every training state must contain an option token")
    option_logits = logits.masked_fill(~option_mask, -torch.inf)
    logp = torch.log_softmax(option_logits, dim=-1)
    # Avoid 0 * -inf in entropy while keeping non-option probabilities exact 0.
    safe_logp = torch.where(option_mask, logp, torch.zeros_like(logp))
    p = torch.where(option_mask, logp.exp(), torch.zeros_like(logp))
    entropy = -(p * safe_logp).sum(-1)
    expected_q = (p * qmean).sum(-1)
    return p, logp, entropy, expected_q


def option_anchor_loss(qmean, value, taken_pos, option_mask):
    """Anchor only untaken selectable options, with the correct denominator."""
    untaken = option_mask.clone()
    untaken[torch.arange(len(taken_pos), device=taken_pos.device), taken_pos] = False
    sqerr = (qmean - value.detach()[:, None]).square()
    return (sqerr * untaken).sum() / untaken.sum().clamp(min=1)


def alpha_tuning_loss(log_alpha, entropy, target_entropy):
    """Tune entropy temperature in log space without a near-zero dead zone.

    Differentiating ``exp(log_alpha)`` here makes the update proportional to
    alpha itself.  Once alpha gets small it then cannot recover when entropy
    drops below target.  The standard log-space objective keeps that corrective
    gradient finite.
    """
    gap = (target_entropy - entropy).detach()
    return -(log_alpha * gap).mean()


def save_model(model, path):
    """Write deployable actor weights plus the training-only quantile head."""
    export_npz(model.base, path)
    torch.save({"q_head": model.q_head.state_dict()},
               path.replace(".npz", "_qhead.pt"))


def iteration_path(path, iteration):
    stem, ext = os.path.splitext(path)
    return f"{stem}_iter{iteration:03d}{ext or '.npz'}"


def load_bc_replay(path, max_samples, seed):
    """Load a bounded rehearsal set of leaderboard/search policy targets.

    Logged games cannot supervise Q for actions that were not chosen, but their
    search visit distributions remain high-value actor targets.  Sampling while
    each compressed shard is open keeps the retained set small enough for a
    laptop even when the full replay corpus is much larger.
    """
    files = sorted(glob.glob(os.path.join(path, "*.npz"))) \
        if os.path.isdir(path) else [path]
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(f"no BC replay shards found at {path}")
    keys = ("kind", "card", "scal", "mask", "ctx", "stype", "pi")
    parts = {k: [] for k in keys}
    rng = np.random.default_rng(seed)
    per_file = max(1, (max_samples + len(files) - 1) // len(files))
    for path_i in files:
        with np.load(path_i) as shard:
            n = len(shard["pi"])
            take = min(n, per_file)
            chosen = rng.choice(n, size=take, replace=False)
            for key in keys:
                parts[key].append(shard[key][chosen])
    data = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    if len(data["pi"]) > max_samples:
        chosen = rng.choice(len(data["pi"]), size=max_samples, replace=False)
        data = {k: v[chosen] for k, v in data.items()}
    return data


def bc_actor_loss(model, data, indices, device):
    """Option-only cross entropy on leaderboard/search policy targets."""
    kind = torch.as_tensor(data["kind"][indices].astype(np.int64), device=device)
    card = torch.as_tensor(data["card"][indices].astype(np.int64), device=device)
    scal = torch.as_tensor(data["scal"][indices], device=device)
    mask = torch.as_tensor(data["mask"][indices], device=device)
    ctx = torch.as_tensor(data["ctx"][indices].astype(np.int64), device=device)
    styp = torch.as_tensor(data["stype"][indices].astype(np.int64), device=device)
    target = torch.as_tensor(data["pi"][indices], device=device)
    logits, _quant, _value = model(kind, card, scal, mask, ctx, styp)
    options = option_token_mask(kind, mask)
    option_logits = logits.masked_fill(~options, -torch.inf)
    logp = torch.log_softmax(option_logits, dim=-1)
    safe_logp = torch.where(options, logp, torch.zeros_like(logp))
    target = torch.where(options, target, torch.zeros_like(target))
    target = target / target.sum(-1, keepdim=True).clamp(min=1e-8)
    return -(target * safe_logp).sum(-1).mean()


def act(model, M, obs, me, rng, epsilon, alpha_log):
    """Sample an action from the entropy regularised policy."""
    sel = obs["select"]
    opts = sel.get("option") or []
    scores = [M._option_score(o, sel) for o in opts]
    kind, card, scal, mask, slot = NF.encode(
        {"current": obs["current"], "select": sel}, me, M.CARD, M.ATTACK, scores)
    device = next(model.parameters()).device
    with torch.no_grad():
        logits, quant, _v = model(
            torch.as_tensor(kind[None].astype(np.int64), device=device),
            torch.as_tensor(card[None].astype(np.int64), device=device),
            torch.as_tensor(scal[None], device=device),
            torch.as_tensor(mask[None], device=device),
            torch.as_tensor([int(sel.get("context") or 0)], device=device),
            torch.as_tensor([int(sel.get("type") or 0)], device=device))
    valid = [(i, slot[i]) for i in range(len(opts)) if slot[i] >= 0]
    if not valid:
        return None, None
    pos = torch.tensor([p for _i, p in valid])
    lp = torch.log_softmax(logits[0][pos], dim=-1)
    if epsilon > 0 and rng.random() < epsilon:
        j = rng.randrange(len(valid))
    else:
        j = int(torch.multinomial(lp.exp(), 1).item())
    kmax = max(1, min(sel.get("maxCount", 1), len(opts)))
    action = [valid[j][0]]
    if kmax > 1:
        for jj in np.argsort(-lp.detach().cpu().numpy()):
            if len(action) >= kmax:
                break
            oi = valid[int(jj)][0]
            if oi not in action:
                action.append(oi)
    feats = (kind, card, scal, mask, int(sel.get("context") or 0),
             int(sel.get("type") or 0), valid[j][1], np.array(pos))
    return action, feats


def rollout(model, M, n_games, rng, epsilon, alpha_log):
    from kaggle_environments import make
    recs, wins, played = [], 0, 0
    names = list(M.META_DECKS.keys())
    for g in range(n_games):
        buf = []

        def make_seat(deck_ids):
            def seat(obs):
                if obs.get("select") is None:
                    return list(deck_ids)
                sel = obs["select"]
                opts = sel.get("option") or []
                if not opts:
                    return []
                if len(opts) == 1:
                    return [0]
                me = (obs.get("current") or {}).get("yourIndex", 0)
                a, f = act(model, M, obs, me, rng, epsilon, alpha_log)
                if a is None:
                    return M._validate(M._heuristic_action(sel, rng), sel) or [0]
                buf.append((f, me))
                return M._validate(a, sel) or [0]
            return seat

        env = make("cabt")
        a0, a1 = make_seat(M.MY_DECK), make_seat(M.META_DECKS[names[g % len(names)]])
        try:
            if g % 2 == 0:
                env.run([a0, a1]); mine = 0
            else:
                env.run([a1, a0]); mine = 1
        except Exception as exc:
            print(f"  game {g} crashed: {exc!r}", flush=True)
            continue
        r0 = env.state[0].reward
        if r0 is None:                      # a seat played an illegal action
            continue
        played += 1
        wins += int((r0 == 1) == (mine == 0))
        for f, me in buf:
            recs.append((f, float(r0) if me == 0 else -float(r0)))
    return recs, wins, played


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=90)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=2,
                    help="passes over the buffer per iteration. The winning "
                         "scratch-DMC run did full 2-epoch sweeps (~1875 "
                         "steps/iter); the original QR-SAC run did one capped "
                         "pass (~400 steps/iter) and was ~8x under-trained on "
                         "gradient updates as a result.")
    ap.add_argument("--save-every", type=int, default=5,
                    help="export an intermediate model every N iterations; "
                         "zero disables periodic checkpoints")
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--anchor", type=float, default=0.3)
    ap.add_argument("--target-entropy", type=float, default=0.7,
                    help="fraction of ln(n_actions) to hold")
    ap.add_argument("--bc-data",
                    default=os.path.join(ROOT, "track1_search", "train", "data_bc"),
                    help="leaderboard/search replay shard or directory used "
                         "for actor rehearsal; empty string disables it")
    ap.add_argument("--bc-weight", type=float, default=0.1,
                    help="weight of option-only supervised actor rehearsal; "
                         "zero disables it")
    ap.add_argument("--bc-samples", type=int, default=20000,
                    help="maximum replay decisions retained in memory")
    ap.add_argument("--bc-batch", type=int, default=64)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                    help="training device; auto uses CUDA when available")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--init", default=os.path.join(ROOT, "track1_search", "train", "model_v4.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "model_qrsac.npz"))
    a = ap.parse_args()

    if a.device == "cuda" and not torch.cuda.is_available():
        ap.error("--device cuda requested but CUDA is unavailable")
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    print(f"device: {device}")

    M = load_agent_module()
    bc_data = None
    if a.bc_weight > 0 and a.bc_data:
        bc_data = load_bc_replay(a.bc_data, a.bc_samples, a.seed)
        print(f"leaderboard actor rehearsal: {len(bc_data['pi'])} decisions, "
              f"weight {a.bc_weight:g}")
    base = TCGNet()
    if not a.scratch and os.path.exists(a.init):
        n = import_weights(base, a.init)
        print(f"trunk + policy head warm started: {n} tensors")
    else:
        print("full scratch initialisation")
    model = QRSACNet(base).to(device)
    print(f"quantile head is FRESH ({N_QUANTILES} quantiles) - never warm "
          f"started from policy logits (see QRSAC_SPEC.md item 2)")

    taus = torch.tensor([(i + 0.5) / N_QUANTILES for i in range(N_QUANTILES)],
                        dtype=torch.float32, device=device)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    opt_alpha = torch.optim.Adam([log_alpha], lr=a.lr)
    rng = random.Random(a.seed)
    buffer = []

    for it in range(a.iters):
        t0 = time.perf_counter()
        model.eval()
        eps = a.epsilon * (1 - it / max(a.iters - 1, 1)) + 0.02
        recs, wins, played = rollout(model, M, a.games, rng, eps, log_alpha)
        buffer.extend(recs)
        if len(buffer) > 60000:
            buffer = buffer[-60000:]
        if not buffer:
            print("no data"); break

        model.train()
        idx = list(range(len(buffer)))
        rng.shuffle(idx)
        # Match scratch-DMC's optimisation budget: multiple epochs over the
        # buffer, capped by --max-steps so wall-clock per iter stays bounded.
        idx = (idx * a.epochs)[:a.max_steps * a.batch]
        tot_c = tot_a = tot_bc = 0.0
        nb = 0
        for s in range(0, len(idx), a.batch):
            chunk = [buffer[i] for i in idx[s:s + a.batch]]
            if len(chunk) < 2:
                continue
            kind = torch.as_tensor(
                np.stack([c[0][0] for c in chunk]).astype(np.int64), device=device)
            card = torch.as_tensor(
                np.stack([c[0][1] for c in chunk]).astype(np.int64), device=device)
            scal = torch.as_tensor(
                np.stack([c[0][2] for c in chunk]), device=device)
            mask = torch.as_tensor(
                np.stack([c[0][3] for c in chunk]), device=device)
            ctx = torch.as_tensor([c[0][4] for c in chunk], device=device)
            styp = torch.as_tensor([c[0][5] for c in chunk], device=device)
            pos = torch.as_tensor([c[0][6] for c in chunk], device=device)
            z = torch.as_tensor(
                [c[1] for c in chunk], dtype=torch.float32, device=device)
            ar = torch.arange(len(chunk), device=device)

            logits, quant, v = model(kind, card, scal, mask, ctx, styp)

            # ---- critic: quantile regression toward the Monte Carlo return
            q_taken = quant[ar, pos]                            # [B, Nq]
            closs = quantile_huber(q_taken, z, taus)
            option_mask = option_token_mask(kind, mask)

            # Anchor untaken OPTIONS so ranking stays defined where exploration
            # has not reached (spec item 6).  The attention mask also includes
            # state tokens; anchoring those was the original over-flattening bug.
            if a.anchor > 0:
                qmean = quant.mean(-1)                          # [B, SEQ]
                closs = closs + a.anchor * option_anchor_loss(
                    qmean, v, pos, option_mask)
            closs = closs + 0.5 * ((v - z) ** 2).mean()

            # ---- actor: maximise Q minus alpha-weighted entropy penalty.
            # Collection and deployment normalize over option tokens only, so
            # actor training and its entropy target must use that same support.
            alpha = log_alpha.exp().detach()
            qmean_d = quant.mean(-1).detach()
            _p, _logp, ent, expected_q = option_policy_stats(
                logits, qmean_d, option_mask)
            aloss = (-alpha * ent - expected_q).mean()

            bcloss = torch.zeros((), device=device)
            if bc_data is not None:
                bc_idx = np.array(
                    [rng.randrange(len(bc_data["pi"])) for _ in range(a.bc_batch)])
                bcloss = bc_actor_loss(model, bc_data, bc_idx, device)

            loss = closs + aloss + a.bc_weight * bcloss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # ---- alpha: hold entropy at a target fraction of ln(n_actions)
            n_act = option_mask.sum(-1).float().clamp(min=2)
            target = a.target_entropy * torch.log(n_act)
            alpha_loss = alpha_tuning_loss(log_alpha, ent, target)
            opt_alpha.zero_grad(); alpha_loss.backward(); opt_alpha.step()
            with torch.no_grad():
                log_alpha.clamp_(min=-10.0, max=5.0)

            tot_c += float(closs.item())
            tot_a += float(aloss.item())
            tot_bc += float(bcloss.item())
            nb += 1

        print(f"iter {it+1}/{a.iters}: steps {nb}, buffer {len(buffer)}, "
              f"winrate {100*wins/max(played,1):.0f}%, eps {eps:.2f}, "
              f"critic {tot_c/max(nb,1):.4f}, actor {tot_a/max(nb,1):.4f}, "
              f"bc {tot_bc/max(nb,1):.4f}, "
              f"alpha {float(log_alpha.detach().exp()):.3f}, "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
        if a.save_every > 0 and (it + 1) % a.save_every == 0:
            # Preserve snapshots for post-training selection; the best search
            # prior need not be the final optimization iterate.  Also refresh
            # the main path so an interrupted run always leaves a usable model.
            save_model(model, iteration_path(a.out, it + 1))
            save_model(model, a.out)
            print(f"  checkpointed {iteration_path(a.out, it + 1)}", flush=True)

    # Export in the deployed format. The agent reads the POLICY head for root
    # priors, so exporting the base is exactly what search consumes.
    save_model(model, a.out)
    print(f"exported {a.out} (+ quantile head alongside)")
    print("\nCHECKPOINT:\n  py tools/ab_test.py <variant_dir> track1_search/agent 30")


if __name__ == "__main__":
    main()

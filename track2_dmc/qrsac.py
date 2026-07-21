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
  py track2_dmc/qrsac.py --iters 60 --games 10
  py track2_dmc/qrsac.py --iters 60 --games 10 --scratch     # no trunk warm start
"""
from __future__ import annotations

import argparse
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


def act(model, M, obs, me, rng, epsilon, alpha_log):
    """Sample an action from the entropy regularised policy."""
    sel = obs["select"]
    opts = sel.get("option") or []
    scores = [M._option_score(o, sel) for o in opts]
    kind, card, scal, mask, slot = NF.encode(
        {"current": obs["current"], "select": sel}, me, M.CARD, M.ATTACK, scores)
    with torch.no_grad():
        logits, quant, _v = model(
            torch.tensor(kind[None].astype(np.int64)),
            torch.tensor(card[None].astype(np.int64)),
            torch.tensor(scal[None]), torch.tensor(mask[None]),
            torch.tensor(np.array([int(sel.get("context") or 0)])),
            torch.tensor(np.array([int(sel.get("type") or 0)])))
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
        for jj in np.argsort(-lp.detach().numpy()):
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
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--anchor", type=float, default=0.3)
    ap.add_argument("--target-entropy", type=float, default=0.7,
                    help="fraction of ln(n_actions) to hold")
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--init", default=os.path.join(ROOT, "track1_search", "train", "model_v4.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "model_qrsac.npz"))
    a = ap.parse_args()

    M = load_agent_module()
    base = TCGNet()
    if not a.scratch and os.path.exists(a.init):
        n = import_weights(base, a.init)
        print(f"trunk + policy head warm started: {n} tensors")
    else:
        print("full scratch initialisation")
    model = QRSACNet(base)
    print(f"quantile head is FRESH ({N_QUANTILES} quantiles) - never warm "
          f"started from policy logits (see QRSAC_SPEC.md item 2)")

    taus = torch.tensor([(i + 0.5) / N_QUANTILES for i in range(N_QUANTILES)],
                        dtype=torch.float32)
    log_alpha = torch.zeros(1, requires_grad=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    opt_alpha = torch.optim.Adam([log_alpha], lr=a.lr)
    rng = random.Random(0)
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
        idx = idx[:a.max_steps * a.batch]
        tot_c = tot_a = 0.0
        nb = 0
        for s in range(0, len(idx), a.batch):
            chunk = [buffer[i] for i in idx[s:s + a.batch]]
            if len(chunk) < 2:
                continue
            kind = torch.tensor(np.stack([c[0][0] for c in chunk]).astype(np.int64))
            card = torch.tensor(np.stack([c[0][1] for c in chunk]).astype(np.int64))
            scal = torch.tensor(np.stack([c[0][2] for c in chunk]))
            mask = torch.tensor(np.stack([c[0][3] for c in chunk]))
            ctx = torch.tensor(np.array([c[0][4] for c in chunk]))
            styp = torch.tensor(np.array([c[0][5] for c in chunk]))
            pos = torch.tensor(np.array([c[0][6] for c in chunk]))
            z = torch.tensor(np.array([c[1] for c in chunk], dtype=np.float32))
            ar = torch.arange(len(chunk))

            logits, quant, v = model(kind, card, scal, mask, ctx, styp)

            # ---- critic: quantile regression toward the Monte Carlo return
            q_taken = quant[ar, pos]                            # [B, Nq]
            closs = quantile_huber(q_taken, z, taus)
            # anchor untaken options so ranking stays defined where exploration
            # has not reached (spec item 6)
            if a.anchor > 0:
                qmean = quant.mean(-1)                          # [B, SEQ]
                valid = mask > 0.5
                diff = (qmean - v.detach()[:, None]) ** 2 * valid
                diff[ar, pos] = 0.0
                closs = closs + a.anchor * (diff.sum() / valid.sum().clamp(min=1))
            closs = closs + 0.5 * ((v - z) ** 2).mean()

            # ---- actor: maximise Q minus alpha-weighted entropy penalty
            alpha = log_alpha.exp().detach()
            logp_all = torch.log_softmax(logits, dim=-1)
            p_all = logp_all.exp() * (mask > 0.5).float()
            qmean_d = quant.mean(-1).detach()
            ent = -(p_all * logp_all.clamp(min=-20)).sum(-1)
            aloss = (alpha * -ent - (p_all * qmean_d).sum(-1)).mean()

            loss = closs + aloss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # ---- alpha: hold entropy at a target fraction of ln(n_actions)
            n_act = (mask > 0.5).float().sum(-1).clamp(min=2)
            target = a.target_entropy * torch.log(n_act)
            alpha_loss = -(log_alpha.exp() * (target - ent.detach()).mean())
            opt_alpha.zero_grad(); alpha_loss.backward(); opt_alpha.step()

            tot_c += float(closs.item()); tot_a += float(aloss.item()); nb += 1

        print(f"iter {it+1}/{a.iters}: steps {nb}, buffer {len(buffer)}, "
              f"winrate {100*wins/max(played,1):.0f}%, eps {eps:.2f}, "
              f"critic {tot_c/max(nb,1):.4f}, actor {tot_a/max(nb,1):.4f}, "
              f"alpha {float(log_alpha.exp()):.3f}, "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    # Export in the deployed format. The agent reads the POLICY head for root
    # priors, so exporting the base is exactly what search consumes.
    export_npz(model.base, a.out)
    torch.save({"q_head": model.q_head.state_dict()},
               a.out.replace(".npz", "_qhead.pt"))
    print(f"exported {a.out} (+ quantile head alongside)")
    print("\nCHECKPOINT:\n  py tools/ab_test.py <variant_dir> track1_search/agent 30")


if __name__ == "__main__":
    main()

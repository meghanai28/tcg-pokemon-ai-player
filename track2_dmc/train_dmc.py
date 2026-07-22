"""Track 2: Deep Monte Carlo (DouZero style).

https://arxiv.org/pdf/2106.06135

No tree search anywhere. Learn Q(state, action) by regressing toward the Monte
Carlo return of self play games, act by argmax over Q of the legal options.
DouZero's argument for preferring this to tree search is that sampling lets you
spend the whole compute budget generating data instead of splitting it between
lookahead and inference, which is precisely the failure we measured in Track 1
(an engine step costs about 0.05 ms, a network call about 3 ms, so each network
evaluation inside search burns roughly 60 simulations).

Action representation reuses `nn_features.py` option tokens, which play the
role of DouZero's card matrices: each legal option already becomes a feature
vector, so the network generalises across option combinations it never saw.

The policy head of the shared network is reinterpreted here as a per-option Q
value, trained with MSE against the realised return rather than cross entropy
against a target distribution.

Usage:
  py track2_dmc/train_dmc.py --iters 10 --games 12
  py track2_dmc/train_dmc.py --iters 10 --games 12 --scratch   # no warm start
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "track1_search", "agent"))
sys.path.insert(0, os.path.join(ROOT, "track1_search", "train"))
sys.path.insert(0, os.path.join(ROOT, "track4_policygrad"))

import nn_features as NF                      # noqa: E402
from model import TCGNet, export_npz          # noqa: E402
from train_dg import load_agent_module, import_weights   # noqa: E402


def q_action(model, M, obs, me, rng, epsilon=0.0, use_heuristic_order=True):
    """Score every legal option with Q and take the argmax (epsilon greedy)."""
    sel = obs["select"]
    opts = sel.get("option") or []
    scores = [M._option_score(o, sel) for o in opts] if use_heuristic_order else None
    kind, card, scal, mask, slot = NF.encode(
        {"current": obs["current"], "select": sel}, me, M.CARD, M.ATTACK, scores)
    with torch.no_grad():
        q, _v = model(torch.tensor(kind[None].astype(np.int64)),
                      torch.tensor(card[None].astype(np.int64)),
                      torch.tensor(scal[None]), torch.tensor(mask[None]),
                      torch.tensor(np.array([int(sel.get("context") or 0)])),
                      torch.tensor(np.array([int(sel.get("type") or 0)])))
    valid = [(i, slot[i]) for i in range(len(opts)) if slot[i] >= 0]
    if not valid:
        return None, None
    qs = q[0][torch.tensor([p for _i, p in valid])]
    if epsilon > 0 and rng.random() < epsilon:
        j = rng.randrange(len(valid))
    else:
        j = int(qs.argmax().item())
    kmax = max(1, min(sel.get("maxCount", 1), len(opts)))
    action = [valid[j][0]]
    if kmax > 1:
        for jj in np.argsort(-qs.detach().numpy()):
            if len(action) >= kmax:
                break
            oi = valid[int(jj)][0]
            if oi not in action:
                action.append(oi)
    feats = (kind, card, scal, mask, int(sel.get("context") or 0),
             int(sel.get("type") or 0), valid[j][1])
    return action, feats


def rollout(model, M, n_games, rng, epsilon):
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
                act, feats = q_action(model, M, obs, me, rng, epsilon)
                if act is None:
                    return M._validate(M._heuristic_action(sel, rng), sel) or [0]
                buf.append((feats, me))
                return M._validate(act, sel) or [0]
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
        if r0 is None:
            # a seat errored out; the episode has no usable outcome label
            print(f"  game {g} had no reward (statuses "
                  f"{[env.state[i].status for i in range(2)]})", flush=True)
            continue
        played += 1
        wins += int((r0 == 1) == (mine == 0))
        for feats, me in buf:
            recs.append((feats, float(r0) if me == 0 else -float(r0)))
    return recs, wins, played


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2, help="passes over buffer")
    ap.add_argument("--max-steps", type=int, default=400,
                    help="cap on gradient steps per iteration")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--scratch", action="store_true", help="no warm start")
    ap.add_argument("--anchor", type=float, default=0.3,
                    help="pull untaken option values toward the state value")
    ap.add_argument("--init", default=os.path.join(ROOT, "track1_search", "train", "model_v4.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "model_dmc.npz"))
    a = ap.parse_args()

    M = load_agent_module()
    model = TCGNet()
    if not a.scratch and os.path.exists(a.init):
        n = import_weights(model, a.init)
        print(f"warm start: {n} tensors from {os.path.basename(a.init)}")
    else:
        print("training Q from scratch (random init)")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    rng = random.Random(0)
    buffer = []                                   # all-history replay buffer

    for it in range(a.iters):
        t0 = time.perf_counter()
        model.eval()
        eps = a.epsilon * (1 - it / max(a.iters - 1, 1)) + 0.02
        recs, wins, played = rollout(model, M, a.games, rng, eps)
        buffer.extend(recs)
        if len(buffer) > 60000:
            buffer = buffer[-60000:]
        if not buffer:
            print("no data"); break

        model.train()
        idx = list(range(len(buffer)))
        rng.shuffle(idx)
        # Bounded work per iteration. Sweeping the whole buffer made each
        # iteration grow without limit (64s -> 217s by iteration 17, heading
        # for ~10 min once the buffer hit its cap). Cap the gradient steps so
        # wall clock per iteration stays flat while still doing plenty of
        # updates, which was the original problem.
        idx = (idx * a.epochs)[:a.max_steps * a.batch]
        tot = 0.0
        nb = 0
        for s in range(0, len(idx), a.batch):
            chunk = [buffer[i] for i in idx[s:s + a.batch]]
            if not chunk:
                continue
            kind = torch.tensor(np.stack([c[0][0] for c in chunk]).astype(np.int64))
            card = torch.tensor(np.stack([c[0][1] for c in chunk]).astype(np.int64))
            scal = torch.tensor(np.stack([c[0][2] for c in chunk]))
            mask = torch.tensor(np.stack([c[0][3] for c in chunk]))
            ctx = torch.tensor(np.array([c[0][4] for c in chunk]))
            styp = torch.tensor(np.array([c[0][5] for c in chunk]))
            pos = torch.tensor(np.array([c[0][6] for c in chunk]))
            z = torch.tensor(np.array([c[1] for c in chunk], dtype=np.float32))
            q, v = model(kind, card, scal, mask, ctx, styp)
            q_taken = q[torch.arange(len(chunk)), pos]
            # Deep Monte Carlo: regress Q(s,a) toward the realised return.
            loss = ((q_taken - z) ** 2).mean() + 0.5 * ((v - z) ** 2).mean()
            if a.anchor > 0:
                # Anchor UNTAKEN options toward the state value. Only the taken
                # action gets a return label, so every other option's output is
                # otherwise unconstrained and free to drift; ranking by argmax
                # over unconstrained outputs is meaningless. Training offline on
                # 136k logged decisions failed exactly this way (10.6 percent
                # top-1, and 37.5 percent in the A/B) because logged data never
                # shows the alternatives. This keeps them calibrated so the
                # ordering stays sane where exploration has not reached.
                # ``mask`` is the transformer attention mask and also includes
                # global, board, and hand tokens.  Only kind-3 tokens are legal
                # options; pulling state-token outputs toward V both wastes most
                # of the regularizer and overwhelms the option ranking signal.
                valid = (kind == 3) & (mask > 0.5)
                anchor_t = v.detach()[:, None].expand_as(q)
                diff = (q - anchor_t) ** 2 * valid
                diff[torch.arange(len(chunk)), pos] = 0.0
                untaken_count = valid.sum() - len(chunk)
                loss = loss + a.anchor * (
                    diff.sum() / untaken_count.clamp(min=1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        print(f"iter {it+1}/{a.iters}: steps {nb}, buffer {len(buffer)}, winrate "
              f"{100*wins/max(played,1):.0f}%, eps {eps:.2f}, loss {tot/max(nb,1):.4f}, "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    export_npz(model, a.out)
    print(f"exported {a.out}")
    print("\nCHECKPOINT (the bar is the HEURISTIC, not Track 1 search):")
    print(f"  py track4_policygrad/eval_policy.py --a {a.out} --heuristic --games 40")


if __name__ == "__main__":
    main()

"""Track 4: Delightful Gradient policy gradient fine tuning.

https://arxiv.org/abs/2603.14608

Standard policy gradient weights each term by the advantage. DG additionally
gates it by a sigmoid of "delight" = advantage * surprisal, where surprisal is
-log pi(a|s). The intent is that gradient budget stops being spent on contexts
the policy already handles confidently and correctly, and moves to the ones it
does not.

That matters here because our decision distribution is extremely lopsided.
Divergence mining against 1050+ Elo players measured ACTIVATE at 95 percent
agreement over 1,106 decisions and DRAW_COUNT at 96 percent over 78, while MAIN
sits at 38 percent over 18,226. A vanilla policy gradient keeps paying for the
solved contexts.

This plays with the RAW POLICY, no search, which is what makes it on-policy.
It is also therefore a much weaker player than the Track 1 agent, so the
checkpoint below compares against the un-fine-tuned policy, not against search.

Usage:
  py track4_policygrad/train_dg.py --iters 3 --games 12 --out track4_policygrad/model_dg.npz
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AGENT = os.path.join(ROOT, "track1_search", "agent")
T1TRAIN = os.path.join(ROOT, "track1_search", "train")
sys.path.insert(0, AGENT)
sys.path.insert(0, T1TRAIN)

import nn_features as NF          # noqa: E402
from model import TCGNet          # noqa: E402


def load_agent_module():
    spec = importlib.util.spec_from_file_location("dg_main", os.path.join(AGENT, "main.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.MY_DECK = m._load_deck()
    m._load_engine()
    m._load_card_db()
    m._ENGINE_TRIED = True
    return m


def import_weights(model, npz_path):
    """Load exported .npz weights back into the torch model."""
    z = np.load(npz_path)
    sd = model.state_dict()
    loaded = 0
    for k in sd:
        if k in z.files and tuple(z[k].shape) == tuple(sd[k].shape):
            sd[k] = torch.tensor(z[k])
            loaded += 1
    model.load_state_dict(sd)
    return loaded


_NO_HEURISTIC = False


def encode_decision(M, obs, me):
    sel = obs["select"]
    opts = sel.get("option") or []
    scores = None if _NO_HEURISTIC else [M._option_score(o, sel) for o in opts]
    kind, card, scal, mask, slot = NF.encode(
        {"current": obs["current"], "select": sel}, me, M.CARD, M.ATTACK, scores)
    return kind, card, scal, mask, slot, sel, opts


def policy_action(model, M, obs, me, rng, sample=True):
    """Pick an action with the raw network. Returns (action, logprob, feats)."""
    kind, card, scal, mask, slot, sel, opts = encode_decision(M, obs, me)
    with torch.no_grad():
        logits, _v = model(torch.tensor(kind[None].astype(np.int64)),
                           torch.tensor(card[None].astype(np.int64)),
                           torch.tensor(scal[None]), torch.tensor(mask[None]),
                           torch.tensor(np.array([int(sel.get("context") or 0)])),
                           torch.tensor(np.array([int(sel.get("type") or 0)])))
    # restrict to real option tokens
    valid = [(i, slot[i]) for i in range(len(opts)) if slot[i] >= 0]
    if not valid:
        return None, None, None
    idx = torch.tensor([p for _i, p in valid])
    lp = torch.log_softmax(logits[0][idx], dim=-1)
    if sample:
        j = int(torch.multinomial(lp.exp(), 1).item())
    else:
        j = int(lp.argmax().item())
    opt_i = valid[j][0]
    kmax = max(1, min(sel.get("maxCount", 1), len(opts)))
    action = [opt_i]
    if kmax > 1:   # fill remaining slots greedily, credit only the first pick
        order = list(np.argsort(-lp.detach().numpy()))
        for jj in order:
            if len(action) >= kmax:
                break
            oi = valid[int(jj)][0]
            if oi not in action:
                action.append(oi)
    return action, float(lp[j].item()), (kind, card, scal, mask,
                                         int(sel.get("context") or 0),
                                         int(sel.get("type") or 0), j, idx.numpy())


def rollout_games(model, M, n_games, rng, opponent="meta"):
    """Self play with the raw policy. Returns per decision records."""
    from kaggle_environments import make
    recs = []
    names = list(M.META_DECKS.keys())
    wins = 0
    for g in range(n_games):
        buf = []

        def make_seat(deck_ids):
            """Single-argument closure.

            kaggle_environments inspects agent arity and passes `config` as a
            second positional argument, so a two-parameter agent silently
            receives the config where the deck was expected.
            """
            def seat(obs):
                if obs.get("select") is None:
                    return list(deck_ids)
                sel = obs["select"]
                opts = sel.get("option") or []
                if len(opts) == 0:
                    return []
                if len(opts) == 1:
                    return [0]
                me = (obs.get("current") or {}).get("yourIndex", 0)
                act, lp, feats = policy_action(model, M, obs, me, rng)
                if act is None:
                    if _NO_HEURISTIC:
                        k = max(1, min(sel.get("maxCount", 1), len(opts)))
                        return rng.sample(range(len(opts)), k)
                    return M._validate(M._heuristic_action(sel, rng), sel) or [0]
                if feats is not None:
                    buf.append((feats, lp, me))
                return M._validate(act, sel) or [0]
            return seat

        opp_deck = M.META_DECKS[names[g % len(names)]]
        agent = make_seat(M.MY_DECK)
        opp = make_seat(opp_deck)

        env = make("cabt")
        try:
            if g % 2 == 0:
                env.run([agent, opp]); mine = 0
            else:
                env.run([opp, agent]); mine = 1
        except Exception as exc:
            print(f"  game {g} crashed: {exc!r}", flush=True)
            continue
        r0 = env.state[0].reward
        wins += int((r0 == 1) == (mine == 0))
        for feats, lp, me in buf:
            z = float(r0) if me == 0 else -float(r0)
            recs.append((feats, lp, z))
    return recs, wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=os.path.join(T1TRAIN, "model_v4.npz"))
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=4, help="passes per rollout")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--plain-pg", action="store_true", help="ablate DG gating")
    ap.add_argument("--scratch", action="store_true",
                    help="pure RL: random init, no BC warm start")
    ap.add_argument("--no-heuristic", action="store_true",
                    help="never consult the hand written scorer, even for "
                         "option ordering or fallback (uniform-random fallback "
                         "instead, so the agent is purely the learned policy)")
    ap.add_argument("--out", default=os.path.join(HERE, "model_dg.npz"))
    a = ap.parse_args()

    M = load_agent_module()
    model = TCGNet()
    if a.scratch:
        print("PURE RL: random initialisation, no behaviour cloning warm start")
    else:
        n = import_weights(model, a.init)
        print(f"warm start from {os.path.basename(a.init)}: {n} tensors loaded")
    if a.no_heuristic:
        # Strip every heuristic dependency from the acting path. encode() uses
        # the scorer only to decide which options survive truncation, so feed
        # it None; and replace the illegal-action fallback with uniform random.
        global _NO_HEURISTIC
        _NO_HEURISTIC = True
        print("PURE RL: heuristic scorer disabled (ordering + fallback)")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    rng = random.Random(0)

    for it in range(a.iters):
        t0 = time.perf_counter()
        model.eval()
        recs, wins = rollout_games(model, M, a.games, rng)
        if not recs:
            print("no data collected"); break
        model.train()
        # PROPER OPTIMISATION. The original version accumulated every
        # decision's loss and called opt.step() ONCE per iteration, i.e. 25
        # gradient updates for an entire run, which is why the loss was flat.
        # Now: batched forwards, minibatch updates, several epochs per rollout.
        gates = []
        tot, nsteps = 0.0, 0
        order = list(range(len(recs)))
        for _ep in range(a.epochs):
            rng.shuffle(order)
            for s0 in range(0, len(order), a.batch):
                chunk = [recs[i] for i in order[s0:s0 + a.batch]]
                if not chunk:
                    continue
                kind = torch.tensor(np.stack([c[0][0] for c in chunk]).astype(np.int64))
                card = torch.tensor(np.stack([c[0][1] for c in chunk]).astype(np.int64))
                scal = torch.tensor(np.stack([c[0][2] for c in chunk]))
                mask = torch.tensor(np.stack([c[0][3] for c in chunk]))
                ctx = torch.tensor(np.array([c[0][4] for c in chunk]))
                styp = torch.tensor(np.array([c[0][5] for c in chunk]))
                zs = torch.tensor(np.array([c[2] for c in chunk], dtype=np.float32))
                logits, v = model(kind, card, scal, mask, ctx, styp)
                losses = []
                for bi, c in enumerate(chunk):
                    j, idx = c[0][6], c[0][7]
                    lp_all = torch.log_softmax(logits[bi][torch.tensor(idx)], dim=-1)
                    lp = lp_all[j]
                    adv = float(zs[bi]) - float(v[bi].item())
                    if a.plain_pg:
                        w = 1.0
                    else:
                        # Delightful Gradient: sigmoid(advantage * surprisal)
                        w = float(torch.sigmoid(torch.tensor(adv) * (-lp.detach())))
                    gates.append(w)
                    ent = -(lp_all.exp() * lp_all).sum()
                    losses.append(-(w * adv) * lp - a.entropy * ent)
                loss = torch.stack(losses).mean() + 0.5 * ((v - zs) ** 2).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss.item()); nsteps += 1
        loss_val = tot / max(nsteps, 1)
        print(f"iter {it+1}/{a.iters}: {len(recs)} decisions, selfplay winrate "
              f"{100*wins/max(a.games,1):.0f}%, loss {loss_val:.4f}, "
              f"steps {nsteps}, gate {np.mean(gates):.3f}, "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    from model import export_npz
    export_npz(model, a.out)
    print(f"exported {a.out}")
    print("\nCHECKPOINT: compare raw policies head to head, e.g.\n"
          "  py track4_policygrad/eval_policy.py --a track4_policygrad/model_dg.npz "
          f"--b {a.init} --games 40")


if __name__ == "__main__":
    main()

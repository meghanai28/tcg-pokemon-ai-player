"""Measure, and then refit, the frozen shell's static leaf evaluator.

Three separate results say the leaf evaluator is what limits this search, not
the amount of search:

  * giving the same search 2.7x the think time lost 8-15
  * a learned value head lost 1-19 and then 0-6 against heuristic leaves
  * a lethality term large enough to dominate dropped the ladder 65% to 40%

All three are the same failure. Deeper search converges harder onto whatever the
evaluator believes, so a biased evaluator gets worse, not better, with more
simulations. Every other lever this project pulled (priors, decks, search time)
left the evaluator's hand-picked constants untouched.

`_evaluate` is a weighted sum of about a dozen features whose weights were
guessed. This script scores real ladder positions with those features, labels
each with the game's actual outcome, and asks two questions:

  1. how well does the shipped evaluator predict the winner?
  2. does refitting only the weights, keeping the exact functional form and
     therefore the exact throughput, predict better?

Keeping the form matters. Heuristic leaves ran ~7,500 simulations where neural
leaves managed ~220, and throughput is what wins here, so the replacement has to
stay a dozen multiply-adds.

Usage:
    py tools/fit_evaluator.py --episodes 600
    py tools/fit_evaluator.py --episodes 2000 --out data/evaluator_fit.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAY_DIR = os.path.join(ROOT, "data", "fresh", "replays")

# Feature order is the order the weights appear in `_evaluate`, so a fitted
# vector can be read straight back into the shell.
FEATURES = [
    "prize_diff",        # (opp_prizes_left - my_prizes_left) / 6      shipped 0.40
    "mon_count_diff",    # board size difference                       shipped 0.05
    "mon_hp_diff",       # sum(hp / maxHp) difference                  shipped 0.06
    "mon_energy_diff",   # sum(len(energies)) difference               shipped 0.015
    "mon_atk_diff",      # sum(min(best_attack, 300) / 10) difference  shipped 0.001
    "mon_stage1_diff",   # stage-1 count difference                    shipped 0.02
    "mon_stage2_diff",   # stage-2 count difference                    shipped 0.04
    "my_hand",           # min(hand, 16)                               shipped 0.010
    "opp_hand",          # min(opp hand, 16)                           shipped -0.006
    "my_deck_out",       # my deckCount == 0                           shipped -0.35
    "my_deck_low",       # my deckCount <= 2                           shipped -0.10
    "opp_deck_out",      # opp deckCount == 0                          shipped 0.35
    "status_diff",       # opponent status flags minus ours            shipped 0.02
]

SHIPPED = np.array([0.40, 0.05, 0.06, 0.015, 0.001, 0.02, 0.04,
                    0.010, -0.006, -0.35, -0.10, 0.35, 0.02])

STATUS_FLAGS = ("poisoned", "burned", "asleep", "paralyzed", "confused")


def load_shell(agent_dir: str) -> dict:
    """Exec the frozen shell the way the Kaggle runner does, and keep its globals.

    We need the shell's own `_card` / `_max_attack_damage`, which are backed by
    the native card database, so that features are computed from exactly the
    tables the agent sees at play time rather than a reimplementation.
    """
    main_py = os.path.join(agent_dir, "main.py")
    with open(main_py, encoding="utf-8") as handle:
        source = handle.read()
    namespace: dict = {"__name__": "shell_under_test", "__file__": main_py}
    sys.path.insert(0, agent_dir)
    cwd = os.getcwd()
    try:
        os.chdir(agent_dir)
        exec(compile(source, main_py, "exec"), namespace)
        # The shell only boots the engine on its first agent() call, and the card
        # tables come from the native library, so do that explicitly here.  With
        # CARD empty the stage and attack features silently read as zero, which
        # looks like a working run rather than a broken one.
        namespace["_load_engine"]()
        namespace["_load_card_db"]()
    finally:
        os.chdir(cwd)
        sys.path.remove(agent_dir)
    if not namespace.get("CARD"):
        raise RuntimeError("card database is empty; stage and attack features "
                           "would silently be zero")
    return namespace


def features_for(state: dict, shell: dict, me: int) -> np.ndarray | None:
    cur = state.get("current") or state
    if cur.get("result", -1) >= 0:
        return None
    players = cur.get("players")
    if not players or len(players) < 2:
        return None
    mine, opp = players[me], players[1 - me]
    card = shell["_card"]
    max_dmg = shell["_max_attack_damage"]

    def board(player):
        return [m for m in list(player.get("active") or []) +
                list(player.get("bench") or []) if m]

    my_board, opp_board = board(mine), board(opp)

    def agg(mons):
        count = len(mons)
        hp = sum((m.get("hp", 0) / (m.get("maxHp") or 1)) for m in mons)
        energy = sum(len(m.get("energies") or []) for m in mons)
        atk = sum(min(max_dmg(m.get("id")), 300) / 10.0 for m in mons)
        s1 = sum(1 for m in mons if card(m.get("id")).get("stage1"))
        s2 = sum(1 for m in mons if card(m.get("id")).get("stage2"))
        return count, hp, energy, atk, s1, s2

    mc, mh, men, mat, ms1, ms2 = agg(my_board)
    oc, oh, oen, oat, os1, os2 = agg(opp_board)

    my_prize = len(mine.get("prize") or [])
    opp_prize = len(opp.get("prize") or [])
    my_hand = len(mine.get("hand") or []) if mine.get("hand") is not None \
        else mine.get("handCount", 0)

    status = sum(1 for f in STATUS_FLAGS if opp.get(f)) - \
        sum(1 for f in STATUS_FLAGS if mine.get(f))

    return np.array([
        (opp_prize - my_prize) / 6.0,
        mc - oc, mh - oh, men - oen, mat - oat, ms1 - os1, ms2 - os2,
        min(my_hand, 16), min(opp.get("handCount", 0), 16),
        1.0 if mine.get("deckCount", 1) == 0 else 0.0,
        1.0 if 0 < mine.get("deckCount", 99) <= 2 else 0.0,
        1.0 if opp.get("deckCount", 1) == 0 else 0.0,
        float(status),
    ], dtype=np.float64)


def harvest(shell: dict, n_episodes: int, step_stride: int, seed: int):
    """Sample states from real ladder episodes, labelled by who actually won."""
    rows, labels, turns = [], [], []
    rng = random.Random(seed)
    archives = sorted(f for f in os.listdir(REPLAY_DIR) if f.endswith(".zip"))
    per_zip = max(1, n_episodes // max(1, len(archives)))
    for name in archives:
        with zipfile.ZipFile(os.path.join(REPLAY_DIR, name)) as zf:
            entries = zf.namelist()
            rng.shuffle(entries)
            taken = 0
            for entry in entries:
                if taken >= per_zip:
                    break
                try:
                    episode = json.loads(zf.read(entry))
                except Exception:
                    continue
                rewards = episode.get("rewards") or []
                if len(rewards) < 2 or rewards[0] is None or rewards[0] == rewards[1]:
                    continue
                steps = episode.get("steps") or []
                taken += 1
                for i in range(1, len(steps), step_stride):
                    try:
                        obs = steps[i][0]["observation"]
                    except (KeyError, IndexError, TypeError):
                        continue
                    turn = (obs.get("current") or {}).get("turn", 0)
                    # Harvest BOTH seats.  `_evaluate(state, me)` is called with
                    # whichever seat we happen to hold, so weights fitted from
                    # seat 0 alone would be applied to seat 1 on half of all
                    # games.  Taking both makes the fit seat-symmetric by
                    # construction and doubles the sample for free.
                    for seat in (0, 1):
                        feat = features_for(obs, shell, seat)
                        if feat is None:
                            continue
                        rows.append(feat)
                        labels.append(1.0 if rewards[seat] > rewards[1 - seat] else 0.0)
                        turns.append(turn)
    return np.array(rows), np.array(labels), np.array(turns)


def logistic_fit(x: np.ndarray, y: np.ndarray, l2: float = 1.0,
                 iters: int = 400) -> tuple[np.ndarray, float]:
    """Plain Newton-free logistic regression, standardised then unstandardised.

    Standardising matters: the raw features differ by three orders of magnitude
    (a prize difference is O(1), a hand size is O(10)), and an unstandardised
    gradient descent would spend all its steps on the largest one.
    """
    mu, sigma = x.mean(0), x.std(0)
    sigma[sigma < 1e-9] = 1.0
    xs = (x - mu) / sigma
    xs = np.hstack([xs, np.ones((len(xs), 1))])
    w = np.zeros(xs.shape[1])
    lr = 0.5
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-xs @ w))
        grad = xs.T @ (p - y) / len(y)
        grad[:-1] += l2 * w[:-1] / len(y)
        w -= lr * grad
    weights = w[:-1] / sigma
    bias = w[-1] - float(np.sum(w[:-1] * mu / sigma))
    return weights, bias


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels.sum(), (1 - labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--step-stride", type=int, default=5)
    parser.add_argument("--agent-dir", default=None,
                        help="a packaged agent directory; defaults to unpacking "
                             "the champion anchor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(ROOT, "data", "evaluator_fit.json"))
    args = parser.parse_args()

    agent_dir = args.agent_dir
    tmp = None
    if agent_dir is None:
        import tarfile, tempfile
        tmp = tempfile.mkdtemp(prefix="evalfit_")
        with tarfile.open(os.path.join(ROOT, "harness", "anchors",
                                       "grpo_tech_grim_972_912_811.tar.gz")) as tf:
            tf.extractall(tmp)
        agent_dir = tmp

    shell = load_shell(agent_dir)
    print(f"engine loaded, cards={len(shell.get('CARD', {}))}")

    x, y, turns = harvest(shell, args.episodes, args.step_stride, args.seed)
    print(f"states={len(x):,} from ~{args.episodes} episodes, "
          f"win rate={y.mean():.3f}")
    if len(x) < 1000:
        sys.exit("too few states harvested to fit anything")

    # split by episode order rather than at random: consecutive states inside one
    # game are highly correlated, and each state now appears twice (once per
    # seat), so a random split would put a state's mirror image in the other half
    # and flatter the fit badly
    cut = int(0.75 * len(x))
    cut -= cut % 2   # keep seat pairs together
    xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]

    shipped_test = xte @ SHIPPED
    fitted_w, fitted_b = logistic_fit(xtr, ytr)
    fitted_test = xte @ fitted_w + fitted_b

    a_ship, a_fit = auc(shipped_test, yte), auc(fitted_test, yte)
    print()
    print(f"{'evaluator':<26} {'AUC':>7}  {'accuracy':>9}")
    print(f"{'shipped (hand-picked)':<26} {a_ship:>7.4f}  "
          f"{np.mean((shipped_test > 0) == (yte > 0.5)):>9.4f}")
    print(f"{'refit (same features)':<26} {a_fit:>7.4f}  "
          f"{np.mean((fitted_test > 0) == (yte > 0.5)):>9.4f}")

    print(f"\n{'feature':<18} {'shipped':>9} {'refit':>9}  {'ratio':>7}")
    for name, old, new in zip(FEATURES, SHIPPED, fitted_w):
        ratio = new / old if abs(old) > 1e-9 else float("inf")
        print(f"{name:<18} {old:>9.4f} {new:>9.4f}  {ratio:>7.2f}")
    print(f"{'bias':<18} {0.0:>9.4f} {fitted_b:>9.4f}")

    # by game phase, because a leaf evaluator is used mid-game, and late states
    # are trivially predictable in a way that flatters any evaluator
    print(f"\n{'turn range':<14} {'n':>7} {'shipped':>9} {'refit':>9}")
    for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 999)):
        m = (turns[cut:] >= lo) & (turns[cut:] < hi)
        if m.sum() > 200:
            print(f"{f'{lo}-{hi}':<14} {int(m.sum()):>7} "
                  f"{auc(shipped_test[m], yte[m]):>9.4f} "
                  f"{auc(fitted_test[m], yte[m]):>9.4f}")

    # Saturation is the trap that killed the learned value head: 46% of its
    # outputs sat above 0.95, so PUCT could not tell sibling moves apart and
    # committed to a line instead of verifying it.  A calibrated probability
    # squashed into [-1, 1] can do exactly the same thing, so pick a temperature
    # that keeps the output in a range where it still discriminates.
    print(f"\n{'temperature':<13} {'|v|>0.95':>9} {'|v|>0.80':>9} {'std':>7}  "
          f"{'AUC':>7}")
    chosen = None
    for temp in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        v = 2.0 / (1.0 + np.exp(-(fitted_test) / temp)) - 1.0
        sat95 = float(np.mean(np.abs(v) > 0.95))
        sat80 = float(np.mean(np.abs(v) > 0.80))
        marker = ""
        if chosen is None and sat95 < 0.05:
            chosen, marker = temp, "   <- chosen"
        print(f"{temp:<13.1f} {sat95:>9.3f} {sat80:>9.3f} {v.std():>7.3f}  "
              f"{auc(v, yte):>7.4f}{marker}")
    if chosen is None:
        chosen = 8.0
    shipped_v = np.clip(shipped_test, -0.97, 0.97)
    print(f"\nshipped evaluator for comparison: |v|>0.95 = "
          f"{float(np.mean(np.abs(shipped_v) > 0.95)):.3f}, "
          f"std = {shipped_v.std():.3f}")

    payload = {
        "states": int(len(x)),
        "episodes_requested": args.episodes,
        "auc_shipped": a_ship,
        "auc_refit": a_fit,
        "features": FEATURES,
        "shipped_weights": SHIPPED.tolist(),
        "fitted_weights": fitted_w.tolist(),
        "fitted_bias": float(fitted_b),
        "temperature": float(chosen),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print(f"\nwrote {args.out}")
    if tmp:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

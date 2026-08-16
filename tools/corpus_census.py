"""Summarize every training corpus so the record survives deleting the bytes.

Prints a markdown table: shards, episodes, decisions, Elo spread, win-label
balance, and distinct decks per corpus.
"""
import glob
import os
import sys

import numpy as np

BC_KEYS = ("group", "elo", "z", "deck")


def census(directory):
    shards = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not shards:
        return None
    rows = 0
    groups = set()
    decks = set()
    elos = []
    wins = 0
    labeled = 0
    dagger_repeats = []
    disagreement = []
    for path in shards:
        with np.load(path) as z:
            n = int(z["z"].shape[0])
            rows += n
            groups.update(np.unique(z["group"]).tolist())
            decks.update(np.unique(z["deck"]).tolist())
            e = z["elo"]
            elos.append(e[np.isfinite(e) & (e > 0)])
            label = z["z"]
            labeled += int((label != 0).sum())
            wins += int((label > 0).sum())
            if "teacher_repeats" in z.files:
                dagger_repeats.append(z["teacher_repeats"].astype(np.float32))
            if "teacher_q_disagreement" in z.files:
                disagreement.append(z["teacher_q_disagreement"])
    elo = np.concatenate(elos) if elos else np.array([0.0])
    out = {
        "shards": len(shards),
        "episodes": len(groups),
        "decisions": rows,
        "decks": len(decks),
        "elo_mean": float(elo.mean()),
        "elo_min": float(elo.min()),
        "elo_max": float(elo.max()),
        "win_frac": (wins / labeled) if labeled else float("nan"),
        "size_mb": sum(os.path.getsize(p) for p in shards) / 1e6,
    }
    if dagger_repeats:
        out["repeats"] = float(np.concatenate(dagger_repeats).mean())
    if disagreement:
        out["disagreement"] = float(np.concatenate(disagreement).mean())
    return out


def main():
    dirs = sys.argv[1:] or sorted(
        d for d in glob.glob("data/*") if os.path.isdir(d))
    print("| corpus | shards | episodes | decisions | decks | mean Elo | "
          "Elo range | win rate | size |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    total = 0
    for d in dirs:
        c = census(d)
        if c is None:
            continue
        total += c["decisions"]
        print(f"| `{os.path.basename(d)}` | {c['shards']} | "
              f"{c['episodes']:,} | {c['decisions']:,} | {c['decks']:,} | "
              f"{c['elo_mean']:.0f} | {c['elo_min']:.0f}-{c['elo_max']:.0f} | "
              f"{c['win_frac']*100:.1f}% | {c['size_mb']:.0f} MB |")
        if "repeats" in c:
            print(f"|   ^ teacher repeats {c['repeats']:.2f}, "
                  f"mean q disagreement {c.get('disagreement', float('nan')):.4f} "
                  f"| | | | | | | | |")
    print(f"\ntotal decisions: {total:,}")


if __name__ == "__main__":
    main()

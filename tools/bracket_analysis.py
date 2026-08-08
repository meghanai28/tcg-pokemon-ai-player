"""Is the agent degrading, or is the bracket?

Four of our submissions are the SAME BYTES.  If their scores differ, the cause
cannot be the agent.  This prints, for each, the opening games that set its
bracket and the field it ended up playing.
"""
import statistics, sys, math
sys.path.insert(0, "tools")
from ladder_status import session, episodes

SAME_BYTES = [55256846, 55290078, 55294353, 55294549]
OTHER = {55264582: "retrained prior", 55294656: "field_9 specialist"}

sess = session()

print("=" * 78)
print("IDENTICAL BYTES (grpo_tech_grim_972_912_811), four uploads")
print("=" * 78)
for ref in SAME_BYTES:
    rows = episodes(sess, ref)
    if not rows:
        continue
    w = sum(1 for r in rows if r["reward"] > 0)
    l = sum(1 for r in rows if r["reward"] < 0)
    opps = [r["opponent"] for r in rows if r["opponent"] is not None]
    mo = statistics.mean(opps)
    edge = rows[-1]["after"] - mo
    print(f"\n  ref {ref}   {len(rows)} eps   {w}-{l} ({100*w/max(w+l,1):.1f}%)   "
          f"final {rows[-1]['after']:.1f}")
    print(f"    mean opponent {mo:.1f}   final score minus mean opponent = {edge:+.1f}")
    first = rows[:10]
    fo = [r['opponent'] for r in first if r['opponent'] is not None]
    print(f"    first 10 games: opponents avg {statistics.mean(fo):.0f}"
          f"   record {sum(1 for r in first if r['reward']>0)}-"
          f"{sum(1 for r in first if r['reward']<0)}"
          f"   rating after 10: {first[-1]['after']:.1f}")
    print("      " + "  ".join(
        f"{(r['opponent'] or 0):.0f}{'W' if r['reward']>0 else 'L'}" for r in first))

print()
print("=" * 78)
print("EDGE OVER OWN FIELD (final score minus mean opponent rating)")
print("=" * 78)
for ref in SAME_BYTES + list(OTHER):
    rows = episodes(sess, ref)
    if not rows or len(rows) < 20:
        continue
    opps = [r["opponent"] for r in rows if r["opponent"] is not None]
    mo = statistics.mean(opps)
    label = OTHER.get(ref, "champion bytes")
    print(f"  ref {ref}  {len(rows):>3} eps  field {mo:>6.1f}  "
          f"score {rows[-1]['after']:>6.1f}  edge {rows[-1]['after']-mo:>+6.1f}  {label}")

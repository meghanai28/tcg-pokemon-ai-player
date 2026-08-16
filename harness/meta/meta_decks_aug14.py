"""Aug-14 field mix for learner-state DAgger.

Repeated entries intentionally approximate the observed top-20 archetype
frequency while retaining deterministic round-robin coverage.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _deck(name):
    cards = [int(line) for line in (ROOT / "grpo_prior" / "decks" / name)
             .read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"{name}: expected 60 cards, got {len(cards)}")
    return cards


_DREEPY = _deck("dreepy_drakloak.csv")
_BUG = _deck("bugcatch_lillies.csv")
_BUG_JUDGE = _deck("bugcatch_lillies_judge2.csv")
_SLOWPOKE = _deck("slowpoke_slowking.csv")
_ALAKAZAM = _deck("alakazam_opp.csv")
_LUCARIO = _deck("mega_lucario.csv")

META_DECKS = {
    "dreepy_a": _DREEPY,
    "dreepy_b": _DREEPY,
    "dreepy_c": _DREEPY,
    "dreepy_d": _DREEPY,
    "dreepy_e": _DREEPY,
    "bug_exact": _BUG,
    "bug_judge": _BUG_JUDGE,
    "slowpoke_a": _SLOWPOKE,
    "slowpoke_b": _SLOWPOKE,
    "alakazam_a": _ALAKAZAM,
    "alakazam_b": _ALAKAZAM,
    "lucario": _LUCARIO,
}


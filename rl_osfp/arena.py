"""Fast sequential game rollouts through the official native engine."""
from __future__ import annotations

from dataclasses import dataclass
import json

from foundation.cg import game
from foundation.cg.engine import get_lib

from .network import ActorCritic
from .policy import Decision, choose_action


@dataclass
class GameResult:
    winner: int | None
    decisions: tuple[list[Decision], list[Decision]]
    engine_steps: int
    invalid_actions: int
    error: str | None = None
    step_capped: bool = False


class Arena:
    def __init__(self) -> None:
        lib = get_lib()
        self.card_db = {
            int(card["cardId"]): card
            for card in json.loads(lib.AllCard().decode())
        }
        self.attack_db = {
            int(attack["attackId"]): attack
            for attack in json.loads(lib.AllAttack().decode())
        }

    @staticmethod
    def _fallback(select: dict) -> list[int]:
        options = select.get("option") or []
        n_options = len(options)
        if not n_options:
            return []
        low = max(0, min(int(select.get("minCount", 1) or 0), n_options))
        high = max(low, min(int(select.get("maxCount", max(low, 1)) or 0), n_options))
        count = low if low > 0 else min(1, high)
        return list(range(count))

    def play(
        self,
        models: tuple[ActorCritic, ActorCritic],
        decks: tuple[list[int], list[int]],
        *,
        record_seats: frozenset[int] = frozenset(),
        sample: bool = True,
        temperature: float = 1.0,
        max_steps: int = 2400,
        max_recorded_decisions: int = 3000,
    ) -> GameResult:
        for deck in decks:
            if len(deck) != 60 or not all(isinstance(card, int) for card in deck):
                raise ValueError("every deck must contain exactly 60 integer card ids")
        records: tuple[list[Decision], list[Decision]] = ([], [])
        invalid = 0
        observation = None
        try:
            observation, start = game.battle_start(decks[0], decks[1])
            if observation is None:
                return GameResult(None, records, 0, 0, f"BattleStart error {start.error}")
            for step in range(max_steps):
                current = observation.get("current") or {}
                result = int(current.get("result", -1))
                if result >= 0:
                    return GameResult(result if result in (0, 1) else None, records, step, invalid)
                seat = int(current.get("yourIndex", 0))
                action, decision = choose_action(
                    models[seat], observation, decks[seat], self.card_db,
                    self.attack_db, sample=sample, temperature=temperature,
                )
                try:
                    next_observation = game.battle_select(action)
                except (IndexError, ValueError):
                    invalid += 1
                    decision = None
                    next_observation = game.battle_select(
                        self._fallback(observation.get("select") or {})
                    )
                if (
                    decision is not None
                    and seat in record_seats
                    and len(records[seat]) < max_recorded_decisions
                ):
                    records[seat].append(decision)
                observation = next_observation
            # This is an artificial rollout horizon, not an engine fault. Keep
            # the trajectory as a zero-reward draw so stalling is not deleted
            # from the data and report the cap separately from real errors.
            return GameResult(None, records, max_steps, invalid, None, True)
        except Exception as exc:  # engine faults must be visible in metrics
            return GameResult(None, records, 0, invalid, repr(exc))
        finally:
            if observation is not None:
                try:
                    game.battle_finish()
                except Exception:
                    pass

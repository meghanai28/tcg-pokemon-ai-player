"""Complete masked action distribution for one PTCG selection callback."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from foundation import nn_features_rich as NF
from .network import ActorCritic, MAX_COUNT


@dataclass
class Decision:
    kind: np.ndarray
    card: np.ndarray
    scal: np.ndarray
    mask: np.ndarray
    ctx: int
    stype: int
    selected_slots: np.ndarray
    selected_count: int
    low_count: int
    high_count: int
    old_logp: float
    old_value: float
    outcome: float = 0.0
    advantage: float = 0.0


def _bounds(select: dict, n_options: int) -> tuple[int, int]:
    low = int(select.get("minCount", 1) or 0)
    high = int(select.get("maxCount", max(low, 1)) or 0)
    low = max(0, min(low, n_options, MAX_COUNT))
    high = max(low, min(high, n_options, MAX_COUNT))
    return low, high


def encode(observation: dict, deck: list[int], card_db: dict, attack_db: dict):
    select = observation.get("select") or {}
    current = observation.get("current") or {}
    me = int(current.get("yourIndex", 0))
    state = {"current": current, "select": select, "decklist": deck}
    return NF.encode(state, me, card_db, attack_db, None)


def choose_action(
    model: ActorCritic,
    observation: dict,
    deck: list[int],
    card_db: dict,
    attack_db: dict,
    *,
    sample: bool,
    temperature: float = 1.0,
) -> tuple[list[int], Decision | None]:
    select = observation.get("select") or {}
    options = select.get("option") or []
    n_options = len(options)
    if n_options == 0:
        return [], None
    low, high = _bounds(select, n_options)
    if low == high == n_options:
        return list(range(n_options)), None

    kind, card, scal, mask, slots = encode(
        observation, deck, card_db, attack_db
    )
    valid = [(index, slot) for index, slot in enumerate(slots) if slot >= 0]
    if len(valid) != n_options:
        # MAX_OPT is deliberately 64; this only protects against a future
        # engine action larger than the current card pool.
        forced = max(low, 1 if high else 0)
        return list(range(min(forced, n_options))), None

    tensors = (
        torch.as_tensor(kind[None].astype(np.int64)),
        torch.as_tensor(card[None].astype(np.int64)),
        torch.as_tensor(scal[None]),
        torch.as_tensor(mask[None]),
        torch.as_tensor([int(select.get("context") or 0)]),
        torch.as_tensor([int(select.get("type") or 0)]),
    )
    with torch.inference_mode():
        option_logits, count_logits, value = model(*tensors)
    option_logits = option_logits[0] / max(float(temperature), 1e-3)
    count_logits = count_logits[0] / max(float(temperature), 1e-3)

    count_mask = torch.zeros(MAX_COUNT + 1, dtype=torch.bool)
    count_mask[low : high + 1] = True
    count_logp = torch.log_softmax(count_logits.masked_fill(~count_mask, -torch.inf), -1)
    if sample:
        count = int(torch.multinomial(count_logp.exp(), 1).item())
    else:
        count = int(torch.argmax(count_logp).item())
    total_logp = float(count_logp[count])

    if count == n_options:
        picked_indices = list(range(n_options))
        picked_slots = [slot for _index, slot in valid]
    else:
        available = torch.zeros_like(option_logits, dtype=torch.bool)
        for _index, slot in valid:
            available[slot] = True
        slot_to_index = {slot: index for index, slot in valid}
        picked_indices, picked_slots = [], []
        for _ in range(count):
            logp = torch.log_softmax(option_logits.masked_fill(~available, -torch.inf), -1)
            slot = int(torch.multinomial(logp.exp(), 1).item()) if sample else int(logp.argmax())
            total_logp += float(logp[slot])
            picked_indices.append(slot_to_index[slot])
            picked_slots.append(slot)
            available[slot] = False

    padded = np.full(MAX_COUNT, -1, dtype=np.int16)
    if picked_slots:
        padded[: len(picked_slots)] = picked_slots
    decision = Decision(
        kind=kind,
        card=card,
        scal=scal,
        mask=mask,
        ctx=int(select.get("context") or 0),
        stype=int(select.get("type") or 0),
        selected_slots=padded,
        selected_count=count,
        low_count=low,
        high_count=high,
        old_logp=total_logp,
        old_value=float(value[0]),
    )
    return picked_indices, decision


def batch_logprob_entropy(
    option_logits: torch.Tensor,
    count_logits: torch.Tensor,
    kind: torch.Tensor,
    mask: torch.Tensor,
    selected_slots: torch.Tensor,
    selected_count: torch.Tensor,
    low_count: torch.Tensor,
    high_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute log pi(a|s) for count + Plackett-Luce option sequence."""
    batch = option_logits.shape[0]
    device = option_logits.device
    counts = torch.arange(MAX_COUNT + 1, device=device)[None, :]
    valid_count = (counts >= low_count[:, None]) & (counts <= high_count[:, None])
    count_logp = torch.log_softmax(count_logits.masked_fill(~valid_count, -torch.inf), -1)
    rows = torch.arange(batch, device=device)
    total = count_logp[rows, selected_count]
    count_prob = torch.where(valid_count, count_logp.exp(), torch.zeros_like(count_logp))
    entropy = -(count_prob * torch.where(valid_count, count_logp, torch.zeros_like(count_logp))).sum(-1)

    available = (kind == 3) & (mask > 0.5)
    option_total = available.sum(-1)
    forced_all = selected_count == option_total
    max_selected = int(selected_count.max().detach().cpu()) if batch else 0
    for index in range(max_selected):
        active = (selected_count > index) & ~forced_all
        if not bool(active.any()):
            continue
        logp = torch.log_softmax(option_logits.masked_fill(~available, -torch.inf), -1)
        probability = torch.where(available, logp.exp(), torch.zeros_like(logp))
        step_entropy = -(probability * torch.where(available, logp, torch.zeros_like(logp))).sum(-1)
        slot = selected_slots[:, index].clamp_min(0)
        total = total + torch.where(active, logp[rows, slot], torch.zeros_like(total))
        entropy = entropy + torch.where(active, step_entropy, torch.zeros_like(entropy))
        available = available.clone()
        available[rows[active], slot[active]] = False
    normalizer = 1.0 + selected_count.float().clamp_min(1.0)
    return total, entropy / normalizer

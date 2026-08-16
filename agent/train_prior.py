"""Shard-streamed, deck-weighted training for a search prior.

Unlike ``bc_train/train_bc.py``, this trainer never concatenates the corpus.
One compressed shard (normally <=60k decisions) is resident at a time, so every
row in the 2.19M-decision source corpus can be used without a 40+ GiB load peak.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import math
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from bc_train import model as model_module
from bc_train import nn_features_rich as features
from bc_train import nn_features_v2 as features_v2
from bc_train import nn_features_v3 as features_v3
from bc_train.model import export_npz
from bc_train.train_bc import CRITICAL_CONTEXTS, option_policy_loss

FIELDS = ("kind", "card", "scal", "mask", "ctx", "stype", "pi", "z")
TEACHER_FIELDS = (
    "action_tokens", "action_sizes", "action_q", "action_visits", "action_mask")
V3_FIELDS = features_v3.V3_FIELDS
V3_DEMO_FIELDS = ("demo_tokens", "demo_size")


def stable_id(value) -> int:
    raw = str(value or "").encode("utf-8", "replace")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def read_deck(path: str) -> list[int]:
    with open(path, encoding="utf-8") as source:
        deck = [int(line) for line in source if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"{path}: expected 60 cards, got {len(deck)}")
    return deck


def variant_ids(deck: list[int], census_path: str, threshold: float) -> dict:
    """Every census list within ``threshold`` multiset-Jaccard of ``deck``.

    The corpus tags each decision with a hash of the exact 60-card list, so
    matching on one list counts only pilots playing that precise build. Real
    archetypes are spread over tech variants: Bug Catching has six census lists
    and Dreepy five, so exact matching found 2,981 Bug Catching rows where the
    archetype actually holds far more. Training on the whole archetype teaches
    the shared game plan, which is what transfers to the one list we ship.
    """
    from collections import Counter
    with open(census_path, encoding="utf-8") as source:
        blob = json.load(source)
    rows = blob["decks"] if isinstance(blob, dict) and "decks" in blob else blob
    target = Counter(deck)
    found = {}
    for row in rows:
        other = row.get("deck") or row.get("representative")
        if not other:
            continue
        counts = Counter(other)
        inter = sum((target & counts).values())
        union = sum((target | counts).values())
        if union and inter / union >= threshold:
            found[stable_id(tuple(sorted(other)))] = (
                (row.get("label") or "")[:40], inter / union, row.get("seats", 0))
    return found


def shard_paths(data_dirs: list[str]) -> list[str]:
    paths = sorted({path for source in data_dirs
                    for path in ([source] if os.path.isfile(source)
                                 and source.endswith(".npz") else
                                 glob.glob(os.path.join(source, "*.npz")))})
    if not paths:
        raise ValueError(f"no .npz shards under {data_dirs}")
    return paths


def load_shard(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        missing = [key for key in FIELDS if key not in source]
        if missing:
            raise ValueError(f"{path} misses {missing}")
        optional = ("elo", "deck", "group", "seat", "pilot") \
            + V3_FIELDS + V3_DEMO_FIELDS
        keys = FIELDS + tuple(key for key in optional if key in source)
        return {key: source[key] for key in keys}


def load_teacher_reservoir(paths: list[str], max_rows: int,
                           seed: int) -> dict[str, np.ndarray]:
    """Load a bounded, uniform pool of usable learner-state Q labels.

    DAgger writes one shard per game so labels can be recovered after an
    interrupted generation run.  Training from thousands of tiny compressed
    shards is needlessly expensive; this one-time bounded reservoir keeps the
    ordinary human corpus streamed while making the much smaller teacher
    stream cheap to mix into every optimizer batch.
    """
    with np.load(paths[0]) as first:
        v3_fields = V3_FIELDS if all(key in first for key in V3_FIELDS) else ()
    counts = []
    total = 0
    for path in paths:
        with np.load(path) as source:
            missing = [key for key in FIELDS + TEACHER_FIELDS + v3_fields
                       if key not in source]
            if missing:
                raise ValueError(f"{path} misses teacher fields {missing}")
            usable = source["action_mask"].sum(axis=1) >= 2
            count = int(usable.sum())
        counts.append(count)
        total += count
    if total == 0:
        return {}

    keep = min(total, max_rows)
    rng = np.random.default_rng(seed)
    chosen = (np.arange(total, dtype=np.int64) if keep == total else
              np.sort(rng.choice(total, size=keep, replace=False)))
    optional = ("group", "seat", "deck", "elo")
    parts: dict[str, list[np.ndarray]] = {
        key: [] for key in FIELDS + TEACHER_FIELDS + v3_fields + optional}
    offset = 0
    for path, count in zip(paths, counts):
        if not count:
            continue
        lo = int(np.searchsorted(chosen, offset, side="left"))
        hi = int(np.searchsorted(chosen, offset + count, side="left"))
        old_offset = offset
        offset += count
        if lo == hi:
            continue
        with np.load(path) as source:
            usable_rows = np.flatnonzero(
                source["action_mask"].sum(axis=1) >= 2)
            local = usable_rows[chosen[lo:hi] - old_offset]
            for key in FIELDS + TEACHER_FIELDS + v3_fields:
                parts[key].append(source[key][local])
            rows = len(local)
            parts["group"].append(
                source["group"][local].astype(np.uint64)
                if "group" in source else
                np.full(rows, stable_id(path), dtype=np.uint64))
            parts["seat"].append(
                source["seat"][local].astype(np.int8)
                if "seat" in source else np.zeros(rows, dtype=np.int8))
            parts["deck"].append(
                source["deck"][local].astype(np.uint64)
                if "deck" in source else np.zeros(rows, dtype=np.uint64))
            parts["elo"].append(
                source["elo"][local].astype(np.float32)
                if "elo" in source else np.full(rows, 1000, dtype=np.float32))
    result = {key: np.concatenate(value, axis=0)
              for key, value in parts.items()}
    result["episode_weight"] = episode_balance_weights(
        result["group"], result["seat"])
    return result


def configure(arch: tuple[int, int, int, int], feature_module=features) -> None:
    dim, layers, heads, d_ff = arch
    if dim % heads:
        raise ValueError(f"dim {dim} is not divisible by heads {heads}")
    model_module.NF = feature_module
    model_module.D_MODEL = dim
    model_module.N_LAYERS = layers
    model_module.N_HEADS = heads
    model_module.D_FF = d_ff
    model_module.DROPOUT = 0.0


def configure_and_load(path: str | None, device: torch.device,
                       arch: tuple[int, int, int, int] | None = None,
                       feature_module=features,
                       ) -> torch.nn.Module:
    """Build the network, optionally warm-starting from a checkpoint.

    With ``path`` the checkpoint's own ``_meta`` defines the architecture, so
    the champion warm-start keeps behaving exactly as before. Without it the
    net is randomly initialised at ``arch``; the deployed ``nn_infer.NumpyNet``
    reads ``_meta`` at load time and loops over ``n_layers``, so any shape it
    can hold is shippable as long as the latency guard in the search shell is
    respected.
    """
    if path is None:
        if arch is None:
            raise ValueError("need --init or an explicit architecture")
        configure(arch, feature_module)
        return model_module.TCGNet().to(device)

    with np.load(path) as weights:
        meta = tuple(map(int, weights.get("_meta", [])))
        if len(meta) != 4:
            raise ValueError(f"{path} has no architecture metadata")
        if arch is not None and tuple(arch) != meta:
            raise ValueError(
                f"--init is {meta} but requested architecture is {tuple(arch)}; "
                "weights cannot be reshaped, drop --init to train from scratch")
        configure(meta, feature_module)
        net = model_module.TCGNet()
        state = net.state_dict()
        missing = [key for key, value in state.items()
                   if key not in weights or weights[key].shape != tuple(value.shape)]
        if missing:
            raise ValueError(f"init has missing/mismatched tensors: {missing[:5]}")
        net.load_state_dict({key: torch.as_tensor(weights[key], dtype=value.dtype)
                             for key, value in state.items()})
    return net.to(device)


def tensor_batch(data: dict[str, np.ndarray], indices: np.ndarray,
                 device: torch.device):
    kind_np = data["kind"][indices]
    card_np = data["card"][indices]
    scal_np = data["scal"][indices]
    mask_np = data["mask"][indices]
    pi_np = data["pi"][indices]
    if getattr(model_module.NF, "FEATURE_VERSION", 1) == 2:
        deck_ids = (data["deck"][indices] if "deck" in data else
                    np.zeros(len(indices), dtype=np.uint64))
        kind_np, card_np, scal_np, mask_np, pi_np = features_v2.upgrade_batch(
            kind_np, card_np, scal_np, mask_np, pi_np, deck_ids)
    tensors = (
        torch.as_tensor(kind_np.astype(np.int64), device=device),
        torch.as_tensor(card_np.astype(np.int64), device=device),
        torch.as_tensor(scal_np, device=device),
        torch.as_tensor(mask_np, device=device),
        torch.as_tensor(data["ctx"][indices].astype(np.int64), device=device),
        torch.as_tensor(data["stype"][indices].astype(np.int64), device=device),
        torch.as_tensor(pi_np, device=device),
        torch.as_tensor(data["z"][indices], device=device),
    )
    if getattr(model_module.NF, "FEATURE_VERSION", 1) == 3:
        missing = [key for key in V3_FIELDS if key not in data]
        if missing:
            raise ValueError(f"feature-v3 shard misses {missing}")
        tensors += (
            torch.as_tensor(data["bag_card"][indices].astype(np.int64),
                            device=device),
            torch.as_tensor(data["bag_count"][indices], device=device),
            torch.as_tensor(data["bag_kind"][indices].astype(np.int64),
                            device=device),
            torch.as_tensor(data["bag_scal"][indices], device=device),
            torch.as_tensor(data["bag_mask"][indices], device=device),
        )
    return tensors


def forward_tensor_batch(net: torch.nn.Module, tensors):
    """Forward a tensor_batch while keeping v1/v2 call sites compatible."""
    kind, card, scal, mask, ctx, stype = tensors[:6]
    extras = tensors[8:]
    return net(kind, card, scal, mask, ctx, stype, *extras)


def teacher_field_tensors(data: dict[str, np.ndarray], indices: np.ndarray,
                          device: torch.device) -> list[torch.Tensor]:
    arrays = [data[key][indices] for key in TEACHER_FIELDS]
    if (getattr(model_module.NF, "FEATURE_VERSION", 1) == 2 and
            data["kind"].shape[1] != features_v2.SEQ):
        arrays[0] = features_v2.upgrade_action_tokens(arrays[0])
    return [torch.as_tensor(value, device=device) for value in arrays]


HOLDOUT_BUCKETS = np.uint64(10000)


def group_holdout_mask(group: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Deterministic per-episode holdout membership.

    Splitting by ``group`` (one game-seat) rather than by row is the whole
    point. Decisions inside a single game are highly correlated, so a row-level
    random split puts the same game on both sides and the validation number
    becomes a memorisation score. Hashing the group id keeps the split stable
    across shards and across processes without a global pass over the corpus.
    """
    if frac <= 0.0 or not len(group):
        return np.zeros(len(group), dtype=bool)
    x = group.astype(np.uint64) ^ np.uint64(
        (seed * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    cutoff = np.uint64(max(1, int(round(frac * float(HOLDOUT_BUCKETS)))))
    return (x % HOLDOUT_BUCKETS) < cutoff


def eligible_row_indices(data: dict[str, np.ndarray],
                         drop_unresolved_looking: bool,
                         min_elo: float = 0.0,
                         holdout_frac: float = 0.0,
                         holdout_seed: int = 0,
                         want_holdout: bool = False) -> np.ndarray:
    rows = np.arange(len(data["ctx"]), dtype=np.int64)
    if holdout_frac > 0.0 and len(rows) and "group" in data:
        in_holdout = group_holdout_mask(
            data["group"], holdout_frac, holdout_seed)
        rows = rows[in_holdout[rows] == want_holdout]
    if min_elo > 0.0 and len(rows) and "elo" in data:
        # --elo-weight only *rewards* pilots above 1000; below that every row
        # carries identical weight, so a 807-Elo mistake trains as hard as a
        # 1000-Elo one. This drops the tail outright rather than reweighting it.
        rows = rows[data["elo"][rows] >= min_elo]
    if not drop_unresolved_looking or not len(rows):
        return rows
    option = (data["kind"] == 3) & (data["mask"] > 0.5)
    looking = np.isclose(data["scal"][:, :, 20], 1.0, atol=1e-6)
    unresolved = option & looking & (data["card"] == 0)
    # A public LOOKING list should always expose its card identities.  Drop
    # only rows where such an option is unresolved; hidden prize options use
    # area 6 and remain valid semantic partial labels.  The mask spans every
    # row in the shard, so restrict it to the rows that survived above.
    return rows[~unresolved.any(axis=1)[rows]]


def add_trajectory_counts(counts: dict[tuple[int, int], int],
                          data: dict[str, np.ndarray],
                          indices: np.ndarray) -> None:
    if "group" not in data or "seat" not in data or not len(indices):
        return
    keys = np.rec.fromarrays(
        (data["group"][indices].astype(np.uint64),
         data["seat"][indices].astype(np.int8)), names=("group", "seat"))
    unique, frequency = np.unique(keys, return_counts=True)
    for key, count in zip(unique, frequency):
        pair = (int(key["group"]), int(key["seat"]))
        counts[pair] = counts.get(pair, 0) + int(count)


def trajectory_batch_weights(data: dict[str, np.ndarray],
                             indices: np.ndarray,
                             counts: dict[tuple[int, int], int],
                             normalization: float) -> np.ndarray:
    if not counts or "group" not in data or "seat" not in data:
        return np.ones(len(indices), dtype=np.float32)
    result = np.empty(len(indices), dtype=np.float32)
    for out_index, row in enumerate(indices):
        key = (int(data["group"][row]), int(data["seat"][row]))
        result[out_index] = normalization / max(counts.get(key, 1), 1)
    return result


EQUIVALENT_COPY_CONTEXTS = (7, 8, 22)


def _equivalent_copy_groups(kind: torch.Tensor, card: torch.Tensor,
                            scal: torch.Tensor, mask: torch.Tensor,
                            ctx: torch.Tensor
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return rows, pairwise groups, and group sizes for safe copy aliases.

    Card ID alone is not enough to prove that two options are interchangeable:
    unresolved actions use card ID zero, and equal attached energies on two
    different Pokemon have different consequences.  Option type/source fields
    retain those semantic distinctions while intentionally ignoring the
    physical hand/deck index of otherwise identical copies.
    """
    active = torch.zeros_like(ctx, dtype=torch.bool)
    for context in EQUIVALENT_COPY_CONTEXTS:
        active |= ctx == context
    options = (kind == 3) & (mask > 0.5)

    same_card = card[:, :, None] == card[:, None, :]
    resolved = (card[:, :, None] > 0) & (card[:, None, :] > 0)

    # Hidden prize positions are deliberately exchangeable from the policy's
    # information state, but generic card-id-zero options are not.  PRIZE is
    # encoded as area 6 / 12 in scalar slot 20.
    prize_area = torch.full_like(scal[:, :, 20], 6.0 / 12.0)
    hidden_prize = (
        (ctx == 7)[:, None]
        & (card == 0)
        & torch.isclose(scal[:, :, 20], prize_area, atol=1e-6, rtol=0.0)
    )
    singleton = torch.eye(
        card.shape[1], device=card.device, dtype=torch.bool)[None, :, :]
    same_identity = singleton | (same_card & (
        resolved | (hidden_prize[:, :, None] & hidden_prize[:, None, :])))

    # Same option type, source area, and player.  Slots 23/25 encode physical
    # card/energy indices and are intentionally ignored for ordinary copies.
    same_descriptor = (
        (scal[:, :, None, :17] == scal[:, None, :, :17]).all(-1)
        & (scal[:, :, None, 20] == scal[:, None, :, 20])
        & (scal[:, :, None, 22] == scal[:, None, :, 22])
    )
    # Energy selections must also refer to the same board source/target.  Only
    # the attached energy's own physical index (slot 25) may differ.
    same_energy_source = (
        (scal[:, :, None, 21] == scal[:, None, :, 21])
        & (scal[:, :, None, 23] == scal[:, None, :, 23])
        & (scal[:, :, None, 24] == scal[:, None, :, 24])
        & (scal[:, :, None, 31] == scal[:, None, :, 31])
    )
    same_descriptor &= torch.where(
        (ctx == 22)[:, None, None], same_energy_source,
        torch.ones_like(same_energy_source))

    same = (options[:, :, None] & options[:, None, :] & same_identity
            & same_descriptor)
    group_size = same.sum(dim=-1).clamp_min(1)
    return active, same, group_size


def canonicalize_equivalent_targets(pi: torch.Tensor, kind: torch.Tensor,
                                    card: torch.Tensor, scal: torch.Tensor,
                                    mask: torch.Tensor,
                                    ctx: torch.Tensor) -> torch.Tensor:
    """Remove arbitrary physical-copy identity from replay labels.

    In deck/prize search (7), discard selection (8), and energy selection (22),
    two options with the same resolved card ID are interchangeable copies. The
    engine records which hand/deck index a pilot clicked, so ordinary one-hot
    BC wastes capacity predicting an outcome-irrelevant index. Spread each
    selected copy's mass across its equivalent copies while retaining the total
    mass assigned to that card identity. Other contexts stay untouched because
    two copies on different board slots can have genuinely different state.
    """
    active, same, group_size = _equivalent_copy_groups(
        kind, card, scal, mask, ctx)
    if not bool(active.any()):
        return pi
    spread = pi / group_size.to(pi.dtype)
    canonical = torch.bmm(same.to(pi.dtype), spread.unsqueeze(-1)).squeeze(-1)
    return torch.where(active[:, None], canonical, pi)


def canonicalize_equivalent_logits(logits: torch.Tensor, kind: torch.Tensor,
                                   card: torch.Tensor, scal: torch.Tensor,
                                   mask: torch.Tensor,
                                   ctx: torch.Tensor) -> torch.Tensor:
    """Make duplicate-card loss depend only on semantic group probability.

    Repeating ``logsumexp(group) - log(group_size)`` at every member preserves
    the group's total softmax probability while making its internal physical-
    copy allocation irrelevant to the objective.
    """
    active, same, group_size = _equivalent_copy_groups(
        kind, card, scal, mask, ctx)
    if not bool(active.any()):
        return logits
    options = (kind == 3) & (mask > 0.5)
    option_logits = logits.masked_fill(~options, -torch.inf)
    members = option_logits[:, None, :].masked_fill(~same, -torch.inf)
    grouped = torch.logsumexp(members, dim=-1) - group_size.to(
        logits.dtype).log()
    return torch.where(active[:, None] & options, grouped, logits)


def equivalent_copy_ce_offset(pi: torch.Tensor, kind: torch.Tensor,
                              card: torch.Tensor, scal: torch.Tensor,
                              mask: torch.Tensor,
                              ctx: torch.Tensor) -> torch.Tensor:
    """Constant introduced by uniformly expanding a semantic group target."""
    active, _same, group_size = _equivalent_copy_groups(
        kind, card, scal, mask, ctx)
    options = (kind == 3) & (mask > 0.5)
    offset = (pi * group_size.to(pi.dtype).log() * options).sum(-1)
    return torch.where(active, offset, torch.zeros_like(offset))


def equivalent_option_policy_loss(logits: torch.Tensor, pi: torch.Tensor,
                                  kind: torch.Tensor, card: torch.Tensor,
                                  scal: torch.Tensor, mask: torch.Tensor,
                                  ctx: torch.Tensor
                                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Semantic CE for contexts with interchangeable physical card copies."""
    canonical_logits = canonicalize_equivalent_logits(
        logits, kind, card, scal, mask, ctx)
    canonical_pi = canonicalize_equivalent_targets(
        pi, kind, card, scal, mask, ctx)
    loss, option_logits = option_policy_loss(
        canonical_logits, canonical_pi, kind, mask)
    return (loss - equivalent_copy_ce_offset(
        pi, kind, card, scal, mask, ctx), option_logits)


def equivalent_top1_hit(option_logits: torch.Tensor, pi: torch.Tensor,
                        kind: torch.Tensor, card: torch.Tensor,
                        mask: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
    """Top-1 agreement that treats a selected identical copy as correct."""
    active = torch.zeros_like(ctx, dtype=torch.bool)
    for context in EQUIVALENT_COPY_CONTEXTS:
        active |= ctx == context
    options = (kind == 3) & (mask > 0.5)
    predicted = option_logits.argmax(-1)
    exact = predicted == pi.argmax(-1)
    predicted_card = card.gather(1, predicted[:, None]).squeeze(1)
    semantic = ((pi > 0) & options
                & (card == predicted_card[:, None])).any(-1)
    return torch.where(active, semantic, exact)


def action_set_metrics(logits: torch.Tensor, pi: torch.Tensor,
                       kind: torch.Tensor, mask: torch.Tensor,
                       cap: int = 16) -> tuple[torch.Tensor, ...]:
    """Deployment-facing recovery of every demonstrated tuple member.

    Marginal CE can improve while the exact multi-select action gets worse.
    For each row, predict the same number of members as the demonstration and
    measure member recall/exact-set agreement.  On rows with more than ``cap``
    options, also report whether all demonstrated members survive cap-16.
    """
    options = (kind == 3) & (mask > 0.5)
    target = (pi > 0) & options
    target_size = target.sum(-1)
    option_count = options.sum(-1)
    scores = logits.float().masked_fill(~options, -torch.inf)
    order = scores.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(order)
    positions = torch.arange(
        scores.shape[1], device=scores.device)[None, :].expand_as(order)
    ranks.scatter_(1, order, positions)
    predicted = options & (ranks < target_size[:, None])
    selected = target_size > 0
    overlap = (predicted & target).sum(-1)
    recall = overlap.to(logits.dtype) / target_size.clamp_min(1).to(logits.dtype)
    exact = ((predicted == target) | ~options).all(-1) & selected
    cap_bound = (option_count > cap) & selected
    in_cap = (~target | (ranks < cap)).all(-1) & cap_bound
    return recall, exact, selected, in_cap, cap_bound


def action_tuple_scores(logits: torch.Tensor, action_tokens: torch.Tensor,
                        action_sizes: torch.Tensor,
                        action_mask: torch.Tensor) -> torch.Tensor:
    """Score exact candidate actions the same way deployed PUCT does.

    ``main.py::_gen_candidates`` converts option logits to an action prior with
    ``exp(mean(member_logits))``.  Training only on per-option marginals loses
    the identity of multi-select actions.  This helper keeps the current
    additive deployment model, but moves the supervised objective to the object
    PUCT actually consumes: a candidate action tuple.
    """
    if logits.ndim != 2 or action_tokens.ndim != 3:
        raise ValueError("expected logits [B,S] and action_tokens [B,C,M]")
    if action_sizes.shape != action_tokens.shape[:2]:
        raise ValueError("action_sizes shape does not match action_tokens")
    if action_mask.shape != action_tokens.shape[:2]:
        raise ValueError("action_mask shape does not match action_tokens")
    safe_tokens = action_tokens.long().clamp_min(0)
    if bool((safe_tokens >= logits.shape[1]).any()):
        raise ValueError("action token index exceeds policy sequence")
    expanded = logits[:, None, :].expand(-1, action_tokens.shape[1], -1)
    members = torch.gather(expanded, 2, safe_tokens)
    member_mask = (
        torch.arange(action_tokens.shape[2], device=logits.device)[None, None, :]
        < action_sizes.long()[:, :, None]
    ) & (action_tokens >= 0)
    scores = (members * member_mask.to(members.dtype)).sum(-1)
    scores /= action_sizes.to(members.dtype).clamp_min(1)
    # The shell assigns an empty legal action exp(0) before normalizing.  A
    # fixed score of zero mirrors that contract and, importantly, still gives
    # gradients to every non-empty alternative relative to "done".
    live = action_mask.bool()
    return scores.masked_fill(~live, -torch.inf)


def teacher_action_policy_loss(logits: torch.Tensor,
                               action_tokens: torch.Tensor,
                               action_sizes: torch.Tensor,
                               action_q: torch.Tensor,
                               action_mask: torch.Tensor,
                               action_visits: torch.Tensor | None = None,
                               q_shrink_visits: float = 0.0,
                               q_temperature: float = 0.10
                               ) -> tuple[torch.Tensor, ...]:
    """Soft BC target from teacher Q over exact candidate action tuples.

    The search is used only to label states; gradients still come from ordinary
    supervised cross entropy.  Centering Q before the softmax is numerically
    stable and makes the target invariant to a state-wise value offset.
    """
    if q_temperature <= 0:
        raise ValueError("q_temperature must be positive")
    if q_shrink_visits < 0:
        raise ValueError("q_shrink_visits must be nonnegative")
    scores = action_tuple_scores(
        logits, action_tokens, action_sizes, action_mask)
    live = action_mask.bool()
    usable = live.sum(-1) >= 2
    safe_q = action_q.float()
    if q_shrink_visits and action_visits is not None:
        visits = action_visits.float().clamp_min(0.0)
        denom = (visits * live).sum(-1, keepdim=True).clamp_min(1.0)
        center = (safe_q * visits * live).sum(-1, keepdim=True) / denom
        safe_q = ((visits * safe_q + q_shrink_visits * center) /
                  (visits + q_shrink_visits).clamp_min(1e-6))
    safe_q = safe_q.masked_fill(~live, -torch.inf)
    q_max = safe_q.max(-1, keepdim=True).values
    centered = torch.where(live, safe_q - q_max, torch.zeros_like(safe_q))
    target_logits = (centered / q_temperature).masked_fill(~live, -torch.inf)
    target = torch.softmax(target_logits, dim=-1)
    logp = torch.log_softmax(scores.float(), dim=-1)
    per_sample = -(target * torch.where(
        live, logp, torch.zeros_like(logp))).sum(-1)
    predicted = scores.argmax(-1)
    best = safe_q.argmax(-1)
    hit = predicted == best
    predicted_q = safe_q.gather(1, predicted[:, None]).squeeze(1)
    best_q = safe_q.gather(1, best[:, None]).squeeze(1)
    regret = best_q - predicted_q
    zero = torch.zeros_like(per_sample)
    return (torch.where(usable, per_sample, zero),
            torch.where(usable, hit, torch.zeros_like(hit)),
            torch.where(usable, regret, zero), usable)


def evaluate(net, paths: list[str], device: torch.device, batch: int,
             target_ids: np.ndarray | None = None,
             target_wins_only: bool = False,
             equivalent_targets: bool = False,
             drop_unresolved_looking: bool = False,
             amp_ctx=contextlib.nullcontext,
             holdout_frac: float = 0.0,
             holdout_seed: int = 0) -> dict:
    """Evaluate globally and on the deck family that will actually ship.

    Global replay CE is useful as a regression guard, but it is a poor model
    selector for a deck specialist: more than 95% of the validation rows may
    come from decks the submitted agent can never play.  The target metrics
    use the same exact/variant IDs as the training weights.
    """
    total = ce = mae = top1 = 0.0
    set_rows = set_recall = exact_sets = 0.0
    cap_rows = cap_hits = 0.0
    target_total = target_ce = target_mae = target_top1 = 0.0
    target_set_rows = target_set_recall = target_exact_sets = 0.0
    target_cap_rows = target_cap_hits = 0.0
    target_clean_total = target_clean_ce = target_clean_top1 = 0.0
    target_win_total = target_win_ce = 0.0
    target_loss_total = target_loss_ce = 0.0
    net.eval()
    with torch.inference_mode():
        for path in paths:
            data = load_shard(path)
            # want_holdout=True: with a group split the validation rows are the
            # held-out episodes inside the same shards the trainer streams.
            eligible = eligible_row_indices(
                data, drop_unresolved_looking, 0.0,
                holdout_frac, holdout_seed, want_holdout=True)
            for start in range(0, len(eligible), batch):
                indices = eligible[start:start + batch]
                batch_tensors = tensor_batch(data, indices, device)
                kind, card, scal, mask, ctx, stype, pi, z = batch_tensors[:8]
                with amp_ctx():
                    logits, value = forward_tensor_batch(net, batch_tensors)
                # Loss in fp32: the policy that ships is a softmax over these
                # logits, and prior sharpness has already cost this project
                # hundreds of ladder points once.
                if equivalent_targets:
                    sample_ce, option_logits = equivalent_option_policy_loss(
                        logits.float(), pi, kind, card, scal, mask, ctx)
                    hit = equivalent_top1_hit(
                        option_logits, pi, kind, card, mask, ctx)
                else:
                    sample_ce, option_logits = option_policy_loss(
                        logits.float(), pi, kind, mask)
                    hit = option_logits.argmax(-1) == pi.argmax(-1)
                (batch_set_recall, batch_exact_set, batch_set_selected,
                 batch_cap_hit, batch_cap_bound) = action_set_metrics(
                    logits.float(), pi, kind, mask)
                count = len(indices)
                total += count
                ce += float(sample_ce.sum())
                mae += float((value - z).abs().sum())
                top1 += float(hit.sum())
                set_rows += int(batch_set_selected.sum())
                set_recall += float(
                    batch_set_recall[batch_set_selected].sum())
                exact_sets += float(batch_exact_set[batch_set_selected].sum())
                cap_rows += int(batch_cap_bound.sum())
                cap_hits += float(batch_cap_hit[batch_cap_bound].sum())
                if (target_ids is not None and len(target_ids) and
                        "deck" in data):
                    target_np = np.isin(
                        data["deck"][indices].astype(np.uint64), target_ids)
                    win_np = target_np & (data["z"][indices] > 0)
                    loss_np = target_np & (data["z"][indices] < 0)
                    for outcome_np, totals in (
                            (win_np, "win"), (loss_np, "loss")):
                        if outcome_np.any():
                            outcome = torch.as_tensor(outcome_np, device=device)
                            if totals == "win":
                                target_win_total += int(outcome_np.sum())
                                target_win_ce += float(sample_ce[outcome].sum())
                            else:
                                target_loss_total += int(outcome_np.sum())
                                target_loss_ce += float(sample_ce[outcome].sum())
                    selected_np = target_np
                    if target_wins_only:
                        selected_np &= data["z"][indices] > 0
                    if selected_np.any():
                        selected = torch.as_tensor(selected_np, device=device)
                        target_total += int(selected_np.sum())
                        target_ce += float(sample_ce[selected].sum())
                        target_mae += float((value[selected] - z[selected]).abs().sum())
                        target_top1 += float(hit[selected].sum())
                        target_set = selected & batch_set_selected
                        target_set_rows += int(target_set.sum())
                        target_set_recall += float(
                            batch_set_recall[target_set].sum())
                        target_exact_sets += float(
                            batch_exact_set[target_set].sum())
                        target_cap = selected & batch_cap_bound
                        target_cap_rows += int(target_cap.sum())
                        target_cap_hits += float(batch_cap_hit[target_cap].sum())
                        clean_np = selected_np & ~np.isin(
                            data["ctx"][indices], EQUIVALENT_COPY_CONTEXTS)
                        if clean_np.any():
                            clean = torch.as_tensor(clean_np, device=device)
                            target_clean_total += int(clean_np.sum())
                            target_clean_ce += float(sample_ce[clean].sum())
                            target_clean_top1 += float(hit[clean].sum())
    result = {"ce": ce / total, "value_mae": mae / total,
              "top1": top1 / total, "decisions": int(total),
              "action_set_rows": int(set_rows),
              "action_set_recall": set_recall / set_rows if set_rows else None,
              "exact_action_set": exact_sets / set_rows if set_rows else None,
              "cap_bind_rows": int(cap_rows),
              "cap_bind_recall": cap_hits / cap_rows if cap_rows else None,
              "target_decisions": int(target_total)}
    if target_total:
        result.update({
            "target_ce": target_ce / target_total,
            "target_value_mae": target_mae / target_total,
            "target_top1": target_top1 / target_total,
            "target_action_set_rows": int(target_set_rows),
            "target_action_set_recall": (
                target_set_recall / target_set_rows if target_set_rows else None),
            "target_exact_action_set": (
                target_exact_sets / target_set_rows if target_set_rows else None),
            "target_cap_bind_rows": int(target_cap_rows),
            "target_cap_bind_recall": (
                target_cap_hits / target_cap_rows if target_cap_rows else None),
            "target_clean_decisions": int(target_clean_total),
            "target_clean_ce": (target_clean_ce / target_clean_total
                                if target_clean_total else None),
            "target_clean_top1": (target_clean_top1 / target_clean_total
                                  if target_clean_total else None),
            "target_win_decisions": int(target_win_total),
            "target_win_ce": (target_win_ce / target_win_total
                              if target_win_total else None),
            "target_loss_decisions": int(target_loss_total),
            "target_loss_ce": (target_loss_ce / target_loss_total
                               if target_loss_total else None),
        })
    else:
        result.update({"target_ce": None, "target_value_mae": None,
                       "target_top1": None, "target_clean_decisions": 0,
                       "target_action_set_rows": 0,
                       "target_action_set_recall": None,
                       "target_exact_action_set": None,
                       "target_cap_bind_rows": 0,
                       "target_cap_bind_recall": None,
                       "target_clean_ce": None, "target_clean_top1": None,
                       "target_win_decisions": 0, "target_win_ce": None,
                       "target_loss_decisions": 0, "target_loss_ce": None})
    result["target_set_error"] = (
        1.0 - result["target_exact_action_set"]
        if result["target_exact_action_set"] is not None else None)
    return result


def evaluate_teacher(net: torch.nn.Module, data: dict[str, np.ndarray],
                     device: torch.device, batch: int,
                     q_temperature: float,
                     q_shrink_visits: float = 0.0,
                     amp_ctx=contextlib.nullcontext) -> dict:
    """Evaluate the deployed object: ranking exact actions by searched Q."""
    total = hit_sum = loss_sum = regret_sum = 0.0
    net.eval()
    with torch.inference_mode():
        for start in range(0, len(data.get("ctx", ())), batch):
            indices = np.arange(start, min(start + batch, len(data["ctx"])))
            batch_tensors = tensor_batch(data, indices, device)
            with amp_ctx():
                logits, _value = forward_tensor_batch(net, batch_tensors)
            tensors = teacher_field_tensors(data, indices, device)
            tokens, sizes, q, visits, live = tensors
            per_sample, hit, regret, usable = teacher_action_policy_loss(
                logits.float(), tokens, sizes, q, live, visits,
                q_shrink_visits=q_shrink_visits,
                q_temperature=q_temperature)
            count = int(usable.sum())
            if count:
                total += count
                loss_sum += float(per_sample[usable].sum())
                hit_sum += float(hit[usable].sum())
                regret_sum += float(regret[usable].sum())
    if not total:
        return {"teacher_decisions": 0, "teacher_ce": None,
                "teacher_top1": None, "teacher_regret": None}
    return {"teacher_decisions": int(total),
            "teacher_ce": loss_sum / total,
            "teacher_top1": hit_sum / total,
            "teacher_regret": regret_sum / total}


def count_target_rows(paths: list[str], target_ids: np.ndarray,
                      wins_only: bool = False,
                      holdout_frac: float = 0.0,
                      holdout_seed: int = 0) -> int:
    """Cheap preflight for a meaningful deck-specific validation split."""
    total = 0
    for path in paths:
        with np.load(path) as source:
            if "deck" not in source:
                continue
            selected = np.isin(source["deck"].astype(np.uint64), target_ids)
            if wins_only:
                selected &= source["z"] > 0
            if holdout_frac > 0.0 and "group" in source:
                # Count only the held-out episodes; the rest are training rows.
                selected &= group_holdout_mask(
                    source["group"], holdout_frac, holdout_seed)
            total += int(selected.sum())
    return total


def load_target_reservoir(paths: list[str], target_ids: np.ndarray,
                          max_rows: int, seed: int,
                          wins_only: bool = False) -> dict[str, np.ndarray]:
    """Load a deterministic, bounded replay pool for the target archetype.

    The broad corpus stays shard-streamed. Only target rows are resident, so
    every optimizer batch can contain specialist practice without loading the
    multi-million-row general dataset. Sampling global target-row positions
    avoids favoring small shards or dates that happen to sort first.
    """
    counts = []
    total = 0
    for path in paths:
        with np.load(path) as source:
            if "deck" in source:
                selected = np.isin(
                    source["deck"].astype(np.uint64), target_ids)
                if wins_only:
                    selected &= source["z"] > 0
                count = int(selected.sum())
            else:
                count = 0
        counts.append(count)
        total += count
    if total == 0:
        return {}

    keep = min(total, max_rows)
    if keep == total:
        chosen = np.arange(total, dtype=np.int64)
    else:
        chosen = np.sort(np.random.default_rng(seed).choice(
            total, size=keep, replace=False))

    parts: dict[str, list[np.ndarray]] = {
        key: [] for key in FIELDS + ("elo", "deck", "group", "seat")}
    offset = 0
    for path, count in zip(paths, counts):
        if not count:
            continue
        lo = int(np.searchsorted(chosen, offset, side="left"))
        hi = int(np.searchsorted(chosen, offset + count, side="left"))
        old_offset = offset
        offset += count
        if lo == hi:
            continue
        with np.load(path) as source:
            selected = np.isin(
                source["deck"].astype(np.uint64), target_ids)
            if wins_only:
                selected &= source["z"] > 0
            target_rows = np.flatnonzero(selected)
            local = target_rows[chosen[lo:hi] - old_offset]
            for key in FIELDS:
                if key not in source:
                    raise ValueError(f"{path} misses {key}")
                parts[key].append(source[key][local])
            parts["deck"].append(source["deck"][local])
            # Elo 1000 is neutral under the trainer's weighting formula.
            parts["elo"].append(
                source["elo"][local] if "elo" in source else
                np.full(len(local), 1000, dtype=np.float32))
            # A trajectory is one player-seat within a game.  Preserve that
            # identity so optional replay weighting can stop long games from
            # dominating merely because they contain more decisions.
            parts["group"].append(
                source["group"][local].astype(np.uint64)
                if "group" in source else chosen[lo:hi].astype(np.uint64))
            parts["seat"].append(
                source["seat"][local].astype(np.int8)
                if "seat" in source else np.zeros(len(local), dtype=np.int8))
    result = {key: np.concatenate(value, axis=0)
              for key, value in parts.items()}
    result["episode_weight"] = episode_balance_weights(
        result["group"], result["seat"])
    return result


def episode_balance_weights(group: np.ndarray, seat: np.ndarray) -> np.ndarray:
    """Per-row weights giving every game-seat equal total replay mass."""
    if len(group) != len(seat):
        raise ValueError("group and seat lengths differ")
    if not len(group):
        return np.empty(0, dtype=np.float32)
    keys = np.column_stack((group.astype(np.uint64), seat.astype(np.uint64)))
    _unique, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True)
    weight = 1.0 / counts[inverse].astype(np.float64)
    weight /= weight.mean()
    return weight.astype(np.float32)


def replay_rows_per_epoch(stream_rows: int, pool_rows: int,
                          fraction: float, max_passes: float) -> int:
    """Bound a requested mixture by unique-pool exposure per epoch."""
    if not stream_rows or not pool_rows or not fraction:
        return 0
    requested = int(round(stream_rows * fraction / (1.0 - fraction)))
    cap = int(math.floor(pool_rows * max_passes + 1e-9))
    return min(requested, cap)


def make_replay_order(pool_rows: int, rows: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Cycle shuffled rows; no item repeats before every item is exposed."""
    if rows < 0 or pool_rows < 0 or (rows and not pool_rows):
        raise ValueError("invalid replay order sizes")
    parts = []
    remaining = rows
    while remaining:
        order = rng.permutation(pool_rows)
        take = min(remaining, pool_rows)
        parts.append(order[:take])
        remaining -= take
    return (np.concatenate(parts).astype(np.int64, copy=False)
            if parts else np.empty(0, dtype=np.int64))


def replay_aware_value_loss(value: torch.Tensor, z: torch.Tensor,
                            broad_rows: int,
                            replay_weight: float) -> torch.Tensor:
    """Value MSE with independently controllable auxiliary replay influence."""
    squared = (value.float() - z) ** 2
    if broad_rows == len(squared) or replay_weight == 1.0:
        return squared.mean()
    weights = torch.ones_like(squared)
    weights[broad_rows:] = replay_weight
    return (squared * weights).sum() / weights.sum().clamp_min(1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--val-data", nargs="+", default=None)
    parser.add_argument(
        "--holdout-group-frac", type=float, default=0.0,
        help="hold out this fraction of episode GROUPS from --data for "
             "validation, instead of reserving a separate day. Splitting by "
             "group is required: rows inside one game are correlated, so a "
             "row-level split leaks the same game into both sides. When set, "
             "--val-data defaults to --data and the split is by hash, not file.")
    parser.add_argument("--holdout-group-seed", type=int, default=20260814)
    parser.add_argument("--deck", required=True)
    parser.add_argument("--init", default=None,
                        help="warm-start checkpoint; its _meta fixes the "
                             "architecture. Omit to train from scratch at "
                             "--dim/--layers/--heads/--d-ff")
    parser.add_argument("--dim", type=int, default=None, help="model width")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--features", choices=("rich", "v2", "v3"), default="rich",
        help="v2 mechanically upgrades old rich shards; v3 requires a fresh "
             "raw ingest and adds public resource-set/structured-action state")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--specialist-weight", type=float, default=2.0)
    parser.add_argument(
        "--target-batch-fraction", type=float, default=0.0,
        help="fraction of each optimizer batch reserved for a bounded target-"
             "family replay stream")
    parser.add_argument("--target-reservoir-max", type=int, default=120000)
    parser.add_argument(
        "--target-replay-max-passes", type=float, default=1.0,
        help="maximum average exposures of each resident target row per epoch; "
             "the requested batch fraction is an upper bound")
    parser.add_argument(
        "--target-replay-balance", choices=("decisions", "episodes"),
        default="episodes",
        help="episodes gives each game-seat equal total auxiliary policy mass")
    parser.add_argument(
        "--target-replay-value-weight", type=float, default=0.0,
        help="relative value-loss weight on auxiliary replay rows; zero keeps "
             "winner/outcome selection from leaking into the shared value head")
    parser.add_argument(
        "--target-replay-outcome", choices=("all", "wins"), default="all",
        help="wins replays only successful target-deck lines in the reserved stream")
    parser.add_argument(
        "--teacher-data", nargs="+", default=None,
        help="DAgger/search-Q shard directories mixed as supervised exact-"
             "action labels; trajectories remain controlled by the learner")
    parser.add_argument(
        "--teacher-val-data", nargs="+", default=None,
        help="frozen, disjoint Q-labelled shards for regret-based selection")
    parser.add_argument("--teacher-batch-fraction", type=float, default=0.0,
                        help="upper bound on optimizer rows from teacher labels")
    parser.add_argument("--teacher-reservoir-max", type=int, default=100000)
    parser.add_argument("--teacher-val-max", type=int, default=30000)
    parser.add_argument(
        "--teacher-max-passes", type=float, default=1.0,
        help="maximum average teacher-label exposures per epoch")
    parser.add_argument("--teacher-q-temperature", type=float, default=0.10)
    parser.add_argument(
        "--teacher-q-shrink-visits", type=float, default=4.0,
        help="shrink noisy action Q toward the state's visit-weighted mean")
    parser.add_argument("--teacher-policy-weight", type=float, default=1.0,
                        help="relative weight per episode-balanced teacher state")
    parser.add_argument(
        "--equivalent-targets", choices=("off", "known"), default="known",
        help="remove arbitrary duplicate-card identity in contexts 7, 8, and 22")
    parser.add_argument(
        "--drop-unresolved-looking", action="store_true",
        help="exclude legacy rich-v1 rows where public area-12 card identities "
             "were incorrectly encoded as zero")
    parser.add_argument("--critical-weight", type=float, default=1.5)
    parser.add_argument("--elo-weight", type=float, default=0.5)
    parser.add_argument(
        "--min-elo", type=float, default=0.0,
        help="drop training rows whose pilot Elo is below this. --elo-weight "
             "only upweights above 1000, so without this every sub-1000 row "
             "trains at identical weight. Applies to the streamed corpus only; "
             "validation is left untouched so metrics stay comparable.")
    parser.add_argument("--winner-weight", type=float, default=1.1)
    parser.add_argument("--value-weight", type=float, default=0.02)
    parser.add_argument(
        "--stream-balance", choices=("decisions", "episodes"),
        default="decisions",
        help="episodes gives each broad-corpus game-seat equal total policy "
             "mass instead of allowing long/stalling games to dominate")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--selection-metric", choices=(
            "target_ce", "ce", "target_set_error", "teacher_regret"),
        default="target_ce",
        help="checkpoint selector. target_ce is the default because the agent "
             "can only play --deck; global CE remains a regression metric")
    parser.add_argument(
        "--selection-scope", choices=("exact", "family"), default="exact",
        help="which rows target_ce uses. exact selects for the submitted 60-card "
             "list; family includes the Jaccard-matched augmentation lists")
    parser.add_argument(
        "--selection-outcome", choices=("all", "wins"), default="all",
        help="which held-out outcomes target_ce selects on")
    parser.add_argument(
        "--min-target-val", type=int, default=1000,
        help="minimum target-family holdout rows required for target_ce selection")
    parser.add_argument("--variant-census", default=None,
                        help="census json; every deck within --variant-jaccard "
                             "of --deck also counts as the target archetype, "
                             "instead of only the one exact 60-card list")
    parser.add_argument("--variant-jaccard", type=float, default=0.80)
    parser.add_argument(
        "--target-deck-ids", nargs="*", type=int, default=None,
        help="explicit stable_id list for the target family, chosen by measured "
             "decision similarity (tools/deck_gameplay_similarity.py). When set, "
             "census jaccard matching is skipped entirely. The exact deck is "
             "always included and need not be listed.")
    parser.add_argument("--amp", choices=("off", "bf16"), default="off",
                        help="bf16 autocast for the matmuls. Master weights, "
                             "gradient accumulation and the softmax/CE stay "
                             "fp32, and export_npz writes fp32 regardless, so "
                             "the shipped model is unaffected in format. bf16 "
                             "rather than fp16 because it keeps fp32's 8-bit "
                             "exponent and needs no GradScaler.")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the net. Numerically equivalent, "
                             "just fused kernels; dynamic=True because the "
                             "last batch of every shard is short and would "
                             "otherwise trigger a recompile.")
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    parser.add_argument("--seed", type=int, default=917)
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 5:
        parser.error("--threads must be in [1,5]")
    # Every prior trained so far was still improving when its schedule ran out,
    # so the old ceiling of 12 was cutting runs short rather than protecting
    # anything. Early stopping on validation CE is the real guard.
    if args.epochs < 1 or args.epochs > 40:
        parser.error("--epochs must be in [1,40]")
    if args.batch < 32 or args.batch > 1024:
        parser.error("--batch must be in [32,1024]")
    if args.specialist_weight < 1:
        parser.error("--specialist-weight must be >=1")
    if not 0.0 <= args.target_batch_fraction < 0.5:
        parser.error("--target-batch-fraction must be in [0,0.5)")
    if args.target_reservoir_max < 1:
        parser.error("--target-reservoir-max must be positive")
    if args.target_replay_max_passes <= 0:
        parser.error("--target-replay-max-passes must be positive")
    if not 0.0 <= args.target_replay_value_weight <= 1.0:
        parser.error("--target-replay-value-weight must be in [0,1]")
    if not 0.0 <= args.teacher_batch_fraction < 0.5:
        parser.error("--teacher-batch-fraction must be in [0,0.5)")
    if args.teacher_batch_fraction and not args.teacher_data:
        parser.error("--teacher-batch-fraction needs --teacher-data")
    if args.teacher_reservoir_max < 1 or args.teacher_val_max < 1:
        parser.error("teacher reservoir limits must be positive")
    if args.teacher_max_passes <= 0:
        parser.error("--teacher-max-passes must be positive")
    if args.teacher_q_temperature <= 0:
        parser.error("--teacher-q-temperature must be positive")
    if args.teacher_q_shrink_visits < 0:
        parser.error("--teacher-q-shrink-visits must be nonnegative")
    if args.teacher_policy_weight <= 0:
        parser.error("--teacher-policy-weight must be positive")
    if args.selection_metric == "teacher_regret" and not args.teacher_val_data:
        parser.error("--selection-metric teacher_regret needs --teacher-val-data")
    if not args.val_data and args.holdout_group_frac <= 0.0:
        parser.error("pass --val-data, or --holdout-group-frac to split "
                     "episode groups out of --data instead")
    if not 0.0 <= args.holdout_group_frac < 0.5:
        parser.error("--holdout-group-frac must be in [0, 0.5)")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.features in ("v2", "v3") and args.init is not None:
        parser.error(f"feature {args.features} changes the feature ABI; train "
                     "it from scratch rather than warm-starting old weights")

    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available()
                          else "cpu")
    torch.set_num_threads(args.threads)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.75)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_paths = shard_paths(args.data)
    if args.holdout_group_frac > 0.0:
        # Group split: train and validation read the same shards and are
        # separated by a hash of the episode id, so no file-level exclusion
        # applies and no day has to be sacrificed.
        val_paths = shard_paths(args.val_data) if args.val_data else train_paths
        print(f"group holdout: {args.holdout_group_frac:.1%} of episodes "
              f"(seed {args.holdout_group_seed}) reserved from "
              f"{len(train_paths)} shards", flush=True)
    else:
        val_paths = shard_paths(args.val_data)
        # A validation shard that also sits under --data is trained on once per
        # epoch, which makes every reported ce/top1/target_* optimistic and lets
        # checkpoint selection pick the epoch that memorised it best. Passing a
        # diagnostic shard from inside the training directory is the natural way
        # to set this up, so drop the overlap here rather than trusting the
        # caller.
        val_keys = {os.path.realpath(path) for path in val_paths}
        kept = [path for path in train_paths
                if os.path.realpath(path) not in val_keys]
        if len(kept) != len(train_paths):
            dropped = len(train_paths) - len(kept)
            if not kept:
                parser.error(
                    "every --data shard is also a --val-data shard; nothing "
                    "left to train on")
            print(f"excluded {dropped} validation shard(s) from the training "
                  f"stream: {len(train_paths)} -> {len(kept)}", flush=True)
            train_paths = kept
    teacher_paths = shard_paths(args.teacher_data) if args.teacher_data else []
    teacher_val_paths = (shard_paths(args.teacher_val_data)
                         if args.teacher_val_data else [])
    teacher_replay = (
        load_teacher_reservoir(
            teacher_paths, args.teacher_reservoir_max, args.seed + 101)
        if args.teacher_batch_fraction else {})
    teacher_val = (
        load_teacher_reservoir(
            teacher_val_paths, args.teacher_val_max, args.seed + 202)
        if teacher_val_paths else {})
    if args.teacher_batch_fraction and not teacher_replay:
        parser.error("teacher stream has no rows with at least two searched actions")
    deck_cards = read_deck(args.deck)
    exact_id = stable_id(tuple(sorted(deck_cards)))
    target_ids = {exact_id}
    if args.target_deck_ids:
        # Explicit family, chosen by measured decision similarity rather than by
        # card overlap. Jaccard is a proxy for deck composition, not for how a
        # deck is played: for Bug Catching the list that actually shares our
        # decision distribution sits at jaccard 0.714 and was being excluded,
        # while two lists at 0.913 and 0.955 that need different logic were
        # being pulled in at 3x specialist weight.
        extra = {int(x) for x in args.target_deck_ids} - {exact_id}
        target_ids |= extra
        print(f"explicit target family: exact deck + {len(extra)} measured-"
              f"similar list(s); census jaccard matching disabled", flush=True)
        for _id in sorted(extra):
            print(f"   deck id {_id}", flush=True)
    elif args.variant_census and os.path.isfile(args.variant_census):
        found = variant_ids(deck_cards, args.variant_census, args.variant_jaccard)
        target_ids |= set(found)
        print(f"archetype variants at jaccard >= {args.variant_jaccard}: "
              f"{len(found)} census lists", flush=True)
        for _id, (label, score, seats) in sorted(
                found.items(), key=lambda kv: -kv[1][2])[:8]:
            print(f"   {seats:6d} seats  jaccard {score:.2f}  {label}", flush=True)
    target_array = np.array(sorted(target_ids), dtype=np.uint64)
    target_replay = (
        load_target_reservoir(
            train_paths, target_array, args.target_reservoir_max, args.seed,
            wins_only=args.target_replay_outcome == "wins")
        if args.target_batch_fraction else {})
    if args.target_batch_fraction and not target_replay:
        parser.error("--target-batch-fraction requested but no matching target "
                     "replay rows were found")
    selection_array = (target_array if args.selection_scope == "family" else
                       np.array([exact_id], dtype=np.uint64))
    hold = dict(holdout_frac=args.holdout_group_frac,
                holdout_seed=args.holdout_group_seed)
    exact_val_rows = count_target_rows(
        val_paths, np.array([exact_id], dtype=np.uint64), **hold)
    family_val_rows = count_target_rows(val_paths, target_array, **hold)
    target_val_rows = count_target_rows(
        val_paths, selection_array,
        wins_only=args.selection_outcome == "wins", **hold)
    print(f"validation decisions: exact={exact_val_rows:,}; "
          f"family={family_val_rows:,}; selection={args.selection_scope} "
          f"outcome={args.selection_outcome} ({target_val_rows:,})", flush=True)
    if (args.selection_metric in ("target_ce", "target_set_error") and
            target_val_rows < args.min_target_val):
        parser.error(
            f"--selection-metric {args.selection_metric} needs at least "
            f"{args.min_target_val:,} {args.selection_scope}-scope validation "
            f"rows; found "
            f"{target_val_rows:,}. Refresh the holdout/census or explicitly "
            "select --selection-metric ce as a documented fallback.")
    arch_parts = (args.dim, args.layers, args.heads, args.d_ff)
    if any(part is not None for part in arch_parts):
        if any(part is None for part in arch_parts):
            parser.error("--dim, --layers, --heads and --d-ff go together")
        arch = tuple(int(part) for part in arch_parts)
    else:
        arch = None
    if args.init is None and arch is None:
        parser.error("need --init to warm-start, or a full architecture to "
                     "train from scratch")
    feature_module = {
        "rich": features, "v2": features_v2, "v3": features_v3,
    }[args.features]
    net = configure_and_load(args.init, device, arch, feature_module)
    active = (model_module.D_MODEL, model_module.N_LAYERS,
              model_module.N_HEADS, model_module.D_FF)
    params = sum(p.numel() for p in net.parameters())
    print(f"architecture {active}; features={args.features} "
          f"(seq={model_module.NF.SEQ}, cards={model_module.NF.N_CARD}); "
          f"{params:,} parameters; "
          f"{'warm start from ' + args.init if args.init else 'random init'}",
          flush=True)
    if args.amp == "bf16" and device.type == "cuda":
        amp_ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        print("mixed precision: bf16 autocast (fp32 master weights)", flush=True)
    else:
        amp_ctx = contextlib.nullcontext
    if args.compile:
        # Inductor compiles lazily at the first forward, so a missing C
        # compiler surfaces mid-training rather than here. This box has no gcc,
        # so the flag is accepted and ignored rather than crashing an epoch in.
        import shutil as _shutil
        if _shutil.which("cc") or _shutil.which("gcc"):
            net = torch.compile(net, dynamic=True)
            print("torch.compile enabled", flush=True)
        else:
            print("torch.compile skipped: no C compiler for Triton", flush=True)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    critical_ids = torch.tensor(CRITICAL_CONTEXTS, device=device)
    rng = np.random.default_rng(args.seed)
    history = []
    best_score = math.inf
    best_metrics = None
    best_state = None
    stale = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"device={device}; {len(train_paths)} streaming train shards; "
          f"{len(val_paths)} validation shards; "
          f"target archetype lists={len(target_ids)}", flush=True)
    stream_rows_total = 0
    unresolved_rows_dropped = 0
    stream_trajectory_counts: dict[tuple[int, int], int] = {}
    for path in train_paths:
        data = load_shard(path)
        # Must apply the SAME filters the training loop does. The replay and
        # teacher schedules pace themselves against stream_rows_total, so an
        # overcount here means they never reach their row budget and the
        # end-of-epoch consistency check fires.
        eligible = eligible_row_indices(
            data, args.drop_unresolved_looking, args.min_elo,
            args.holdout_group_frac, args.holdout_group_seed,
            want_holdout=False)
        stream_rows_total += len(eligible)
        unresolved_rows_dropped += len(data["pi"]) - len(eligible)
        if args.stream_balance == "episodes":
            add_trajectory_counts(stream_trajectory_counts, data, eligible)
    trajectory_normalization = (
        stream_rows_total / len(stream_trajectory_counts)
        if stream_trajectory_counts else 1.0)
    if args.stream_balance == "episodes":
        if not stream_trajectory_counts:
            parser.error("--stream-balance episodes needs group and seat fields")
        print(f"broad stream balanced across "
              f"{len(stream_trajectory_counts):,} game-seats", flush=True)
    if unresolved_rows_dropped:
        print(f"dropped {unresolved_rows_dropped:,} legacy unresolved LOOKING "
              f"rows from the broad stream", flush=True)
    replay_rows_epoch = replay_rows_per_epoch(
        stream_rows_total, len(target_replay.get("pi", ())),
        args.target_batch_fraction, args.target_replay_max_passes)
    teacher_rows_epoch = replay_rows_per_epoch(
        stream_rows_total, len(teacher_replay.get("pi", ())),
        args.teacher_batch_fraction, args.teacher_max_passes)
    effective_replay_fraction = (
        replay_rows_epoch / (stream_rows_total + replay_rows_epoch)
        if replay_rows_epoch else 0.0)
    epoch_rows_total = stream_rows_total + replay_rows_epoch + teacher_rows_epoch
    replay_per_full_batch = int(round(
        args.batch * replay_rows_epoch / epoch_rows_total))
    teacher_per_full_batch = int(round(
        args.batch * teacher_rows_epoch / epoch_rows_total))
    stream_batch = args.batch - replay_per_full_batch - teacher_per_full_batch
    if stream_batch < 1:
        parser.error("auxiliary streams leave no broad rows in a batch")
    if target_replay:
        print(f"two-stream replay: {len(target_replay['pi']):,} "
              f"{args.target_replay_outcome} target rows; "
              f"{replay_rows_epoch:,}/epoch "
              f"({replay_rows_epoch / len(target_replay['pi']):.2f} "
              f"passes, {effective_replay_fraction:.2%} effective; "
              f"requested <= {args.target_batch_fraction:.1%}); "
              f"batch~{stream_batch} broad + {replay_per_full_batch} target; "
              f"balance={args.target_replay_balance}; "
              f"replay_value_weight={args.target_replay_value_weight:g}",
              flush=True)
    if teacher_replay:
        effective_teacher = teacher_rows_epoch / epoch_rows_total
        print(f"teacher stream: {len(teacher_replay['pi']):,} usable labels; "
              f"{teacher_rows_epoch:,}/epoch "
              f"({teacher_rows_epoch / len(teacher_replay['pi']):.2f} passes, "
              f"{effective_teacher:.2%} effective; "
              f"batch~{teacher_per_full_batch} teacher; "
              f"qT={args.teacher_q_temperature:g}; "
              f"shrink={args.teacher_q_shrink_visits:g}; "
              f"weight={args.teacher_policy_weight:g})", flush=True)
    if teacher_val:
        print(f"frozen teacher validation: {len(teacher_val['pi']):,} states",
              flush=True)
    if args.equivalent_targets == "known":
        print("equivalent-copy targets enabled for contexts 7, 8, 22", flush=True)
    for epoch in range(1, args.epochs + 1):
        net.train()
        paths = list(train_paths)
        rng.shuffle(paths)
        epoch_replay_order = make_replay_order(
            len(target_replay.get("pi", ())), replay_rows_epoch, rng)
        epoch_teacher_order = make_replay_order(
            len(teacher_replay.get("pi", ())), teacher_rows_epoch, rng)
        replay_cursor = teacher_cursor = 0
        seen = replay_seen = teacher_seen = specialist_seen = steps = 0
        teacher_usable_seen = teacher_hit_sum = 0
        teacher_regret_sum = 0.0
        loss_sum = policy_sum = 0.0
        for shard_i, path in enumerate(paths, 1):
            data = load_shard(path)
            eligible = eligible_row_indices(
                data, args.drop_unresolved_looking, args.min_elo,
                args.holdout_group_frac, args.holdout_group_seed,
                want_holdout=False)
            order = rng.permutation(eligible)
            for start in range(0, len(order), stream_batch):
                indices = order[start:start + stream_batch]
                batch_tensors = tensor_batch(data, indices, device)
                replay_count = 0
                if replay_rows_epoch:
                    # A cumulative schedule spreads a bounded epoch budget
                    # across shards.  The shuffled order never repeats a row
                    # until the resident pool has been exhausted once.
                    desired = int(round(
                        replay_rows_epoch * (seen + len(indices)) /
                        stream_rows_total))
                    replay_count = desired - replay_seen
                    replay_indices = epoch_replay_order[
                        replay_cursor:replay_cursor + replay_count]
                    replay_cursor += replay_count
                    replay_tensors = tensor_batch(
                        target_replay, replay_indices, device)
                    batch_tensors = tuple(
                        torch.cat((broad, target), dim=0)
                        for broad, target in zip(
                            batch_tensors,
                            replay_tensors))

                normal_count = len(indices) + replay_count
                teacher_count = 0
                if teacher_rows_epoch:
                    desired_teacher = int(round(
                        teacher_rows_epoch * (seen + len(indices)) /
                        stream_rows_total))
                    teacher_count = desired_teacher - teacher_seen
                    teacher_indices = epoch_teacher_order[
                        teacher_cursor:teacher_cursor + teacher_count]
                    teacher_cursor += teacher_count
                    teacher_tensors = tensor_batch(
                        teacher_replay, teacher_indices, device)
                    batch_tensors = tuple(
                        torch.cat((ordinary, teacher), dim=0)
                        for ordinary, teacher in zip(
                            batch_tensors,
                            teacher_tensors))

                kind, card, scal, mask, ctx, stype, pi, z = batch_tensors[:8]

                elo_np = (data["elo"][indices] if "elo" in data else
                          np.full(len(indices), 1000, dtype=np.float32))
                deck_np = (data["deck"][indices] if "deck" in data else
                           np.zeros(len(indices), dtype=np.uint64))
                if replay_count:
                    elo_np = np.concatenate(
                        (elo_np, target_replay["elo"][replay_indices]))
                    deck_np = np.concatenate(
                        (deck_np, target_replay["deck"][replay_indices]))
                with amp_ctx():
                    logits, value = forward_tensor_batch(net, batch_tensors)
                ordinary = slice(0, normal_count)
                if args.equivalent_targets == "known":
                    per_sample, _ = equivalent_option_policy_loss(
                        logits[ordinary].float(), pi[ordinary], kind[ordinary],
                        card[ordinary], scal[ordinary], mask[ordinary],
                        ctx[ordinary])
                else:
                    per_sample, _ = option_policy_loss(
                        logits[ordinary].float(), pi[ordinary], kind[ordinary],
                        mask[ordinary])
                weight = torch.ones_like(per_sample)
                if args.stream_balance == "episodes":
                    broad_policy_weight = torch.as_tensor(
                        trajectory_batch_weights(
                            data, indices, stream_trajectory_counts,
                            trajectory_normalization),
                        device=device, dtype=weight.dtype)
                    weight[:len(indices)] *= broad_policy_weight
                if replay_count and args.target_replay_balance == "episodes":
                    replay_policy_weight = torch.as_tensor(
                        target_replay["episode_weight"][replay_indices],
                        device=device, dtype=weight.dtype)
                    weight[-replay_count:] *= replay_policy_weight
                critical = (
                    ctx[ordinary, None] == critical_ids[None, :]).any(-1)
                weight *= torch.where(
                    critical,
                    torch.full_like(weight, args.critical_weight),
                    torch.ones_like(weight))
                if "elo" in data or replay_count:
                    elo = torch.as_tensor(elo_np, device=device)
                    weight *= 1.0 + args.elo_weight * torch.clamp(
                        (elo - 1000.0) / 250.0, 0.0, 1.0)
                if "deck" in data or replay_count:
                    is_target_np = np.isin(
                        deck_np.astype(np.uint64), target_array)
                    is_target = torch.as_tensor(is_target_np, device=device)
                    weight *= torch.where(
                        is_target,
                        torch.full_like(weight, args.specialist_weight),
                        torch.ones_like(weight))
                    specialist_seen += int(is_target_np.sum())
                weight *= torch.where(
                    z[ordinary] > 0,
                    torch.full_like(weight, args.winner_weight),
                    torch.where(
                        z[ordinary] < 0,
                        torch.full_like(weight, 1.0 / args.winner_weight),
                        torch.ones_like(weight)))
                policy_numerator = (per_sample * weight).sum()
                policy_denominator = weight.sum()
                if teacher_count:
                    teacher_slice = slice(normal_count, normal_count + teacher_count)
                    teacher_arrays = teacher_field_tensors(
                        teacher_replay, teacher_indices, device)
                    tokens, sizes, action_q, action_visits, action_mask = teacher_arrays
                    teacher_ce, teacher_hit, teacher_regret, teacher_usable = (
                        teacher_action_policy_loss(
                            logits[teacher_slice].float(), tokens, sizes,
                            action_q, action_mask, action_visits,
                            q_shrink_visits=args.teacher_q_shrink_visits,
                            q_temperature=args.teacher_q_temperature))
                    teacher_weight = torch.as_tensor(
                        teacher_replay["episode_weight"][teacher_indices],
                        device=device, dtype=policy_numerator.dtype)
                    teacher_weight *= teacher_usable.to(teacher_weight.dtype)
                    teacher_weight *= args.teacher_policy_weight
                    policy_numerator += (teacher_ce * teacher_weight).sum()
                    policy_denominator += teacher_weight.sum()
                    teacher_usable_seen += int(teacher_usable.sum())
                    teacher_hit_sum += int(teacher_hit[teacher_usable].sum())
                    teacher_regret_sum += float(
                        teacher_regret[teacher_usable].sum().detach())
                policy_loss = policy_numerator / policy_denominator.clamp_min(1.0)
                value_loss = replay_aware_value_loss(
                    value[ordinary], z[ordinary], len(indices),
                    args.target_replay_value_weight)
                loss = policy_loss + args.value_weight * value_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.8)
                optimizer.step()
                seen += len(indices)
                replay_seen += replay_count
                teacher_seen += teacher_count
                steps += 1
                loss_sum += float(loss.detach())
                policy_sum += float(policy_loss.detach())
            print(f"  epoch {epoch}: shard {shard_i}/{len(paths)}; "
                  f"broad={seen:,}; target_replay={replay_seen:,}; "
                  f"teacher={teacher_seen:,}", flush=True)
        if replay_seen != replay_rows_epoch or replay_cursor != replay_rows_epoch:
            raise RuntimeError(
                f"replay schedule consumed {replay_seen:,}/{replay_rows_epoch:,} rows")
        if (teacher_seen != teacher_rows_epoch or
                teacher_cursor != teacher_rows_epoch):
            raise RuntimeError(
                f"teacher schedule consumed {teacher_seen:,}/"
                f"{teacher_rows_epoch:,} rows")
        metrics = evaluate(
            net, val_paths, device, max(args.batch, 512), selection_array,
            target_wins_only=args.selection_outcome == "wins",
            equivalent_targets=args.equivalent_targets == "known",
            drop_unresolved_looking=args.drop_unresolved_looking,
            amp_ctx=amp_ctx,
            holdout_frac=args.holdout_group_frac,
            holdout_seed=args.holdout_group_seed)
        if teacher_val:
            metrics.update(evaluate_teacher(
                net, teacher_val, device, max(args.batch, 512),
                args.teacher_q_temperature, args.teacher_q_shrink_visits,
                amp_ctx=amp_ctx))
        else:
            metrics.update({"teacher_decisions": 0, "teacher_ce": None,
                            "teacher_top1": None, "teacher_regret": None})
        metrics.update({"epoch": epoch,
                        "train_decisions": seen + replay_seen + teacher_seen,
                        "train_stream_decisions": seen,
                        "train_replay_decisions": replay_seen,
                        "train_replay_unique_decisions": int(
                            np.unique(epoch_replay_order).size),
                        "train_replay_passes": (
                            replay_seen / len(target_replay["pi"])
                            if replay_seen else 0.0),
                        "train_teacher_decisions": teacher_seen,
                        "train_teacher_unique_decisions": int(
                            np.unique(epoch_teacher_order).size),
                        "train_teacher_passes": (
                            teacher_seen / len(teacher_replay["pi"])
                            if teacher_seen else 0.0),
                        "train_teacher_top1": (
                            teacher_hit_sum / teacher_usable_seen
                            if teacher_usable_seen else None),
                        "train_teacher_regret": (
                            teacher_regret_sum / teacher_usable_seen
                            if teacher_usable_seen else None),
                        "train_target_decisions": specialist_seen,
                        "train_loss": loss_sum / max(steps, 1),
                        "train_policy_loss": policy_sum / max(steps, 1),
                        "lr": optimizer.param_groups[0]["lr"]})
        history.append(metrics)
        stem, extension = os.path.splitext(args.out)
        export_npz(net, f"{stem}_epoch{epoch:02d}{extension}")
        print(f"epoch {epoch}: {metrics}", flush=True)
        selection_score = metrics[args.selection_metric]
        if selection_score is None:
            raise RuntimeError(
                f"selection metric {args.selection_metric} is unavailable")
        if selection_score < best_score - 1e-5:
            best_score = selection_score
            best_metrics = dict(metrics)
            best_state = {key: value.detach().cpu().clone()
                          for key, value in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if args.patience and stale >= args.patience:
            print(f"early stop after {stale} non-improving validation epochs")
            break

    if best_state is None:
        raise RuntimeError("no prior checkpoint was produced")
    net.load_state_dict(best_state)
    export_npz(net, args.out)
    with open(args.out + ".metrics.json", "w", encoding="utf-8") as target:
        json.dump({"config": vars(args), "selection_metric": args.selection_metric,
                   "best_score": best_score,
                   "best_ce": best_metrics["ce"] if best_metrics else None,
                   "best_metrics": best_metrics, "history": history},
                  target, indent=2)
    print(f"exported best prior to {args.out}; "
          f"{args.selection_metric}={best_score:.6f}")


if __name__ == "__main__":
    main()

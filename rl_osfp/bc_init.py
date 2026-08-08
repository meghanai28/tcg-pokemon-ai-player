"""Start PPO from the behaviour-cloned prior instead of from random weights.

This closes the one cell of the experiment table that was never runnable:

    RL from random init, full scale (6.4M decisions)      ladder 480
    BC alone                                              ladder 967 / 917 / 873
    BC + 26,642 decisions of GRPO (642 optimizer steps)   ladder 972
    BC init + full-scale RL                               NEVER RUN

The last row was not tried and rejected; `train.py` simply had no way to load
anything but its own `training_state.pt`, so every RL run in this repo started
from noise.  That is also the only measured difference between the arm that
reached 972 and the arm that reached 480.

The two architectures are the same trunk.  `bc_train/model.py`'s TCGNet and
`rl_osfp/network.py`'s ActorCritic share every embedding, every block and the
final layer norm, name for name and shape for shape.  Only three things differ:

    pol_head   -> option_head     same shape, renamed
    val_fc1/2  -> value_fc1/2     same shape, renamed
    count_head                    new, absent from the BC model

**Sequence length does not have to match.**  There is no positional embedding
anywhere in the trunk: a token is `card_emb + kind_emb + scal_proj(scalars)`,
and attention is full.  The network is therefore permutation-equivariant over
tokens, so weights trained with the champion's encoder (MAX_OPT 24, SEQ 53)
load and run unchanged under rl_osfp's (MAX_OPT 64, SEQ 93).  The wider encoder
only means fewer options get truncated.

`count_head` is zero-initialised, which leaves a uniform distribution over the
legal counts.  That is deliberate and cheap: `policy.py` masks the count to
[minCount, maxCount], and 84.85% of real decisions have minCount == maxCount ==
1 (measured over 257,223 ladder selections), so the mask decides the count
outright and the head only matters on the remaining sixth.
"""
from __future__ import annotations

import numpy as np
import torch

from .network import ActorCritic, NetworkConfig

# BC checkpoint name -> ActorCritic name.  Everything not listed maps to itself.
RENAMES = {
    "pol_head.weight": "option_head.weight",
    "pol_head.bias": "option_head.bias",
    "val_fc1.weight": "value_fc1.weight",
    "val_fc1.bias": "value_fc1.bias",
    "val_fc2.weight": "value_fc2.weight",
    "val_fc2.bias": "value_fc2.bias",
}
FRESH = ("count_head.weight", "count_head.bias")


def export_champion_npz(model: ActorCritic, path: str) -> dict:
    """Write an ActorCritic back out in the champion archive's own format.

    This is the shipping path, and it is why the whole exercise stays a clean
    single-variable experiment: the result drops into
    `harness/anchors/grpo_tech_grim_972_912_811.tar.gz` by replacing `model.npz`
    and nothing else, so a gate against the champion isolates the weights.

    `count_head` is dropped rather than serialised.  The frozen shell never asks
    for a count: it generates candidate option sets itself and uses the network
    only for per-option priors, so the count head is training machinery that has
    no deployment.  Keeping it would also break the shell's loader, which
    expects exactly the champion's 75 tensors.

    Only valid for a model trained under `PTCG_MAX_OPT=24`.  At the default 64
    the encoder computes a different `g[17]` and a longer sequence than the
    archive's `nn_features.py` will produce at inference, which is a silent
    mismatch rather than an error.
    """
    inverse = {v: k for k, v in RENAMES.items()}
    out: dict[str, np.ndarray] = {}
    for name, tensor in model.state_dict().items():
        if name in FRESH:
            continue
        out[inverse.get(name, name)] = tensor.detach().cpu().numpy().astype(np.float32)
    cfg = model.cfg
    out["_meta"] = np.array([cfg.d_model, cfg.layers, cfg.heads, cfg.d_ff], dtype=np.int64)
    np.savez(path, **out)
    return {"path": path, "tensors": len(out),
            "dropped": list(FRESH), "meta": out["_meta"].tolist()}


def config_from_meta(meta: np.ndarray) -> NetworkConfig:
    """BC checkpoints carry `_meta = [d_model, layers, heads, d_ff]`."""
    if len(meta) < 4:
        raise ValueError(f"_meta has {len(meta)} entries, expected at least 4")
    d_model, layers, heads, d_ff = (int(x) for x in meta[:4])
    return NetworkConfig(d_model=d_model, layers=layers, heads=heads, d_ff=d_ff)


def load_bc_checkpoint(path: str) -> tuple[ActorCritic, NetworkConfig, dict]:
    """Build an ActorCritic holding the BC weights.

    Raises rather than warns on any tensor that does not line up.  A silently
    partial load is the exact failure this repo keeps hitting: the archive is
    well formed, nothing raises, and the agent plays a different policy than the
    one that was selected.
    """
    with np.load(path) as data:
        if "_meta" not in data.files:
            raise ValueError(f"{path} has no _meta; not a checkpoint this loader understands")
        cfg = config_from_meta(data["_meta"])
        source = {k: np.asarray(data[k]) for k in data.files if k != "_meta"}

    model = ActorCritic(cfg)
    target = model.state_dict()

    mapped: dict[str, torch.Tensor] = {}
    for name, array in source.items():
        dest = RENAMES.get(name, name)
        if dest not in target:
            raise ValueError(f"{path}: tensor {name!r} maps to {dest!r}, "
                             f"which ActorCritic does not have")
        if tuple(target[dest].shape) != tuple(array.shape):
            raise ValueError(f"{path}: {name} is {array.shape}, "
                             f"but {dest} expects {tuple(target[dest].shape)}")
        mapped[dest] = torch.from_numpy(array.astype(np.float32))

    missing = [k for k in target if k not in mapped]
    if sorted(missing) != sorted(FRESH):
        raise ValueError(f"{path}: expected only {FRESH} to be missing, got {missing}")
    for name in FRESH:
        mapped[name] = torch.zeros_like(target[name])

    model.load_state_dict(mapped)
    report = {
        "source": path,
        "d_model": cfg.d_model, "layers": cfg.layers,
        "heads": cfg.heads, "d_ff": cfg.d_ff,
        "tensors_copied": len(source),
        "tensors_zero_initialised": list(FRESH),
    }
    return model, cfg, report

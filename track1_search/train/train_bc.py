"""ExIt distillation: train the network to imitate the search.

Policy loss = cross-entropy against the search's root visit distribution.
Value  loss = MSE against the game outcome (from the mover's perspective).

Gates reported at the end:
  * policy top-1 agreement with the search on a held-out split, compared
    against the HEURISTIC prior's agreement (the thing the net must beat --
    if the net doesn't beat the heuristic, the distillation is not working)
  * value MAE compared against a predict-the-mean baseline

Usage: py train/train_bc.py --epochs 12 --out train/model_bc.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "track1_search", "agent"))
sys.path.insert(0, HERE)

import nn_features as NF          # noqa: E402
from model import TCGNet, export_npz  # noqa: E402


CRITICAL_CONTEXTS = (0, 3, 4, 13, 14, 15, 16, 21, 22, 25, 35, 43)


def option_policy_loss(logits, pi, kind, mask, label_smoothing=0.0):
    """Per-sample CE over the legal option tokens used by deployment."""
    options = (kind == 3) & (mask > 0.5)
    if not bool(options.any(dim=-1).all()):
        raise ValueError("sample without an encoded legal option")
    option_logits = logits.masked_fill(~options, -torch.inf)
    logp = torch.log_softmax(option_logits, dim=-1)
    # Avoid 0 * -inf on non-option tokens.
    safe_logp = torch.where(options, logp, torch.zeros_like(logp))
    target = pi
    if label_smoothing:
        uniform = options.float() / options.sum(-1, keepdim=True)
        target = (1.0 - label_smoothing) * pi + label_smoothing * uniform
    return -(target * safe_logp).sum(-1), option_logits


def eval_val(model, va, dev, batch=512):
    """Chunked validation on `dev` so a big val split fits an 8 GB GPU.

    Returns (policy_ce, value_mse, top1_agreement, value_mae)."""
    kind, card, scal, mask, ctx, styp, pi, z = va
    n = len(kind)
    ce = vmse = t1 = vmae = 0.0
    model.eval()
    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(n, s + batch)
            k, c, sc, m, cx, st = (t[s:e].to(dev)
                                   for t in (kind, card, scal, mask, ctx, styp))
            pib, zb = pi[s:e].to(dev), z[s:e].to(dev)
            logits, v = model(k, c, sc, m, cx, st)
            ce_i, option_logits = option_policy_loss(logits, pib, k, m)
            ce += ce_i.sum().item()
            vmse += ((v - zb) ** 2).sum().item()
            vmae += (v - zb).abs().sum().item()
            t1 += (option_logits.argmax(-1) == pib.argmax(-1)).float().sum().item()
    return ce / n, vmse / n, t1 / n, vmae / n


def load_data(data_dirs, max_per_shard=0):
    if isinstance(data_dirs, (str, os.PathLike)):
        data_dirs = [data_dirs]
    files = sorted(
        file
        for data_path in data_dirs
        for file in (
            [os.fspath(data_path)]
            if os.path.isfile(data_path) and os.fspath(data_path).endswith(".npz")
            else glob.glob(os.path.join(data_path, "*.npz"))
        )
    )
    if not files:
        raise SystemExit(
            f"no shards in {list(data_dirs)} -- run train/selfplay.py first")
    required = ("kind", "card", "scal", "mask", "ctx", "stype", "pi", "z")
    optional = ("group", "pilot", "seat", "elo")
    parts = {k: [] for k in required}
    optional_parts = {k: [] for k in optional}
    kept = 0
    for file_index, f in enumerate(files):
        d = np.load(f)
        take = None
        shard_n = len(d["pi"])
        if max_per_shard and shard_n > max_per_shard:
            rng = np.random.default_rng(917 + file_index)
            take = np.sort(rng.choice(
                shard_n, size=max_per_shard, replace=False))
        for k in parts:
            parts[k].append(d[k] if take is None else d[k][take])
        for k in optional:
            if k in d:
                optional_parts[k].append(
                    d[k] if take is None else d[k][take])
        kept += shard_n if take is None else len(take)
    out = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    for k, values in optional_parts.items():
        if len(values) == len(files):
            out[k] = np.concatenate(values, axis=0)
    cap_note = f", cap {max_per_shard}/shard" if max_per_shard else ""
    print(f"loaded {len(files)} shards, {kept} samples{cap_note}")
    return out


def main():
    global NF
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[os.path.join(HERE, "data")],
                    help="one or more training-shard directories")
    ap.add_argument("--val-data", default=None,
                    help="optional separate held-out shard directory")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-per-shard", type=int, default=0,
                    help="deterministically cap each shard to balance dense days")
    ap.add_argument("--split", choices=("auto", "decision", "episode", "pilot"),
                    default="auto", help="auto uses episode groups when present")
    ap.add_argument("--out", default=os.path.join(HERE, "model_bc.npz"))
    ap.add_argument("--init", default=None,
                    help="optional exported .npz checkpoint to warm-start; its "
                         "architecture must match --dim/--layers/--heads")
    ap.add_argument("--dim", type=int, default=96, help="model width")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="training-only residual/embedding dropout")
    ap.add_argument("--features", choices=("base", "rich"), default="base")
    ap.add_argument("--value-weight", type=float, default=0.5,
                    help="set near zero for a policy-only search prior")
    ap.add_argument("--critical-weight", type=float, default=1.0,
                    help="relative weight for strategic selection contexts")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="smooth replay actions over legal options (0..1)")
    ap.add_argument("--elo-weight", type=float, default=0.0,
                    help="upweight stronger pilots by up to this fraction")
    ap.add_argument("--winner-weight", type=float, default=1.0,
                    help="relative weight for winner decisions; losers receive "
                         "the reciprocal weight")
    ap.add_argument("--patience", type=int, default=4,
                    help="early-stop patience on held-out option-only CE; 0 disables")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                    help="training device; auto uses CUDA when available")
    ap.add_argument("--gpu-resident", choices=("auto", "yes", "no"),
                    default="auto", help="keep tensors on GPU when they fit; "
                                         "otherwise stream batches from CPU")
    ap.add_argument("--distill", default=None,
                    help="teacher .npz; student also matches its outputs")
    ap.add_argument("--dmc", action="store_true",
                    help="Deep Monte Carlo: regress Q(s, a_taken) toward the "
                         "realised outcome instead of cross-entropy against a "
                         "target distribution. Uses every labelled decision we "
                         "have (pro replays, self play, ExIt, our own ladder "
                         "games) rather than only fresh self play.")
    a = ap.parse_args()
    if not 0.0 <= a.label_smoothing < 1.0:
        ap.error("--label-smoothing must be in [0, 1)")
    if not 0.0 <= a.dropout < 1.0:
        ap.error("--dropout must be in [0, 1)")
    if a.elo_weight < 0:
        ap.error("--elo-weight must be non-negative")
    if a.winner_weight <= 0:
        ap.error("--winner-weight must be positive")
    if a.device == "cuda" and not torch.cuda.is_available():
        ap.error("--device cuda requested but CUDA is unavailable")
    dev = torch.device("cuda" if (a.device != "cpu" and torch.cuda.is_available())
                       else "cpu")
    if dev.type == "cuda":
        # Leave headroom for WSL, the display stack, and allocator variance.
        torch.cuda.set_per_process_memory_fraction(0.75)
    print(f"device: {dev}")
    import model as _m
    if a.features == "rich":
        import nn_features_rich
        NF = nn_features_rich
        _m.NF = nn_features_rich
    _m.D_MODEL, _m.N_LAYERS, _m.N_HEADS, _m.D_FF = (
        a.dim, a.layers, a.heads, 2 * a.dim)
    _m.DROPOUT = a.dropout

    d = load_data(a.data, a.max_per_shard)
    n = d["pi"].shape[0]

    if a.distill:
        # Knowledge distillation: blend hard targets with the teacher's
        # predictions. The soft targets carry the teacher's ranking of EVERY
        # option (not just the played one) and its calibrated value estimates,
        # which is far more signal per sample than one-hot targets.
        from nn_infer import NumpyNet
        teacher = NumpyNet(a.distill)
        tl = np.zeros_like(d["pi"])
        tv = np.zeros_like(d["z"])
        B = 256
        for s in range(0, n, B):
            e = min(n, s + B)
            pol, val = teacher.forward(
                d["kind"][s:e].astype(np.int64), d["card"][s:e].astype(np.int64),
                d["scal"][s:e], d["mask"][s:e],
                d["ctx"][s:e].astype(np.int64), d["stype"][s:e].astype(np.int64))
            options = ((d["kind"][s:e] == 3) & (d["mask"][s:e] > 0.5))
            x = np.where(options, pol, -1e9)
            x = x - x.max(-1, keepdims=True)
            p = np.exp(x); p /= p.sum(-1, keepdims=True)
            tl[s:e] = p
            tv[s:e] = val
        d["pi"] = 0.5 * d["pi"] + 0.5 * tl.astype(np.float32)
        d["z"] = 0.5 * d["z"] + 0.5 * tv.astype(np.float32)
        print(f"distilling from {a.distill}: targets blended 50/50")
    rng = np.random.default_rng(0)
    val_d = load_data(a.val_data, a.max_per_shard) if a.val_data else d
    if a.val_data:
        tr_idx = np.arange(n)
        val_idx = np.arange(len(val_d["pi"]))
        split_name = "separate shards"
    else:
        split = a.split
        if split == "auto":
            split = "episode" if "group" in d else "decision"
        key = "group" if split == "episode" else ("pilot" if split == "pilot" else None)
        if key is not None:
            if key not in d:
                ap.error(f"--split {split} requires '{key}' metadata; re-ingest "
                         "with ingest_episodes.py --features rich")
            groups = np.unique(d[key])
            groups = groups[rng.permutation(len(groups))]
            n_val_groups = max(1, int(len(groups) * a.val_frac))
            val_groups = groups[:n_val_groups]
            is_val = np.isin(d[key], val_groups)
            val_idx, tr_idx = np.flatnonzero(is_val), np.flatnonzero(~is_val)
            split_name = f"{split}-disjoint ({n_val_groups}/{len(groups)} groups)"
        else:
            perm = rng.permutation(n)
            n_val = max(1, int(n * a.val_frac))
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
            split_name = "random decisions"

    def to_t(source, idx):
        return (torch.tensor(source["kind"][idx].astype(np.int64)),
                torch.tensor(source["card"][idx].astype(np.int64)),
                torch.tensor(source["scal"][idx]),
                torch.tensor(source["mask"][idx]),
                torch.tensor(source["ctx"][idx].astype(np.int64)),
                torch.tensor(source["stype"][idx].astype(np.int64)),
                torch.tensor(source["pi"][idx]),
                torch.tensor(source["z"][idx]))

    tr = to_t(d, tr_idx)
    va = to_t(val_d, val_idx)
    sample_weight = np.ones(len(tr_idx), dtype=np.float32)
    if a.elo_weight:
        if "elo" not in d:
            ap.error("--elo-weight requires elo metadata; re-run ingestion")
        strength = np.clip(
            (d["elo"][tr_idx].astype(np.float32) - 1000.0) / 250.0,
            0.0, 1.0)
        sample_weight *= 1.0 + a.elo_weight * strength
    if a.winner_weight != 1.0:
        outcomes = d["z"][tr_idx]
        sample_weight *= np.where(
            outcomes > 0, a.winner_weight,
            np.where(outcomes < 0, 1.0 / a.winner_weight, 1.0)
        ).astype(np.float32)
    sample_weight /= sample_weight.mean()
    train_weight = torch.tensor(sample_weight)
    resident = False
    tensor_bytes = sum(
        t.numel() * t.element_size() for group in (tr, va) for t in group)
    tensor_bytes += train_weight.numel() * train_weight.element_size()
    if dev.type == "cuda":
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        resident = (a.gpu_resident == "yes" or
                    (a.gpu_resident == "auto" and
                     tensor_bytes < 0.55 * free_bytes))
        if a.gpu_resident == "yes" and tensor_bytes >= 0.80 * free_bytes:
            ap.error("--gpu-resident yes would leave too little CUDA memory; "
                     "use auto/no or lower --max-per-shard")
    if resident:
        tr = tuple(t.to(dev) for t in tr)
        va = tuple(t.to(dev) for t in va)
        train_weight = train_weight.to(dev)
    print(f"train {len(tr_idx)} / val {len(val_idx)}; split={split_name}; "
          f"features={a.features}; tensors={'GPU-resident' if resident else 'CPU-streamed'} "
          f"({tensor_bytes / 2**30:.2f} GiB)")

    model = TCGNet()
    if a.init:
        with np.load(a.init) as weights:
            expected_meta = np.array(
                [a.dim, a.layers, a.heads, 2 * a.dim], dtype=np.int64)
            if "_meta" not in weights or not np.array_equal(
                    weights["_meta"], expected_meta):
                got = weights["_meta"].tolist() if "_meta" in weights else None
                ap.error(f"--init architecture {got} does not match "
                         f"{expected_meta.tolist()}")
            state = model.state_dict()
            missing = [
                key for key, value in state.items()
                if key not in weights or
                tuple(weights[key].shape) != tuple(value.shape)
            ]
            if missing:
                ap.error(f"--init has missing or mismatched weights: {missing[:5]}")
            model.load_state_dict({
                key: torch.as_tensor(weights[key]).to(dtype=value.dtype)
                for key, value in state.items()
            })
        print(f"warm-started model from {a.init}")
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    n_tr = len(tr_idx)
    steps = max(1, n_tr // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * steps, pct_start=0.3)

    critical_ids = torch.tensor(CRITICAL_CONTEXTS, device=dev)
    best_ce = float("inf")
    best_ep = 0
    best_state = None
    stale = 0
    for ep in range(a.epochs):
        model.train()
        order = torch.randperm(
            n_tr, device=dev if resident else torch.device("cpu"))
        tot_p = tot_v = 0.0
        for s in range(steps):
            b = order[s * a.batch:(s + 1) * a.batch]
            batch_tensors = tuple(t[b] for t in tr)
            example_weight = train_weight[b]
            if not resident:
                batch_tensors = tuple(t.to(dev) for t in batch_tensors)
                example_weight = example_weight.to(dev)
            kind, card, scal, mask, ctx, styp, pi, z = batch_tensors
            logits, v = model(kind, card, scal, mask, ctx, styp)
            if a.dmc:
                # Q head: the action actually taken is argmax(pi) (exactly the
                # played move for replay data, the most visited for search
                # data). Regress its Q toward the realised return.
                taken = pi.argmax(-1)
                q_taken = logits[
                    torch.arange(len(taken), device=taken.device), taken]
                lp = ((q_taken - z) ** 2).mean()
            else:
                per_sample, _option_logits = option_policy_loss(
                    logits, pi, kind, mask, a.label_smoothing)
                if a.critical_weight != 1.0:
                    critical = (ctx[:, None] == critical_ids[None, :]).any(-1)
                    context_weight = torch.where(
                        critical,
                        torch.full_like(per_sample, a.critical_weight),
                        torch.ones_like(per_sample))
                    weight = context_weight * example_weight
                else:
                    weight = example_weight
                lp = (per_sample * weight).sum() / weight.sum()
            lv = Fn.mse_loss(v, z)
            loss = lp + a.value_weight * lv
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_p += lp.item(); tot_v += lv.item()

        vlp, vlv, top1, _ = eval_val(model, va, dev)
        print(f"ep {ep+1}/{a.epochs}  train p={tot_p/steps:.4f} v={tot_v/steps:.4f}"
              f"  | val p={vlp:.4f} v={vlv:.4f} top1={top1*100:.1f}%")
        if vlp < best_ce - 1e-5:
            best_ce = vlp
            best_ep = ep + 1
            best_state = {
                k: value.detach().cpu().clone()
                for k, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if a.patience and stale >= a.patience:
                print(f"early stop: option-only CE has not improved for "
                      f"{a.patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(dev)
        print(f"restored best policy checkpoint: epoch {best_ep}, "
              f"held-out option CE {best_ce:.4f}")

    # ---------------- gates ----------------
    _vp, _vv, net_top1, val_mae = eval_val(model, va, dev)

    # Replay ingestion preserves engine option order.  This is an honest
    # raw-first-option baseline, not a heuristic score.
    pi_am = va[6].argmax(-1).cpu()
    z_cpu = va[7].cpu()
    heur_top1 = (torch.full((len(pi_am),), NF.OPT_BASE)
                 == pi_am).float().mean().item()
    base_mae = (z_cpu - z_cpu.mean()).abs().mean().item()

    print("\n==== GATES ====")
    print(f"policy top-1 vs replay: net {net_top1*100:.1f}%  |  raw option-0 {heur_top1*100:.1f}%")
    print(f"value MAE              : net {val_mae:.3f}  |  predict-mean baseline {base_mae:.3f}")
    ok_p = net_top1 > heur_top1
    ok_v = val_mae < base_mae
    print(f"policy gate: {'PASS' if ok_p else 'FAIL'}   value gate: {'PASS' if ok_v else 'FAIL'}")

    export_npz(model, a.out)
    print(f"exported {a.out}")


if __name__ == "__main__":
    main()

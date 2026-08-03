"""Select a genuinely updated GRPO checkpoint using exact temporal gates."""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "track1_search", "train"))
sys.path.insert(0, os.path.join(ROOT, "track5_grpo"))

from train_bc import eval_val, load_data  # noqa: E402
from train_grpo import load_model  # noqa: E402

BASE = os.path.join(HERE, "model_tech_awr_bc.npz")
FINAL = os.path.join(HERE, "model_tech_awr_grpo_final.npz")
HOLDOUT = os.path.join(ROOT, "track6_controlled", "data_tech_grim_holdout")


def tensors(data):
    return tuple(torch.tensor(value) for value in (
        data["kind"].astype(np.int64), data["card"].astype(np.int64),
        data["scal"], data["mask"], data["ctx"].astype(np.int64),
        data["stype"].astype(np.int64), data["pi"], data["z"],
    ))


def policy_kl(model, reference, holdout, batch=512):
    total = 0.0
    n = len(holdout[0])
    model.eval(); reference.eval()
    with torch.no_grad():
        for start in range(0, n, batch):
            end = min(start + batch, n)
            kind, card, scal, mask, ctx, stype = (
                value[start:end] for value in holdout[:6])
            logits, _ = model(kind, card, scal, mask, ctx, stype)
            ref_logits, _ = reference(kind, card, scal, mask, ctx, stype)
            options = (kind == 3) & (mask > 0.5)
            logp = torch.log_softmax(logits.masked_fill(~options, -torch.inf), -1)
            ref_logp = torch.log_softmax(
                ref_logits.masked_fill(~options, -torch.inf), -1)
            probs = torch.where(options, logp.exp(), torch.zeros_like(logp))
            safe = torch.where(options, logp, torch.zeros_like(logp))
            safe_ref = torch.where(options, ref_logp, torch.zeros_like(ref_logp))
            total += (probs * (safe - safe_ref)).sum(-1).sum().item()
    return total / n


def main() -> None:
    holdout = tensors(load_data(HOLDOUT))
    reference = load_model(BASE, "cpu")
    base_ce, _mse, base_top1, _mae = eval_val(reference, holdout, "cpu", 512)
    rows = []
    for path in sorted(glob.glob(os.path.join(
            HERE, "model_tech_awr_grpo_iter*.npz"))):
        metrics_path = path + ".metrics.json"
        metrics = json.load(open(metrics_path, encoding="utf-8"))
        if int(metrics.get("optimizer_steps", 0)) <= 0:
            continue
        model = load_model(path, "cpu")
        ce, _mse, top1, _mae = eval_val(model, holdout, "cpu", 512)
        kl = policy_kl(model, reference, holdout)
        passed = (kl <= 0.0015 and top1 >= base_top1 - 0.005 and
                  ce <= base_ce + 0.01)
        rows.append({"path": path, "iteration": metrics.get("iteration"),
                     "ce": ce, "top1": top1, "kl": kl,
                     "active_groups": metrics.get("active_groups"),
                     "passed": passed})
        print(f"iter={metrics.get('iteration')} CE={ce:.5f} "
              f"top1={top1*100:.2f}% KL={kl:.6f} passed={passed}")
    candidates = [row for row in rows if row["passed"]]
    if not candidates:
        raise SystemExit("no genuinely updated GRPO checkpoint passed the gates")
    chosen = min(candidates, key=lambda row: (row["ce"], -row["top1"], row["kl"]))
    shutil.copy2(chosen["path"], FINAL)
    report = {"base": {"ce": base_ce, "top1": base_top1},
              "candidates": rows, "selected": chosen}
    with open(os.path.join(HERE, "selection.json"), "w", encoding="utf-8") as out:
        json.dump(report, out, indent=2)
    print(f"selected iteration {chosen['iteration']} -> {FINAL}")


if __name__ == "__main__":
    main()

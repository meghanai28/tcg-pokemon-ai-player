"""Gate test: numpy inference must exactly mirror the torch model."""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "track1_search", "agent"))
sys.path.insert(0, HERE)

import nn_features as NF          # noqa: E402
from model import TCGNet, export_npz  # noqa: E402
from nn_infer import NumpyNet     # noqa: E402

torch.manual_seed(0)
rng = np.random.default_rng(0)

m = TCGNet().eval()
path = os.path.join(HERE, "_parity_tmp.npz")
export_npz(m, path)
nnet = NumpyNet(path)

B, S, F = 8, NF.SEQ, NF.F
kind = rng.integers(0, NF.N_KIND, (B, S))
card = rng.integers(0, NF.N_CARD, (B, S))
scal = rng.standard_normal((B, S, F)).astype(np.float32)
mask = (rng.random((B, S)) > 0.3).astype(np.float32)
mask[:, 0] = 1.0
ctx = rng.integers(0, NF.N_CTX, (B,))
styp = rng.integers(0, NF.N_STYPE, (B,))

with torch.no_grad():
    tp, tv = m(torch.tensor(kind), torch.tensor(card), torch.tensor(scal),
               torch.tensor(mask), torch.tensor(ctx), torch.tensor(styp))
npp, npv = nnet.forward(kind, card, scal, mask, ctx, styp)

# compare only on unmasked positions (masked logits are -1e9 sentinels)
sel = mask > 0.5
dp = np.abs(tp.numpy()[sel] - npp[sel]).max()
dv = np.abs(tv.numpy() - npv).max()
print(f"float32: max |policy diff| = {dp:.2e}   max |value diff| = {dv:.2e}")

# Structural check: if the graphs match, float64 on both sides collapses the
# gap to ~1e-12. A real architecture mismatch would NOT shrink with precision.
with torch.no_grad():
    m64 = TCGNet().double()
    m64.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
    tp64, tv64 = m64(torch.tensor(kind), torch.tensor(card),
                     torch.tensor(scal, dtype=torch.float64),
                     torch.tensor(mask, dtype=torch.float64),
                     torch.tensor(ctx), torch.tensor(styp))
nnet.w = {k: (v.astype(np.float64) if v.dtype == np.float32 else v)
          for k, v in nnet.w.items()}
npp64, npv64 = nnet.forward(kind, card, scal.astype(np.float64), mask.astype(np.float64),
                            ctx, styp)
dp64 = np.abs(tp64.numpy()[sel] - npp64[sel]).max()
dv64 = np.abs(tv64.numpy() - npv64).max()
print(f"float64: max |policy diff| = {dp64:.2e}   max |value diff| = {dv64:.2e}")

assert dp64 < 1e-9 and dv64 < 1e-9, "PARITY FAIL — architectures differ"
assert dp < 1e-3 and dv < 1e-3, "PARITY FAIL — float32 drift too large"
print("PARITY OK (float32 gap is accumulation noise; graphs are identical)")

t0 = time.perf_counter()
for _ in range(20):
    nnet.forward(kind, card, scal, mask, ctx, styp)
dt = (time.perf_counter() - t0) / 20
print(f"numpy forward batch={B}: {dt*1000:.2f} ms  ({dt/B*1e6:.0f} us/sample)")
os.remove(path)

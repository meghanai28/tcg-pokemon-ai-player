"""Focused regression tests for QRSAC's action-support math."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from track2_dmc.qrsac import (
    alpha_tuning_loss,
    bc_actor_loss,
    option_anchor_loss,
    option_policy_stats,
    option_token_mask,
)


class OptionSupportTest(unittest.TestCase):
    def setUp(self):
        # global, board, hand, two options, padding
        self.kind = torch.tensor([[0, 1, 2, 3, 3, 0]])
        self.mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
        self.options = option_token_mask(self.kind, self.mask)

    def test_option_mask_excludes_state_tokens(self):
        self.assertEqual(self.options.tolist(),
                         [[False, False, False, True, True, False]])

    def test_policy_normalizes_only_over_options(self):
        # Huge state-token logits must not steal probability from legal options.
        logits = torch.tensor([[100.0, 80.0, 60.0, 0.0, 0.0, -1e9]])
        qmean = torch.tensor([[99.0, 99.0, 99.0, -1.0, 1.0, 99.0]])
        p, _logp, entropy, expected_q = option_policy_stats(
            logits, qmean, self.options)
        self.assertTrue(torch.allclose(p[0, 3:5], torch.tensor([0.5, 0.5])))
        self.assertEqual(float(p[0, :3].sum()), 0.0)
        self.assertAlmostEqual(float(entropy), 0.693147, places=5)
        self.assertAlmostEqual(float(expected_q), 0.0, places=6)

    def test_anchor_uses_only_untaken_options(self):
        qmean = torch.tensor([[100.0, 100.0, 100.0, 2.0, 4.0, 100.0]])
        value = torch.tensor([1.0])
        # Position 3 was taken, so only position 4 contributes: (4 - 1)^2.
        loss = option_anchor_loss(qmean, value, torch.tensor([3]), self.options)
        self.assertEqual(float(loss), 9.0)

    def test_bc_rehearsal_normalizes_only_over_options(self):
        class FixedModel(torch.nn.Module):
            def forward(self, kind, card, scal, mask, ctx, styp):
                logits = torch.tensor(
                    [[100.0, 80.0, 60.0, 0.0, 0.0, -1e9]],
                    device=kind.device, requires_grad=True)
                quant = torch.zeros((*logits.shape, 2), device=kind.device)
                value = torch.zeros(len(kind), device=kind.device)
                return logits, quant, value

        data = {
            "kind": self.kind.numpy(),
            "card": np.zeros((1, 6), dtype=np.int64),
            "scal": np.zeros((1, 6, 1), dtype=np.float32),
            "mask": self.mask.numpy(),
            "ctx": np.zeros(1, dtype=np.int64),
            "stype": np.zeros(1, dtype=np.int64),
            "pi": np.array([[0, 0, 0, 0.5, 0.5, 0]], dtype=np.float32),
        }
        loss = bc_actor_loss(FixedModel(), data, np.array([0]), torch.device("cpu"))
        self.assertAlmostEqual(float(loss.detach()), 0.693147, places=5)

    def test_alpha_can_recover_from_near_zero(self):
        log_alpha = torch.tensor([-10.0], requires_grad=True)
        loss = alpha_tuning_loss(
            log_alpha, entropy=torch.tensor([0.5]), target_entropy=torch.tensor([1.0]))
        loss.backward()
        # Gradient descent subtracts this negative, increasing log(alpha).
        self.assertAlmostEqual(float(log_alpha.grad), -0.5)


if __name__ == "__main__":
    unittest.main()

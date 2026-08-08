from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from foundation import nn_features_rich as NF
from rl_osfp.audit_run import audit
from rl_osfp.network import ActorCritic, NetworkConfig
from rl_osfp.policy import Decision, batch_logprob_entropy
from rl_osfp.train import ppo_update


class PPOInvariantTests(unittest.TestCase):
    def test_temperature_log_probability_round_trip(self) -> None:
        """A rollout at temperature != 1 must still start PPO at ratio one."""
        torch.manual_seed(7)
        model = ActorCritic(NetworkConfig(32, 1, 4, 64, 0.0)).eval()
        kind = np.zeros(NF.SEQ, dtype=np.int8)
        card = np.zeros(NF.SEQ, dtype=np.int16)
        scal = np.zeros((NF.SEQ, NF.F), dtype=np.float32)
        mask = np.zeros(NF.SEQ, dtype=np.float32)
        mask[0] = 1.0
        option_slots = [NF.OPT_BASE, NF.OPT_BASE + 1, NF.OPT_BASE + 2]
        for slot in option_slots:
            kind[slot] = 3
            mask[slot] = 1.0
        selected_slots = np.full(60, -1, dtype=np.int16)
        selected_slots[0] = option_slots[1]
        tensors = (
            torch.as_tensor(kind[None].astype(np.int64)),
            torch.as_tensor(card[None].astype(np.int64)),
            torch.as_tensor(scal[None]),
            torch.as_tensor(mask[None]),
            torch.as_tensor([0]),
            torch.as_tensor([0]),
        )
        temperature = 0.63
        with torch.inference_mode():
            option, count, value = model(*tensors)
            logp, _ = batch_logprob_entropy(
                option / temperature,
                count / temperature,
                tensors[0], tensors[3],
                torch.as_tensor(selected_slots[None].astype(np.int64)),
                torch.as_tensor([1]), torch.as_tensor([1]), torch.as_tensor([1]),
            )
        decision = Decision(
            kind, card, scal, mask, 0, 0, selected_slots, 1, 1, 1,
            float(logp[0]), float(value[0]), outcome=1.0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
        args = Namespace(
            epochs=1, batch=1, temperature=temperature, clip=0.2,
            value_coef=1.0, entropy_coef=0.01, ref_kl_coef=0.0,
            grad_clip=0.8, target_kl=0.04,
        )
        report = ppo_update(
            model, [decision], optimizer, torch.device("cpu"), random.Random(9), args
        )
        self.assertLess(report["approx_kl"], 1e-10)
        self.assertEqual(report["clip_fraction"], 0.0)


class AuditInvariantTests(unittest.TestCase):
    def test_stale_policy_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = root / "pool.json"
            deck = list(range(60))
            pool.write_text(json.dumps({
                "learner_decks": [{"name": "a", "cards": deck}],
                "field_decks": [{"name": "b", "cards": deck}],
            }), encoding="utf-8")
            metrics = {
                "config": {"pool": str(pool), "async_rollout": True},
                "periods": [{
                    "period": 2,
                    "games": [{"period": 1, "learner_deck": "a", "opponent_deck": "b"}],
                    "decisions": 500, "errors": 0, "invalid_actions": 0,
                    "update": {},
                }],
            }
            (root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            report = audit(root)
            self.assertFalse(report["ready_to_scale"])
            self.assertEqual(report["policy_lag_counts"], {1: 1})
            self.assertTrue(any("unsafe" in item for item in report["failures"]))


if __name__ == "__main__":
    unittest.main()

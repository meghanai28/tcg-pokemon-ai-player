import unittest
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_grpo import group_advantages


class GroupAdvantageTest(unittest.TestCase):
    def test_centered_and_ordered(self):
        advantage = group_advantages([-1, 1, -1, 1])
        self.assertAlmostEqual(float(advantage.mean()), 0.0, places=5)
        self.assertGreater(advantage[1], advantage[0])

    def test_no_variance_is_no_update(self):
        np.testing.assert_array_equal(
            group_advantages([1, 1, 1, 1]), np.zeros(4, dtype=np.float32))

    def test_single_rollout_is_rejected_as_signal(self):
        np.testing.assert_array_equal(
            group_advantages([1]), np.zeros(1, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

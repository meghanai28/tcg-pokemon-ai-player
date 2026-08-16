import os
import tarfile
import tempfile
import unittest

import numpy as np
import torch

from bc_train import nn_features_rich
from bc_train import nn_features_v2
from agent import build_submissions
from agent.train_prior import (
    FIELDS,
    action_set_metrics,
    canonicalize_equivalent_targets,
    count_target_rows,
    episode_balance_weights,
    equivalent_option_policy_loss,
    action_tuple_scores,
    load_teacher_reservoir,
    load_target_reservoir,
    make_replay_order,
    replay_aware_value_loss,
    replay_rows_per_epoch,
    stable_id,
    add_trajectory_counts,
    trajectory_batch_weights,
    teacher_action_policy_loss,
)
from tools.ladder_harness import wilson_interval
from tools.dagger_generate import merge_stable_visits
from tools.exit_generate import (
    soften_prior,
    visits_to_action_targets,
    visits_to_q,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GrpoPriorTests(unittest.TestCase):
    def test_feature_v2_exact_deck_token_and_slot_upgrade(self):
        deck = [1] * 59 + [96]
        deck_id = nn_features_v2.stable_deck_id(deck)
        self.assertEqual(deck_id, stable_id(tuple(sorted(deck))))
        kind = np.zeros((1, 53), dtype=np.int8)
        card = np.zeros((1, 53), dtype=np.int16)
        scal = np.zeros((1, 53, 32), dtype=np.float32)
        mask = np.zeros((1, 53), dtype=np.float32)
        pi = np.zeros((1, 53), dtype=np.float32)
        kind[0, 29] = 3
        mask[0, [0, 29]] = 1
        pi[0, 29] = 1
        scal[0, 0, 21] = 1094 / 1299.0
        got = nn_features_v2.upgrade_batch(
            kind, card, scal, mask, pi,
            np.array([deck_id], dtype=np.uint64))
        new_kind, new_card, _new_scal, new_mask, new_pi = got
        self.assertEqual(new_kind.shape, (1, 55))
        self.assertEqual(new_card[0, 0],
                         nn_features_v2.deck_token_from_id(deck_id))
        self.assertEqual(new_card[0, nn_features_v2.CONTEXT_SLOT], 1094)
        self.assertEqual(new_mask[0, nn_features_v2.CONTEXT_SLOT], 1)
        self.assertEqual(new_kind[0, nn_features_v2.OPT_BASE], 3)
        self.assertEqual(new_pi[0, nn_features_v2.OPT_BASE], 1)

    def test_rich_features_resolve_public_looking_cards(self):
        cur = {
            "looking": [{"id": 1094}, {"id": 96}],
            "players": [{"hand": [], "discard": [], "active": [],
                         "bench": [], "prize": []}],
        }
        sel = {"deck": None}
        option = {"area": nn_features_rich.LOOKING, "index": 1,
                  "playerIndex": 0, "type": nn_features_rich.OT_CARD}
        self.assertEqual(
            nn_features_rich._zone_card(cur, sel, option, 0)["id"], 96)

    def test_champion_package_changes_only_model_and_deck(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = build_submissions.build(
                "smoke",
                os.path.join(ROOT, "agent", "champion", "model.npz"),
                os.path.join(ROOT, "agent", "decks", "crispin.csv"),
                tmp,
            )
            with tarfile.open(archive, "r:gz") as bundle:
                names = set(bundle.getnames())
            self.assertIn("main.py", names)
            self.assertIn("cg/libcg.so", names)

    def test_target_validation_count_uses_deck_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shard.npz")
            np.savez(path, deck=np.array([10, 20, 20, 30], dtype=np.uint64))
            self.assertEqual(
                count_target_rows([path], np.array([20, 40], dtype=np.uint64)), 2)

    def test_target_validation_can_select_winning_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shard.npz")
            np.savez(path, deck=np.array([20, 20, 30], dtype=np.uint64),
                     z=np.array([1, -1, 1], dtype=np.float32))
            self.assertEqual(count_target_rows(
                [path], np.array([20], dtype=np.uint64), wins_only=True), 1)

    def test_equivalent_card_copies_share_target_mass(self):
        kind = torch.tensor([[0, 3, 3, 3, 0]])
        card = torch.tensor([[0, 1, 1, 1, 0]])
        scal = torch.zeros((1, 5, 32))
        scal[:, 1:4, 3] = 1.0
        mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.float32)
        pi = torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.float32)
        got = canonicalize_equivalent_targets(
            pi, kind, card, scal, mask, torch.tensor([22]))
        torch.testing.assert_close(
            got, torch.tensor([[0, 1 / 3, 1 / 3, 1 / 3, 0]]))
        unchanged = canonicalize_equivalent_targets(
            pi, kind, card, scal, mask, torch.tensor([21]))
        torch.testing.assert_close(unchanged, pi)

    def test_equivalent_loss_uses_group_probability_not_copy_identity(self):
        kind = torch.tensor([[3, 3, 3]])
        card = torch.tensor([[11, 11, 22]])
        scal = torch.zeros((1, 3, 32))
        scal[:, :, 4] = 1.0
        mask = torch.ones((1, 3))
        pi = torch.tensor([[1.0, 0.0, 0.0]])
        ctx = torch.tensor([8])
        logits_a = torch.tensor([[0.0, -2.0, 0.4]])
        # Preserve exp(logit0)+exp(logit1) while redistributing it equally.
        tied = torch.logsumexp(logits_a[:, :2], dim=-1) - np.log(2.0)
        logits_b = torch.cat((tied[:, None], tied[:, None], logits_a[:, 2:]), 1)
        loss_a, _ = equivalent_option_policy_loss(
            logits_a, pi, kind, card, scal, mask, ctx)
        loss_b, _ = equivalent_option_policy_loss(
            logits_b, pi, kind, card, scal, mask, ctx)
        torch.testing.assert_close(loss_a, loss_b)

    def test_unresolved_or_different_source_options_do_not_merge(self):
        kind = torch.tensor([[3, 3, 3]])
        card = torch.tensor([[0, 0, 7]])
        scal = torch.zeros((1, 3, 32))
        scal[0, 0, 1] = 1.0
        scal[0, 1, 2] = 1.0
        scal[0, 2, 1] = 1.0
        mask = torch.ones((1, 3))
        pi = torch.tensor([[1.0, 0.0, 0.0]])
        got = canonicalize_equivalent_targets(
            pi, kind, card, scal, mask, torch.tensor([8]))
        torch.testing.assert_close(got, pi)

        # Equal attached energies on different Pokemon remain distinct.
        card = torch.tensor([[7, 7, 9]])
        scal.zero_()
        scal[:, :, 5] = 1.0
        scal[0, 0, 23] = 0.1
        scal[0, 1, 23] = 0.2
        got = canonicalize_equivalent_targets(
            pi, kind, card, scal, mask, torch.tensor([22]))
        torch.testing.assert_close(got, pi)

    def test_winner_target_reservoir_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shard.npz")
            rows = 6
            arrays = {
                "kind": np.zeros((rows, 2), dtype=np.int8),
                "card": np.zeros((rows, 2), dtype=np.int16),
                "scal": np.zeros((rows, 2, 3), dtype=np.float32),
                "mask": np.ones((rows, 2), dtype=np.float32),
                "ctx": np.zeros(rows, dtype=np.int16),
                "stype": np.zeros(rows, dtype=np.int16),
                "pi": np.zeros((rows, 2), dtype=np.float32),
                "z": np.array([1, -1, 1, 1, -1, 1], dtype=np.float32),
                "elo": np.arange(rows, dtype=np.float32) + 1000,
                "deck": np.array([20, 20, 20, 30, 20, 20], dtype=np.uint64),
            }
            self.assertTrue(all(key in arrays for key in FIELDS))
            np.savez(path, **arrays)
            pool = load_target_reservoir(
                [path], np.array([20], dtype=np.uint64), max_rows=2,
                seed=7, wins_only=True)
            self.assertEqual(len(pool["pi"]), 2)
            self.assertTrue(np.all(pool["deck"] == 20))
            self.assertTrue(np.all(pool["z"] > 0))

    def test_replay_budget_caps_passes_and_avoids_early_repeats(self):
        self.assertEqual(replay_rows_per_epoch(
            stream_rows=1000, pool_rows=40, fraction=0.10,
            max_passes=1.0), 40)
        self.assertEqual(replay_rows_per_epoch(
            stream_rows=1000, pool_rows=40, fraction=0.10,
            max_passes=2.0), 80)
        order = make_replay_order(40, 80, np.random.default_rng(3))
        self.assertEqual(len(np.unique(order[:40])), 40)
        self.assertEqual(len(np.unique(order[40:])), 40)

    def test_episode_balance_and_policy_only_replay_value(self):
        weights = episode_balance_weights(
            np.array([10, 10, 10, 20], dtype=np.uint64),
            np.array([0, 0, 0, 1], dtype=np.int8))
        self.assertAlmostEqual(float(weights[:3].sum()), float(weights[3]))
        value = torch.tensor([0.0, -1.0])
        z = torch.tensor([0.0, 1.0])
        self.assertEqual(float(replay_aware_value_loss(
            value, z, broad_rows=1, replay_weight=0.0)), 0.0)

    def test_stream_episode_balance_uses_global_trajectory_counts(self):
        data = {
            "group": np.array([10, 10, 10, 20], dtype=np.uint64),
            "seat": np.array([0, 0, 0, 1], dtype=np.int8),
        }
        counts = {}
        indices = np.arange(4)
        add_trajectory_counts(counts, data, indices)
        weights = trajectory_batch_weights(
            data, indices, counts, normalization=4 / 2)
        self.assertAlmostEqual(float(weights[:3].sum()), float(weights[3]))
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_wilson_interval_exposes_small_gate_uncertainty(self):
        lo, hi = wilson_interval(32, 28)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_teacher_temperature_is_idempotent(self):
        namespace = {"_net_scores": lambda *_args: [9.0, -1e9]}
        soften_prior(namespace, 3.0)
        self.assertEqual(namespace["_net_scores"](None, None, None, None, None)[0], 3.0)
        soften_prior(namespace, 3.0)
        self.assertEqual(namespace["_net_scores"](None, None, None, None, None)[0], 3.0)
        soften_prior(namespace, 1.0)
        self.assertEqual(namespace["_net_scores"](None, None, None, None, None)[0], 9.0)

    def test_teacher_q_keeps_neutral_actions_and_exact_tuples(self):
        visits = {(0, 2): (10, 0.0), (1,): (5, -0.2)}
        slots = [7, 8, 9]
        q, q_mask = visits_to_q(visits, slots, 12)
        self.assertTrue(q_mask[7])
        self.assertTrue(q_mask[9])
        self.assertEqual(q[7], 0.0)
        targets = visits_to_action_targets(visits, slots)
        action_tokens, action_sizes, action_q, action_visits, action_mask = targets
        self.assertTrue(action_mask[0])
        self.assertEqual(action_sizes[0], 2)
        np.testing.assert_array_equal(action_tokens[0, :2], [7, 9])
        self.assertEqual(action_q[0], 0.0)
        self.assertEqual(action_visits[0], 10)

    def test_replicated_teacher_abstains_on_unstable_top_action(self):
        left = {(0,): (8, 0.4), (1,): (8, 0.1)}
        right = {(0,): (6, 0.3), (1,): (6, 0.2)}
        merged = merge_stable_visits(left, right)
        self.assertIsNotNone(merged)
        visits, disagreement = merged
        self.assertEqual(visits[(0,)][0], 14)
        self.assertGreater(disagreement, 0)
        flipped = {(0,): (6, 0.0), (1,): (6, 0.5)}
        self.assertIsNone(merge_stable_visits(left, flipped))

    def test_teacher_loss_scores_exact_action_tuples(self):
        logits = torch.tensor([[0.0, 2.0, 1.0, 3.0]])
        tokens = torch.tensor([[[1, 2], [1, 3], [-1, -1]]])
        sizes = torch.tensor([[2, 2, 0]])
        live = torch.tensor([[True, True, False]])
        scores = action_tuple_scores(logits, tokens, sizes, live)
        torch.testing.assert_close(scores[:, :2], torch.tensor([[1.5, 2.5]]))
        q = torch.tensor([[0.8, 0.1, 0.0]])
        loss, hit, regret, usable = teacher_action_policy_loss(
            logits, tokens, sizes, q, live, q_temperature=0.1)
        self.assertTrue(usable.item())
        self.assertFalse(hit.item())
        self.assertAlmostEqual(float(regret), 0.7, places=6)
        self.assertGreater(float(loss), 0.0)

    def test_action_set_metric_catches_marginal_pair_failure(self):
        kind = torch.tensor([[3, 3, 3, 3]])
        mask = torch.ones((1, 4))
        pi = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        logits = torch.tensor([[4.0, 1.0, 3.0, 0.0]])
        recall, exact, selected, _cap_hit, cap_bound = action_set_metrics(
            logits, pi, kind, mask, cap=2)
        self.assertTrue(selected.item())
        self.assertAlmostEqual(float(recall), 0.5)
        self.assertFalse(exact.item())
        self.assertTrue(cap_bound.item())

    def test_teacher_loss_can_learn_relative_to_empty_action(self):
        logits = torch.tensor([[2.0, -1.0]], requires_grad=True)
        tokens = torch.tensor([[[-1], [0]]])
        sizes = torch.tensor([[0, 1]])
        live = torch.tensor([[True, True]])
        q = torch.tensor([[0.8, -0.2]])
        visits = torch.tensor([[8, 8]])
        scores = action_tuple_scores(logits, tokens, sizes, live)
        torch.testing.assert_close(scores, torch.tensor([[0.0, 2.0]]))
        loss, _hit, _regret, usable = teacher_action_policy_loss(
            logits, tokens, sizes, q, live, visits,
            q_shrink_visits=0.0, q_temperature=0.1)
        self.assertTrue(usable.item())
        loss.sum().backward()
        self.assertGreater(float(logits.grad[0, 0]), 0.0)

    def test_teacher_reservoir_filters_unusable_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "teacher.npz")
            rows, seq, candidates, members = 3, 4, 3, 2
            arrays = {
                "kind": np.zeros((rows, seq), dtype=np.int8),
                "card": np.zeros((rows, seq), dtype=np.int16),
                "scal": np.zeros((rows, seq, 32), dtype=np.float32),
                "mask": np.ones((rows, seq), dtype=np.float32),
                "ctx": np.zeros(rows, dtype=np.int16),
                "stype": np.zeros(rows, dtype=np.int16),
                "pi": np.zeros((rows, seq), dtype=np.float32),
                "z": np.zeros(rows, dtype=np.float32),
                "action_tokens": np.zeros(
                    (rows, candidates, members), dtype=np.int16),
                "action_sizes": np.ones((rows, candidates), dtype=np.int8),
                "action_q": np.zeros((rows, candidates), dtype=np.float32),
                "action_visits": np.ones((rows, candidates), dtype=np.int32),
                "action_mask": np.array(
                    [[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=np.bool_),
                "group": np.array([10, 10, 20], dtype=np.uint64),
                "seat": np.zeros(rows, dtype=np.int8),
            }
            np.savez(path, **arrays)
            pool = load_teacher_reservoir([path], max_rows=10, seed=3)
            self.assertEqual(len(pool["ctx"]), 2)
            self.assertTrue(np.all(pool["action_mask"].sum(1) >= 2))


if __name__ == "__main__":
    unittest.main()

# Copyright 2026 Viktor Brunclík
# SPDX-License-Identifier: Apache-2.0
"""Rules of coldcap/porridge.py: temperature, rewards, early ejection, controller."""

import unittest

from coldcap.porridge import (COLD, HOT, ICY, Controller, ControllerConfig, Memory,
                              PorridgeConfig, advantages, rewards, temperature)


class Temperature(unittest.TestCase):
    config = PorridgeConfig()

    def test_hard_and_middle_prompts_are_hot(self):
        self.assertEqual(temperature([0, 0, 0, 0], [900] * 4, self.config), HOT)
        self.assertEqual(temperature([1, 1, 0, 0], [300, 400, 900, 900], self.config), HOT)

    def test_reliable_but_rambling_is_cold(self):
        self.assertEqual(temperature([1, 1, 1, 1], [200, 600, 300, 900], self.config), COLD)
        self.assertEqual(temperature([1, 1, 1, 0], [200, 210, 205, 900], self.config), COLD)

    def test_always_right_and_already_tight_is_icy(self):
        self.assertEqual(temperature([1, 1, 1, 1], [200, 205, 210, 215], self.config), ICY)


class Rewards(unittest.TestCase):
    def test_shortest_correct_earns_most_and_wrong_earns_minus_beta(self):
        got = rewards([1, 1, 0], [100, 200, 50], alpha=1.0, beta=0.5)
        self.assertEqual(got, [2.0, 1.5, -0.5])

    def test_history_sets_a_tighter_reference(self):
        got = rewards([1], [200], alpha=1.0, beta=0.0, l_history=100)
        self.assertEqual(got, [1.5])

    def test_advantages_are_centred(self):
        adv = advantages([2.0, 1.0, 0.0])
        self.assertAlmostEqual(sum(adv), 0.0)
        self.assertGreater(adv[0], adv[1])


class EarlyEjection(unittest.TestCase):
    def test_cap_follows_shortest_correct(self):
        memory = Memory(PorridgeConfig(eject_factor=2.0, min_cap=64, max_cap=2048))
        self.assertEqual(memory.cap(7), 2048)  # unknown prompt: full budget
        memory.update(7, [1, 1, 0, 1], [300, 500, 2048, 400], step=0)
        self.assertEqual(memory.cap(7), 600)

    def test_hot_prompts_are_mostly_skipped(self):
        memory = Memory(PorridgeConfig(recheck=0.0))
        memory.update(3, [0, 0, 0, 0], [900] * 4, step=0)
        self.assertFalse(memory.worth_generating(3))
        self.assertTrue(memory.worth_generating(4))

    def test_state_round_trips(self):
        memory = Memory(PorridgeConfig())
        memory.update(1, [1, 1], [10, 30], step=2)
        other = Memory(PorridgeConfig())
        other.load(memory.state())
        self.assertEqual(other.prompts, memory.prompts)


class ControllerRules(unittest.TestCase):
    def test_pushes_harder_inside_budget_and_backs_off_outside(self):
        controller = Controller(ControllerConfig(alpha=1.0, slack=0.0))
        # 30% shorter, 2 points lost: allowed 6 -> push
        controller.observe(10, base_acc=0.80, base_tokens=1000, acc=0.78, tokens=700)
        self.assertGreater(controller.alpha, 1.0)
        pushed = controller.alpha
        # 10% shorter, 5 points lost: allowed 2 -> back off and punish errors
        record = controller.observe(20, base_acc=0.80, base_tokens=1000, acc=0.75, tokens=900)
        self.assertFalse(record["within_budget"])
        self.assertLess(controller.alpha, pushed)
        self.assertGreater(controller.beta, 0.0)


if __name__ == "__main__":
    unittest.main()

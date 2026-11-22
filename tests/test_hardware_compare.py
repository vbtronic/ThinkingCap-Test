# Copyright 2026 Viktor Brunclík
# SPDX-License-Identifier: Apache-2.0
"""compare.py across cards: differing GPU or dtype is compared, never refused."""

import unittest

from utils.eval.compare import check, hardware_note


def meta(gpu, dtype, adapter=None):
    return {"selection": "full", "cohort_sha256": "x", "seed": 1, "samples": 8,
            "max_tokens": {"svamp": 4096}, "model": "Qwen/Qwen3-0.6B", "adapter": adapter,
            "benchmarks": ["svamp"], "hardware": {"gpu": gpu, "dtype": dtype}}


def side(m):
    return {"model": m["model"], "adapter": m["adapter"], "hardware": m["hardware"]}


class AcrossHardware(unittest.TestCase):
    def test_different_dtype_and_gpu_are_compared(self):
        base, mine = meta("Tesla T4", "float16"), meta("NVIDIA L4", "bfloat16", "lora")
        self.assertEqual(check(base, mine), ["svamp"])
        note = hardware_note({"base": side(base), "candidate": side(mine)})
        self.assertIn("different setups", note)

    def test_same_checkpoint_on_two_cards_is_a_calibration(self):
        a, b = meta("Tesla T4", "float16"), meta("NVIDIA L4", "bfloat16")
        self.assertEqual(check(a, b), ["svamp"])
        self.assertIn("A/A CALIBRATION", hardware_note({"base": side(a), "candidate": side(b)}))

    def test_same_checkpoint_on_same_card_is_still_refused(self):
        with self.assertRaises(ValueError):
            check(meta("Tesla T4", "float16"), meta("Tesla T4", "float16"))

    def test_same_setup_has_no_note(self):
        a, b = meta("Tesla T4", "float16"), meta("Tesla T4", "float16", "lora")
        self.assertEqual(hardware_note({"base": side(a), "candidate": side(b)}), "")


if __name__ == "__main__":
    unittest.main()

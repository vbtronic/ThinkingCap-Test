# Copyright 2026 Viktor Brunclík
# SPDX-License-Identifier: Apache-2.0
"""EnginePool splits work between GPUs without a GPU: fake engines echo their requests."""

import unittest

from coldcap.rollout import EnginePool


class Fake:
    def __init__(self, name, speed=None):
        self.info = {"gpu": name, "dtype": "float16"}
        self.speed = speed
        self.got = []

    def submit(self, requests, adapter=None):
        self.got.append(requests)

    def collect(self):
        return [{"tokens": [r], "finished": True, "by": self.info["gpu"]} for r in self.got[-1]]


class PoolSplit(unittest.TestCase):
    def test_keeps_order_and_splits_evenly_when_fixed(self):
        a, b = Fake("A", speed=300.0), Fake("B", speed=100.0)
        pool = EnginePool([a, b])
        out = pool.generate(list(range(10)), fixed=True)
        self.assertEqual([r["tokens"][0] for r in out], list(range(10)))
        self.assertEqual(len(a.got[-1]), 5)

    def test_follows_speed_for_training(self):
        a, b = Fake("A", speed=300.0), Fake("B", speed=100.0)
        out = EnginePool([a, b]).generate(list(range(8)))
        self.assertEqual([len(a.got[-1]), len(b.got[-1])], [6, 2])
        self.assertEqual([r["tokens"][0] for r in out], list(range(8)))

    def test_unknown_speed_means_even_and_empty_slices_are_skipped(self):
        a, b = Fake("A"), Fake("B", speed=5.0)
        pool = EnginePool([a, b])
        pool.generate([1])
        self.assertEqual(len(a.got) + len(b.got), 1)
        self.assertEqual(pool.info["gpu"], "A + B")


if __name__ == "__main__":
    unittest.main()

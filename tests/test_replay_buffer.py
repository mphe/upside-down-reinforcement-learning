#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import unittest
from ..replay_buffer import ReplayBuffer, Trajectory


class ReplayBufferTest(unittest.TestCase):
    MAX_SIZE = 10

    def setUp(self):
        self.buffer = ReplayBuffer(self.MAX_SIZE)

    def _addElements(self, n: int):
        for _ in range(n):
            self.buffer.add([ None ], [ None ], [ np.random.random_sample() ])

    def test_add(self):
        self.assertEqual(0, len(self.buffer))
        self._addElements(1)
        self.assertEqual(1, len(self.buffer))

    def test_sample(self):
        self._addElements(10)
        sampled = self.buffer.sample(5)
        self.assertEqual(len(sampled), 5)
        for i in sampled:
            self.assertIsInstance(i, Trajectory)

    def test_k_best(self):
        def is_sorted(l):  # noqa
            return all(a >= b for a, b in zip(l, l[1:]))

        self._addElements(10)
        k_best = self.buffer.k_best(self.MAX_SIZE)
        self.assertEqual(self.MAX_SIZE, len(k_best))
        self.assertTrue(is_sorted([ i.summed_rewards for i in k_best ]))

    def test_max_size(self):
        smallest_element = -1
        self.buffer.add([ None ], [ None ], [ smallest_element ])
        self._addElements(9)

        self.assertEqual(self.buffer.k_best(self.MAX_SIZE)[-1].summed_rewards, smallest_element)

        self._addElements(10)

        self.assertEqual(len(self.buffer), self.MAX_SIZE)
        self.assertNotEqual(self.buffer.k_best(self.MAX_SIZE)[-1].summed_rewards, smallest_element)

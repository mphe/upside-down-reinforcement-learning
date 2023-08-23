#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import heapq
import numpy as np
import sys
from typing import List, Iterable
from sortedcontainers import SortedList

# Last test results:
# Heap
# 'test_buffer'  8489.42 ms
# SortedList
# 'test_buffer'  27.94 ms
# Sorted internal list
# 'test_buffer'  4.84 ms


class Buffer:
    def __init__(self, max_size: int):
        self._max_size: int = max_size

    def add(self, _value: int) -> None:
        raise NotImplementedError

    def get_n_largest(self, _n: int) -> List[int]:
        raise NotImplementedError


class Heap(Buffer):
    def __init__(self, max_size: int):
        super().__init__(max_size)
        self._buffer: List[int] = []

    def add(self, value: int) -> None:
        heapq.heappush(self._buffer, value)

        if len(self._buffer) == self._max_size:
            heapq.heappushpop(self._buffer, value)
        else:
            heapq.heappush(self._buffer, value)

    def get_n_largest(self, n: int) -> List[int]:
        return heapq.nlargest(n, self._buffer)


class Sorted(Buffer):
    def __init__(self, max_size: int):
        super().__init__(max_size)
        self._buffer: List[int] = []

    def add(self, value: int) -> None:
        self._buffer.append(value)

        if len(self._buffer) == self._max_size:
            self._buffer.sort(reverse=True)
            self._buffer = self._buffer[:self._max_size]

    def get_n_largest(self, n: int) -> List[int]:
        return self._buffer[:n]


class SortedListBuffer(Buffer):
    def __init__(self, max_size: int):
        super().__init__(max_size)
        self._buffer = SortedList()

    def add(self, value: int) -> None:
        self._buffer.add(value)

        if len(self._buffer) == self._max_size:
            self._buffer.pop(0)

    def get_n_largest(self, n: int) -> List[int]:
        return self._buffer[-n:]


def timeit(method):
    import time

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


@timeit
def test_buffer(buffer: Buffer, values: Iterable[int], n_largest: int) -> None:
    for i in values:
        buffer.add(i)
        buffer.get_n_largest(n_largest)


def main() -> int:
    NUM_SAMPLES = 10000
    N_LARGEST = 100
    BUFFER_SIZE = 700

    values = np.random.sample(NUM_SAMPLES)

    heap = Heap(BUFFER_SIZE)
    sorted_list = SortedListBuffer(BUFFER_SIZE)
    sorted_internal_list = Sorted(BUFFER_SIZE)

    print("Heap")
    test_buffer(heap, values, N_LARGEST)

    print("SortedList")
    test_buffer(sorted_list, values, N_LARGEST)

    print("Sorted internal list")
    test_buffer(sorted_internal_list, values, N_LARGEST)

    return 0


if __name__ == "__main__":
    sys.exit(main())

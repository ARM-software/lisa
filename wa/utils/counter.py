"""
An auto incremeting value (kind of like an AUTO INCREMENT field in SQL).
Optionally, the name of the counter to be used is specified (each counter
increments separately).

Counts start at 1, not 0.

"""
from collections import defaultdict

__all__ = [
    'next',
    'reset',
    'reset_all',
    'counter',
]

__counters = defaultdict(int)


def next(name=None):
    __counters[name] += 1
    value = __counters[name]
    return value


def reset_all(value=0):
    for k in __counters:
        reset(k, value)


def reset(name=None, value=0):
    __counters[name] = value


class counter(object):

    def __init__(self, name):
        self.name = name

    def next(self):
        return next(self.name)

    def reset(self, value=0):
        return reset(self.name, value)


"""
Generate diverse number series (fibonacci, lucas, tribonacci, ...)
"""
from __future__ import annotations
from typing import List, Generator

PHI = 1.618033988749894848204586834


def fibonacci(n:int, a=1, b=1) -> List[int]:
    """
    Calculate the first *n* numbers of the the fibonacci series
    """
    out = [a, b]
    for _ in range(n - 2):
        c = a + b
        out.append(c)
        a, b = b, c
    return out


def ifibonacci(a=1, b=1) -> Generator[int, None, None]:
    """
    Returns an iterator over the numbers of the fibonacci series
    """
    yield a
    yield b
    while True:
        c = a + b
        yield c
        a, b = b, c


def lucas(n: int, a=2, b=1) -> List[int]:
    """
    Calculate the first *n* numbers of the luca series
    """
    return fibonacci(n, a, b)


def tribonacci(n:int, a=0, b=0, c=1) -> List[int]:
    """
    Calculate the first *n* numbers of the tribonacci series
    """
    out = [a, b, c]
    for _ in range(n-3):
        d = a + b + c
        out.append(d)
        a, b, c = b, c, d
    return out


def padovan(n:int, a=1, b=1, c=1) -> List[int]:
    """
    Generate *n* elements of the padovan sequence

    https://en.wikipedia.org/wiki/Padovan_sequence
    """
    out = [a, b, c]
    for _ in range(n-3):
        d = a + b
        out.append(d)
        a, b, c = b, c, d
    return out


def geometric(n:int, start=1, expon=PHI) -> List[int]:
    """
    Generates a geometric series. With expon==PHI, results in a fibonacci series

    Args:
        n: number of items to generate
        start: the starting number of the series
        expon: the exponential of the series

    Returns:
        a list of n elements
    """
    x = start
    out = [start]
    for _ in range(n-1):
        x = round(x * expon)
        out.append(x)
    return out

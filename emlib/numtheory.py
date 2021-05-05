"""
A subset of NZMATH to detect and calculate primes without having
MZMATH as a dependency

I don't claim to understand any of this code, but I often need
to check for primes and find a coprime or next prime to a given
number
"""
from __future__ import annotations
import math
from math import gcd
from typing import Iterator as Iter, Tuple

PRIMES_LE_31 = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
PRIMONIAL_31 = 200560490130


def isprime(n:int, pdivisors=None) -> bool:
    """
    Return True iff n is prime.

    The check is done without APR.
    Assume that n is very small (less than 10**12) or
    prime factorization of n - 1 is known (prime divisors are passed to
    the optional argument pdivisors as a sequence).
    """
    if gcd(n, PRIMONIAL_31) > 1:
        return (n in PRIMES_LE_31)
    elif n < 10 ** 12:
        # 1369 == 37**2
        # 1662803 is the only prime base in smallSpsp which has not checked
        return n < 1369 or n == 1662803 or small_spsp(n)
    else:
        raise ValueError("only numbers below 10**12 are supported")

"""
def gcd(a:int, b:int) -> int:
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a
"""

def small_spsp(n:int, s:int=None, t:int=None) -> bool:
    if s is None or t is None:
        s, t = vp(n - 1, 2)
    for p in (2, 13, 23, 1662803):
        if not spsp(n, p, s, t):
            return False
    return True    


def vp(n:int, p:int, k=0) -> Tuple[int, int]:
    """
    Return p-adic valuation and indivisible part of given integer.

    For example:
    >>> vp(100, 2)
    (2, 25)

    That means, 100 is 2 times divisible by 2, and the factor 25 of
    100 is indivisible by 2.

    The optional argument k will be added to the valuation.
    """
    q = p
    while not (n % q):
        k += 1
        q *= p
    return (k, n // (q // p))


def spsp(n:int, base:int, s:int=None, t:int=None) -> bool:
    """
    Strong Pseudo-Prime test.  Optional third and fourth argument
    s and t are the numbers such that n-1 = 2**s * t and t is odd.
    """
    if s is None or t is None:
        s, t = vp(n-1, 2)
    z = pow(base, t, n)
    if z != 1 and z != n-1:
        j = 0
        while j < s:
            j += 1
            z = pow(z, 2, n)
            if z == n-1:
                break
        else:
            return False
    return True


def nextprime(n:int) -> int:
    """
    Return the smallest prime bigger than the given integer.
    """
    if n <= 1:
        return 2
    if n == 2:
        return 3
    n += (1 + (n & 1))   # make n be odd.
    while not primeq(n):
        n += 2
    return n


def primeq(n:int) -> bool:
    """
    A convenient function for primatilty test. It uses one of
    trialDivision, smallSpsp or apr depending on the size of n.
    """
    if int(n) != n:
        raise ValueError("non-integer for primeq()")
    if n <= 1:
        return False
    if gcd(n, PRIMONIAL_31) > 1:
        return (n in PRIMES_LE_31)
    if n < 2000000:
        return trial_division(n)
    if not small_spsp(n):
        return False
    if n < 10 ** 12:
        return True
    else:
        raise ValueError("only numbers below 10**12 are supported")
    

def trial_division(n:int, bound:int=0) -> bool:
    """
    Trial division primality test for an odd natural number.
    Optional second argument is a search bound of primes.
    If the bound is given and less than the sqaure root of n
    and True is returned, it only means there is no prime factor
    less than the bound.
    """
    if bound:
        m = min(bound, floorsqrt(n))
    else:
        m = floorsqrt(n)
    #for p in bigrange.range(3, m+1, 2):
    for p in range(3, m+1, 2):
        if not (n % p):
            return False
    return True


def floorsqrt(a:int) -> int:
    """
    Return the floor of square root of the given integer.
    """
    if a < (1 << 59):
        return int(math.sqrt(a))
    else:
        # Newton method
        x = pow(10, (math.log(a, 10) // 2) + 1)   # compute initial value
        while True:
            x_new = (x + a//x) // 2
            if x <= x_new:
                return int(x)
            x = x_new


def primes_generator() -> Iter[int]:
    """
    Generate primes from 2 to infinity.
    """
    yield 2
    yield 3
    yield 5
    coprimeTo30 = (7, 11, 13, 17, 19, 23, 29, 31)
    times30 = 0
    while True:
        for i in coprimeTo30:
            if primeq(i + times30):
                yield i + times30
        times30 += 30


def coprime(a:int, b:int) -> bool:
    """
    Return True if a and b are coprime, False otherwise.

    For Example:
    >>> coprime(8, 5)
    True
    >>> coprime(-15, -27)
    False
    >>>
    """
    return gcd(a, b) == 1


def lcm(a:int, b:int) -> int:
    """
    Return the least common multiple of given 2 integers.
    If both are zero, it raises an exception.
    """
    return a // gcd(a, b) * b


PHI = 1.618033988749894848204586834


def fibonacci(n:int, a=1, b=1):
    out = [a, b]
    for _ in range(n - 2):
        c = a + b
        out.append(c)
        a, b = b, c
    return out


def ifibonacci(a=1, b=1):
    yield a
    yield b
    while True:
        c = a + b
        yield c
        a, b = b, c


def lucas(n, a=2, b=1):
    return fibonacci(n, a, b)


def tribonacci(n:int, a=0, b=0, c=1):
    out = [a, b, c]
    for _ in range(n-3):
        d = a + b + c
        out.append(d)
        a, b, c = b, c, d
    return out


def padovan(n:int, a=1, b=1, c=1):
    """
    https://en.wikipedia.org/wiki/Padovan_sequence

    Args:
        n: the number of elements to generate
        a: first value of the seq.
        b: second value of the seq.
        c: third value of the seq
    """
    out = [a, b, c]
    for _ in range(n-3):
        d = a + b
        out.append(d)
        a, b, c = b, c, d
    return out


def geometric(n:int, start=1, expon=PHI):
    """
    Generaes a geometric series. With expon==PHI, results in a fibonacci series

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

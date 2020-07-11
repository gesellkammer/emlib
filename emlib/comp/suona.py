def fingeringHash(fingering):
    """
    fingering: x=closed, o=open, /=half
    """
    accum = 0
    for i, f in enumerate(fingering):
        if f == "x":
            accum += (3**i) * 2
        elif f == "/":
            accum += 3**i
    return accum
     
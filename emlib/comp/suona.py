from emlib import lib


def _normalize_fingering(fingering:str) -> str:
    return fingering.lower().replace("|", "")


def fingeringHash(fingering:str) -> int:
    """
    fingering: x=closed, o=open, /=half
    """
    accum = 0
    fingering = _normalize_fingering(fingering)
    for i, f in enumerate(fingering):
        if f == "x":
            accum += (3**i) * 2
        elif f == "/":
            accum += 3**i
    return accum
 

_replacer = lib.makereplacer({'2': 'x', '1': '/', '0': 'o'})


def hash2fingering(hash:int) -> str:
    """
    The inverse of fingeringHash: convert the int representation
    to the original fingering
    """
    base3 = lib.convert_base(str(hash), 10, 3)
    fingering = _replacer(base3)
    fingering = fingering[-1::-1]
    fingering = fingering[:4] + '|' + fingering[4:]
    return fingering

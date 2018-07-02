class Null:
    def __getitem__(self, item): pass
    def __getattr__(self, item): pass
    def __add__(self, other): pass
    def __mul__(self, other): pass
    def __int__(self): pass
    def __eq__(self, other): return False




null = Null()


def getitem(seq, idx):
    try:
        return seq[idx]
    except IndexError:
        return null


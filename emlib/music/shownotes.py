#!/usr/bin/env python3
import liblo
from emlib.mus import Chord
from emlib.music import combtones
from emlib.pitch import m2f, f2m, m2n, n2m
import subprocess
import tempfile
import os
from emlib import conftools

_DEFAULTSERVER = None


_defaultcfg = {
    'm21.showformat': 'lily.png',
    'linux.imageview': 'feh --reload 0.5 {path}',
    'noteserver.verbose': True,
    'noteserver.port': 90109
}

_validator = {
    'm21.showformat::choices': ['musicxml', 'musicxml.png', 'lily.pdf', 'lily.png']
}

config = conftools.ConfigDict("emlib:shownotes", default=_defaultcfg, validator=_validator)


def getConfig():
    return config


def getServer():
    global _DEFAULTSERVER
    if _DEFAULTSERVER is None:
        _DEFAULTSERVER = NoteServer()
    return _DEFAULTSERVER


def _asmidi(seq):
    if isinstance(seq[0], str):
        seq = [n2m(p) for p in seq]
    seq = [p for p in seq if p > 0]
    return seq


def showChord(pitches):
    pitches = _asmidi(pitches)
    getServer().showChord(pitches)


def showChords(chords):
    chords = [_asmidi(chord) for chord in chords]
    getServer().showChords(chords)


def showDifftones(pitches):
    pitches = _asmidi(pitches)
    return getServer().showDifftones(pitches)

    
class NoteServer:
    def __init__(self, port=None, verbose=None):
        cfg = getConfig()
        self.port = port if port is not None else cfg['noteserver.port']
        self.imgproc = None
        fd, path = tempfile.mkstemp(prefix="shownotes-", suffix=".png")
        self.imgfile = path
        self.imgbase = os.path.splitext(self.imgfile)[0]
        self.verbose = verbose if verbose is not None else cfg['noteserver.verbose']

    def showm21(self, stream, fmt=None):
        if fmt is None:
            fmt = 'lily.png'
            # at the moment musescore crashed with some simple chords
        stream.write(fmt, self.imgbase)
        if self.imgproc is None or self.imgproc.poll() is not None:
            # TODO
            img = self.imgbase + '.png'
            args = ["feh", "--reload", "0.5", img]
            self.imgproc = subprocess.Popen(args)    

    def cmd_chord(self, path, args, types):
        pitches = args
        self.showChord(pitches)
        
    def cmd_difftones(self, path, args, types):
        pitches = args
        self.showDifftones(pitches)
        
    def showDifftones(self, pitches):
        pitches = [p for p in pitches if p > 0]
        freqs = combtones.difftones(*[m2f(p) for p in pitches])
        freqs = [max(f, 1) for f in freqs]
        diffpitches = [f2m(f) for f in freqs]
        self.showChords([pitches, diffpitches])
        if self.verbose:
            printChord(diffpitches, label="Difference Tones")
        
    def showChord(self, pitches):
        pitches = [p for p in pitches if p > 0]
        chord = Chord(pitches)
        m = chord.asmusic21(split=True)
        self.showm21(m)

    def showChords(self, chords):
        """ each chord is a list of pitches """
        m = splitchords([Chord(chord) for chord in chords])
        self.showm21(m)
    
    def serve(self):
        s = liblo.Server(self.port)
        s.add_method("/chord", None, self.cmd_chord)
        s.add_method("/difftones", None, self.cmd_difftones)    
        while True:
            s.recv(50)

def printChord(chord, label=None):
    print()
    if label is not None:
        print(label)
        print()
    import tabulate
    chord = sorted(chord, reverse=True)
    rows = [[m2n(p), p, m2f(p)] for p in chord]
    s = tabulate.tabulate(rows)
    print(s)

    
def serve(port=None, verbose=None):
    return NoteServer(port=port, verbose=verbose).serve()
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=90109, type=int)
    args = parser.parse_args()
    serve(verbose=True, port=args.port)

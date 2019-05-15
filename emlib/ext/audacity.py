"""
Implements some helper functions to interact with audacity, in
particular to read markers and labels and convert 
them to useful representations in python

dependencies:
* peach (convert between midi-notes and frequencies)
* bpf4 for bpf support

"""

import os
from collections import namedtuple
from emlib.pitchtools import f2n, f2m

_Label = namedtuple("Label", "start end label")
_Bin = namedtuple("Bin", "freq level")


# some helper functions to read info from audacity


def read_labels(filename, convert_line_ending=True, ignore_name=True):
    """
    import the labels generated in audacity

    returns an iterator of tuples, each represents a label 

    a label is a tuple of the form: (begin_float, end_float, label_string)

    if ignore_name is True, only the times are returned
    """
    assert os.path.exists(filename)
    if convert_line_ending:
        f = open(filename, 'r')
        # read the beginning
        data = f.read(100)
        # look if there are CR
        if '\r' in data:
            # read all in memory, this files should not be
            # so big anyway
            data = (data + f.read()).replace('\r', '\n')
            f.close()
            # write it back, now with LF
            open(filename, 'w').write(data)
        else:
            f.close()
    f = open(filename, 'r')
    if not ignore_name:
        for line in f:
            words = line.split()
            if len(words) == 2:
                begin, end = words
                label = ''
            else:
                begin, end, label = words
            begin = float(begin)
            end = float(end)
            yield (begin, end, label)
    else:
        for line in f:
            words = line.split()
            if len(words) == 2:
                begin, end = words
                label = ''
            else:
                begin, end, label = words
            begin = float(begin)
            end = float(end)
            yield (begin, end)


def write_labels(outfile, markers):
    """
    markers : a sequence of tuples (start, end) or (start, end, name)
    """
    labels = []
    for i, marker in enumerate(markers):
        if len(marker) == 2:
            start, end = marker
            name = str(i)
        elif len(marker) == 3:
            start, end, name = marker
        else:
            raise ValueError(
                "a Marker is a tuple of the form (start, end) or (start, end, label)")
        labels.append((start, end, name))
    if outfile is not None:
        with open(outfile, 'w') as f:
            for label in labels:
                f.write("\t".join(map(str, label)))
                f.write("\n")
    

def read_spectrum(path):
    f = open(path)
    lines = f.readlines()[1:]
    out = []
    for line in lines:
        freq, level = list(map(float, line.split()))
        out.append(_Bin(freq, level))
    return out


def read_spectrum_as_chords(path, split=8, max_notes_per_chord=float('inf')):
    from bpf4 import bpf
    Note = namedtuple("Note", "note midi freq level step")
    data = read_spectrum(path)
    step = (
        bpf.expon(
            -120, 0,
            -60, 0.0,
            -40, 0.1,
            -30, 0.4,
            -18, 0.9,
            -6, 1,
            0, 1,
            exp=0.3333333
        ) * split
    ).apply(int)
    notes = [] 
    for bin in data:
        note = Note(f2n(bin.freq), f2m(bin.freq), bin.freq, bin.level, int(step(bin.level)))
        notes.append(note)
    chords = [[] for i in range(split)]
    notes2 = sorted(notes, key=lambda n:n.level, reverse=True)
    for note in notes2:
        chord = chords[note.step]
        if len(chord) <= max_notes_per_chord:
            chord.append(note)
    for chord in chords:
        chord.sort(key=lambda n:n.level, reverse=True)
    return chords


def read_spectrum_as_bpf(path):
    try:
        from bpf4 import bpf
    except ImportError:
        raise ImportError("bpf support not found. bpf4 should be installed.")
    freqs = []
    levels = []
    f = open(path)
    lines = f.readlines()[1:]
    for line in lines:
        freq, level = list(map(float, line.split()))
        freqs.append(freq)
        levels.append(level)
    return bpf.core.Linear(freqs, levels)

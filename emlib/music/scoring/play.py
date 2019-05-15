import sys
import logging
import subprocess
import re
from functools import lru_cache

import bpf4 as bpf

from emlib.pitchtools import m2n
from emlib.music.scoring import *
import emlib.typehints as t
from emlib import iterlib

logger = logging.getLogger("emlib.scoring")

@lru_cache(maxsize=1)
def _get_fluidsf2():
    if sys.platform == 'linux':
        sf2path = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
    else:
        raise RuntimeError("only works for linux right now")
    return sf2path


_orc = """
sr = {sr}
ksmps = 128
0dbfs = 1
nchnls = 2

gisf  sfload "{sf2path}"

instr piano
    idur, inote0, inote1, idb passign 3
    ivel bpf idb, -120, 0, -90, 10, -70, 20, -24, 90, 0, 127
    knote = linseg(inote0, idur, inote1)
    kfreq = mtof(knote)
    kamp = 2/32768
    a0, a1 sfinstr ivel, inote0, kamp, kfreq, 147, gisf, 1
    aenv linsegr 1, 0.0001, 1, 0.2, 0
    a0 *= aenv
    a1 *= aenv
    outs a0, a1
endin 

instr sine
    idur, inote0, inote1, idb passign 3
    iamp ampdb idb
    iamp *= 0.7
    knote = linseg(inote0, idur, inote1)
    kfreq = mtof(knote)
    a0 oscili iamp, kfreq
    iamp0 = 0.0
    ishape = 0.8
    aenv linsegr 0, 0.05, 1*iamp, 0.1, 0.8*iamp, 0.2, 0
    ; aenv transegr 0, 0.2, ishape, iamp, 0.2, 0, ishape
    ; aenv expsegr iamp0, 0.1, 0.8*iamp, 0.2, iamp0
    a0 *= aenv
    outs a0, a0
endin 

instr sine_trem
    idur, inote0, inote1, idb passign 3
    iamp ampdb idb
    iamp *= 0.7
    knote = linseg(inote0, idur, inote1)
    kfreq = mtof(knote)
    a0 oscili iamp, kfreq
    itremfreq = 20
    atrem = oscili:a(0.5, itremfreq) * 2 - 1 + 0.1
    ; atrem = vco:a(1, 20, 2, 0.5, -1) * 2 - 1 + 0.1
    aenv linsegr 0, 0.05, 1*iamp, 0.1, 0.8*iamp, 0.1, 0
    iamp0 = 0.00001
    ; aenv expsegr iamp0, 0.05, 1*iamp, 0.1, 0.8*iamp, 0.2, iamp0
    aenv *= atrem
    a0 *= aenv
    outs a0, a0
endin 

instr saw
    idur, inote0, inote1, idb passign 3
    iamp ampdb idb
    knote = linseg(inote0, idur, inote1)
    kfreq = mtof(knote)
    a0 oscili iamp, kfreq
    a0 vco iamp, kfreq, 1, 0.5, -1
    imaxamp = 0.1 * iamp
    aenv linsegr 0, 0.05, imaxamp, 0.15, 0.8*imaxamp, 0.12, 0
    a0 *= aenv
    outs a0, a0
endin 

instr square
    idur, inote0, inote1, idb passign 3
    iamp ampdb idb
    knote = linseg(inote0, idur, inote1)
    kfreq = mtof(knote)
    a0 oscili iamp, kfreq
    a0 vco iamp, kfreq, 2, 0.5, -1
    imaxamp = 0.1 * iamp
    aenv linsegr 0, 0.05, imaxamp, 0.15, 0.8*imaxamp, 0.2, 0
    a0 *= aenv
    outs a0, a0
endin 

instr saw_trem
    idur = p3
    inote0 = p4
    inote1 = p5
    idb = p6
    iamp ampdb idb
    knote = linseg(inote0, idur, inote1)
    kfreq = mtof(knote)
    ;a0 oscili iamp, kfreq
    a0 vco iamp, kfreq, 1, 0.5, -1
    imaxamp = 0.3 * iamp
    aenv linsegr 0, 0.01, imaxamp, 0.1, 0.8*imaxamp, 0.1, 0
    atrem = oscili:a(0.5, 20) * 2 - 1 + 0.1
    ; atrem = vco:a(1, 20, 2, 0.5, -1) * 2 - 1 + 0.1
    aenv *= atrem
    a0 *= aenv
    outs a0, a0
endin 
"""

@lru_cache()
def makeCsoundOrc(sr=44100, instrs=None):
    """
    instrs: a list of csound instr definitions
    """
    orc = _orc.format(sr=sr, sf2path=_get_fluidsf2())
    if instrs:
        allinstrs = "\n".join(instrs)
        orc = orc + allinstrs
    return orc


def makeInstr(name, audiogen):
    template = r"""
    instr {name}
        idur, inote0, inote1, idb passign 3
        iamp ampdb idb
        iamp *= 0.7
        knote = linseg(inote0, idur, inote1)
        kfreq = mtof(knote)
        {audiogen}
        ; a0 oscili iamp, kfreq
        aenv cossegr 0, 0.05, 1*iamp, 0.1, 0.8*iamp, 0.2, 0
        a0 *= aenv
        outs a0, a0
    endin
    """
    return template.format(name=name, audiogen=audiogen)


def _get_possible_instrs(orc=None):
    matches = re.findall(r"\binstr\s+\b\S+\b", _orc)
    return [match.split()[1] for match in matches]


_possible_instrs = _get_possible_instrs()

_db2vel = bpf.linear(
    -120, 0,
    -90, 10,
    -70, 20,
    -24, 90,
    0, 127
)


def makeCsoundScore(notes: t.List[Note], defaultinstr='piano',
                    allowDiscontinuousGliss=True) -> str:
    notes = sorted(notes, key=lambda note: note.offset)
    lines = []

    def scoreline(note0:Note, note1:t.Opt[Note]):
        if note0.db <= -120:
            return None
        if note0.gliss and note1 is not None:
            if note0.offset + note0.dur >= note1.offset:
                step1 = note1.step
            else:
                if allowDiscontinuousGliss:
                    step1 = note1.step
                else:
                    logger.debug("asked gliss, but notes are not contiguous!")
                    logger.debug(f"    {note0}  ->  {note1}")
        elif note0.stepend > 0:
            step1 = note0.stepend
        else:
            step1 = note0.step
        start = float(note0.offset)
        dur = float(note0.dur)
        comment = f"{m2n(note0.step)} - {m2n(step1)}"
        instr = note0.instr or defaultinstr
        if instr not in _possible_instrs:
            logger.error(f"note with unknown instr {instr}, falling back to default.\n"
                         f"Possible instruments are: {_possible_instrs}")
            instr = defaultinstr
        return f'i "{instr}"\t{start:.6f}\t{dur:.6f}\t{note0.step}\t{step1}\t{note0.db}\t\t  ; {comment}'

    for note0, note1 in iterlib.pairwise(notes):
        line = scoreline(note0, note1)
        if line:
            lines.append(line)
    lastline = scoreline(notes[-1], None)
    if lastline:
        lines.append(lastline)
    return "\n".join(lines)


def playNotes(events: t.List[Event], defaultinstr=None, sr=None, instrs=None) -> subprocess.Popen:
    """
    gliss:
        if True, the metadata value "gliss" is honoured and a gliss is generated
        between adjacent events wherever this value is set and True
        The .stepend value is also taken into account, so if both are set
        the gliss value has priority
    defaultinstr:
        the default instr used when a Note has no assigned instr
    sr:
        sample rate
    instrs:
        a list of csound instr definitions

    """
    from emlib.snd import csound
    defaultinstr = defaultinstr or 'piano'
    sco = makeCsoundScore(events, defaultinstr=defaultinstr)
    if sr is None:
        sr = csound.get_sr()
    orc = makeCsoundOrc(sr, instrs)
    proc = csound.run_csd(orc=orc, sco=sco, output="dac", piped=True, extradur=0.5)
    return proc

import math as _math
import os 
import sys
import subprocess as _subprocess
import re
from collections import namedtuple
import shutil as _shutil
import logging as _logging
import textwrap as _textwrap
import tempfile
import typing as t

import numpy as np

import emlib.lib as _lib


"""
helper functions to work with csound
"""

logger = _logging.getLogger("emlib.csound")  # type: _logging.Logger


class PlatformNotSupported(Exception):
    pass


def nextpow2(n:int) -> int:
    return int(2 ** _math.ceil(_math.log(n, 2)))
    

def table_size_sndfile(filename:str) -> int:
    """
    return the length of the table for GEN01 (read sound file)
    the length must be the next power of 2 bigger than the number 
    of samples in the sound file
    """
    import sndfileio
    info = sndfileio.sndinfo(filename)
    return nextpow2(info.nframes)
    

def find_csound() -> t.Optional[str]:
    csound = _shutil.which("csound")
    if csound:
        return csound
    logger.error("csound is not in the path!")
    if sys.platform.startswith("linux") or sys.platform == 'darwin':
        for path in ['/usr/local/bin/csound', '/usr/bin/csound']:
            if os.path.exists(path) and not os.path.isdir(path):
                return path
        return None
    elif sys.platform == 'win32':
        return None
    else:
        raise PlatformNotSupported
    

def get_version():
    """
    Returns the csound version as tuple (major, minor, patch) so that '6.03.0' is (6, 3, 0)

    Raises IOError if either csound is not present or its version 
    can't be parsed
    """
    csound = find_csound()
    if not csound:
        raise IOError("Csound not found")
    cmd = '{csound} --help'.format(csound=csound).split()
    proc = _subprocess.Popen(cmd, stderr=_subprocess.PIPE)
    proc.wait()
    lines = proc.stderr.readlines()
    if not lines:
        raise IOError("Could not read csounds output")
    for line in lines:
        if line.startswith("Csound version"):
            matches = re.findall("(\d+\.\d+(\.\d+)?)", line)
            if matches:
                version = matches[0]
                if isinstance(version, tuple):
                    version = version[0]
                points = version.count(".")
                if points == 1:
                    major, minor = list(map(int, version.split(".")))
                    patch = 0
                else:
                    major, minor, patch = list(map(int, version.split(".")[:3]))
                return (major, minor, patch)
    else:
        raise IOError("Did not found a csound version")

_OPCODES = None


def csound_subproc(args, piped=True):
    """
    Calls csound with the given args in a subprocess, returns
    such subprocess. 

    """
    csound = find_csound()
    if not csound:
        return
    p = _subprocess.PIPE if piped else None
    callargs = [csound]
    callargs.extend(args)
    logger.debug(f"csound_subproc> args={callargs}")
    return _subprocess.Popen(callargs, stderr=p, stdout=p)
    

def get_default_backend():
    """
    Get the default backend for platform. Check if the backend
    is available and running (in the case of Jack)

                Default
    -------------------------------------------------
    linux       jack if present, pa_cb otherwise
    mac         auhal (coreaudio)
    win32       pa_cb
    """
    backends = get_audiobackends()
    if "jack" in backends:
        return "jack"
    return backends[0]


def run_csdfile(csdfile:str, backend="", output="dac", input="", 
                piped=False, extra=None) -> _subprocess.Popen:
    """
    csdfile: the path to a .csd file
    output : "dac" to output to the default device, the label of the
             device (dac0, dac1, ...), or a filename to render offline
    input  : The input to use (for realtime)
    backend: The name of the backend to use. If no backend is given, the default
             for the platform is used (this is only meaningful if running in realtime)
    piped  : if True, the output of the csound process is piped and can be accessed
             through the Popen object (.stdout, .stderr)
    extra  : a list of extra arguments to be passed to csound

    Returns the subprocess
    """
    rendertofile = "." in output
    args = []
    args.extend(["-o", output])
    
    if not rendertofile and backend:
        args.append(f"-+rtaudio={backend}")
    if input:
        args.append(f"-i {input}")
    if extra:
        args.extend(extra)
    args.append(csdfile)
    return csound_subproc(args, piped=piped)
    

def _join_csd(orc, sco="", options="", outfile=None):
    """
    """
    csd = r"""
<CsoundSynthesizer>
<CsOptions>
{options}
</CsOptions>
<CsInstruments>

{orc}

</CsInstruments>
<CsScore>

{sco}

</CsScore>
</CsoundSynthesizer>
    """.format(options=options, orc=orc, sco=sco)
    csd = _textwrap.dedent(csd)
    if outfile is None:
        outfile = tempfile.mktemp(suffix=".csd")
    with open(outfile, "w") as f:
        f.write(csd)
    logger.debug(f"_join_csd: saving csd to {outfile}")
    return outfile


def testcsound(dur=8, nchnls=2, backend=None, device="dac", sr=None, verbose=True):
    backend = backend or get_default_backend()
    sr = sr or get_sr(backend)
    printchan = "printk2 kchn" if verbose else ""
    orc = f"""
sr = {sr}
ksmps = 128
nchnls = {nchnls}

instr 1
    iperiod = 1
    kchn init -1
    ktrig metro 1/iperiod
    kchn = (kchn + ktrig) % nchnls 
    anoise pinker
    outch kchn+1, anoise
    {printchan}
endin
    """
    sco = f"i1 0 {dur}"
    orc = _textwrap.dedent(orc)
    logger.debug(orc)
    return run_csd(orc, sco=sco, backend=backend, output=device, extra=["-d", "-m 0"])


ScoreEvent = namedtuple("ScoreEvent", "type name start dur args")

def parse_sco(sco):
    for line in sco.splitlines():
        words = line.split()
        w0 = words[0]
        if w0 == 'i':
            name = words[1]
            t0 = float(words[2])
            dur = float(words[3])
            rest = words[4:]
        elif w0[0] == 'i':
            name = w0[1:]
            t0 = float(words[1])
            dur = float(words[2])
            rest = words[3:]
        else:
            continue
        yield ScoreEvent("i", name, t0, dur, rest)


def run_csd(orc:str, sco="", backend="", output="dac", input="", 
            piped=False, extra=[], extradur=0, supressdisplay=False) -> _subprocess.Popen:
    """
    orc     : the text which normally goes inside <CsInstruments>
    sco     : the text which normally goes inside <CsScore>
    backend : the realtime backend to use (if applicable)
    output  : the soundfile to generate or the output device to use for realtime
    input   : the input device to use for realtime (if applicable)
    piped   : Should the subprocess be piped?
    supressdisplay : if True, all output of csound is supressed (implies piped)
    extradur: Should the duration of the process be extended by some amount?
    """
    def sco_get_end(sco):
        events = parse_sco(sco)
        return max(event.start+event.dur for event in events)

    if extradur:
        t1 = sco_get_end(sco)
        sco += f'\nf0 {t1 + extradur}'

    tmpcsd = _join_csd(orc, sco)
    extraArgs = []
    if supressdisplay:
        extraArgs.extend(['-d', '-m', '0'])
        piped = True
    else:
        piped = False
    if extra:
        extraArgs.extend(extra)
    return run_csdfile(tmpcsd, backend=backend, output=output, input=input,
                       piped=piped, extra=extraArgs)


def get_opcodes(force=False):
    """
    Return a list of the opcodes present
    """
    global _OPCODES
    if _OPCODES is not None and not force:
        return _OPCODES
    s = csound_subproc(['-z'])
    lines = s.stderr.readlines()
    allopcodes = []
    for line in lines:
        if line.startswith("end of score"):
            break
        opcodes = line.split()
        if opcodes:
            allopcodes.extend(opcodes)
    _OPCODES = allopcodes
    return allopcodes

   
def _calculate_table_size(X):
    mindenom = _lib.mindenom(X)
    size = int(max(X) * mindenom + 1)
    return size, mindenom


def save_as_gen23(data, outfile, fmt="%.12f", header=""):
    """
    Saves the points to a gen23 table

    NB: gen23 is a 1D list of numbers in text format, sepparated
        by a space

    data: seq
        A 1D sequence (list or array) of floats
    outfile: path
        The path to save the data to. Recommended extension: '.gen23'
    fmt: 
        If saving frequency tables, fmt can be "%.1f" and save space,
        for amplitude the default if "%.12f" is best
    header: str
        If specified it is included as a comment as the first line
        Csound will skip it. It is there just to document what is
        in the table

    Example:

    >>> a = bpf.linear(0, 0, 1, 10, 2, 300)
    >>> sampling_period = 0.01
    >>> points_to_gen23(a[::sampling_period].ys, "out.gen23", header=f"dt={sampling_period}")
    
    In csound

    gi_tab ftgen 0, 0, 0, -23, "out.gen23"
 
    instr 1
      itotaldur = ftlen(gi_tab) * 0.01
      ay poscil 1, 1/itotaldur, gi_tab
    endin
    """
    if header:
        np.savetxt(outfile, data, fmt=fmt, header="# " + header)
    else:
        np.savetxt(outfile, data, fmt=fmt)


def matrix_as_wav(wavfile, m, dt, t0=0):
    """
    Save the data in m as a wav file. This is not a real soundfle
    but it is used to transfer the data in binary form to be 
    read in csound

    Format:

        header: headerlength, dt, numcols, numrows
        rows: each row has `numcol` number of items

    wavfile: str
        the path where the data is written to
    m: a numpy array of shape (numcols, numsamples)
        a 2D matrix representing a series of streams sampled at a 
        regular period (dt)
    dt: float
        metadata: the sampling period of the matrix
    t0: float
        metadata: the sampling offset of the matrix
        (t = row*dt + t0)
    """
    assert isinstance(wavfile, str)
    assert isinstance(dt, float)
    assert isinstance(t0, float)
    import sndfileio
    sndwriter = sndfileio.sndwrite_chunked(sr=44100, outfile=wavfile, encoding="flt32")
    numrows, numcols = m.shape
    header = np.array([5, dt, numcols, numrows, t0], dtype=float)
    sndwriter.write(header)
    mflat = m.ravel()
    sndwriter.write(mflat)


def matrix_as_gen23(outfile, m, dt, t0=0, header=True):
    numrows, numcols = m.shape
    header = np.array([5, dt, numcols, numrows, t0], dtype=float)
    m = m.round(6)
    with open(outfile, "w") as f:
        if header:
            f.write(" ".join(header.astype(str)))
            f.write("\n")
        for row in m:
            rowstr = " ".join(row.astype(str))
            f.write(rowstr)
            f.write("\n")



_Dev = namedtuple("Dev", "index label name")


def get_audiodevices(backend:str=""):
    """
    Returns (indevices, outdevices), where each of these lists 
    is a tuple (index, label, name)

    backend: 
        specify a backend supported by your installation of csound
        None to use a default for you OS
    label: 
        is something like 'adc0', 'dac1' and is what you
        need to pass to csound to its -i or -o methods. 
    name: 
        the name of the device. Something like "Built-in Input"

    Backends:

            OSX  Linux  Win   Multiple-Devices    Description
    jack     x      x    -     -                  Jack
    auhal    x      -    -     x                  CoreAudio
    pa_cb    x      x    x     x                  PortAudio (Callback)
    pa_bl    x      x    x     x                  PortAudio (blocking)
    """
    if not backend:
        backend = get_default_backend()
    indevices, outdevices = [], []
    proc = csound_subproc(['-+rtaudio=%s' % backend, '--devices'])
    proc.wait()
    lines = proc.stderr.readlines()
    # regex_all = r"([0-9]+):\s(adc[0-9]+|dac[0-9]+)\s\((.+)\)"
    regex_all = r"([0-9]+):\s((?:adc|dac).+)\s\((.+)\)"
    for line in lines:
        line = line.decode("ascii")
        match = re.search(regex_all, line)
        if not match:
            continue
        idxstr, devid, devname = match.groups()
        dev = _Dev(int(idxstr), devid, devname)
        if devid.startswith("adc"):
            indevices.append(dev)
        else:
            outdevices.append(dev)
    return indevices, outdevices


def get_sr(backend:str="") -> float:
    """
    Returns the samplerate reported by the given backend, or
    0 if failed

    If no backend is specified, the default backend is used
    (returned by get_default_backend)

    """
    failed_sr = 0
    if not backend:
        backend = get_default_backend()
    if backend == 'jack':
        sr = int(_subprocess.getoutput("jack_samplerate"))
        return sr 
    else:
        proc = csound_subproc(f"-odac -+rtaudio={backend} --get-system-sr".split())
        proc.wait()
    srlines = [line for line in proc.stdout.readlines() 
               if line.startswith(b"system sr:")]
    if not srlines:
        logger.error(f"get_sr: Failed to get sr with backend {backend}")
        return failed_sr
    sr = float(srlines[0].split(b":")[1].strip())
    logger.debug(f"get_sr: sample rate query output: {srlines}")
    if sr < 0:
        return failed_sr
    return sr


def is_backend_available(backend):
    if backend == 'jack':
        if sys.platform == 'linux':
            status = int(_subprocess.getstatusoutput("jack_control status")[0])
            return status == 0
        else:
            proc = csound_subproc(['+rtaudio=jack', '--get-system-sr'])
            proc.wait()
            return b'JACK module enabled' in proc.stderr.read()
    else:
        indevices, outdevices = get_audiodevices(backend=backend)
        return bool(indevices or outdevices)
        

_platform_backends = {
    'linux': ["jack", "pa_cb", "alsa", "pa_bl", "pulse"],
    'darwin': ["auhal", "pa_cb", "pa_bl", "auhal"],
    'win32': ["pa_cb", "pa_bl"]
}

_backends_always_on = {'pa_cb', 'pa_bl', 'auhal', 'alsa'}


def get_audiobackends(checkall=False):
    """ 
    Return a list of supported audio backends as they would be passed to -+rtaudio

    This is supported by csound >= 6.3.0

    if checkall is False, only those backends which can be available
    in csound but might not be present (jack) are checked
    """
    backends = _platform_backends[sys.platform]
    if checkall:
        backends = [b for b in backends if is_backend_available(b)]
    else:
        backends = [b for b in backends if b in _backends_always_on or is_backend_available(b)]
    return backends


_backends_which_support_systemsr = {'jack', 'alsa', 'coreaudio'}
_backends_which_need_realtime = {'alsa', 'pa_cb'}


Event = namedtuple("Event", "instr start dur args")


class Score(object):
    def __init__(self):
        self.events = []
        self.instrs = {}

    def addevent(self, instr, start, dur, *args):
        event = Event(instr=instr, start=start, dur=dur, args=args)
        self.events.append(event)

    def writescore(self, stream=None):
        stream = stream or sys.stdout

        def write_event(stream, event):
            args = [str(arg) if not isinstance(arg, str) else ('"%s"' % arg) for arg in event.args]
            argstr = " ".join(args) 
            s = "i {instr} {start} {dur} {args}\n".format(
                instr=event.instr, start=event.start, dur=event.dur,
                args=argstr)
            stream.write(s)
        
        self.events.sort(key=lambda ev: ev.start)
        for event in self.events:
            write_event(stream, event)

    def addinstr(self, instr, instrstr):
        self.instrs[instr] = instrstr

    def writecsd(self, outfile, sr, ksmps=64, nchnls=2):
        stream = open(outfile, "w")
        stream.write("<CsoundSynthesizer>\n")
        footer = "</CsoundSynthesizer>"
        txt = f"""
            <CsInstruments>
            sr = {sr}
            ksmps = {ksmps}
            0dbfs = 1
            nchnls = 2

            """
        txt = _textwrap.dedent(txt)
        stream.write(txt)
        instrkeys = sorted(self.instrs.keys())
        if instrkeys[0] == 0:
            stream.write(self.instrs[0])
            instrkeys = instrkeys[1:]
        for instrkey in instrkeys:
            stream.write("instr {key}\n".format(key=instrkey))
            for line in self.instr[instrkeys].splitlines():
                stream.write("    {line}\n".format(line=line))
            stream.write("endin\n")
            stream.write(self.instrs[instrkeys])
        stream.write("</CsInstruments>\n")
        stream.write("<CsScore>\n")
        self.writescore(stream)
        stream.write("</CsScore>\n")
        for line in footer.splitlines():
            stream.write(line)
            stream.write("\n")
        

class Timeline(object):
    def __init__(self, sr=None):
        self.events = []
        self.sndfiles = {}
        self._ftablenum = 1
        self.sr = sr

    def _get_ftable_num(self):
        self._ftablenum += 1
        return self._ftablenum

    def add(self, sndfile, time, gain=1, start=0, end=-1, fadein=0, fadeout=0):
        """
        time: start of playback
        start, end: play a slice of the sndfile
        fadein, fadeout: fade time in seconds
        """
        sndfile = os.path.relpath(sndfile)
        if sndfile in self.sndfiles:
            ftable = self.sndfiles[sndfile]["ftable"]
        else:
            ftable = self._get_ftable_num()
            self.sndfiles[sndfile] = {"ftable": ftable}
        if end < 0:
            info = self._sndinfo(sndfile)
            end = info.duration
        event = {
            'time': time,
            'sndfile': sndfile, 
            'start': start, 
            'end': end, 
            'fadein': fadein, 
            'fadeout': fadeout,
            'ftable': ftable,
            'gain': gain

        }
        self.events.append(event)

    def _sndinfo(self, sndfile):
        if sndfile not in self.sndfiles:
            raise ValueError("No event was added witht hthe given sndfile")
        info = self.sndfiles[sndfile].get("info")
        if info is not None:
            return info
        else:
            import sndfileio
            info = sndfileio.sndinfo(sndfile)
            self.sndfiles[sndfile]["info"] = info
        return info

    def _guess_samplerate(self):
        sr = max(self._sndinfo(sndfile).samplerate for sndfile in self.sndfiles)
        return sr

    def totalduration(self):
        return max(event["time"] + (event["end"] - event["start"]) for event in self.events)

    def writecsd(self, outfile, sr=None, ksmps=64):
        self.events.sort(key=lambda event:event['time'])
        orc = """
<CsInstruments>
sr = {sr}
ksmps = {ksmps}
0dbfs = 1
nchnls = 2

instr 1
    Spath, ioffset, igain, ifadein, ifadeout, ienvpow passign  4
    iatt = ifadein > 0.00001 ? ifadein : 0.00001
    irel = ifadeout > 0.00001 ? ifadeout : 0.00001
    ifilelen = filelen(Spath)
    idur = ifilelen < p3 ? ifilelen : (p3 > 0 ? p3 : ifilelen)
    ipow = ienvpow > 0 ? ienvpow : 1
    ichnls = filenchnls(Spath)
    aenv linseg 0.000000001, iatt, 1, idur - (iatt+irel), 1, irel, 0.000000001
    aenv = pow(aenv, ipow) * igain
    ktime = line(0, idur*2, 2)
    if ( ktime > 1 ) then
        turnoff
    endif
    if (ichnls == 1) then
        a0 diskin2 Spath, 1, ioffset, 0, 0, 4
        a0 = a0 * aenv
        a1 = a0
    else
        a0, a1 diskin2 Spath, 1, ioffset, 0, 0, 4
        a0 = a0 * aenv
        a1 = a1 * aenv
    endif
    outs a0, a1
endin
</CsInstruments>
        """
        if sr is None:
            sr = self._guess_samplerate()
        self.sr = sr
        orc = orc.format(sr=sr, ksmps=ksmps)
        scorelines = ["<CsScore>"]
        for event in self.events:
            line = 'i {instr} {time} {dur} "{path}" {offset} {gain} {fadein} {fadeout} 1.5'.format(
                instr=1, time=event["time"], path=event["sndfile"], 
                dur=event['end'] - event['start'],
                gain=event["gain"], offset=event["start"], fadein=event["fadein"],
                fadeout=event["fadeout"]
                )
            scorelines.append(line)
        scorelines.append("f 0 {totalduration}".format(totalduration=self.totalduration()))       
        scorelines.append("</CsScore>")
        
        def writeline(f, line):
            f.write(line)
            if not line.endswith("\n"):
                f.write("\n")

        header = "<CsoundSynthesizer>"
        footer = "</CsoundSynthesizer>"

        with open(outfile, "w") as out:
            for line in header.splitlines():
                writeline(out, line)
            for line in orc.splitlines():
                writeline(out, line)
            for line in scorelines:
                writeline(out, line)
            for line in footer.splitlines():
                writeline(out, line)        


def mincer(sndfile, timecurve, pitchcurve, outfile=None, dt=0.002, 
           lock=False, fftsize=2048, ksmps=128, debug=False):
    """
    sndfile: the path to a soundfile
    timecurve: a bpf mapping time to playback time or a scalar indicating a timeratio
               (2 means twice as fast)
               1 to leave unmodified
    pitchcurve: a bpf mapping x=time, y=pitchscale. or a scalar indicating a freqratio
                (2 means an octave higher) 
                1 to leave unmodified

    outfile: the path to a resulting outfile

    Returns: a dictionary with information about the process 

    NB: if the mapped time excedes the bounds of the sndfile,
        silence is generated. For example, a negative time
        or a time exceding the duration of the sndfile

    NB2: the samplerate and number of channels of of the generated file matches 
         that of the input file

    NB3: the resulting file is always a 32-bit float .wav file

    ** Example 1: stretch a soundfile 2x

       timecurve = bpf.linear(0, 0, totaldur*2, totaldur)
       outfile = mincer(sndfile, timecurve, 1)
    """
    import bpf4 as bpf
    import sndfileio
    
    if outfile is None:
        outfile = _lib.add_suffix(sndfile, "-mincer")
    info = sndfileio.sndinfo(sndfile)
    sr = info.samplerate
    nchnls = info.channels
    pitchbpf = bpf.asbpf(pitchcurve)
    
    if isinstance(timecurve, (int, float)):
        t0, t1 = 0, info.duration / timecurve
        timebpf = bpf.linear(0, 0, t1, info.duration)
    elif isinstance(timecurve, bpf.core._BpfInterface):
        t0, t1 = timecurve.bounds()
        timebpf = timecurve
    else:
        raise TypeError("timecurve should be either a scalar or a bpf")
    
    assert isinstance(pitchcurve, (int, float, bpf.core._BpfInterface))
    ts = np.arange(t0, t1+dt, dt)
    fmt = "%.12f"
    _, time_gen23 = tempfile.mkstemp(prefix='time-', suffix='.gen23')
    np.savetxt(time_gen23, timebpf.map(ts), fmt=fmt, header=str(dt), comments="")
    _, pitch_gen23 = tempfile.mkstemp(prefix='pitch-', suffix='.gen23')
    np.savetxt(pitch_gen23, pitchbpf.map(ts), fmt=fmt, header=str(dt), comments="")
    if outfile is None:
        outfile = _lib.add_suffix(sndfile, '-mincer')
    csd = f"""
    <CsoundSynthesizer>
    <CsOptions>
    -o {outfile}
    </CsOptions>
    <CsInstruments>

    sr = {sr}
    ksmps = {ksmps}
    nchnls = {nchnls}
    0dbfs = 1.0

    gi_snd   ftgen 0, 0, 0, -1,  "{sndfile}", 0, 0, 0
    gi_time  ftgen 0, 0, 0, -23, "{time_gen23}"
    gi_pitch ftgen 0, 0, 0, -23, "{pitch_gen23}"

    instr vartimepitch
        idt tab_i 0, gi_time
        ilock = {int(lock)}
        ifftsize = {fftsize}
        ikperiod = ksmps/sr
        isndfiledur = ftlen(gi_snd) / ftsr(gi_snd)
        isndchnls = ftchnls(gi_snd)
        ifade = ikperiod*2
        inumsamps = ftlen(gi_time)
        it1 = (inumsamps-2) * idt           ; account for idt and last value
        kt timeinsts
        aidx    linseg 1, it1, inumsamps-1
        at1     tablei aidx, gi_time, 0, 0, 0
        kpitch  tablei k(aidx), gi_pitch, 0, 0, 0
        kat1 = k(at1)
        kgate = (kat1 >= 0 && kat1 <= isndfiledur) ? 1 : 0
        agate = interp(kgate) 
        aenv linseg 0, ifade, 1, it1 - (ifade*2), 1, ifade, 0
        aenv *= agate
        if isndchnls == 1 then
            a0  mincer at1, 1, kpitch, gi_snd, ilock, ifftsize, 8
            outch 1, a0*aenv
        else
            a0, a1   mincer at1, 1, kpitch, gi_snd, ilock, ifftsize, 8
            outs a0*aenv, a1*aenv
        endif
        
      if (kt >= it1 + ikperiod) then
        event "i", "exit", 0.1, 1
            turnoff     
        endif
    endin

    instr exit
        puts "exiting!", 1
        exitnow
    endin

    </CsInstruments>
    <CsScore>
    i "vartimepitch" 0 -1
    f 0 36000

    </CsScore>
    </CsoundSynthesizer>
    """
    _, csdfile = tempfile.mkstemp(suffix=".csd")
    with open(csdfile, "w") as f:
        f.write(csd)
    _subprocess.call(["csound", "-f", csdfile])
    if not debug:
        os.remove(time_gen23)
        os.remove(pitch_gen23)
        os.remove(csdfile)
    return {'outfile': outfile, 'csdstr': csd, 'csd': csdfile}


def _instrAsOrc(instrid, body, initstr, sr, ksmps, nchnls):
    orc = """
sr = {sr}
ksmps = {ksmps}
nchnls = {nchnls}
0dbfs = 1

{initstr}

instr {instrid}
    {body}
endin

    """.format(sr=sr, ksmps=ksmps, instrid=instrid, body=body, nchnls=nchnls, initstr=initstr)
    return orc


def extractPargs(body:str) -> t.Set[int]:
    regex = r"\bp\d+"
    pargs = re.findall(regex, body)
    nums = [int(parg[1:]) for parg in pargs]
    for line in body.splitlines():
        if not re.search(r"\bpassign\b", line):
            continue
        left, right = line.split("passign")
        numleft = len(left.split(","))
        pargstart = int(right) if right else 1
        ps = list(range(pargstart, pargstart + numleft))
        nums.extend(ps)
    return set(nums)


def numPargs(body:str) -> int:
    """
    analyze body to determine the number of pargs needed for this instrument
    """
    try:
        pargs = extractPargs(body)
    except ValueError:
        pargs = None
    if not pargs:
        return 0
    pargs = [parg for parg in pargs if parg >= 4]
    if not pargs:
        return 0
    maxparg = max(pargs)
    minparg = min(pargs)
    if maxparg - minparg > len(pargs):
        skippedpargs = [n for n in range(minparg, maxparg + 1) if n not in pargs]
        raise ValueError(f"pargs {skippedpargs} skipped")
    return len(pargs)

def pargNames(body:str) -> t.Dict[int, str]:
    """
    Analyze body to determine the names (if any) of the pargs used

    iname = p6
    kfoo, ibar passign 4
    """
    argnames = {}

    for line in body.splitlines():
        if re.search(r"\bpassign\b", line):
            names, firstidx = line.split("passign")
            firstidx = int(firstidx)
            names = names.split(",")
            for i, name in enumerate(names):
                argnames[i + firstidx] = name
        words = line.split()
        if len(words) == 3 and words[1] == "=":
            w2 = words[2]
            if w2.startswith("p") and all(ch.isdigit() for ch in w2[1:]):
                idx = int(w2[1:])
                argnames[idx] = words[0]
    # remove p1, p2 and p3, if present
    for idx in (1, 2, 3):
        argnames.pop(idx, None)
    return argnames

def numPargsMatchDefinition(instrbody: str, args: t.List) -> bool:
    lenargs = 0 if args is None else len(args)
    numargs = numPargs(instrbody)
    if numargs != lenargs:
        msg = f"Passed {lenargs} pargs, but instrument expected {numargs}"
        logger.error(msg)
        return False
    return True

def recInstr(body:str, events:t.List, init="", outfile="",
             sr=44100, ksmps=64, nchnls=2, a4=442, samplefmt='float',
             dur=None, comment=None, quiet=True) -> t.Tuple[str, _subprocess.Popen]:
    """
    Record one instrument for a given duration

    dur:
        the duration of the recording
    body:
        the body of the instrument
    init:
        the initialization code (ftgens, global vars, etc)
    outfile:
        the generated soundfile, or None to generate a temporary file
    events:
        a seq. of events, where each event is a list of pargs passed to the instrument,
        beginning with p2: delay, dur, [p4, p5, ...]
    sr, ksmps, nchnls: ...
    samplefmt: defines the sample format used for outfile, one of (16, 24, 32, 'float')
    """
    assert isinstance(body, str)
    if init is None: 
        init = ""
    instrnum = 100
    
    orc = f"""
sr = {sr}
ksmps = {ksmps}
nchnls = {nchnls}
0dbfs = 1
A4 = {a4}

{init}

instr {instrnum}
    {body}
endin"""
    score = []
    if not isinstance(events, list) and all(isinstance(event, (tuple, list)) for event in events):
        raise ValueError("events is a seq., where each item is a list of pargs passed to"
                         "the instrument, beginning with p2: [delay, dur, ...]"
                         f"Got {events} instead")
    for event in events:
        ok = numPargsMatchDefinition(body, event[2:])
        if not ok:
            logger.error(f"mismatch in number of pargs passed to instrument. pargs={event}")
        argstr = ' '.join(str(arg) for arg in event) if event else ''
        score.append(f'i {instrnum} {argstr}')
    if dur is not None:
        score.append(f'e {dur}')
    sco = "\n".join(score)
    if not outfile:
        outfile = tempfile.mktemp(suffix='.wav', prefix='csdengine-rec-')
    outfile = normalizePath(outfile)
    if outfile[-4:] != ".wav":
        raise ValueError(f"only .wav files are supported as output at the moment (given: {outfile})")
    fmtoption = {16: '', 24: '-3', 32: '-f', 'float': '-f'}.get(samplefmt)
    if fmtoption is None:
        raise ValueError("samplefmt should be one of 16, 24, 32, or 'float'")
    extra = [fmtoption]
    if comment:
        extra.append(f'-+id_comment="{comment}"')
    proc = run_csd(orc=orc, sco=sco, backend="", output=outfile, extra=extra, supressdisplay=quiet)
    return outfile, proc


def normalizePath(path):
    return os.path.abspath(os.path.expanduser(path))

    
def genBodyStaticSines(numsines, sineinterp=True, attack=0.05, release=0.1, curve='cos', extend=False):
    """
    Generates the body of an instrument for additive synthesis. In order to be
    used it must be wrapped inside "instr xx" and "endin"

    numsines: the number of sines to generate
    sineinterp: if True, the oscilators use interpolation (oscili)
    extend: extend duration for the release, using linsegr (otherwise a fixed envelope is used)

    It takes the following p-fields:

    chan, gain, freq1, amp1, ..., freqN, ampN

    Where:
        N is numsines
        gain is a gain factor affecting all sines
        ampx represent the relative amplitude of each sine

    Example:

    freqs = [440, 660, 880]
    amps = [0.5, 0.3, 0.2]
    body = bodySines(len(freqs))
    recInstr(dur=2, outfile="out.wav", body=body, args=[1, 1] + list(flatten(zip(freqs, amps))))
    """
    lines = [
        "idur = p3",
        "ichan = p4",
        "igain = p5",
        "aout = 0",
        "itot = 0"
    ]
    sinopcode = "oscili" if sineinterp else "oscil"
    _ = lines.append
    for i in range(numsines):
        _(f"ifreq_{i} = p{i*2+6}")
        _(f"iamp_{i}  = p{i*2+7}")
        _(f"iamp_{i} *= ifreq_{i} > 20 ? 1 : 0")
        _(f"itot += ifreq_{i} > 20 ? 1 : 0")
    for i in range(numsines-1):
        _(f"aout += {sinopcode}:a(iamp_{i}, ifreq_{i})")
        _(f"if itot == {i+1} goto exit")
    _(f"aout += {sinopcode}:a(iamp_{numsines-1}, ifreq_{numsines-1})")
    _("exit:")
    if curve == 'linear':
        if extend:
            env = f"linsegr:a(0, {attack}, igain, {release}, 0)"
        else:
            env = f"linseg:a(0, {attack}, igain, idur-{attack+release}, igain, {release}, 0)"
    elif curve == 'cos':    
        if extend:
            env = f"cossegr:a(0, {attack}, igain, {release}, 0)"
        else:
            env = f"cosseg:a(0, {attack}, igain, idur-{attack+release}, igain, {release}, 0)"
    else:
        raise ValueError(f"curve should be one of 'linear', 'cos', got {curve}")
    _( f"aout *= {env}" )
    _("outch ichan, aout")
    body = "\n".join(lines)
    return body


def genSoundfontInstr(sf2path):
    """
    returns (body, init)
    """
    pass


def _ftsave_read_text(path):
    # a file can have multiple tables saved
    lines = iter(open(path))
    tables = []
    while True:
        tablength = -1
        try:
            headerstart = next(lines)
            if not headerstart.startswith("===="):
                raise IOError(f"Expecting header start, got {headerstart}")
        except StopIteration:
            # no more tables
            break
        # Read header
        for line in lines:
            if line.startswith("flen:"):
                tablength = int(line[5:])
            if 'END OF HEADER' in line:
                break
        if tablength < 0:
            raise IOError("Could not read table length")
        values = np.zeros((tablength+1,), dtype=float)
        # Read data
        for i, line in enumerate(lines):
            if line.startswith("---"):
                break
            values[i] = float(line)
        tables.append(values)
    return tables
        
def ftsave_read(path, mode="text"):
    """
    Read a file saved by ftsave, returns a list of tables
    """
    if mode == "text":
        return _ftsave_read_text(path)
    else:
        raise ValueError("mode not supported")

"""
audiosample
~~~~~~~~~~~

Implements two classes, Sample and TSample

a Sample containts the audio of a soundfile as a
numpy and it knows about its samplerate
It can also perform simple actions (fade-in/out,
cut, insert, reverse, normalize, etc)
on its own audio destructively or return a new Sample.
a TSample is a Sample with a time-tag, it has a starting
point which can be other than 0.
"""

from __future__ import division as _division, absolute_import, print_function

import numpy as np
import os
import tempfile
import subprocess
import bpf4 as _bpf
import pysndfile
import logging
import sys
import fnmatch as _fnmatch

from emlib.snd import sndfiletools
from emlib.snd.resample import resample as _resample
from emlib.pitchtools import amp2db, db2amp
from emlib.conftools import ConfigDict
from emlib.snd.sndfile import sndread
from emlib import numpytools
from emlib import lib
from emlib import typehints as t


logger = logging.getLogger("emlib:audiosample")


def _configCheck(config, key, oldvalue, newvalue):
    if key == 'editor':
        if lib.binary_exists(newvalue):
            return None
        logger.error("Setting editor to {newvalue}, but it could not be found!")
        return oldvalue

config = ConfigDict(
    name='emlib:audiosample',
    default={
        'editor': '/usr/bin/audacity',
        'fade.shape': 'linear'
    },
    precallback=_configCheck
)


def _increase_suffix(filename:str) -> str:
    name, ext = os.path.splitext(filename)
    tokens = name.split("-")
    newname = None
    if len(tokens) > 1:
        suffix = tokens[-1]
        try:
            suffixnum = int(suffix)
            newname = "{}-{}".format(name[:-len(suffix)], suffixnum+1)
        except ValueError:
            pass
    if newname is None:
        newname = name + '-1'
    return newname + ext


def _arrays_match_length(a, b, mode='longer', pad=0):
    # type: (np.ndarray, np.ndarray, str, int) -> t.Tup[np.ndarray, np.ndarray]
    """
    match the lengths of arrays a and b

    if mode is 'longer', then the shorter array is padded with 'pad'
    if mode is 'shorter', the longer array is truncated
    """
    assert isinstance(a, np.ndarray), ("a: Expected Array, got "+str(a.__class__))
    assert isinstance(b, np.ndarray), ("b: Expected Array, got "+str(b.__class__))
    lena = len(a)
    lenb = len(b)
    if lena == lenb:
        return a, b
    if mode == 'longer':
        maxlen = max(lena, lenb)
        if pad == 0:
            func = np.zeros_like
        else:
            func = lambda arr:np.ones_like(arr)*pad
        if lena < maxlen:
            tmp = func(b)
            tmp[:lena] = a
            return tmp, b
        else:
            tmp = func(a)
            tmp[:lenb] = b
            return a, tmp
    elif mode == 'shorter':
        minlen = min(lena, lenb)
        if lena > minlen:
            return a[:minlen], b
        else:
            return a, b[:minlen]

    else:
        raise ValueError("Mode not understood"
                         "It must be either 'shorter' or 'longer'")


def split_channels(sndfile, labels=None, basename=None):
    # type: (str, t.Opt[t.List[str]], t.Opt[str]) -> t.List[Sample]
    """
    split a multichannel sound-file and name the individual files
    with a suffix specified by labels

    sndfile: the path to a soundfile
    labels: a list of labels (strings)
    basename: if given, used as the base name for the new files

    """
    if isinstance(sndfile, str):
        if basename is None:
            basename = sndfile
    else:
        raise TypeError("sndfile must be the path to a soundfile")
    s = Sample(sndfile)
    if labels is None:
        labels = ["%0d" % (ch + 1) for ch in range(s.channels)]
    assert s.channels == len(labels)
    base, ext = os.path.splitext(basename)
    sndfiles = []
    for ch, label in enumerate(labels):
        filename = "%s-%s%s" % (base, label, ext)
        s_ch = s.get_channel(ch)
        s_ch.write(filename)
        sndfiles.append(s_ch)
    return sndfiles


def _normalize_path(path):
    path = os.path.expanduser(path)
    return os.path.abspath(path)


def open_in_editor(filename, wait=False, app=None):
    # type: (str, bool) -> None
    editor = app or config['editor']
    filename = _normalize_path(filename)
    if sys.platform == 'darwin':
        os.system(f'open -a "{editor}" "{os.path.abspath(filename)}"')
    elif sys.platform == 'linux':
        proc = subprocess.Popen(args=[editor, filename])
        if wait:
            logger.debug("open_in_editor: waiting until finished")
            proc.wait()
    else:
        logger.debug("open_in_editor: using windows routine")
        proc = subprocess.Popen(f'"{editor}" "{filename}"', shell=True)
        if wait:
            proc.wait()


def select_audio_device(device):
    """
    Select the output device for playback

    * See list_devices for a list of devices

    device: the index in the list of devices, or the name of the device

  
    Example
    ~~~~~~~

    >>> list_audio_devices()

      0 HDA Intel PCH: CX20590 Analog (hw:0,0), ALSA (2 in, 0 out)
      1 HDA Intel PCH: HDMI 0 (hw:0,3), ALSA (0 in, 8 out)
      2 sysdefault, ALSA (128 in, 128 out)
      3 hdmi, ALSA (0 in, 8 out)
      4 pulse, ALSA (32 in, 32 out)
      5 dmix, ALSA (0 in, 2 out)
    * 6 default, ALSA (32 in, 32 out)
      7 system, JACK Audio Connection Kit (6 in, 6 out)
      8 PulseAudio JACK Sink, JACK Audio Connection Kit (2 in, 0 out)
      9 PulseAudio JACK Source, JACK Audio Connection Kit (0 in, 2 out

    # select device 'system'
    >>> select_audio_device(7)
    """
    import sounddevice as sd 
    sd.default.device = device 


def list_audio_devices():
    """
    Print a list of possible output sound devices, which can be
    selected via select_device
    """
    import sounddevice as sd 
    return sd.query_devices()


def find_audio_device(glob_pattern):
    import sounddevice as sd
    devs = sd.query_devices()
    for i, dev in enumerate(devs):
        if _fnmatch.fnmatch(dev['name'], glob_pattern):
            return i
    return None


def stop():
    """
    Stop all playing sounds
    """
    import sounddevice as sd
    sd.stop()


class Sample(object):

    def __init__(self, sound, samplerate=None, start=0, end=0):
        # type: (t.U[str, np.ndarray], t.Opt[int]) -> None
        """
        sound: str or np.array
            either sample data or a path to a soundfile
        start, end: sec
            if a path is given, it is possible to read a fragment of the data
            end can be negative, in which case it starts counting from the end
        """
        if isinstance(sound, str):
            sndfile = sound
            tmp = Sample.read(sndfile, start=start, end=end)
            self.samples = tmp.samples
            self.samplerate = tmp.samplerate
        elif isinstance(sound, np.ndarray):
            assert samplerate is not None
            self.samples = sound
            self.samplerate = samplerate
        else:
            raise TypeError(
                "sound should be a path to a sndfile or a seq. of samples")
        self.channels = numchannels(self.samples)  # type: int
        self._asbpf = None                    # type: t.Opt[_bpf.BpfInterface]
        self._sdplayers = []

    @property
    def nframes(self):
        # type: () -> int
        return self.samples.shape[0]

    def __repr__(self):
        s = "Sample: dur=%f sr=%d ch=%d" % (
            self.duration, self.samplerate, self.channels)
        return s

    @property
    def duration(self):
        # type: () -> float
        return len(self.samples) / self.samplerate

    @classmethod
    def read(cls, filename, start=0, end=0):
        # type: (str, float, float) -> 'Sample'
        samples, sr = sndread(filename, start=start, end=end)
        return cls(samples, samplerate=sr)
    
    def play(self, device=None, loop=False, chan=1):
        # type: (bool) -> None
        """
        Play the samples on the default sound device

        device: either the device number as returned by list_devices,
                or a string matching a device (via fnmatch)
                None will use the default device

        loop: if True, loop endlessly (use audiosample.stop() to stop it)
        chan: the channel to play to. For a stereo file, playback is always
              on successive channels
                
        To select the sound device, see: 
            * list_audio_devices
            * select_audio_device
        """
        import sounddevice as sd
        mapping = list(range(chan, chan + self.channels))
        if isinstance(device, str):
            device = find_audio_device(device)
            if not device:
                raise KeyError(f"Device {device} not found, see list_audio_devices()")
        return sd.play(self.samples, self.samplerate, loop=loop, device=device, mapping=mapping)
        
    def asbpf(self):
        # type: () -> _bpf.BpfInterface
        if self._asbpf not in (None, False):
            return self._asbpf
        else:
            self._asbpf = _bpf.Sampled(self.samples, 1 / self.samplerate)
            return self._asbpf

    def plot(self, profile='medium'):
        """
        plot the sample data

        profile: one of 'low', 'medium', 'high'
        """
        from . import plotting
        plotting.plot_samples(self.samples, self.samplerate, profile=profile)

    def plot_spectrograph(self, framesize=2048, window='hamming', at=0, dur=0):
        """
        window: As passed to scipy.signal.get_window
                `blackman`, `hamming`, `hann`, `bartlett`, `flattop`, `parzen`, `bohman`, 
                `blackmanharris`, `nuttall`, `barthann`, `kaiser` (needs beta), 
                `gaussian` (needs standard deviation)

        Plots the spectrograph of the entire sample (slice before to use only
        a fraction)

        See Also: .spectrum
        """
        from . import plotting
        if self.channels > 1:
            samples = self.samples[:,0]
        else:
            samples = self.samples
        s0 = 0 if at == 0 else int(at*self.samplerate)
        s1 = self.nframes if dur == 0 else min(self.nframes, int(dur*self.samplerate) - s0)
        if s0 > 0 or s1 != self.nframes:
            samples = samples[s0:s1]
        plotting.plot_power_spectrum(samples, self.samplerate, framesize=framesize, window=window)

    def plot_spectrogram(self, fftsize=2048, window='hamming', overlap=4, mindb=-120):
        """
        fftsize: the size of the fft
        window: window type. One of 'hamming', 'hanning', 'blackman', ... 
                (see scipy.signal.get_window)
        mindb: the min. amplitude to plot
        """
        from . import plotting
        if self.channels > 1:
            samples = self.samples[:,0]
        else:
            samples = self.samples
        return plotting.spectrogram(samples, self.samplerate, window=window, fftsize=fftsize,
                                    overlap=overlap, mindb=mindb)

    def open_in_editor(self, wait=True, app=None, format='wav') -> t.Opt['Sample']:
        """
        Open the sample in an external editor. The original
        is not changed.

        wait:
            if wait, the editor is opened in blocking mode, the results of the edit are returned as a new Sample
        app:
            if given, this application is used to open the sample.
            Otherwise, the application configured via the key 'editor' is used

        """
        assert format in {'wav', 'aiff', 'aif', 'flac'}
        sndfile = tempfile.mktemp(suffix="." + format)
        self.write(sndfile)
        logger.debug(f"open_in_editor: opening {sndfile}")
        open_in_editor(sndfile, wait=wait, app=app)
        if wait:
            return Sample.read(sndfile)
        return None

    def write(self, outfile:str, bits:int=None, **metadata) -> str:
        """
        write the samples to outfile

        outfile: the name of the soundfile. The extension
                 determines the file format
        bits: the number of bits. 32 bits and 64 bits are
              floats, if the format allows it.
              If None, the best resolution is chosen
        """
        ext = (os.path.splitext(outfile)[1][1:]).lower()
        if bits is None:
            if ext in ('wav', 'aif', 'aiff'):
                bits = 32
            elif ext == 'flac':
                bits = 24
            else:
                raise ValueError("extension should be wav, aif or flac")
        o = open_sndfile_to_write(outfile, channels=self.channels,
                                  samplerate=self.samplerate, bits=bits)
        o.write_frames(self.samples)
        if metadata:
            _modify_metadata(outfile, metadata)
        return outfile

    def copy(self):
        # type: () -> 'Sample'
        """
        return a copy of this Sample
        """
        return Sample(self.samples.copy(), self.samplerate)
        
    def __add__(self, other):
        # type: (t.U[float, 'Sample']) -> 'Sample'
        if isinstance(other, Sample):
            s0, s1, sr = broadcast_samplerate(self, other)
            assert isinstance(s0, np.ndarray)
            assert isinstance(s1, np.ndarray)
            s0, s1 = _arrays_match_length(s0, s1, mode='longer')
            return Sample(s0 + s1, samplerate=sr)
        else:
            # not a Sample
            return Sample(self.samples + other, self.samplerate)

    def __iadd__(self, other):
        # type: (t.U[float, 'Sample']) -> 'Sample'
        if isinstance(other, Sample):
            s0, s1, sr = broadcast_samplerate(self, other)
            s0, s1 = _arrays_match_length(s0, s1, mode='longer')
            s0 += s1
            self.samples = s0
            self.samplerate = sr
        else:
            self.samples += other
        return self

    def __mul__(self, other):
        # type: (t.U[float, 'Sample']) -> 'Sample'
        if isinstance(other, Sample):
            s0, s1, sr = broadcast_samplerate(self, other)
            s0, s1 = _arrays_match_length(s0, s1)
            return Sample(s0 * s1, sr)
        elif callable(other):
            other = _mapn_between(other, len(self.samples), 0, self.duration)
        return Sample(self.samples * other, self.samplerate)

    def __pow__(self, other:float) -> 'Sample':
        return Sample(self.samples ** other, self.samplerate)

    def __imul__(self, other):
        # type: (t.U[float, 'Sample']) -> 'Sample'
        if isinstance(other, Sample):
            s0, s1, sr = broadcast_samplerate(self, other)
            s0, s1 = _arrays_match_length(s0, s1)
            s0 *= s1
            self.samples = s0
            self.samplerate = sr
            return self
        elif callable(other):
            other = _mapn_between(other, len(self.samples), 0, self.duration)
        self.samples *= other
        return self

    def __len__(self):
        # type: () -> int
        return len(self.samples)

    def __getitem__(self, item):
        # type: (slice) -> 'Sample'
        """
        samples support slicing

        sample[start:stop] will return a new Sample consisting of a slice
        of this sample between the times start and stop. As it is a slice
        of this Sample, any changes inplace will be reflected in the original
        samples. To avoid this, use .copy:

        # Copy the fragment between seconds 1 and 3
        fragment = original[1:3].copy()

        To slice at the sample level, access .samples directly

        s = Sample("path")

        # This is a Sample
        sliced_by_time = s[fromtime:totime]

        # This is a numpy array
        sliced_by_samples = s.samples[fromsample:tosample]
        """
        if not isinstance(item, slice):
            raise ValueError("We only support sample[start:end]."
                             "To access individual samples, use sample.samples[index]")
        start, stop, step = item.start, item.stop, item.step
        if stop is None:
            stop = self.duration
        if start is None:
            start = 0
        if step is not None:
            return self.resample(step)[start:stop]
        stop = min(stop, self.duration)
        start = min(start, self.duration)
        assert 0 <= start <= stop
        frame0 = int(start * self.samplerate)
        frame1 = int(stop * self.samplerate)
        return Sample(self.samples[frame0:frame1], self.samplerate)

    def fade(self, fadetime, mode='inout', shape=None):
        if shape is None:
            shape = config['fade.shape']
        sndfiletools.fade_array(self.samples, self.samplerate, fadetime=fadetime,
                                mode=mode, shape=shape)
        return self

    def prepend_silence(self, dur):
        # type: (float) -> 'Sample'
        """
        Return a new Sample with silence of given dur at the beginning
        """
        silence = gen_silence(dur, self.channels, self.samplerate)
        return concat([silence, self])

    def normalize(self, headroom=0):
        # type: (float) -> 'Sample'
        """Normalize in place 
        
        headroom: maximum peak in dB
        """
        max_peak_possible = db2amp(headroom)
        peak = np.abs(self.samples).max()
        ratio = max_peak_possible / peak
        self.samples *= ratio
        return self

    def peak(self):
        # type: () -> float
        """return the highest sample value (dB)"""
        return amp2db(np.abs(self.samples).max())

    def peaksbpf(self, res=0.01):
        # type: (float) -> _bpf.Linear
        """
        Return a BPF representing the peaks envelope of the sndfile with the
        resolution given

        res: resolution in seconds
        """
        samplerate = self.samplerate
        chunksize = int(samplerate * res)
        X, Y = [], []
        data = self.samples if self.channels == 1 else self.samples[:, 0]
        data = np.abs(data)
        for pos in np.arange(0, self.nframes, chunksize):
            maximum = np.max(data[pos:pos+chunksize])
            X.append(pos/samplerate)
            Y.append(maximum)
        return _bpf.Linear.fromxy(X, Y)

    def reverse(self):
        # type: () -> 'Sample'
        """ reverse the sample in-place """
        self.samples[:] = self.samples[-1::-1]
        return self

    def rmsbpf(self, dt=0.005):
        # type: (float) -> _bpf.Sampled
        """ 
        Return a bpf representing the rms of this sample as a function of time
        """
        s = self.samples
        period = int(self.samplerate * dt + 0.5)
        dt = period / self.samplerate
        periods = int(len(s) / period)
        values = []  # type: t.List[float]
        _rms = rms
        for i in range(periods):
            chunk = s[i * period:(i + 1) * period]
            values.append(_rms(chunk))
        return _bpf.Sampled(values, x0=0, dx=dt)

    def mono(self):
        # type: () -> 'Sample'
        """
        Return a new Sample with this sample downmixed to mono 
        Returns self if already mono
        """
        if self.channels == 1:
            return self
        return Sample(mono(self.samples), samplerate=self.samplerate)

    def remove_silence_left(self, threshold=-120.0, margin=0.01, window=0.02):
        # type: (float, float, float) -> 'Sample'
        """
        See remove_silence
        """
        period = int(window * self.samplerate)
        first_sound_sample = first_sound(self.samples, threshold, period)
        if first_sound is not None:
            time = max(first_sound_sample / self.samplerate - margin, 0)
            return self[time:]
        return self

    def remove_silence_right(self, threshold=-120.0, margin=0.01, window=0.02):
        # type: (float, float, float) -> 'Sample'
        """
        See remove_silence
        """
        period = int(window * self.samplerate)
        lastsample = last_sound(self.samples, threshold, period)
        if first_sound is not None:
            time = min(lastsample / self.samplerate + margin, self.duration)
            return self[:time]
        return self

    def remove_silence(self, threshold=-120.0, margin=0.01, window=0.02):
        # type: (float, float, float) -> 'Sample'
        """
        Remove silence from the sides. Returns a new Sample

        threshold: dynamic of silence, in dB
        margin: leave at list this amount of time between the first/last sample
                and the beginning of silence or
        window: the duration of the analysis window, in seconds
        """
        out = self.remove_silence_left(threshold, margin, window)
        out = out.remove_silence_right(threshold, margin, window)

    def resample(self, samplerate:int) -> 'Sample':
        """
        Return a new Sample with the given samplerate
        """
        if samplerate == self.samplerate:
            return self
        samples = _resample(self.samples, self.samplerate, samplerate)
        return Sample(samples, samplerate=samplerate)

    def scrub(self, bpf):
        # type: (_bpf.BpfInterface, bool) -> 'Sample'
        """
        bpf: a bpf mapping time -> time
        
        Example 1: Read sample at half speed

        dur = sample1.duration
        sample2 = sample1.scrub(bpf.linear((0, 0), 
                                           (dur*2, dur)
                                           ))
        """
        samples, sr = sndfiletools.scrub((self.samples, self.samplerate), bpf, 
                                         rewind=False)
        return Sample(samples, self.samplerate)

    def get_channel(self, n):
        # type: (int) -> 'Sample'
        """
        return a new mono Sample with the given channel
        """
        if self.channels == 1 and n == 0:
            return self
        if n > (self.channels - 1):
            raise ValueError("this sample has only %d channel(s)!"
                             % self.channels)
        newsamples = self.samples[:, n]
        return Sample(newsamples, self.samplerate)

    def estimate_freq(self, start=0.2, dur=0.15, strategy='autocorr'):
        # type: (float, float, str) -> float
        """
        estimate the frequency of the sample (in Hz)

        start: where to start the analysis (the beginning
               of a sample is often not very clear)
        dur: duration of the fragment to analyze
        strategy: one of 'autocorr' or 'fft'
        """
        t0 = start
        t1 = min(self.duration, t0 + dur)
        s = self.get_channel(0)[t0:t1]
        from .freqestimate import freq_from_autocorr, freq_from_fft
        func = {
            'autocorr': freq_from_autocorr,
            'fft': freq_from_fft
        }.get(strategy, freq_from_autocorr)
        return func(s.samples, s.samplerate)

    def spectrum(self, resolution=30, **kws):
        import sndtrck
        return sndtrck.analyze_samples(self.samples, self.samplerate, 
                                       resolution=resolution, **kws)

    def chord_at(self, t:float, resolution=30, **kws):
        margin = 0.15
        t0 = max(0, t-margin)
        t1 = min(self.duration, t+margin)
        import sndtrck
        s = sndtrck.analyze_samples(self[t0:t1].samples, self.samplerate, 
                                    resolution=resolution, hop=2)
        chord = s.chord_at(t - t0)
        return chord

    def chunks(self, chunksize, hop=None, pad=False):
        """
        Iterate over the samples in chunks of chunksize. If pad is True,
        the last chunk will be zeropadded, if necessary 
        """
        return numpytools.chunks(self.samples, chunksize=chunksize, hop=hop, padwith=(0 if pad else None))


def first_sound(samples, threshold=-120.0, period=256, hopratio=0.5):
    # type: (np.ndarray, float, int, float) -> int
    """
    Find the first sample in samples whith a rms
    exceeding the given threshold

    Returns: time of the first sample holding sound or -1 if
             no sound found
    """
    threshold_amp = db2amp(threshold)
    hopsamples = int(period * hopratio)
    numperiods = int((len(samples) - period) / hopratio)
    for i in range(numperiods):
        i0 = i * hopsamples
        i1 = i0 + period
        chunk = samples[i0:i1]
        rms_now = rms(chunk)
        if rms_now > threshold_amp:
            return i0
    return -1


def last_sound(samples, threshold=-120.0, period=256, hopratio=1.0):
    # type: (np.ndarray, float, int, float) -> int
    """
    Find the end of the last sound in the samples.
    (the last time where the rms is lower than the given threshold)
    
    Returns -1 if no sound is found
    """
    samples1 = samples[::-1]
    i = first_sound(samples1, threshold=threshold,
                    period=period, hopratio=hopratio)
    if i < 0:
        return i
    return len(samples) - (i + period)


def rms(arr):
    # type: (np.ndarray) -> float
    """
    calculate the root-mean-square of the arr
    """
    return ((abs(arr) ** 2) / len(arr)).sum()


def broadcast_samplerate(a, b):
    # type: (Sample, Sample) -> t.Tup[Sample, Sample, int]
    """
    Match the samplerates of audio samples a and b to the highest one
    the audio sample with the lowest samplerate is resampled to the
    higher one.

    a, b: Sample instances

    Returns: (resampled_a, resampled_b, new_samplerate)
    """
    assert isinstance(a, Sample)
    assert isinstance(b, Sample)
    if a.samplerate == b.samplerate:
        return a.samples, b.samples, a.samplerate
    sr = max(a.samplerate, b.samplerate)
    if sr == a.samplerate:
        samples0, samples1 = a.samples, _resample(b.samples, b.samplerate, sr)
    else:
        samples0, samples1 = _resample(a.samples, a.samplerate, sr), b.samples
    assert isinstance(samples0, np.ndarray)
    assert isinstance(samples1, np.ndarray)
    return samples0, samples1, sr


def numchannels(samples):
    # type: (np.ndarray) -> int
    """
    Returns the number of channels held by the `samples` array
    """
    assert isinstance(samples, np.ndarray)
    return 1 if len(samples.shape) == 1 else samples.shape[1]
    

def _as_numpy_samples(samples):
    # type: (t.U[Sample, np.ndarray]) -> np.ndarray
    if isinstance(samples, Sample):
        return samples.samples
    elif isinstance(samples, np.ndarray):
        return samples
    else:
        return np.asarray(samples, dtype=float)


def as_sample(source):
    # type: (t.U[str, Sample, t.Tup[np.ndarray, int]]) -> Sample
    """
    return a Sample instance

    input can be a filename, a Sample or a tuple (samples, samplerate)
    """
    if isinstance(source, str):
        return Sample.read(source)
    if isinstance(source, Sample):
        return source
    if isinstance(source, tuple) and isinstance(source[0], np.ndarray):
        samples, sr = source
        return Sample(samples, sr)
    else:
        raise TypeError("can't convert source to Sample")


def mono(samples):
    # type: (t.U[Sample, np.ndarray]) -> np.ndarray
    """
    If samples are multichannel, it mixes down the samples
    to one channel.
    """
    samples = _as_numpy_samples(samples)
    channels = numchannels(samples)
    if channels == 1:
        return samples
    return np.sum(samples, axis=1)/channels


def concat(sampleseq):
    # type: (t.Seq[Sample]) -> Sample
    """
    sampleseq: a seq. of Samples

    concat the given Samples into one Sample.
    Samples should share samplingrate and numchannels 
    """
    s = np.concatenate([s.samples for s in sampleseq])
    return Sample(s, samplet.Seq[0].samplerate)


def _mapn_between(func, n, t0, t1):
    # type: (Callable, int, float, float) -> np.ndarray
    """
    Returns: a numpy array of n-size, mapping func between t0-t1
             at a rate of n/(t1-t0)
    """
    if hasattr(func, 'mapn_between'):
        ys = func.mapn_between(n, t0, t1)  # is it a Bpf?
    else:
        X = np.linspace(t0, t1, n)
        ufunc = np.vectorize(func)
        Y = ufunc(X)
        return Y
    return ys


def _modify_metadata(path, metadata):
    # type: (str, dict) -> None
    """
    possible keys:

    description     \
    originator       |
    orig-ref         |
    umid             | bext
    orig-date        |
    orig-time        |
    coding-hist     /

    title
    copyright
    artist
    comment
    date
    album
    license
    """
    possible_keys = {
        "description": "bext-description",
        "originator": "bext-originator",
        "orig-ref": "bext-orig-ref",
        "umid": "bext-umid",
        "orig-date": "bext-orig-time",
        "coding-hist": "bext-coding-hist",
        "title": "str-title",
        "copyright": "str-copyright",
        "artist": "str-artist",
        "comment": "str-comment",
        "date": "str-date",
        "album": "str-album"
    }
    args = []
    for key, value in metadata.items():
        key2 = possible_keys.get(key)
        if key2 is not None:
            args.append(' --%s "%s"' % (key2, str(value)))
    os.system('sndfile-metadata-set %s "%s"' % (" ".join(args), path))


def open_sndfile_to_write(filename, channels=1, samplerate=48000, bits=None):
    # type: (str, int, int, t.Opt[int]) -> pysndfile.PySndfile
    """
    The format is inferred from the extension (wav, aiff, flac, etc.)

    if bits is given, it is used. otherwise it is inferred from the format
    """
    encodings = {
        'wav': {
            16: "pcm16",
            24: "pcm24",
            32: "float32"
        },
        'aif': {
            16: "pcm16",
            24: "pcm24",
            32: "float32",
        },
        'flac': {
            16: "pcm16",
            24: "pcm24",
            32: "pcm24"
        }
    }
    base, ext = os.path.splitext(filename)
    ext = ext[1:4].lower()
    if not ext or ext not in encodings:
        raise ValueError("The extension of the file is not supported")
    
    encoding = encodings[ext].get(bits)
    if encoding is None:
        raise ValueError("no format possible for %s with %d bits" %
                         (ext, bits))
    fmt = pysndfile.construct_format(ext, encoding)
    return pysndfile.PySndfile(filename, 'w', format=fmt,
                               channels=channels, samplerate=samplerate)


def gen_silence(dur, channels, sr):
    # type: (float, int, int) -> Sample
    """
    Generate a silent Sample with the given characteristics
    """
    if channels == 1:
        samples = np.zeros((int(dur*sr),), dtype=float)
    else:
        samples = np.zeros((int(dur*sr), channels), dtype=float)
    return Sample(samples, sr)

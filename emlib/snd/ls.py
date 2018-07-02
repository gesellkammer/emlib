#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import sndfileio
from colorama import init as colorama_init, Fore, Style
import os
import sys


amp2db = None
maximum_peak = None
DEFAULT_OPTIONS = ('--size',)


def is_soundfile(f):
    name, ext = os.path.splitext(f)
    return ext.lower() in ('.wav', '.aiff', '.aif', '.flac', '.mp3', '.ogg')


def s2m(secs):
    m = int(secs / 60)
    s = (secs % 60) / 100
    return m + s


def init(args):
    colorama_init(autoreset=True)
    args = args[1:] if len(args) > 1 else []
    options = [f for f in args if f.startswith("-")]
    if not options:
        options = DEFAULT_OPTIONS
    files = [f for f in args if not f.startswith("-")]
    if len(files) == 0:
        import glob
        files = glob.glob("*")
    files = [f for f in files if is_soundfile(f)]
    if not files:
        sys.exit(0)
    files.sort()
    if '--help' in options or '-h' in options:
        print("""
    snd-ls [options] files

    --peak            : output the highest peak in dB
    --sortdur         : sort by descending duration
    """)
        sys.exit(0)

    if '--all' in options or '-a' in options:
        options = ('--peak', '--size')

    if '--peak' in options:
        from em.snd.sndfiletools import maximum_peak as _maximum_peak
        from em.pitchtools import amp2db as _amp2db
        global maximum_peak, amp2db
        maximum_peak = _maximum_peak
        amp2db = _amp2db
        calculate_peak = True
    else:
        calculate_peak = False

    calculate_size = True

    if '--sortdur' in options:
        files.sort(key=lambda f: get_dur(f))
    largest_filename = min(max(len(f) for f in files) + 2, 54)
    
    return files, largest_filename, calculate_size, calculate_peak


def get_dur(f):
    snd = audiolab.Sndfile(f)
    dur = snd.nframes / float(snd.samplerate)
    return dur


def get_info_str(f, largest_filename, calculate_size, calculate_peak=False):
    ext = os.path.splitext(f)[1]
    if ext == '.mp3':
        return (f, None)
    try:
        info = sndfileio.sndinfo(f)
        snd_channels = info.channels
        
        snd_sr = info.samplerate
        snd_nframes = snd.nframes
        name = os.path.split(f)[-1].ljust(largest_filename)[:largest_filename]
        name = name.decode('utf-8', 'ignore')
        name = Style.BRIGHT + name
        ch_sr = Fore.BLUE + "{ch}/{sr}".format(ch=snd_channels, sr=snd_sr)
        mins = s2m(float(snd_nframes) / snd_sr)
        secs = (mins - int(mins)) * 100
        ms = int((secs - int(secs)) * 1000)
        secs = int(secs)
        mins = int(mins)
        dur = "%02d:%02d.%03d" % (mins, secs, ms)
        dur = Fore.RED + str(dur)
        encoding = Fore.GREEN + str(snd.encoding).replace("float32", "flt32").ljust(6)
        # encoding = Fore.BLUE + str(snd.encoding).ljust(7)
        extras = []
        if calculate_size:
            size_str = ("%.1fM" % (os.stat(f).st_size / 1000000.)).rjust(5)
            size_in_megabytes = Fore.CYAN + (size_str)
            extras.append(size_in_megabytes)
        if calculate_peak:
            peak_str = ("%.1fdB" % (amp2db(maximum_peak(f)))).rjust(6)
            peak = Fore.MAGENTA + peak_str
            extras.append(peak)
        extras_string = " ".join(extras)
        s = "  ".join((name, ch_sr, dur, encoding, extras_string))
    except ValueError:
        name = os.path.split(f)[-1].ljust(largest_filename)
        name = Style.BRIGHT + name.decode('utf-8', 'ignore')
        s = "%s   Format not supported" % (name)
    return (f, s)


def _get_info_str(xxx_todo_changeme):
    (f, largest_filename, calc_size, calc_peak) = xxx_todo_changeme
    return get_info_str(f, largest_filename, calc_size, calc_peak)


def do_it(files, largest_filename, calculate_size, calculate_peak):
    for f in files:
        if os.path.splitext(f)[1].lower() == '.mp3':
            pass  # mp3 still not supported
        else:
            print(get_info_str(f, largest_filename, calculate_size, calculate_peak)[1])


def ls(args):
    files, largest_filename, calculate_size, calculate_peak = init(args)
    # do_it_conc(files, largest_filename, calculate_size, calculate_peak)
    # print(largest_filename)
    do_it(files, largest_filename, calculate_size, calculate_peak)


if __name__ == '__main__':
    ls(sys.argv)

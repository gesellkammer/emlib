from collections import namedtuple
import math
import midi
from .util import cents2pitchbend
from emlib.lib import snap_to_grid
import warnings


def generate_chrom_scale(midinote0, midinote1, time0, time1, curve='linear', 
                         velocity=100, outfile='scale.mid'):
    """
    generate a chromatic scale from midinote0 to midinoe1 between the given times

    midinote0: the initial note
    midinote1: the final note included
    """
    import numpy as np
    from bpf4 import bpf
    notecurve = bpf.util.get_bpf_constructor(curve)(time0, midinote0, time1, midinote1).floor()
    times = np.arange(time0, time1, (time1 - time0) / abs(midinote1 - midinote0))
    notes = list(map(int, notecurve.map(times)))
    ends = list(times[1:])
    ends.append(time1)
    velocities = list(map(int, bpf.asbpf(velocity).map(times)))
    track = midi.Track()
    R = 1920  # resolution in ticks

    def time2tick(time):
        return int(time * R + 0.5)

    for note, time, end, vel in zip(notes, times, ends, velocities):
        track.append(midi.NoteOnEvent(tick=time2tick(time), pitch=note, velocity=vel))
        track.append(midi.NoteOffEvent(tick=time2tick(end), pitch=note))
    track.append(midi.EndOfTrackEvent(tick=time2tick(ends[-1])))
    track.make_ticks_rel()
    midifile = midi.Pattern([track], resolution=R)
    midi.write_midifile(outfile, midifile)
    return notes, times


class _Note(namedtuple("Note", "midinote start dur velocity")):
    def __new__(cls, midinote, start, dur, velocity=90):
        return super(_Note, cls).__new__(cls, midinote, start, dur, velocity)


class _Track(list):
    def remove_overlap(self):
        pass


class Score(object):
    def __init__(self, resolution=480, tempo=60):
        self.resolution = resolution
        self.tempo = tempo
        self.tracks = {}

    def addnote(self, midinote, start, dur, velocity=90, trackid=None):
        """
        midinote: normally an integer. If a float with some fractional part is used
                  then it is possible to write this as a pitchwheel alteration
                  (see .write)
        start: start time in seconds
        dur: duration in seconds
        """
        if trackid is None:
            trackid = self.default_trackid()
        if trackid not in self.tracks:
            self.tracks[trackid] = _Track()
        track = self.tracks[trackid]
        track.append(_Note(midinote, start, dur, velocity))

    def default_trackid(self):
        if not self.tracks:
            return 0
        return min(self.tracks.keys())

    def sort(self):
        for trackid, track in self.tracks.items():
            track.sort(key=lambda n:n.start)
        
    def dump(self):
        self.sort()
        trackids = sorted(self.tracks.keys())
        for trackid in trackids:
            print(("Track: {0}".format(trackid)))
            track = self.tracks[trackid]
            for note in track:
                print(("   " + str(note)))

    def time2tick(self, t):
        return int(t * self.resolution + 0.5)

    def get_overlaps(self):
        raise NotImplementedError
        overlaps = []
        for trackid, track in self.tracks.items():
            pass

    def snap_to_grid(self, dt=0.125, offset=0):
        warnings.warn("Deprecated. Use quantize_time")
        return self.quantize_time(dt=dt, offset=offset)
        
    def quantize_time(self, dt=0.125, offset=0):
        """
        Snap all times (starts, ends, durations) to a time grid defined
        by t=offset+k*dt
        """
        tracks = {}
        for trackid, track in self.tracks.items():
            newtrack = []
            for note in track:
                start = snap_to_grid(note.start, dt, offset=offset)
                end = snap_to_grid(note.start + note.dur, dt, offset=offset)
                newtrack.append(note._replace(start=start, dur=end-start))
            tracks[trackid] = newtrack
        self.tracks = tracks
        return self

    def quantize_pitch(self, quant=0.5):
        tracks = {}
        for trackid, track in self.tracks.items():
            newtrack = []
            for note in track:
                midinote = snap_to_grid(note.midinote, quant)
                newtrack.append(note._replace(midinote=midinote))
            tracks[trackid] = newtrack
        self.tracks = tracks
        return self
        
    def asmusic21(self):
        import music21
        sc = music21.stream.Score()
        now = 0
        for trackid in sorted(self.tracks.keys()):
            notes = self.tracks[trackid]
            voice = music21.stream.Voice()
            for note in notes:
                if note.start > now:
                    voice.append(music21.note.Rest(duration=music21.duration.Duration(note.start-now)))
                voice.append(music21.note.Note(note.midinote, duration=music21.duration.Duration(note.dur)))
                now = note.start + note.dur
                # voice.insert(note.start, music21.note.Note(note.midinote, duration=music21.duration.Duration(note.dur)))
            #gaps = voice.findGaps()
            #for gap in gaps:
            #    if gap.duration.quarterLength > 0.001:
            #        print("insering gap: {start} {dur}".format(start=gap.offset, dur=gap.duration))
            #        voice.insert(gap.offset, music21.note.Rest(duration=gap.duration))
            voice.sliceByBeat(inPlace=True)
            maxoffset = 1 + int(voice.duration.quarterLength)
            voice.sliceAtOffsets(list(range(maxoffset)), inPlace=True)
            sc.append(voice)
        sc.voices[0].insert(0, music21.tempo.MetronomeMark(number=60))
        return sc

    def write_musicxml(self, outfile):
        score = self.asmusic21()
        score.write("musicxml", outfile)
        return score
        
    def write(self, outfile, microtonal=True):
        self.sort()
        miditracks = []
        for trackid in sorted(self.tracks.keys()):
            miditrack = midi.Track()
            notes = self.tracks[trackid]
            events = []
            lastpitchbend = 0
            for note in notes:
                start = self.time2tick(note.start)
                end = self.time2tick(note.start + note.dur)
                # an event is (time, priority, midi.event)
                frac, integer = math.modf(note.midinote)
                if microtonal:
                    pitchbend = cents2pitchbend(int(frac * 100)) - 8192
                    midinote = int(integer)
                    if pitchbend != lastpitchbend:
                        events.append((start, 2, midi.PitchWheelEvent(tick=-1, pitch=pitchbend)))
                        lastpitchbend = pitchbend
                        start += 1
                else:
                    midinote = int(note.midinote)
                events.append((start, 10, midi.NoteOnEvent(tick=-1, pitch=midinote, velocity=note.velocity)))
                events.append((end, 0, midi.NoteOffEvent(tick=-1, pitch=midinote, velocity=0)))
            events.sort()
            ticknow = 0
            for tick, priority, event in events:
                event.tick = tick - ticknow
                miditrack.append(event)
                ticknow = tick
            trackend = max(note.start + note.dur for note in notes)
            miditrack.append(midi.EndOfTrackEvent(tick=self.time2tick(trackend)))
            # miditrack.make_ticks_rel()
            miditracks.append(miditrack)
        midifile = midi.Pattern(miditracks, resolution=self.resolution)
        midi.write_midifile(outfile, midifile)
        return events

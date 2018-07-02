from emlib.iterlib import pairwise
from fractions import Fraction
import typing as t

"""
Implements a general routine for packing a seq. of
items (notes, partials, etc) into a series of tracks.

A Track is a seq. of non-overlapping items

In order to be attached to a Track, each object (note, partial, etc.)
must be wrapped inside an Item, defining an offset, duration and step
"""

def _overlap(x0, x1, y0, y1):
    if x0 < y0:
        return x1 > y0
    return y1 > x0


class Item(t.NamedTuple):
    obj: t.Any
    offset: Fraction
    dur: Fraction
    step: float

    @property
    def end(self):
        return self.offset + self.dur


class Track(list):

    def append(self, item: Item) -> None:
        super().append(item)

    def __getitem__(self, idx:int) -> Item:
        out = super().__getitem__(idx)
        assert isinstance(out, Item)
        return out

    def __iter__(self) -> t.Iterator[Item]:
        return super().__iter__()


def pack_in_tracks(items: t.Iterator[Item], maxrange=36) -> t.List[Track]:
    """
    items: a seq. of Items
    maxrange: the maximum pitch range of a track, in semitones
    """
    tracks: t.List[Track] = []
    items2: t.List[Item] = sorted(items, key=lambda itm: itm.offset)
    for item in items2:
        track = _best_track(tracks, item, maxrange=maxrange)
        if track is None:
            track = Track()
            tracks.append(track)
        track.append(item)
    assert all(_checktrack(track) for track in tracks)
    return tracks


def dumptrack(track: Track) -> None:
    print("--------")
    for item in track:
        print(f"{float(item.offset):.4f} - {float(item.end):.4f} {item.step}")


def dumptracks(tracks: t.List[Track]):
    for track in tracks:
        dumptrack(track)

# ------------------------------------------------------------------------


def _track_getrange(track):
    if not track:
        return None, None
    note0 = 99999999999
    note1 = 0
    for item in track:
        step = item.step
        if step < note0:
            note0 = step
        elif step > note1:
            note1 = step
    return note0, note1


def _best_track(tracks: t.List[Track], item: Item, maxrange: int):
    """
    tracks: list of tracks
    node: node to fit
    trackrange: the maximum range a track can have
    """
    possibletracks = [track for track in tracks if _fits_in_track(track, item, maxrange=maxrange)]
    if not possibletracks:
        return None
    results = [(_rate_fit(track, item), track) for track in possibletracks]
    results.sort()
    rating, track = results[0]
    return track


def _fits_in_track(track: Track, item: Item, maxrange: int):
    if len(track) == 0:
        return True
    for packednode in track:
        if _overlap(packednode.offset, packednode.end, item.offset, item.end):
            return False
    tracknote0, tracknote1 = _track_getrange(track)
    step = item.step
    n0 = min(tracknote0, step)
    n1 = max(tracknote1, step)
    if n1 - n0 < maxrange:
        return True
    return False


def _rate_fit(track: Track, item: Item):
    """
    Return a value representing how goog this item fits in track
    Assumes that it fits both horizontally and vertically.
    The lower the value, the best the fit
    """
    assert isinstance(track, list)
    assert isinstance(item, Item)
    if len(track) == 0:
        t1 = 0
    else:
        t1 = track[-1].end
    assert t1 <= item.offset
    rating = item.offset - t1
    return rating


def _checktrack(track: Track) -> bool:
    if not track:
        return True
    for item0, item1 in pairwise(track):
        if item0.end > item1.offset:
            return False
    return True



"""
Utilities to read regions and markers from REAPER's .RPP files
"""

from collections import namedtuple
from .. import lib as _lib

Region = namedtuple("Region", "start end id label")
Marker = namedtuple("Marker", "start end id label")


def get_regions(rpp_file):
    """
    given a REAPER .rpp, extract the regions as a list of Region instances
    (start, end, id, label)
    """
    f = open(rpp_file)
    regions = []

    def is_region(line):
        # expects that line has been stripped
        return line.startswith("MARKER") and line[-1] == '1'

    def parse_region(line):
        words = line.split()
        MARKER = words[0]
        id = int(words[1])
        time = float(words[2])
        track = int(words[-1])
        label = " ".join(words[3:-1])
        label = label.replace('"', '')
        return id, time, label
    # skip until we find markers
    region_started = False
    for line in f:
        line = _lib.unicode_force(line).strip()
        if is_region(line):
            id, start, label = parse_region(line)
            region_started = True
            break
    for line in f:
        line = _lib.unicode_force(line).strip()
        if not is_region(line):
            break
        id, time, label = parse_region(line)
        if region_started:
            end = time
            regions.append(Region(start, end, id, label))
            region_started = False
        else:
            region_started = True
            start = time
    return regions


def get_markers(rpp_file):
    """
    given a REAPER .rpp, extract the markers as a list of Markers
    (start, end, id, label)
    """
    f = open(rpp_file)
    regions = []

    def is_marker(line):
        # expects that line has been stripped
        return line.startswith("MARKER") and line[-1] == '0'

    def parse_marker(line):
        words = line.split()
        MARKER = words[0]
        id = int(words[1])
        time = float(words[2])
        kind = int(words[-1])
        label = " ".join(words[3:-1])
        label = label.replace('"', '')
        return id, time, label

    markers = []
    for line in f:
        line = _lib.unicode_force(line).strip()
        if is_marker(line):
            id, start, label = parse_marker(line)
            markers.append(Marker(start, start, id, label))
    return markers


def get_markers_and_regions(rpp_file):
    """
    get all markers and regions in .RPP file as a flat list
    """
    markers = get_markers(rpp_file)
    regions = get_regions(rpp_file)
    out = markers + regions
    out.sort(key=lambda x:x.start)
    return out


def write_markers(csvfile, markers):
    """
    markers: a list of (name, start, [end])
             Each marker/region is a tuple of two or three elements
             if an end is given, a Region is created

    start and end can be either a floating point, in which case
    they are interpreted as absolute time, or a string of the type
    "MM.BB.XXX" with MM=measure, BB=beat, XXX=subdivision    

    Reaper exchanges markers and regions as .csv with format

    id, Name, Start, End, Length, Color

    id: M1, M2, R1, etc, donde Mxx identifica un marker y Rxx identifica
        una region
    """
    import csv
    markerid = 1
    regionid = 1
    rows = []
    for marker in markers:
        if len(marker) == 2:
            name, start = marker
            id = "M" + str(markerid)
            markerid += 1
            end = ""
            length = ""
            color = ""
        elif len(marker) == 3:
            name, start, end = marker
            length = ""
            color = ""
            id = "R" + str(regionid)
            regionid += 1
        row = (id, name, start, end, length, color)
        rows.append(row)
    with open(csvfile, "w") as fileobj:
        writer = csv.writer(fileobj)
        writer.writerow("# Name Start End Length Color".split())
        for row in rows:
            writer.writerow(row)

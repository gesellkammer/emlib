""" video routines based on ffmpeg and moviepy """

from emlib.pitchtools import db2amp
from emlib.lib import add_suffix
import numpy as np
import moviepy.editor
import moviepy


def moviepy_normalize(clip, peakdb=0.0):
    """
    clip: a moviepy clip
    peakdb: db value for highest peak

    Returns: a moviepy clip
    """
    samples = clip.audio.to_soundarray()
    maxamp = np.abs(samples).max()
    peakamp = db2amp(peakdb)
    ratio = peakamp / maxamp
    return clip.volumex(ratio)


def normalize_audio(videopath, peakdb=0, outfile=None):
    clip = moviepy.editor.VideoFileClip(videopath)
    clip2 = moviepy_normalize(clip, peakdb)
    outfile = outfile or add_suffix(videopath, "-norm")
    clip2.write_video(outfile)
    return outfile

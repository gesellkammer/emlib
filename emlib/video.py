""" video routines based on ffmpeg and moviepy """

from .pitch import db2amp
from .lib import add_suffix
import numpy as np


def moviepy_normalize(clip, peakdb=0):
    """
    clip: a moviepy clip
    """
    samples = clip.audio.to_soundarray()
    maxamp = np.abs(samples).max()
    peakamp = db2amp(peakdb)
    ratio = peakamp / maxamp
    return clip.volumex(ratio)


def normalize_audio(videopath, peakdb=0, outfile=None):
    import moviepy.editor as e
    clip = e.VideoFileClip(videopath)
    clip2 = moviepy_normalize(clip, peakdb)
    outfile = outfile or add_suffix(videopath, "-norm")
    clip2.write_video(outfile)
    return outfile
""" video routines based on ffmpeg and moviepy """

from pitchtools import db2amp
from emlib.filetools import addSuffix
import numpy as np

try:
    import moviepy.editor
    import moviepy
except ImportError:
    raise ImportError("This module needs moviepy")


def normalizeClip(clip, peakdb=0.0):
    """
    Normalize a moviepy  clip

    Args:
        clip: a moviepy clip
        peakdb: db value for highest peak

    Returns:
        a moviepy clip
    """
    samples = clip.audio.to_soundarray()
    maxamp = np.abs(samples).max()
    peakamp = db2amp(peakdb)
    ratio = peakamp / maxamp
    return clip.volumex(ratio)


def normalizeAudio(videopath: str, peakdb=0., outfile:str=None) -> str:
    """
    Normalize the audio of a movie file

    Args:
        videopath: the path to the video file
        peakdb: the peak to normalize to
        outfile: the output  path of the video file with the audio normalized

    Returns:
        the path of the output video file. If no output file was given, this will
        be a file in the same folder as *videopath* with an added suffix
    """
    clip = moviepy.editor.VideoFileClip(videopath)
    clip2 = normalizeClip(clip, peakdb)
    outfile = outfile or addSuffix(videopath, "-norm")
    clip2.write_video(outfile)
    return outfile

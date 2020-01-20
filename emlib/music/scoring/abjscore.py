from . import core
import os

import abjad as abj
import abjadtools as abjtools

from emlib import lib
import emlib.typehints as t


def _openPdf(pdf:str) -> None:
    pdf = os.path.abspath(os.path.expanduser(pdf))
    lib.open_with_standard_app(pdf)


class AbjScore(core.AbstractScore):
    """
    Not to be created directly
    """

    def __init__(self, abjscore,
                 pageSize: str=None,
                 orientation: str=None,
                 staffSize: int=None,
                 includeBranch: bool=None,
                 eventTracks=None) -> None:
        super().__init__(abjscore, pageSize=pageSize, orientation=orientation, staffSize=staffSize,
                         includeBranch=includeBranch, eventTracks=eventTracks)

    def show(self):
        return abj.show(self.score)

    def dump(self):
        return abj.f(self.score)

    def _toLily(self) -> abj.lilypondfile.LilyPondFile:
        lilyfile = abjtools.scoreToLily(score=self.score, pageSize=self.pageSize,
                                        orientation=self.orientation, staffSize=self.staffSize)
        return lilyfile

    def _saveLily(self, outfile:str=None) -> str:
        outfile = abjtools.saveLily(self.score, outfile, pageSize=self.pageSize,
                                    orientation=self.orientation, staffSize=self.staffSize)
        return outfile

    def save(self, outfile:str) -> None:
        """
        Save this score.

        Supported formats are: lilypond (.ly), pdf (.pdf)
        """
        outfile = os.path.expanduser(outfile)
        base, ext = os.path.splitext(outfile)
        if ext == '.ly':
            self._saveLily(outfile)
        elif ext == '.pdf':
            lilyfile = self._saveLily(outfile)
            abjtools.lilytools.lily2pdf(lilyfile, outfile)
        else:
            raise ValueError(f"format {ext} not supported. Possible formats: .ly, .pdf")
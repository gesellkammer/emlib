import os
from emlib import lib
from emlib.music import m21tools
import emlib.typehints as t
from . import core


def _openPdf(pdf:str) -> None:
    pdf = os.path.abspath(os.path.expanduser(pdf))
    lib.open_with_standard_app(pdf)


class Music21Score(core.AbstractScore):
    """
    Not to be created directly
    """

    def __init__(self, m21score,
                 pageSize: str=None,
                 orientation: str=None,
                 staffSize: int=None,
                 includeBranch: bool=None,
                 tracks=None) -> None:
        super().__init__(m21score, pageSize=pageSize, orientation=orientation, staffSize=staffSize,
                         includeBranch=includeBranch)
        self._tracks = tracks

    def show(self, **options):
        return self.score.show(**options)

    def dump(self):
        return self.score.show('text')

    def save(self, outfile:str) -> None:
        """
        Save this score.

        Supported formats are: lilypond (.ly), pdf (.pdf), xml
        """
        outfile = os.path.expanduser(outfile)
        base, ext = os.path.splitext(outfile)
        if ext == '.ly':
            m21tools.saveLily(self.score, outfile)
        elif ext == '.pdf':
            return self.score.write('pdf', outfile)
        elif ext == '.xml':
            return self.score.write('xml', outfile)
        else:
            raise ValueError(f"format {ext} not supported. Possible formats: .ly, .pdf")

    def musicxml(self):
        return m21tools.getXml(self.score)

    def extractTracks(self) -> t.List[core.Track]:
        """
        This extracts back a list of tracks from the current score
        """
        raise NotImplemented()




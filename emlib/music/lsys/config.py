from emlib.conftools import ConfigDict

APPNAME = 'emlib:lsys'

config = ConfigDict(
    APPNAME,
    default = {
        'plot.node.linewidth': 4,
        'score.backend': 'abjad',
        'score.showbranch': True,
        'score.pagesize': 'a4',
        'score.orientation': 'portrait',
        'score.track.maxrange': 36,
        'score.gliss.allowsamenote': False,
        'score.showmeta': True
    },
    validator = {
        'score.orientation::choices': ('portrait', 'landscape'),
        'score.pagesize::choices': ('a3', 'a4', 'A3', 'A4'),
        'score.track.maxrange::range': (2, 100),
        'plot.node.linewidth::range': (1, 100),
        'score.backend::choices': ('abjad',)
    }
)
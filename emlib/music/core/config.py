from emlib import conftools
import logging
import sys


_defaultconfig = {
    'A4': 442,
    'defaultDuration': 1.0,
    'repr.showfreq': True,
    'chord.arpeggio': 'auto',
    'chord.adjustGain': True,
    'm21.displayhook.install': True,
    'm21.displayhook.format': 'xml.png',
    'm21.fixstream': True,
    'show.semitoneDivisions':2,
    'show.lastBreakpointDur':1/8,
    'show.cents': True,
    'show.centsMethod': 'lyric',
    'show.split': True,
    'show.gliss': True,
    'show.centSep': ',',
    'show.scaleFactor': 1.0,
    'show.format': 'xml.png',
    'show.external': False,
    'show.cacheImages': True,
    'show.seqDuration': 1,
    'show.defaultDuration': 1,
    'show.arpeggioDuration': 0.5,
    'show.label.fontSize': 12.0,
    'use_musicxml2ly': True,
    'app.png': 'feh --image-bg white' if sys.platform == 'linux' else '',
    'displayhook.install': True,
    'play.dur': 2.0,
    'play.gain': 0.5,
    'play.group': 'emlib.mus2',
    'play.instr': 'sine',
    'play.fade': 0.02,
    'play.numChannels': 2,
    'play.unschedFadeout': 0.05,
    'rec.block': False,
    'rec.gain': 1.0
}

_validator = {
    'show.semitoneDivisions::choices': [1, 2, 4],
    'm21.displayhook.format::choices': ['xml.png', 'lily.png'],
    'show.format::choices':
        ['xml.png', 'xml.pdf', 'lily.png', 'lily.pdf', 'repr'],
    'chord.arpeggio::choices': ['auto', True, False],
    'play.instr::choices': ['sine', 'piano', 'tri', 'clarinet'],
    'play.gain::range': (0, 1),
    'play.numChannels::type': int,
    'show.centsMethod::choices': ['lyric', 'expression']
}

_help = {
    'defaultDuration': 
        "Value used when a duration is needed and has not been set (Note, Chord). Not the same as play.dur",
    'repr.showfreq':
        "Show frequency when calling printing a Note in the console",
    'chord.arpeggio':
        "Arpeggiate notes of a chord when showing. In auto mode, only arpeggiate when needed",
    'chord.adjustGain':
        "Adjust the gain of a chord according to the number of notes, to prevent clipping",
    'show.external':
        "Force opening images with an external tool, even when inside a Jupyter notebook",
    'show.split':
        "Should a voice be split between two stafs. A midinumber can be given instead",
    'show.lastBreakpointDur':
        "Dur of a note representing the end of a line/gliss, which has no duration per se",
    'show.semitoneDivisions':
        "The number of divisions per semitone (2=quarter-tones, 4=eighth-tones)",
    'show.scaleFactor':
        "Affects the size of the generated image",
    'show.format':
        "Used when no explicit format is passed to .show",
    'show.gliss':
        "If true, show a glissando line where appropriate",
    'play.numChannels':
        "Default number of channels (channels can be set explicitely when calling startPlayEngine",
    'rec.block':
        "Default value when calling .rec (True=.rec will block until finished, otherwise recording is done async)",
    'use_musicxml2ly':
        "Use musicxml2ly when converting xml 2 lily, instead of the builtin conversion in music21",
    'play.group':
        "Name of the play engine used",
    'm21.fixstream':
        "If True, fix the streams returned by .asmusic21 (see m21fix)",
    'show.seqDuration':
        "Default duration of each element of a NoteSeq or ChordSeq when shown",
    'show.label.fontSize':
        "Font size to use for labels"
}

def _checkConfig(cfg, key, oldvalue, value):
    if key == 'notation.semitoneDivisions' and value == 4:
        showformat = cfg.get('show.format')
        if showformat and showformat.startswith('lily'):
            newvalue = oldvalue if oldvalue is not None else 2
            msg = ("\nlilypond backend (show.format) does not support 1/8 tones yet.\n"
                   "Either set config['notation.semitoneDivisions'] to 2 or\n"
                   "set config['show.format'] to 'xml.png'."
                   "Setting notation.semitoneDivision to {newvalue}")
            logger.error(msg)
            return newvalue


config = conftools.ConfigDict(f'emlib:music_core', _defaultconfig, _validator, help=_help, precallback=_checkConfig)


logger = logging.getLogger(f"emlib.music_core")

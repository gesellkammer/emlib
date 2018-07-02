from __future__ import print_function
import six as _six
import os

try:
    import hyphen
except ImportError:
    print("PyHyphen is needed to split syllables. These functionality will not be available")

_HYPHENATION_DIR = None
_HYPHENATION_INIT = False
_HYPHENATORS = {}
_REMOVE_PUNCT_TABLE = dict((ord(ch), None) for ch in u',;.:-_+*()[]/=?!^@')
_DICTS = {
    'de':'data/openthesaurus-de.txt'
}

def _first(seq):
    "returns the first element of seq or None if the seq is empty"
    try:
        return next(seq)
    except StopIteration:
        return None

def _flatten(s, exclude=_six.string_types):
    """
    return an iterator to the flattened items of sequence s
    strings are not flattened
    """
    try:
        it = iter(s)
    except TypeError:
        yield s
        raise StopIteration
    for elem in it:
        if isinstance(elem, exclude):
            yield elem
        else:
            for subelem in flatten(elem, exclude):
                yield subelem

def detect_hyphenation_dir():
    """
    try to detect the location of the hyphenation directory.
    if not found it returns None, either the path of the directory
    """
    HOME = os.getenv("HOME")
    possible_dirs = [os.path.join(HOME, '.hyphenation')]
    return _first(possible_dir for possible_dir in possible_dirs if os.path.exists(possible_dir))

def _get_hyphenator(language, hyphenation_dir):
    global _HYPHENATORS, _HYPHENATION_DIR, _HYPHENATION_INIT
    h = _HYPHENATORS.get(language)
    if h:
        return h
    #hyphenation_init(hyphenation_dir)
    if not _HYPHENATION_INIT:
        if hyphenation_dir == 'detect':
            hyphenation_dir = _HYPHENATION_DIR if _HYPHENATION_DIR is not None else detect_hyphenation_dir()
        if os.path.exists(hyphenation_dir):
            _HYPHENATION_DIR = hyphenation_dir
        else:
            raise RuntimeError("hyphenation directory not found or not detected. See PyHyphen")
        _HYPHENATION_INIT = True
    h = hyphen.hyphenator(language, directory=_HYPHENATION_DIR)
    _HYPHENATORS[language] = h
    return h

def split_syllables(s, language='en_US', hyphenation_dir='detect'):
    """
    split the string into syllables (the string should be a unicode string!)

    >>> split_syllables(u"esto es una prueba", 'es_ES')
    [u'es', u'to', u'es', u'u', u'na', u'prue', u'ba']

    """
    hyphenator = _get_hyphenator(language, hyphenation_dir)
    out = []
    out_extend = out.extend
    out_append = out.append
    # strip punctuation # FIXIT # TODO
    s = s.translate(_REMOVE_PUNCT_TABLE)
    words = s.split()
    
    for word in words:
        syllables = hyphenator.syllables(word)
        if len(syllables) == 0:
            out_append(word)
        else:
            out_extend(syllables)
    return out
        
    for syllable in hyphenator.syllables(s):
        if u' ' in syllable:
            out_extend(syllable.strip().split())
        else:
            out_append(syllable)
    return out

def syllable_splitter(language):
    from functools import partial
    return partial(split_syllables, language=language)

def hyphenation_by_decomposition(text, language='DE'):
    """
    currently this works only for german language
    """
    if language != 'DE':
        raise ValueError("at the moment only DE is supported")
    from wordaxe.DCWHyphenator import DCWHyphenator
    from wordaxe.PyHnjHyphenator import PyHnjHyphenator
    from em.iterlib import pairwise
    from em.lib import flatten

    HYPHENATORS = {}

    def _get_hyphenator(language):
        hyphenator = None
        if language.upper() == 'DE':
            # hyphenator = DCWHyphenator('DE', 5)
            hyphenator = PyHnjHyphenator('DE', 4)
        return hyphenator

    def hyphenate_syllables(word, language='DE'):
        H = HYPHENATORS.get(language)
        if H is None:
            H = _get_hyphenator(language)
            HYPHENATORS[language] = H
        hword = H.hyphenate(word)
        splits = [0] + [h.indx for h in hword.hyphenations] + [len(hword)]
        syllables = [hword[s0:s1] for s0, s1 in pairwise(splits)]
        return syllables

    def hyphenate_syllables_fragment(text, language='DE'):
        lines = [l for l in text.splitlines() if l]
        words = flatten(l.split() for l in lines)
        allsyllables = []
        for word in words:
            try:
                syllables = split_syllables(word, language)
                allsyllables.append(syllables)
            except:
                allsyllables.append([word])
        return allsyllables

    if " " in text:
        return split_syllables_fragment(text, language)
    else:
        return split_syllables(text, language)

def german_dict():
    """
    get a list of german words
    """
    thesfile_relpath = _DICTS.get('de')
    moduledir = os.path.split(__file__)[0]
    thesfile = os.path.join(moduledir, thesfile_relpath)
    if not os.path.exists(thesfile):
        raise IOError("could not find thesaurus file")
    allwords = []
    with open(thesfile) as thes:
        for line in thes:
            words = [w.decode("UTF-8").strip() for w in line.split(";")]
            allwords.extend(words)
    return allwords



"""
help write out XML documents

>>> import xmlprinter
>>> import StringIO
>>> fp = StringIO.StringIO()
>>> xp = xmlprinter.xmlprinter(fp) # The fp need only have a write() method
>>> xp.startDocument()
>>> xp.notationDecl("html", "-//W3C//DTD XHTML 1.1//EN", "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd")
>>> xp.startElement('html',
...         {'xmlns': "http://www.w3.org/1999/xhtml",
...          'xml:lang': "en-us"})
>>> xp.data("\\n")
>>> xp.startElement('head')
>>> xp.startElement('title')
>>> xp.data("This is the title")
>>> xp.endElement()    # we may omit the element name ('title')
>>> xp.endElement('head')  # or we can include it
>>> xp.data("\\n")
>>> xp.startElement('body')
>>> xp.data("\\n")
>>> xp.startElement('p')
>>> xp.data("This is some information in a paragraph.")
>>> xp.endElement('p')
>>> xp.data("\\n")
>>> xp.emptyElement('hr', {'style': 'color: red'})
>>> xp.data("\\n")
>>> xp.endDocument()       # by default closes remaining tags

<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en-us">
<head><title>This is the title</title></head>
<body>
<p>This is some information in a paragraph.</p>
<hr style="color: red" />
</body></html>

This module does nothing fancy like indenting.

Distributions for this module can be downloaded at
https://sourceforge.net/project/showfiles.php?group_id=60881


Copyright (C) 2002 Frank J. Tobin, ftobin@neverending.org

This library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation; either version 2.1 of the
License, or (at your option) any later version.

The idea for this module was taken from Perl's XML::Writer.
"""

__version__ = "0.2.1"

_TAB = '  '


class WellFormedError(Exception):
    pass


class xmlprinter(object):
    """We try to ensure a well-formed document, but won't check
    things like the validity of element names.
    Method raise WellFormedError if there are well-formed-ness problems.
    """
    xml_version = '1.0'

    def __init__(self, fp):
        """fp is a file-like object, needing only a write() method"""
        self.fp = fp
        self._finished = False
        self._past_doctype = False
        self._past_decl = False
        self._elstack = []
        self._inroot = True
        self._tabify = True
        self._last = "start"

    def __call__(self, *args):
        self.data(*args)
        return True

    def tag(self, *args, **kws):
        """
        To be used as a with statement:

        with parser.tag("person"):
            parser.tag1("name", "John")
            parser.tag1("surname", "Smith")
        """
        self.startElement(*args, **kws)
        return self

    def tag1(self, name, *data):
        """
        Allows to construct simple tags

        parser.tag1('name', 'John') --> <name>John</name>
        """
        self.startElement(name)
        for datum in data:
            self.data(datum)
        self.endElement()

    def __enter__(self):
        pass
    
    def __exit__(self, *args):
        self.endElement()


    def startDocument(self, encoding='UTF-8'):
        """Begin writing out a document, including the XML declaration.
        Currently the encoding header can be changed from the default,
        but it won't affect how the rest of the document is encoded.
        """
        self.fp.write('<?xml version=%s encoding=%s?>\n' % (
            quoteattr(self.xml_version), quoteattr(encoding)))
        self._past_decl = True

    def notationDecl(self, name, public_id=None, system_id=None):
        """Insert DOCTYPE declaration.
        Can only be added right after document start.
        Optional for a well-formed document.
        At least a public_id or system_id must be specified if called."""
        if self._past_doctype:
            raise WellFormedError("past allowed point for doctype")

        self.fp.write('<!DOCTYPE %s' % name)

        if public_id is not None:
            if system_id is None:
                raise TypeError("must have system_id with public_id")
            self.fp.write(" PUBLIC %s %s" % (quoteattr(public_id), quoteattr(system_id)))
        else:
            self.fp.write(" SYSTEM %s" % quoteattr(system_id))
        self.fp.write(">\n")
        self._past_doctype = True

    def startElement(self, name, **attrs):
        """Start element 'name' with attributes 'attrs'. (<example>)"""
        self._past_doctype = True
        self._past_decl = True
        if self._finished:
            raise WellFormedError("attempt to add second root element")
        # tabulizar
        if self._tabify or self._last == "end":
            self.fp.write('\n')
        self.fp.write(_TAB * len(self._elstack))
        self.fp.write("<%s" % name)
        for attr, val in attrs.items():
            val = str(val)
            self.fp.write(" %s=%s" % (attr, quoteattr(val)))
        self.fp.write(">")
        self._elstack.append(name)
        self._inroot = True
        self._tabify = True
        self._last = "start"

    def data(self, data):
        """Add text 'data'."""
        data = str(data)
        # self.fp.write(escape(data).encode('UTF-8'))
        self.fp.write(escape(data))
        self._tabify = False
        self._last = "data"

    def separator(self, s = "--=======================================================--"):
        self.fp.write(_TAB*len(self._elstack))
        self.fp.write('\n<!'+s+'>\n')

    def empty(self, name, **attrs):
        """Add an empty element (<example />)"""
        if not self._inroot:
            raise WellFormedError("attempt to add element outside of root")
        self.fp.write('\n'+_TAB*len(self._elstack))
        self.fp.write("<%s" % name)
        for attr, val in attrs.items():
            val = str(val)
            self.fp.write(" %s=%s" % (attr, quoteattr(val)))
        self.fp.write("/>")

    def endElement(self, name=None):
        """End the element 'name'.
        If 'name' is None, then end the most recently-opened element.
        (</example>).

        If the last element is being closed, then it
        """
        popel = self._elstack.pop()
        if name is not None and name != popel:
            raise WellFormedError("ending an unstarted element %s" % name)
        if name is None:
            name = popel
        if self._last == "end":
            self.fp.write("\n")
        if self._tabify or self._last == "end":
            self.fp.write(_TAB*len(self._elstack))
            self.fp.write("</%s>" % name)
        else:
            self.fp.write("</%s>" % name)

        if len(self._elstack) == 0:
            self._inroot = False
            # ensures a newline at the end of a text file
            self.fp.write("\n")
        self._tabify = True
        self._last = "end"

    def endDocument(self, autoclose=True):
        """Finish up a document.
        If autoclose is True, then also close any unclosed elements.
        Else, all elements must already be closed.
        """
        if self._finished:
            raise WellFormedError("attempt to re-end a finished document")
        if autoclose:
            while len(self._elstack) > 0:
                self.endElement()
        if len(self._elstack) > 0:
            raise WellFormedError("attempt to re-end a _finished document")
        self._finished = True


def escape(data):
    """Escape &, <, and > in a string of data; used for character data."""
    return data.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def quoteattr(data):
    """Escape and quote an attribute value."""
    data = escape(data)
    # We don't just turn " into &quot;, we'll use single quotes
    # if possible to retain the 'look' better.
    if '"' in data:
        if "'" in data:
            data = '"%s"' % data.replace('"', "&quot;")
        else:
            data = "'%s'" % data
    else:
        data = '"%s"' % data

    return data

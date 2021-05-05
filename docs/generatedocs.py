import emlib
from emlib import doctools
import importlib
import textwrap

modules = [
    "misc",
    "textlib",
    "filetools",
    "iterlib",
    "doctools",
    "csvtools",
    "jsontools",
    "containers",
    "dialogs",
    "mathlib",
    "numpytools",
    "matplotting",
    "calculus",
    "combinatorics",
    "numberseries",
    "graphlib",
    "bpfx", 
    "minizinctools",
    "net",
]

def get_abstract(docstr):
    for line in docstr.splitlines():
        line = line.strip()
        if line:
            line, *_ = line.split("(", maxsplit=1)
            return line
    return "<docstring>"

def get_skipped(module):
    external = doctools.externalModuleMembers(module)
    return external    

def generate_template(module):
    m = importlib.import_module(f"emlib.{module}")
    docstr = m.__doc__
    if docstr:
        abstract = get_abstract(docstr)
    else:
        abstract = "???"
        docstr = ""

    skippedItems = get_skipped(m)
    if skippedItems:
        skipstr = "\n".join(f":skip: {skippedItem}" for skippedItem in skippedItems)
        skipstr = textwrap.indent(skipstr, prefix="    ")
    else:
        skipstr = ""

    heading = f"{module}: {abstract}"
    s = \
f"""
{heading}
{'=' * len(heading)}

.. automodapi:: emlib.{module}
    :no-inheritance-diagram:
    :no-heading:
    {skipstr}

"""
    open(module + ".rst", "w").write(s)

for module in modules:
    generate_template(module)

submodulestr = textwrap.indent("\n".join(modules), prefix="    ")

emlibdocstr = emlib.__doc__

indexstr = \
f"""
emlib
=====

{emlibdocstr}

-----

Modules
=======

.. toctree::
    :maxdepth: 1

{submodulestr}
"""

open("index.rst", "w").write(indexstr)

from __future__ import annotations

import sys as _sys


_cache = {}


def session_type() -> str:
    """
    Returns the kind of python session

    .. note::
        See also `is_interactive_session` to check if we are inside a REPL

    Returns:
        Returns one of "jupyter", "ipython-terminal" (if running ipython
        in a terminal), "ipython" (if running ipython outside a terminal),
        "python" if running normal python.

    """
    if out := _cache.get('session_type'):
        return out

    try:
        # get_ipython should be available within an ipython/jupyter session
        shell = get_ipython().__class__.__name__   # type: ignore
        if shell == 'ZMQInteractiveShell':
            out = "jupyter"
        elif shell == 'TerminalInteractiveShell':
            out = "ipython-terminal"
        else:
            out = "ipython"
        _cache['session_type'] = out
        return out

    except NameError:
        return "python"


def inside_jupyter() -> bool:
    """
    Are we running inside a jupyter notebook?
    """
    return session_type() == 'jupyter'


def inside_ipython() -> bool:
    """
    Are we running inside ipython?

    This includes any ipython session (ipython in terminal, jupyter, etc.)
    """
    if out := _cache.get('inside_ipython'):
        return out
    _cache['inside_ipython'] = out = session_type() in ('jupyter', 'ipython', 'ipython-terminal')
    return out


def is_interactive_session() -> bool:
    """ Are we running inside an interactive session? """
    return _sys.flags.interactive == 1


def running_inside_terminal() -> bool:
    """
    Are we running inside a terminal and not in the background?

    """
    return _sys.stdin and _sys.stdin.isatty()


def ipython_qt_eventloop_started() -> bool:
    """
    Are we running ipython / jupyter and the qt event loop has been started?
    ( %gui qt )
    """
    session = session_type()
    if session == 'ipython-terminal' or session == 'jupyter':
        # we are inside ipython so we can just call 'get_ipython'
        ip = get_ipython()   # type: ignore
        return ip.active_eventloop == "qt"
    else:
        return False


def get_platform() -> tuple[str, str]:
    """
    Return a tuple (osname, architecture)

    This attempts to improve upon `sysconfig.get_platform` by fixing some
    issues when running a Python interpreter with a different architecture than
    that of the system (e.g. 32bit on 64bit system, or a multiarch build),
    which should return the machine architecture of the currently running
    interpreter rather than that of the system (which didn't seem to work
    properly). The reported machine architectures follow platform-specific
    naming conventions (e.g. "x86_64" on Linux, but "x64" on Windows).

    Returns:
        a tuple (osname: str, architecture: str)


    Example output strings for common platforms::

        ("darwin", one of ppc|ppc64|i368|x86_64|arm64)
        ("linux", one of i686|x86_64|armv7l|aarch64)
        ("windows", one of x86|x64|arm32|arm64

    """
    if out := _cache.get('get_platform'):
        return out

    import platform
    import sysconfig

    system = platform.system().lower()
    machine = sysconfig.get_platform().split("-")[-1].lower()
    is_64bit = _sys.maxsize > 2 ** 32

    if system == "darwin": # get machine architecture of multiarch binaries
        if any([x in machine for x in ("fat", "intel", "universal")]):
            machine = platform.machine().lower()

    elif system == "linux":  # fix running 32bit interpreter on 64bit system
        if not is_64bit and machine == "x86_64":
            machine = "i686"
        elif not is_64bit and machine == "aarch64":
            machine = "armv7l"

    elif system == "windows": # return more precise machine architecture names
        if machine == "amd64":
            machine = "x64"
        elif machine == "win32":
            if is_64bit:
                machine = platform.machine().lower()
            else:
                machine = "x86"

    # some more fixes based on examples in https://en.wikipedia.org/wiki/Uname
    if not is_64bit and machine in ("x86_64", "amd64"):
        if any([x in system for x in ("cygwin", "mingw", "msys")]):
            machine = "i686"
        else:
            machine = "i386"

    _cache['get_platform'] = out = (system, machine)
    return out


def get_base_prefix_compat() -> str:
    """Get base/real prefix, or sys.prefix if there is none."""
    return getattr(_sys, "base_prefix", None) or getattr(_sys, "real_prefix", None) or _sys.prefix



def in_virtualenv() -> bool:
    """
    Are we inside a virtual environment?
    """
    return get_base_prefix_compat() != _sys.prefix


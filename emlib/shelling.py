from __future__ import annotations
import os
import re


class Proc:
    def __init__(self, pid, args):
        self.pid = pid
        self.args = args
        self._cmdline = None
        self._repr = None
        self._basebin = None

    @property
    def cmdline(self):
        if self._cmdline is None:
            self._cmdline = " ".join(self.args)
        return self._cmdline

    @property
    def basebin(self):
        if self._basebin is None:
            self._basebin = self.args[0].split("/")[-1]
        return self._basebin

    def matchfull(self, pattern):
        """
        Similar to pgrep -f
        """
        match = re.search(pattern, self.cmdline)
        return True if match else False

    def matchexe(self, exe):
        return exe == self.args[0] or exe == self.basebin

    def __repr__(self):
        if self._repr is None:
            self._repr = f"Proc(pid={self.pid}, args={self.args})"    
        return self._repr


def allpids():
    """
    This works only on unixy systems
    """
    return [int(pid) for pid in os.listdir('/proc') if pid != 'curproc' and pid.isdigit()]
        

def readargs(pid):
    """
    This only works only on unixy systems
    """
    args = None
    try:
        with open(f'/proc/{pid}/cmdline', mode='rb') as fd:
            args = fd.read().decode().split('\x00')[:-1]
    except Exception:
        pass
    return args


def allprocs(skipempty=True, pids=None):
    """
    Get a list of all processes running (does not work on Windows)

    returns a list of Procs 
    """
    pids = pids or allpids()
    out = []
    for pid in pids:
        args = readargs(pid)
        if args or not skipempty:
            out.append(Proc(pid=pid, args=args))
    return out


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def fzf(cmd=None, pattern=None):
    if pattern is not None and cmd is None:
        cmdline = f'fd {pattern} | fzf'
    elif cmd and pattern:
        cmdline = f'{cmd} | fzf --query {pattern}'
    elif cmd:
        cmdline = f'{cmd} | fzf'
    else:
        cmdline = 'fzf'
    return os.popen(cmdline).read().strip()

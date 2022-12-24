"""
Utilities around python's own logging framework
"""
from __future__ import annotations
import logging
import logging.handlers
import os


def prettylog(logger: logging.Logger, obj, level='DEBUG') -> None:
    """
    Log obj using pprint's representation
    """
    import pprint
    msg = pprint.pformat(obj)
    levelint = logging.getLevelName(level)
    assert isinstance(levelint, int), f"Unknown logging level {level}"
    logger.log(level=levelint, msg=msg)


def fileLogger(path:str, level='DEBUG', fmt='%(levelname)s: %(message)s'
               ) -> logging.Logger:
    """
    Create a logger outputting to a (rotating) file

    Args:
        path: the path of the output file
        level: the level of the logger
        fmt: the format used

    Returns:
        a Logger
    """
    name = os.path.splitext(os.path.split(path)[1])[0]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=80*2000, backupCount=1)
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def reloadBasicConfig(level='DEBUG', otherLoggers: list[logging.Logger] = None
                      ) -> None:
    """
    Setup basicConfig for the logging module

    Inside an interactive session often logging is already setup when you
    get to setup your own logging. In such a case it is needed to reload
    the logging module to setup the desired logging level

    Args:
        level: the level to setup the basic config
        otherLoggers: a list of logging.Loggers  to also set at the given level
    """
    import importlib
    importlib.reload(logging)
    logging.basicConfig(level=level)
    if otherLoggers:
        for logger in otherLoggers:
            logger.setLevel(level)

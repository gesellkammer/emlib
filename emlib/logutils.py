from __future__ import annotations
import logging
import logging.handlers
import os


def fileLogger(path:str, level='DEBUG', fmt='%(levelname)s: %(message)s'
               ) -> logging.Logger:
    """
    Create a logger outputting to a file

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


def reloadBasicConfig(level='DEBUG', otherLoggers=[]) -> None:
    import importlib
    importlib.reload(logging)
    logging.basicConfig(level="DEBUG")
    for logger in otherLoggers:
        logger.setLevel(level)

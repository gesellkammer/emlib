import logging
import logging.handlers
import os


def fileLogger(path, level='DEBUG', fmt='%(levelname)s: %(message)s'):
    """
    Returns: a logger
    """
    name = os.path.splitext(os.path.split(path)[1])[0]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=80*2000, backupCount=1)
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

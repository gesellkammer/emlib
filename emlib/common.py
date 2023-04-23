from __future__ import annotations


class runonce:
    """
    To be used as decorator. `func` will run only once

    Example::

        # get_config() will only read the file the first time,
        # return the resulting dict for any further calls

        @runonce
        def get_config():
            config = json.load(open("/path/to/config.json"))
            return config

        config = get_config()

    """
    __slots__ = ('func', 'result', 'has_run')

    def __init__(self, func):
        self.func = func
        self.result = None
        self.has_run = False

    def __call__(self, *args, **kwargs):
        if self.has_run:
            return self.result

        self.result = self.func(*args, **kwargs)
        self.has_run = True
        return self.result
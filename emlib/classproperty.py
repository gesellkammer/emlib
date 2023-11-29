"""
Very simple implementation of a class property

Example
~~~~~~~

    class Foo:
        _active: Foo | None = None
        _initdone = False

        def __init__(self, bar=None):
            self.bar = bar

        @classmethod
        def _initclass(self):
            if Foo._initdone:
                return
            Foo._initdone = True
            Foo._active = Foo()

        @classproperty
        def active(cls) -> Foo
            assert Foo._initdone and Foo._active is not None
            return Foo._active

    Foo._initclass()
"""


class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)

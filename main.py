"""Backward-compatible entry point for the IR-AQML package."""
import sys
import types

import app as _app


class _MainModule(types.ModuleType):

    def __getattr__(self,name):
        return getattr(_app,name)

    def __setattr__(self,name,value):
        if name in _app.__dict__:
            setattr(_app,name,value)
            return
        super().__setattr__(name,value)

    def __delattr__(self,name):
        if name in _app.__dict__:
            delattr(_app,name)
            return
        super().__delattr__(name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(_app)))


sys.modules[__name__].__class__=_MainModule

__all__=sorted(
    name for name in dir(_app)
    if not name.startswith("_")
)


def __getattr__(name):
    return getattr(_app,name)


def __dir__():
    return sorted(set(globals()) | set(dir(_app)))


if __name__=="__main__":
    _app.menu()

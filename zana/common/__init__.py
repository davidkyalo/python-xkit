from collections import abc
from functools import reduce
import typing as t

from importlib import import_module


_empty = object()


# _P = ParamSpec("_P")
_R = t.TypeVar("_R")
_T = t.TypeVar("_T")
_R_Co = t.TypeVar("_R_Co", covariant=True)
_T_Co = t.TypeVar("_T_Co", covariant=True)


class class_property(t.Generic[_R_Co]):
    def __init__(
        self,
        getter: abc.Callable[..., _R_Co],
        # setter: abc.Callable[..., t.NoReturn] = None,
    ) -> None:
        self.__fget__ = getter

        if getter:
            info = getter
            self.__doc__ = info.__doc__
            self.__name__ = info.__name__
            self.__module__ = info.__module__

    def __get__(self, obj: _T, typ: type = None) -> _R_Co:
        return self.__fget__(typ if obj is None else obj.__class__)

    # def getter(self, getter: abc.Callable[..., _R_Co]) -> "class_property[_R_Co]":
    #     return self.__class__(getter, self.fset)

    # def __set__(self, obj: _T, value: t.Any):
    #     if self.fset is None:
    #         raise AttributeError("")
    #     self.fset(obj.__class__, value)

    # def setter(self, setter: abc.Callable[..., t.NoReturn]):
    #     return self.__class__(self.fget, setter)


def try_import(modulename: str, qualname: str = None, default=_empty):
    """Try to import and return module object.

    Returns None if the module does not exist.
    """
    if not isinstance(modulename, str):
        return modulename

    if qualname is None:
        modulename, _, qualname = modulename.partition(":")

    try:
        module = import_module(modulename)
    except ImportError:
        if not qualname:
            modulename, _, qualname = modulename.rpartition(".")
            if modulename:
                return try_import(modulename, qualname, default=default)
        if default is _empty:
            raise
        return default
    else:
        if qualname:
            try:
                return reduce(getattr, qualname.split("."), module)
            except AttributeError:
                if default is _empty:
                    raise
                return default
        return module

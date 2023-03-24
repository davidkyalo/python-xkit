import typing as t
from collections import abc
from functools import reduce
from importlib import import_module
from threading import RLock
from unittest import skip

from typing_extensions import ParamSpec, Self

from zana.types import NotSet

# _P = ParamSpec("_P")
_R = t.TypeVar("_R")
_T = t.TypeVar("_T")
_KT = t.TypeVar("_KT")
_VT = t.TypeVar("_VT")
_T_Co = t.TypeVar("_T_Co", covariant=True)

_P = ParamSpec("_P")


class descriptor(property, t.Generic[_T_Co]):
    attrname = None

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)

    @property
    def __name__(self):
        return self.attrname or self.fget.__name__

    @property
    def takes_self(self) -> None:
        return self

    def __set_name__(self, owner: type, name: str):
        if name != (self.attrname or name):
            raise TypeError(
                "Cannot assign the same class_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )
        self.attrname = name

    def __get__(self, obj, cls=None):
        return self.fget(obj, self, cls)

    def __call__(this, /, self, *args, **kwargs):
        return this.__get__(self)(*args, **kwargs)


def _dict_not_set_error(self, obj: object):
    msg = (
        f"No '__dict__' attribute on {obj.__class__.__name__!r} "
        f"instance to cache {self.attrname!r} property."
    )
    return TypeError(msg)


def _dict_not_mutable_error(self, obj: object):
    msg = (
        f"No '__dict__' attribute on {obj.__class__.__name__!r} "
        f"instance to cache {self.attrname!r} property."
    )
    return TypeError(msg)


def _dictset(self, obj: object, val: t.Any):
    try:
        obj.__dict__[self.attrname] = val
    except AttributeError:
        raise self._dict_not_set_error(obj) from None
    except TypeError:
        raise self._dict_not_mutable_error(obj) from None


def _dictpop(self, obj: object):
    try:
        del obj.__dict__[self.attrname]
    except AttributeError:
        raise self._dict_not_set_error(obj) from None
    except TypeError:
        raise self._dict_not_mutable_error(obj) from None
    except KeyError:
        pass


def subclasses(*bases: type[_T], inclusive: bool = True, depth: int = -1, __skip=None, __lvl=0):
    """ """
    __skip = set() if __skip is None else __skip
    kwds = {"depth": depth, "__lvl": __lvl + 1, "__skip": __skip}
    for base in bases:
        if not (inclusive is False or base in __skip or __skip.add(base)):
            yield base
        if depth - __lvl:
            yield from subclasses(*base.__subclasses__(), **kwds)


class class_property(t.Generic[_R]):
    attrname: str = None

    _dict_not_mutable_error = _dict_not_mutable_error
    _dict_not_set_error = _dict_not_set_error

    def __init__(
        self: Self,
        getter: abc.Callable[..., _R] = None,
    ) -> None:
        self.__fget__ = getter

        if getter:
            if hasattr(getter, "__name__"):
                self.__doc__ = getter.__doc__
                self.__name__ = getter.__name__
                self.__module__ = getter.__module__

    def __set_name__(self, owner: type, name: str):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same class_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def getter(self, getter: abc.Callable[..., _R]) -> "_R | class_property[_R]":
        return self.__class__(getter)

    def __get__(self, obj: _T, typ: type = None) -> _R:
        if not obj is None:
            if not (name := self.attrname) is None:
                try:
                    return obj.__dict__[name]
                except (AttributeError, KeyError):
                    pass
            typ = type(obj)

        return self.__fget__(typ)

    __set__ = _dictset
    __delete__ = _dictpop


class cached_attr(property, t.Generic[_T_Co]):
    _lock: RLock
    attrname: str

    _dict_not_mutable_error = _dict_not_mutable_error
    _dict_not_set_error = _dict_not_set_error

    if not t.TYPE_CHECKING:

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)
            self._lock = RLock()
            self.attrname = None

    def __set_name__(self, owner: type, name: str):
        supa = super()
        if hasattr(supa, "__set_name__"):
            supa.__set_name__(owner, name)

        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def __get__(self, obj: _T, cls: t.Union[type, None] = ...):
        if obj is None:
            return self
        name = self.attrname
        try:
            return obj.__dict__[name]
        except AttributeError:
            raise self._dict_not_set_error(obj) from None
        except KeyError:
            cache = obj.__dict__
            with self._lock:
                if name in cache:
                    return cache[name]
                else:
                    if not (fget := self.fget):
                        raise AttributeError(f"{name!r} not set.")
                    val = fget(obj)
                    try:
                        cache[name] = val
                    except TypeError:
                        raise self._dict_not_mutable_error(obj) from None
                    return val

    def __set__(self, obj: _T, val: t.Any) -> None:
        with self._lock:
            if fset := self.fset:
                return fset(obj, val)

            _dictset(self, obj, val)

    def __delete__(self, obj: _T) -> None:
        with self._lock:
            if fdel := self.fdel:
                return fdel(obj)

            _dictpop(self, obj)


def try_import(modulename: t.Any, qualname: str = None, *, default=NotSet):
    """Try to import and return module object.

    Returns None if the module does not exist.
    """
    if not isinstance(modulename, str):
        if default is NotSet:
            raise TypeError(f"cannot import from {modulename.__class__.__name__!r} objects")
        return default

    if qualname is None:
        modulename, _, qualname = modulename.partition(":")

    try:
        module = import_module(modulename)
    except ImportError:
        if not qualname:
            modulename, _, qualname = modulename.rpartition(".")
            if modulename:
                return try_import(modulename, qualname, default=default)
        if default is NotSet:
            raise
        return default
    else:
        if qualname:
            try:
                return reduce(getattr, qualname.split("."), module)
            except AttributeError:
                if default is NotSet:
                    raise
                return default
        return module


def _kwarg_map(func, it: abc.Iterable[abc.Mapping[str, _T] | abc.Iterable[tuple[str, _T]]]):
    """Like `itertools.starmap` but for keyword arguments"""
    for x in it:
        yield func(**(x if isinstance(x, (dict, abc.Mapping)) else dict(x)))


def _arg_kwarg_map(
    func,
    it: abc.Iterable[
        abc.Iterable[tuple[abc.Iterable[_VT], abc.Mapping[str, _T] | abc.Iterable[tuple[str, _T]]]]
    ],
):
    """A combination of both `kwarg_map` and `itertools.starmap`."""
    for a, kw in it:
        yield func(*a, **(kw if isinstance(it, (dict, abc.Mapping)) else dict(kw)))


def iteritems(*items: abc.Mapping[_KT, _VT] | abc.Iterable[tuple[_KT, _VT]], **kwds: _VT):
    """Iterator over (key, value) pairs from mappings, iterables and/or keywords."""
    its = (it.items() if isinstance(it, (dict, abc.Mapping)) else it for it in items)
    for k, v in ((kv for kv in its if kv[0] not in kwds), kwds.items()) if kwds else (its,):
        yield k, v

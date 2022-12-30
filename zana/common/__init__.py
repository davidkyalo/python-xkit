import typing as t
from collections import abc
from functools import cache, cached_property, reduce
from importlib import import_module
from threading import RLock

from typing_extensions import Self

from zana.types import NotSet

# _P = ParamSpec("_P")
_R = t.TypeVar("_R")
_T = t.TypeVar("_T")
_T_Co = t.TypeVar("_T_Co", covariant=True)


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

class class_property(t.Generic[_R]):
    attrname: str = None
    
    _dict_not_mutable_error = _dict_not_mutable_error
    _dict_not_set_error = _dict_not_set_error

    def __init__(
        self,
        getter: abc.Callable[..., _R]=None,
    ) -> None:
        self.__fget__ = getter

        if getter:
            info = getter
            self.__doc__ = info.__doc__
            self.__name__ = info.__name__
            self.__module__ = info.__module__

    def __set_name__(self, owner: type, name: str):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same class_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    contribute_to_class = __set_name__

    def getter(self, getter: abc.Callable[..., _R]) -> "_R | class_property[_R]":
        return self.__class__(getter)

    def __get__(self, obj: _T, typ: type = None) -> _R:
        if not obj is None:
            if not (name := self.attrname) is None:
                try:
                    return obj.__dict__[name]
                except (AttributeError, KeyError):
                    pass
            typ = obj.__class__
                
        return self.__fget__(typ)

    __set__ = _dictset
    __delete__ = _dictpop


class lazyattr(property, t.Generic[_T_Co]):

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
            cache = obj.__dict__
        except AttributeError:
            raise self._dict_not_set_error(obj) from None

        val = cache.get(name, NotSet)
        if val is NotSet:
            with self._lock:
                val = cache.get(name, NotSet)
                if val is NotSet:
                    val = super().__get__(obj, cls)
                    try:
                        cache[name] = val
                    except TypeError:
                        raise self._dict_not_mutable_error(obj) from None

        return val

    def __set__(self, obj: _T, val: t.Any) -> None:
        with self._lock:
            if self.fset:
                return super().__set__(obj, val)

            _dictset(self, obj, val)

    def __delete__(self, obj: _T) -> None:
        with self._lock:
            if self.fdel:
                return super().__delete__(obj)

            _dictpop(self, obj)
    


def try_import(modulename: str, qualname: str = None, *, default=NotSet):
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

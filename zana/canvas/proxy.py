"""A set of python utilities

"""

"""Proxy/PromiseProxy implementation.

This module contains critical utilities that needs to be loaded as
soon as possible, and that shall not load any third party modules.

Parts of this module is Copyright by Werkzeug Team.
"""

import typing as t
from abc import abstractmethod
from collections import abc
from threading import RLock

from typing_extensions import ParamSpec

from zana.types.collections import FrozenDict

__all__ = ("Proxy", "CachingProxy", "try_import", "unproxy")

__module__ = __name__  # used by Proxy class body

_object_new = object.__new__
_object_setattr = object.__setattr__
_object_getattribute = object.__getattribute__
_object_delattr = object.__delattr__

_empty = object()
_empty1 = object()

_P = ParamSpec("_P")
_R = t.TypeVar("_R")
_T_Co = t.TypeVar("_T_Co", covariant=True)
_T_Factory = abc.Callable[_P, _R]


def _default_cls_attr(name, type_, cls_value=_empty):
    # Proxy uses properties to forward the standard
    # class attributes __module__, __name__ and __doc__ to the real
    # object, but these needs to be a string when accessed from
    # the Proxy class directly.  This is a hack to make that work.
    # -- See Issue #1087.

    def __new__(cls, getter):
        instance = type_.__new__(cls, cls_value)
        instance.__getter = getter
        return instance

    def __get__(self, obj, cls=None):
        return self.__getter(obj) if obj is not None else self

    if t.TYPE_CHECKING:
        return t.cast(type_, cls_value)

    return type(
        name,
        (type_,),
        {
            "__new__": __new__,
            "__get__": __get__,
        },
    )


def unproxy(obj: t.Union[_R, "SupportsProxy[_R]"]) -> _R:
    """Attempt to ."""
    try:
        fn = obj.__get_proxy_target__
    except AttributeError:
        return obj
    else:
        return fn()


class SupportsProxy(t.Protocol[_R]):
    @abstractmethod
    def __get_proxy_target__(self, *a, **kw) -> _R:
        pass

    @abstractmethod
    def __proxy_cached_target__(self, default: ... = _empty) -> _R:
        ...

    @abstractmethod
    def __proxy_target_is_cached__(self) -> bool:
        pass

    @abstractmethod
    def __proxy_target_evaluate__(self) -> None:
        pass

    @abstractmethod
    def __proxy_reset_cache__(self) -> None:
        pass


class Proxy(t.Generic[_R]):
    """Proxy to another object."""

    # Code stolen from werkzeug.local.Proxy.
    __slots__ = (
        "__local",
        "__args",
        "__kwargs",
        "__orig_class",
        "__name",
        "__module",
        "__qualname",
        "__doc",
        "__weakref__",
    )

    __local: abc.Callable[_P, _R]
    __args: tuple
    __kwargs: dict[str, t.Any]
    __bypass_names: t.Final = frozenset(["__orig_class__"])
    __name: str
    __qualname: str
    __module: str
    __doc: str

    def __new__(
        cls,
        local: abc.Callable[_P, _R],
        /,
        *args: _P.args,
        __name__: str = None,
        __qualname__: str = None,
        __module__: str = None,
        __doc__: str = None,
        **kwargs: _P.kwargs,
    ) -> _R | SupportsProxy[_R]:
        self = _object_new(cls)
        _object_setattr(self, "_Proxy__local", local)
        _object_setattr(self, "_Proxy__args", args)
        _object_setattr(self, "_Proxy__kwargs", FrozenDict(kwargs))

        _object_setattr(self, "_Proxy__name", __name__)
        _object_setattr(self, "_Proxy__module", __module__)
        _object_setattr(self, "_Proxy__qualname", __qualname__)
        _object_setattr(self, "_Proxy__doc", __doc__)

        return self

    @_default_cls_attr("name", str, "Proxy")
    def __name__(self):
        if (val := self.__name) is None:
            val = self.__get_proxy_target__().__name__
        return val

    @_default_cls_attr("qualname", str, "Proxy")
    def __qualname__(self):
        if (val := self.__qualname) is None:
            val = self.__get_proxy_target__().__qualname__
        return val

    @_default_cls_attr("module", str, __module__)
    def __module__(self):
        if (val := self.__module) is None:
            val = self.__get_proxy_target__().__module__
        return val

    @_default_cls_attr("doc", str, __doc__)
    def __doc__(self):
        if (val := self.__doc) is None:
            val = self.__get_proxy_target__().__doc__
        return val

    @property
    def __class__(self):
        return self.__get_proxy_target__().__class__

    @property
    def __orig_class__(self):
        return self.__get_proxy_target__().__orig_class__

    @__orig_class__.setter
    def __orig_class__(self, value):
        _object_setattr(self, "_Proxy__orig_class", value)

    def __get_proxy_target__(self, *a, **kw) -> _R:
        """Get current object.
        This is useful if you want the real
        object behind the proxy at a time for performance reasons or because
        you want to pass the object into a different context.
        """
        return self.__local(*self.__args, *a, **self.__kwargs | kw)

    @property
    def __dict__(self):
        try:
            return self.__get_proxy_target__().__dict__
        except RuntimeError:  # pragma: no cover
            raise AttributeError("__dict__")

    if True:

        def __repr__(self):
            try:
                obj = self.__get_proxy_target__()
            except RuntimeError:  # pragma: no cover
                return f"<{self.__class__.__name__} unbound>"
            return repr(obj)

        def __bool__(self):
            try:
                return bool(self.__get_proxy_target__())
            except RuntimeError:  # pragma: no cover
                return False

        __nonzero__ = __bool__  # Py2

        def __dir__(self):
            try:
                return dir(self.__get_proxy_target__())
            except RuntimeError:  # pragma: no cover
                return []

        def __getattr__(self, name):
            if name == "__members__":
                return dir(self.__get_proxy_target__())
            return getattr(self.__get_proxy_target__(), name)

        def __setitem__(self, key, value):
            self.__get_proxy_target__()[key] = value

        def __delitem__(self, key):
            del self.__get_proxy_target__()[key]

        def __setattr__(self, name, value):
            if name in type(self).__bypass_names:
                _object_setattr(self, name, value)
            else:
                setattr(self.__get_proxy_target__(), name, value)

        def __delattr__(self, name):
            delattr(self.__get_proxy_target__(), name)

        def __str__(self):
            return str(self.__get_proxy_target__())

        def __lt__(self, other):
            return self.__get_proxy_target__() < other

        def __le__(self, other):
            return self.__get_proxy_target__() <= other

        def __eq__(self, other):
            return self.__get_proxy_target__() == other

        def __ne__(self, other):
            return self.__get_proxy_target__() != other

        def __gt__(self, other):
            return self.__get_proxy_target__() > other

        def __ge__(self, other):
            return self.__get_proxy_target__() >= other

        def __hash__(self):
            return hash(self.__get_proxy_target__())

        def __call__(self, *a, **kw):
            return self.__get_proxy_target__()(*a, **kw)

        def __len__(self):
            return len(self.__get_proxy_target__())

        def __getitem__(self, i):
            return self.__get_proxy_target__()[i]

        def __iter__(self):
            return iter(self.__get_proxy_target__())

        def __contains__(self, i):
            return i in self.__get_proxy_target__()

        def __add__(self, other):
            return self.__get_proxy_target__() + other

        def __sub__(self, other):
            return self.__get_proxy_target__() - other

        def __mul__(self, other):
            return self.__get_proxy_target__() * other

        def __floordiv__(self, other):
            return self.__get_proxy_target__() // other

        def __mod__(self, other):
            return self.__get_proxy_target__() % other

        def __divmod__(self, other):
            return self.__get_proxy_target__().__divmod__(other)

        def __pow__(self, other):
            return self.__get_proxy_target__() ** other

        def __lshift__(self, other):
            return self.__get_proxy_target__() << other

        def __rshift__(self, other):
            return self.__get_proxy_target__() >> other

        def __and__(self, other):
            return self.__get_proxy_target__() & other

        def __xor__(self, other):
            return self.__get_proxy_target__() ^ other

        def __or__(self, other):
            return self.__get_proxy_target__() | other

        def __div__(self, other):
            return self.__get_proxy_target__().__div__(other)

        def __truediv__(self, other):
            return self.__get_proxy_target__().__truediv__(other)

        def __neg__(self):
            return -(self.__get_proxy_target__())

        def __pos__(self):
            return +(self.__get_proxy_target__())

        def __abs__(self):
            return abs(self.__get_proxy_target__())

        def __invert__(self):
            return ~(self.__get_proxy_target__())

        def __complex__(self):
            return complex(self.__get_proxy_target__())

        def __int__(self):
            return int(self.__get_proxy_target__())

        def __float__(self):
            return float(self.__get_proxy_target__())

        def __oct__(self):
            return oct(self.__get_proxy_target__())

        def __hex__(self):
            return hex(self.__get_proxy_target__())

        def __index__(self):
            return self.__get_proxy_target__().__index__()

        def __coerce__(self, other):
            return self.__get_proxy_target__().__coerce__(other)

        def __enter__(self):
            return self.__get_proxy_target__().__enter__()

        def __exit__(self, *a, **kw):
            return self.__get_proxy_target__().__exit__(*a, **kw)

        def __reduce__(self):
            return self.__get_proxy_target__().__reduce__()

        async def __aenter__(self):
            return await self.__get_proxy_target__().__aenter__()

        async def __aexit__(self, *a, **kw):
            return await self.__get_proxy_target__().__aexit__(*a, **kw)


class CachingProxy(Proxy[_R]):
    """Proxy that evaluates object once.

    :class:`Proxy` will evaluate the object each time, while the
    promise will only evaluate it once.
    """

    __slots__ = "__thing", "__lock"
    __lock: RLock

    if not t.TYPE_CHECKING:

        def __new__(cls, /, *args, **kwargs) -> _R | SupportsProxy[_R]:
            self = Proxy.__new__(cls, *args, **kwargs)
            _object_setattr(self, "_CachingProxy__lock", RLock())
            return self

    def __get_proxy_target__(self) -> _R:
        try:
            return _object_getattribute(self, "_CachingProxy__thing")
        except AttributeError:
            return self.__proxy_target_evaluate__()

    def __proxy_cached_target__(self, default: ... = _empty):
        try:
            return _object_getattribute(self, "_CachingProxy__thing")
        except AttributeError:
            if default is _empty:
                raise
            return default

    def __proxy_target_is_cached__(self):
        return self.__proxy_cached_target__(_empty1) is not _empty1

    def __proxy_reset_cache__(self) -> None:
        with self.__lock:
            try:
                _object_delattr(self, "_CachingProxy__thing")
            except AttributeError:
                pass

    def __proxy_target_evaluate__(self) -> _R:
        with self.__lock:
            if self.__proxy_target_is_cached__():
                return _object_getattribute(self, "_CachingProxy__thing")
            thing = Proxy.__get_proxy_target__(self)
            _object_setattr(self, "_CachingProxy__thing", thing)
            return thing

"""A set of python utilities

"""

__version__ = 0, 0, 0

"""Proxy/PromiseProxy implementation.

This module contains critical utilities that needs to be loaded as
soon as possible, and that shall not load any third party modules.

Parts of this module is Copyright by Werkzeug Team.
"""

import sys
from abc import abstractmethod
from collections import abc
from functools import cache, reduce
from types import GenericAlias, ModuleType
import typing as t

from typing_extensions import Self, ParamSpec, Concatenate

__all__ = ("Proxy", "PromiseProxy", "try_import", "unproxy")

__module__ = __name__  # used by Proxy class body

_object_new = object.__new__
_object_setattr = object.__setattr__
_object_getattribute = object.__getattribute__
_object_delattr = object.__delattr__

_empty = object()

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

    return type(
        name,
        (type_,),
        {
            "__new__": __new__,
            "__get__": __get__,
        },
    )


def unproxy(obj: t.Union[_R, "GetsCurrent[_R]"]) -> _R:
    """Attempt to ."""
    try:
        fn = obj._get_current_object
    except AttributeError:
        return obj
    else:
        return fn()


class GetsCurrent(t.Protocol[_R]):
    @abstractmethod
    def _get_current_object(self, *a, **kw) -> _R:
        pass

    @abstractmethod
    def __evaluated__(self) -> bool:
        pass

    @abstractmethod
    def __maybe_evaluate__(self) -> _R:
        pass

    @abstractmethod
    def __evaluate__(self) -> None:
        pass

    @abstractmethod
    def __reset_promise__(self) -> None:
        pass


class Proxy(t.Generic[_R]):
    """Proxy to another object."""

    # Code stolen from werkzeug.local.Proxy.
    __slots__ = "__local", "__args", "__kwargs", "__orig_class", "__weakref__", "__dict__"

    __local: abc.Callable[_P, _R]
    __args: tuple
    __kwargs: dict[str, t.Any]
    __bypass_names: t.Final = frozenset(["__orig_class__"])

    def __new__(
        cls,
        local: abc.Callable[_P, _R] = _empty,
        /,
        *args: _P.args,
        __name__=None,
        __doc__=None,
        **kwargs: _P.kwargs,
    ):
        self = _object_new(cls)
        if not local is _empty:
            _object_setattr(self, "_Proxy__local", local)
            _object_setattr(self, "_Proxy__args", args)
            _object_setattr(self, "_Proxy__kwargs", kwargs)

        if __name__ is not None:
            _object_setattr(self, "__custom_name__", __name__)
        if __doc__ is not None:
            _object_setattr(self, "__doc__", __doc__)

        return t.cast(t.Union[_R, GetsCurrent[_R]], self)

    @_default_cls_attr("name", str, "Proxy")
    def __name__(self):
        try:
            return self.__custom_name__
        except AttributeError:
            return self._get_current_object().__name__

    @_default_cls_attr("qualname", str, __name__)
    def __qualname__(self):
        try:
            return self.__custom_name__
        except AttributeError:
            return self._get_current_object().__qualname__

    @_default_cls_attr("module", str, __module__)
    def __module__(self):
        return self._get_current_object().__module__

    @_default_cls_attr("doc", str, __doc__)
    def __doc__(self):
        return self._get_current_object().__doc__

    # def _get_class(self):
    #     return self._get_current_object().__class__

    @property
    def __class__(self):
        return self._get_current_object().__class__

    @property
    def __orig_class__(self):
        return self._get_current_object().__orig_class__

    @__orig_class__.setter
    def __orig_class__(self, value):
        _object_setattr(self, "_Proxy__orig_class", value)

    def _get_current_object(self, *a, **kw) -> _R:
        """Get current object.

        This is useful if you want the real
        object behind the proxy at a time for performance reasons or because
        you want to pass the object into a different context.
        """
        loc: abc.Callable[_P, _R] = _object_getattribute(self, "_Proxy__local")
        if not hasattr(loc, "__release_local__"):
            return loc(*self.__args, *a, **self.__kwargs | kw)
        try:  # pragma: no cover
            # not sure what this is about
            return getattr(loc, self.__name__)
        except AttributeError:  # pragma: no cover
            raise RuntimeError(f"no object bound to {self.__name__}")

    @property
    def __dict__(self):
        try:
            return self._get_current_object().__dict__
        except RuntimeError:  # pragma: no cover
            raise AttributeError("__dict__")

    if True:

        def __repr__(self):
            try:
                obj = self._get_current_object()
            except RuntimeError:  # pragma: no cover
                return f"<{self.__class__.__name__} unbound>"
            return repr(obj)

        def __bool__(self):
            try:
                return bool(self._get_current_object())
            except RuntimeError:  # pragma: no cover
                return False

        __nonzero__ = __bool__  # Py2

        def __dir__(self):
            try:
                return dir(self._get_current_object())
            except RuntimeError:  # pragma: no cover
                return []

        def __getattr__(self, name):
            if name == "__members__":
                return dir(self._get_current_object())
            return getattr(self._get_current_object(), name)

        def __setitem__(self, key, value):
            self._get_current_object()[key] = value

        def __delitem__(self, key):
            del self._get_current_object()[key]

        def __setattr__(self, name, value):
            if name in type(self).__bypass_names:
                _object_setattr(self, name, value)
            else:
                setattr(self._get_current_object(), name, value)

        def __delattr__(self, name):
            delattr(self._get_current_object(), name)

        def __str__(self):
            return str(self._get_current_object())

        def __lt__(self, other):
            return self._get_current_object() < other

        def __le__(self, other):
            return self._get_current_object() <= other

        def __eq__(self, other):
            return self._get_current_object() == other

        def __ne__(self, other):
            return self._get_current_object() != other

        def __gt__(self, other):
            return self._get_current_object() > other

        def __ge__(self, other):
            return self._get_current_object() >= other

        def __hash__(self):
            return hash(self._get_current_object())

        def __call__(self, *a, **kw):
            return self._get_current_object()(*a, **kw)

        def __len__(self):
            return len(self._get_current_object())

        def __getitem__(self, i):
            return self._get_current_object()[i]

        def __iter__(self):
            return iter(self._get_current_object())

        def __contains__(self, i):
            return i in self._get_current_object()

        def __add__(self, other):
            return self._get_current_object() + other

        def __sub__(self, other):
            return self._get_current_object() - other

        def __mul__(self, other):
            return self._get_current_object() * other

        def __floordiv__(self, other):
            return self._get_current_object() // other

        def __mod__(self, other):
            return self._get_current_object() % other

        def __divmod__(self, other):
            return self._get_current_object().__divmod__(other)

        def __pow__(self, other):
            return self._get_current_object() ** other

        def __lshift__(self, other):
            return self._get_current_object() << other

        def __rshift__(self, other):
            return self._get_current_object() >> other

        def __and__(self, other):
            return self._get_current_object() & other

        def __xor__(self, other):
            return self._get_current_object() ^ other

        def __or__(self, other):
            return self._get_current_object() | other

        def __div__(self, other):
            return self._get_current_object().__div__(other)

        def __truediv__(self, other):
            return self._get_current_object().__truediv__(other)

        def __neg__(self):
            return -(self._get_current_object())

        def __pos__(self):
            return +(self._get_current_object())

        def __abs__(self):
            return abs(self._get_current_object())

        def __invert__(self):
            return ~(self._get_current_object())

        def __complex__(self):
            return complex(self._get_current_object())

        def __int__(self):
            return int(self._get_current_object())

        def __float__(self):
            return float(self._get_current_object())

        def __oct__(self):
            return oct(self._get_current_object())

        def __hex__(self):
            return hex(self._get_current_object())

        def __index__(self):
            return self._get_current_object().__index__()

        def __coerce__(self, other):
            return self._get_current_object().__coerce__(other)

        def __enter__(self):
            return self._get_current_object().__enter__()

        def __exit__(self, *a, **kw):
            return self._get_current_object().__exit__(*a, **kw)

        def __reduce__(self):
            return self._get_current_object().__reduce__()

        async def __aenter__(self):
            return await self._get_current_object().__aenter__()

        async def __aexit__(self, *a, **kw):
            return await self._get_current_object().__aexit__(*a, **kw)


class PartialProxy(Proxy[_R]):
    __slots__ = ()

    def __call__(self, *a, **kw):
        return self._get_current_object(*a, **kw)


class PromiseProxy(Proxy[_R]):
    """Proxy that evaluates object once.

    :class:`Proxy` will evaluate the object each time, while the
    promise will only evaluate it once.
    """

    __slots__ = "__pending", "__thing"

    def _get_current_object(self) -> _R:
        try:
            return _object_getattribute(self, "_PromiseProxy__thing")
        except AttributeError:
            return self.__evaluate__()

    def __then__(self, fun, *args, **kwargs):
        if self.__evaluated__():
            return fun(*args, **kwargs)
        from collections import deque

        try:
            pending = _object_getattribute(self, "_PromiseProxy__pending")
        except AttributeError:
            pending = None
        if pending is None:
            pending = deque()
            _object_setattr(self, "_PromiseProxy__pending", pending)
        pending.append((fun, args, kwargs))

    def __evaluated__(self):
        try:
            _object_getattribute(self, "_PromiseProxy__thing")
        except AttributeError:
            return False
        return True

    def __maybe_evaluate__(self) -> _R:
        return self._get_current_object()

    def __reset_promise__(self) -> None:
        try:
            _object_delattr(self, "_PromiseProxy__thing")
        except AttributeError:
            pass

    def __evaluate__(self, _clean: t.Union[bool, abc.Sequence[str]] = None) -> _R:
        try:
            thing = Proxy._get_current_object(self)
        except Exception:
            raise
        else:
            _object_setattr(self, "_PromiseProxy__thing", thing)
            # if _clean is True:
            #     _clean = ("_Proxy__local", "_Proxy__args", "_Proxy__kwargs")
            # for attr in _clean or ():
            #     try:
            #         _object_delattr(self, attr)
            #     except AttributeError:  # pragma: no cover
            #         # May mask errors so ignore
            #         pass
            try:
                pending = _object_getattribute(self, "_PromiseProxy__pending")
            except AttributeError:
                pass
            else:
                # try:
                while pending:
                    fun, args, kwargs = pending.popleft()
                    fun(*args, **kwargs)
                # finally:
                #     try:
                #         _object_delattr(self, "_PromiseProxy__pending")
                #     except AttributeError:  # pragma: no cover
                #         pass
            return thing

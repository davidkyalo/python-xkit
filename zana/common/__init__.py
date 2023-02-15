import operator
import sys
import typing as t
from collections import abc
from functools import reduce
from importlib import import_module
from itertools import chain
from operator import attrgetter
from threading import RLock

import attr
from typing_extensions import ParamSpec, Self

from zana.types import NotSet
from zana.types.collections import FrozenDict

# _P = ParamSpec("_P")
_R = t.TypeVar("_R")
_T = t.TypeVar("_T")
_KT = t.TypeVar("_KT")
_VT = t.TypeVar("_VT")
_T_Co = t.TypeVar("_T_Co", covariant=True)

_P = ParamSpec("_P")


@attr.define(slots=True, weakref_slot=True, hash=True, cache_hash=True)
class Callback(t.Generic[_R]):
    func: abc.Callable[[t.Any], _R]
    args: tuple = attr.ib(default=(), converter=tuple)
    kwargs: dict = attr.ib(default=FrozenDict(), converter=FrozenDict)

    def __iter__(self):
        yield self.func
        yield self.args
        yield self.kwargs

    def __call__(self, /, *args, **kwds):
        return self.func(*args, *self.args, **self.kwargs | kwds)

    @property
    def __wrapped__(self):
        return self.func

    def deconstruct(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}", [
            self.func,
            self.args,
            self.kwargs,
        ]


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
        self: Self,
        getter: abc.Callable[..., _R] = None,
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


def pipe(pipes, /, *args, **kwargs):
    """
    Pipes values through given pipes.

    When called on a value, it runs all wrapped callable, returning the
    *last* value.

    Type annotations will be inferred from the wrapped callables', if
    they have any.

    :param pipes: A sequence of callables.
    """
    return pipeline(pipes)(*args, **kwargs)


def _to_slice(val):
    if isinstance(val, slice):
        return val
    elif isinstance(val, int):
        return slice(val, (val + 1) or None)
    elif val is not None:
        return slice(*val)


_slice_to_tuple = attrgetter("start", "stop", "step")


@attr.define(slots=True, weakref_slot=True, hash=True, cache_hash=True)
class pipeline(abc.Sequence[Callback[_R]], t.Generic[_R]):
    """A callable that composes multiple callables into one.

    When called on a value, it runs all wrapped callable, returning the
    *last* value.

    Type annotations will be inferred from the wrapped callables', if
    they have any.

    :param pipes: A sequence of callables.
    """

    pipes: tuple[Callback[_R], ...] = attr.ib(default=(), converter=tuple)
    args: tuple = attr.ib(default=(), converter=tuple)
    kwargs: dict = attr.ib(default=FrozenDict(), converter=FrozenDict)
    tap: int | slice | tuple[int, int, int] = attr.ib(
        default=_to_slice(0), converter=_to_slice, cmp=_slice_to_tuple
    )

    def __call__(self, /, *args, **kwds) -> _R:
        pre, args, kwds, pipes = args[:1], args[1:] + self.args, self.kwargs | kwds, self.pipes
        count = len(self)
        if taps := count and (args or kwds) and self.tap:
            start, stop, _ = taps.indices(count)
            for i, cb in enumerate(pipes):
                if start <= i < stop:
                    pre = (cb(*pre, *args, **kwds),)
                else:
                    pre = (cb(*pre),)
        elif count:
            for cb in pipes:
                pre = (cb(*pre),)
        elif not pre:
            pre = (None,)
        return pre[0]

    @property
    def __wrapped__(self):
        return self.pipes and self[-1]

    def __or__(self, o: abc.Callable):
        if isinstance(o, pipeline):
            return self.__class__((self, o), tap=slice(None))
        elif callable(o):
            return self.evolve(pipes=self.pipes + (o,))
        else:
            return self.evolve(pipes=chain(self.pipes, o))

    def __ror__(self, o: abc.Callable):
        if isinstance(o, pipeline):
            return self.__class__((o, self), tap=slice(None))
        elif callable(o):
            return self.evolve(pipes=(o,) + self.pipes)
        else:
            return self.evolve(pipes=chain(o, self.pipes))

    def __contains__(self, o):
        return o in self.pipes

    def __iter__(self):
        return iter(self.pipes)

    def __reversed__(self, o):
        return reversed(self.pipes)

    def __bool__(self):
        return True

    def __len__(self):
        return len(self.pipes)

    @t.overload
    def __getitem__(self, key: slice) -> Self:
        ...

    @t.overload
    def __getitem__(self, key: int) -> Callback[_R]:
        ...

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.evolve(pipes=self.pipes[key])
        return self.pipes[key]

    def deconstruct(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}", [
            self.pipes,
            self.args,
            self.kwargs,
            _slice_to_tuple(self.tap),
        ]

    if t.TYPE_CHECKING:

        def evolve(
            self,
            *,
            pipes: abc.Iterable[Callback[_R]] = None,
            args: tuple = None,
            kwargs: dict = None,
            tap: int | slice | tuple[int, int, int] = None,
            **kwds,
        ) -> Self:
            ...

    else:
        evolve: t.ClassVar = attr.evolve


def kw_apply(func, kwds: abc.Mapping[str, _T] | abc.Iterable[tuple[str, _T]]):
    """Call given function with keyword arguments"""
    return func(**(kwds if isinstance(kwds, (dict, abc.Mapping)) else dict(kwds)))


def apply(
    func,
    args: abc.Iterable[_VT] = (),
    kwargs: abc.Mapping[str, _VT] | abc.Iterable[tuple[str, _VT]] = None,
):
    """Call given function with both positional and keyword arguments"""
    if kwargs is None:
        return func(*args)
    else:
        return func(*args, **(kwargs if isinstance(kwargs, (dict, abc.Mapping)) else dict(kwargs)))


if sys.version_info < (3, 11):

    def call(obj: abc.Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
        return obj(*args, **kwargs)

else:
    call = operator.call


def kwarg_map(func, it: abc.Iterable[abc.Mapping[str, _T] | abc.Iterable[tuple[str, _T]]]):
    """Like `itertools.starmap` but for keyword arguments"""
    for x in it:
        yield func(**(x if isinstance(x, (dict, abc.Mapping)) else dict(x)))


def arg_kwarg_map(
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

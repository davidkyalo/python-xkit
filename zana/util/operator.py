import operator
import sys
import typing as t
from collections import abc
from itertools import chain

import attr
from typing_extensions import ParamSpec, Self

_R = t.TypeVar("_R")
_T = t.TypeVar("_T")
_T_Co = t.TypeVar("_T_Co", covariant=True)
_T_Except = t.TypeVar("_T_Except", bound=BaseException, covariant=True)
_T_Raise = t.TypeVar("_T_Raise", bound=BaseException, covariant=True)

_P = ParamSpec("_P")
_object_new = object.__new__


def apply(
    obj: abc.Callable[_P, _R],
    /,
    args: abc.Iterable[t.Any] = (),
    kwargs: abc.Mapping[str, t.Any] = None,
):
    """Same as `obj(*args, **kwargs)`"""
    if kwargs is None:
        return obj(*args)
    else:
        return obj(*args, **kwargs)


def kapply(obj: abc.Callable[_P, _R], /, kwargs: abc.Mapping[str, t.Any]):
    """Same as `obj(**kwargs)`"""
    return obj(**kwargs)


if sys.version_info < (3, 11):

    def call(obj: abc.Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Same as `obj(*args, **kwargs)`"""
        return obj(*args, **kwargs)

else:
    call = operator.call


def identity(obj: _T_Co = None, /, *a, **kw) -> _T_Co:
    """Same as `obj`. A Noop operator. Returns the first positional argument."""
    return obj


@t.overload
def noop(obj: Self = None, /, *a, **kw) -> _T_Co:
    ...


noop = identity


def finalize(
    obj: abc.Callable[_P, _R],
    finalizer: abc.Callable,
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
):
    """Same as:
    try:
        rv = obj(*args, **kwargs)
        return rv
    except catch as e:
        finalizer(rv)
    """

    r_args = ()
    try:
        r_args = (obj(*args, **kwargs),)
        return r_args[0]
    finally:
        finalizer(*r_args)


def suppress(
    obj: abc.Callable[_P, _R],
    catch: type[_T_Except] | tuple[_T_Except] = Exception,
    default: _T_Co = None,
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
):
    """Same as:
    try:
        return obj(*args, **kwargs)
    except catch as e:
        return default
    """
    try:
        return obj(*args, **kwargs)
    except catch as e:
        return default


def throw(
    obj: abc.Callable[_P, _R],
    throw: type[_T_Raise] | _T_Raise | abc.Callable[[_T_Except], _T_Raise],
    catch: type[_T_Except] | tuple[_T_Except] = Exception,
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
):
    """Same as:
    try:
        return obj(*args, **kwargs)
    except catch as e:
        raise (throw() if callable(throw) else throw) from e
    """
    try:
        return obj(*args, **kwargs)
    except catch as e:
        raise (throw() if callable(throw) else throw) from e


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


_slice_to_tuple = operator.attrgetter("start", "stop", "step")


def _to_slice(val):
    if isinstance(val, slice):
        return val
    elif isinstance(val, int):
        return slice(val, (val + 1) or None)
    elif val is not None:
        return slice(*val)


def _frozen_dict(*a):
    from zana.types.collections import FrozenDict

    return FrozenDict(*a)


_attr_define = attr.define
if not t.TYPE_CHECKING:

    def _attr_define(*a, init=False, **kw):
        return attr.define(*a, init=init, **kw)


@_attr_define(slots=True, weakref_slot=True, hash=True)
class Callback(t.Generic[_P, _R]):
    func: abc.Callable[_P, _R] = attr.field(validator=attr.validators.is_callable())
    args: tuple = attr.field(default=(), converter=tuple)
    kwargs: dict = attr.field(factory=_frozen_dict, converter=_frozen_dict)

    @t.overload
    @classmethod
    def define(cls, callback: abc.Callable[_P, _R], args: tuple = ..., kwargs: dict = ...) -> Self:
        ...

    @classmethod
    def define(cls, *a, **kw):
        self = _object_new(cls)
        self.__attr_init__(*a, **kw)
        return self

    def __new__(cls: type[Self], func, /, *args, **kwargs) -> Self:
        return cls.define(func, args, kwargs)

    def __iter__(self):
        yield self.func
        yield self.args
        yield self.kwargs

    def __call__(self, /, *args: _P.args, **kwds: _P.kwargs):
        return self.func(*args, *self.args, **self.kwargs | kwds)

    def __reduce__(self):
        return self.__class__.define, (self.func, self.args, self.kwargs)

    @property
    def __wrapped__(self):
        return self.func

    def deconstruct(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}", [
            self.func,
            self.args,
            self.kwargs,
        ]


@attr.define(slots=True, weakref_slot=True, hash=True, cache_hash=True)
class pipeline(abc.Sequence[Callback[_P, _R]], t.Generic[_P, _R]):
    """A callable that composes multiple callables into one.

    When called on a value, it runs all wrapped callable, returning the
    *last* value.

    Type annotations will be inferred from the wrapped callables', if
    they have any.

    :param pipes: A sequence of callables.
    """

    pipes: tuple[Callback[_P, _R], ...] = attr.ib(default=(), converter=tuple)
    args: tuple = attr.ib(default=(), converter=tuple)
    kwargs: dict = attr.ib(factory=_frozen_dict, converter=_frozen_dict)
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
    def __getitem__(self, key: int) -> Callback[_P, _R]:
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
            pipes: abc.Iterable[Callback[_P, _R]] = None,
            args: tuple = None,
            kwargs: dict = None,
            tap: int | slice | tuple[int, int, int] = None,
            **kwds,
        ) -> Self:
            ...

    else:
        evolve: t.ClassVar = attr.evolve

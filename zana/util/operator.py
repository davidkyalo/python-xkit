import builtins
import operator
import sys
import typing as t
from builtins import delattr, getattr, setattr
from collections import abc
from functools import partial
from itertools import chain
from operator import (
    abs,
    add,
    and_,
    attrgetter,
    concat,
    contains,
    countOf,
    delitem,
    eq,
    floordiv,
    ge,
    getitem,
    gt,
    iadd,
    iand,
    iconcat,
    ifloordiv,
    ilshift,
    imatmul,
    imod,
    imul,
    index,
    indexOf,
    inv,
    invert,
    ior,
    ipow,
    irshift,
    is_,
    is_not,
    isub,
    itemgetter,
    itruediv,
    ixor,
    le,
    length_hint,
    lshift,
    lt,
    matmul,
    mod,
    mul,
    ne,
    neg,
    not_,
    or_,
    pos,
    pow,
    rshift,
    setitem,
    sub,
    truediv,
    truth,
    xor,
)

import attr
from typing_extensions import ParamSpec, Self

__all__ = [
    "getattr",
    "setattr",
    "delattr",
    "abs",
    "add",
    "and_",
    "attrgetter",
    "concat",
    "contains",
    "countOf",
    "delitem",
    "eq",
    "floordiv",
    "ge",
    "getitem",
    "gt",
    "iadd",
    "iand",
    "iconcat",
    "ifloordiv",
    "ilshift",
    "imatmul",
    "imod",
    "imul",
    "index",
    "indexOf",
    "inv",
    "invert",
    "ior",
    "ipow",
    "irshift",
    "is_",
    "is_not",
    "isub",
    "itemgetter",
    "itruediv",
    "ixor",
    "le",
    "length_hint",
    "lshift",
    "lt",
    "matmul",
    "methodcaller",
    "mod",
    "mul",
    "ne",
    "neg",
    "not_",
    "or_",
    "pos",
    "pow",
    "rshift",
    "setitem",
    "sub",
    "truediv",
    "truth",
    "xor",
]

_R = t.TypeVar("_R")
_T = t.TypeVar("_T")
_T_Co = t.TypeVar("_T_Co", covariant=True)
_T_Except = t.TypeVar("_T_Except", bound=BaseException, covariant=True)
_T_Raise = t.TypeVar("_T_Raise", bound=BaseException, covariant=True)

_P = ParamSpec("_P")
_object_new = object.__new__

getattr = builtins.getattr
delattr = builtins.delattr
setattr = builtins.setattr


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


def none(obj=None, /, *a, **kw):
    """Returns `None` whenever called."""
    return None


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

    Same As:
        it = iter(pipes)
        obj = next(it)(*args, **kwargs)
        for pipe in it:
            obj = pipe(obj)
        return obj

    When called on a value, it runs all wrapped callable, returning the
    *last* value.

    Type annotations will be inferred from the wrapped callables', if
    they have any.

    :param pipes: A sequence of callables.
    """
    return pipeline(pipes)(*args, **kwargs)


def tap(pipes, /, *args, **kwargs):
    """
    Pipes first argument (`*args[:1]`) through `pipes[:-1]` then calls `pipes[-1]`
    with the `result` and `*arg[1:]` and `**kwargs`.

    Same As:
        pipes = tuple(pipes)
        for pipe in pipes[:-1]:
            obj = pipe(*args[:1])
            args = obj, *args[1:],

        return pipes[-1](*args, **kwargs)
    """
    return pipeline(pipes, tap=-1)(*args, **kwargs)


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
class callback(t.Generic[_P, _R]):
    func: abc.Callable[_P, _R] = attr.field(default=call, validator=attr.validators.is_callable())
    args: tuple = attr.field(default=(), converter=tuple)
    kwargs: dict = attr.field(factory=_frozen_dict, converter=_frozen_dict)

    def __new__(
        cls: type[Self], func: abc.Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> Self:
        self = _object_new(cls)
        self.__attrs_init__(func, *(x for x in (args, kwargs) if x))
        return self

    def __call__(self, /, *args: _P.args, **kwargs: _P.kwargs):
        return self.func(*args, *self.args, **self.kwargs | kwargs)

    def __reduce__(self):
        if kwargs := self.kwargs:
            return partial(self.__class__, self.func, **kwargs), self.args
        return self.__class__, (self.func,) + self.args

    @property
    def __wrapped__(self):
        return self.func

    def deconstruct(self):
        return (
            f"{self.__class__.__module__}.{self.__class__.__name__}",
            [self.func, *self.args],
            self.kwargs,
        )


@_attr_define(slots=True, weakref_slot=True, hash=True)
class caller(t.Generic[_P, _R]):
    args: tuple = attr.field(default=(), converter=tuple)
    kwargs: dict = attr.field(factory=_frozen_dict, converter=_frozen_dict)

    def __new__(cls: type[Self], /, *args: _P.args, **kwargs: _P.kwargs) -> Self:
        self = _object_new(cls)
        self.__attrs_init__(*(x for x in (args, kwargs) if x))
        return self

    def __call__(self, obj: abc.Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs):
        return obj(*self.args, *args, **self.kwargs | kwargs)

    def __reduce__(self):
        if kwargs := self.kwargs:
            return partial(self.__class__, **kwargs), self.args
        return self.__class__, self.args

    def deconstruct(self):
        return (
            f"{self.__class__.__module__}.{self.__class__.__name__}",
            [self.function, *self.args],
            self.kwargs,
        )


@_attr_define(slots=True, weakref_slot=True, hash=True)
class methodcaller(t.Generic[_P, _R]):
    name: str = attr.field(validator=attr.validators.instance_of(str))
    args: tuple = attr.field(default=(), converter=tuple)
    kwargs: dict = attr.field(factory=_frozen_dict, converter=_frozen_dict)

    def __new__(cls: type[Self], name: str, /, *args: _P.args, **kwargs: _P.kwargs) -> Self:
        self = _object_new(cls)
        self.__attrs_init__(name, *(x for x in (args, kwargs) if x))
        return self

    def __call__(self, obj: object, /, *args: _P.args, **kwargs: _P.kwargs):
        return getattr(obj, self.name)(*self.args, *args, **self.kwargs | kwargs)

    def __reduce__(self):
        if kwargs := self.kwargs:
            return partial(self.__class__, self.name, **kwargs), self.args
        return self.__class__, (self.name,) + self.args

    def deconstruct(self):
        return (
            f"{self.__class__.__module__}.{self.__class__.__name__}",
            [self.name, *self.args],
            self.kwargs,
        )


@attr.define(slots=True, weakref_slot=True, hash=True, cache_hash=True)
class pipeline(abc.Sequence[callback[_P, _R]], t.Generic[_P, _R]):
    """A callable that composes multiple callables into one.

    When called on a value, it runs all wrapped callable, returning the
    *last* value.

    Type annotations will be inferred from the wrapped callables', if
    they have any.

    :param pipes: A sequence of callables.
    """

    pipes: tuple[callback[_P, _R], ...] = attr.ib(default=(), converter=tuple)
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
    def __getitem__(self, key: int) -> callback[_P, _R]:
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
            pipes: abc.Iterable[callback[_P, _R]] = None,
            args: tuple = None,
            kwargs: dict = None,
            tap: int | slice | tuple[int, int, int] = None,
            **kwds,
        ) -> Self:
            ...

    else:
        evolve: t.ClassVar = attr.evolve

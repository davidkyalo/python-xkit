import builtins
import keyword
import operator as builtin_operator
import typing as t
import weakref
from abc import abstractmethod
from collections import abc
from functools import partial, reduce, singledispatchmethod, update_wrapper
from itertools import chain
from logging import getLogger
from operator import attrgetter, methodcaller
from pickle import TRUE
from types import GenericAlias, new_class

import attr
from typing_extensions import Self

from zana.types import Interface
from zana.types.collections import (
    Atomic,
    ChainMap,
    Composite,
    DefaultKeyDict,
    FallbackDict,
    FrozenDict,
    UserTuple,
)
from zana.types.enums import IntEnum
from zana.util import cached_attr, operator

_T = t.TypeVar("_T")
_T_Co = t.TypeVar("_T_Co", covariant=True)


_T_Arg = t.TypeVar("_T_Arg")
_T_Kwarg = t.TypeVar("_T_Kwarg", bound=FrozenDict, covariant=True)
_T_Key = t.TypeVar("_T_Key", slice, int, str, t.SupportsIndex, abc.Hashable)
_T_Fn = t.TypeVar("_T_Fn", bound=abc.Callable)
_T_Attr = t.TypeVar("_T_Attr", bound=str)


logger = getLogger(__name__)
_object_new = object.__new__
_empty_dict = FrozenDict()

_repr_str = attr.converters.pipe(str, repr)

_operator_modules = (operator, builtin_operator)

_notset = t.TypeVar("_notset")


def _field_transformer(fn=None):
    def hook(cls: type, fields: list[attr.Attribute]):
        fields = fn(cls, fields) if fn else fields
        cls.__positional_attrs__ = {
            f.alias or f.name.lstrip("_"): f.name
            for f in fields
            if f.init and not f.kw_only and f.metadata.get("var_pas")
        }
        return fields

    return hook


_define = t.overload(attr.s)

if not t.TYPE_CHECKING:

    def _define(*a, no_init=None, field_transformer=None, **kw):
        return attr.define(
            *a,
            **dict(
                frozen=True,
                init=not no_init,
                cache_hash=True,
                field_transformer=_field_transformer(field_transformer),
            )
            | kw,
        )


class OperationType(IntEnum):
    UNARY = 1, "Unary (1 operant)"
    BINARY = 2, "Binary (2 operants)"
    TERNARY = 3, "Ternary (3 operants)"
    GENERIC = 5, "Generic (unknown operants)"

    def __bool__(self) -> bool:
        return True


UNARY_OP = OperationType.UNARY
BINARY_OP = OperationType.BINARY
TERNARY_OP = OperationType.TERNARY
GENERIC_OP = OperationType.GENERIC


def compose(*signatures: "Composable"):
    return +Composition(map(ensure_composable, signatures))


def maybe_composable(obj) -> "Operation":
    return obj.__compose__() if isinstance(obj, Composable) else obj


def ensure_composable(obj) -> "Operation":
    if isinstance(obj, Composable):
        return obj.__compose__()
    elif callable(obj):
        return +Callback(obj)
    else:
        return +Ref(obj)


def operation(name: str, /, *args, **kwargs):
    return ops[name].define(*args, **kwargs)


class SignatureError(Exception):
    pass


class AttributeSignatureError(SignatureError, AttributeError):
    ...


class LookupSignatureError(SignatureError, LookupError):
    ...


class KeySignatureError(LookupSignatureError, KeyError):
    ...


class IndexSignatureError(LookupSignatureError, IndexError):
    ...


@_define()
class Operation(t.Generic[_T_Co]):
    __class_getitem__ = classmethod(GenericAlias)
    __positional_attrs__: t.ClassVar[dict[str, str]] = ...

    lazy: t.ClassVar[bool] = False
    # operator: t.ClassVar["OperatorInfo[Self]"] = attr.field(repr=attrgetter("name"))
    operator: t.ClassVar["OperatorInfo[Self]"] = None
    callback: t.ClassVar[abc.Callable] = None
    __abstract__: t.ClassVar[abc.Callable] = True
    __final__: t.ClassVar[abc.Callable] = False
    terminal: t.ClassVar[bool] = False

    # callback: t.ClassVar[abc.Callable] = attr.ib(
    #     init=False,
    #     cmp=False,
    #     repr=False,
    #     default=attr.Factory(attrgetter("operator.callback"), True),
    # )

    @property
    def name(self):
        return self.operator.name

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.__abstract__ = cls.__dict__.get("__abstract__", False)

    def __pos__(self):
        return self

    def __compose__(self) -> "Operation[_T_Co]":
        return +self

    # @property
    # def atomic(self) -> tuple[Self, ...]:
    #     return (self,)

    @abstractmethod
    def __call__(self, *a, **kw) -> _T_Co:
        raise NotImplementedError(f"{self!r} getter not supported")

    def __add__(self, o) -> Self:
        return NotImplemented

    def __or__(self, o):
        if isinstance(o, Operation):
            return +Composition((self, o))
        elif isinstance(o, abc.Iterable):
            return +Composition(chain((self,), o))
        return NotImplemented

    def __ror__(self, o):
        if isinstance(o, abc.Iterable):
            return +Composition(chain(o, (self,)))
        return NotImplemented

    def evolve(self, *args, **kwds):
        args = dict(zip(self.__positional_attrs__, args)) if args else _empty_dict
        return attr.evolve(self, **args, **kwds)

    if t.TYPE_CHECKING:
        evolve: t.ClassVar[type[Self]]

    def deconstruct(self):
        factory = operation
        path = f"{factory.__module__}.{factory.__name__}"
        fields: list[attr.Attribute] = attr.fields(self.__class__)
        args, kwargs, seen = [], {}, set()

        for field in fields:
            seen |= {field.name, field.alias}
            if field.init:
                default, value = field.default, getattr(self, field.name)
                if not default is attr.NOTHING:
                    if isinstance(default, attr.Factory):
                        default = default.factory(*((self,) if default.takes_self else ()))
                    if value == default:
                        continue
                if field.kw_only:
                    kwargs[field.alias or field.name] = value
                else:
                    args.append(value)

        name, lazy = self.name, self.lazy

        if "name" not in seen:
            args.insert(0, name)

        if lazy and "lazy" not in seen:
            kwargs["lazy"] = True

        return path, args, kwargs


TSignatureOrIterable = Operation | abc.Iterable


@_define()
class LazyOperation(Operation[_T_Co]):
    __abstract__ = True
    lazy: t.ClassVar[t.Literal[True]] = True
    source: Operation = attr.ib()


@_define()
class UnaryOperation(Operation[_T_Co]):
    __abstract__ = True

    def __call__(self, obj, /):
        return self.callback(obj)


@_define()
class BinaryOperation(Operation[_T_Co]):
    __abstract__ = True
    operant: t.Any = attr.ib()

    def __call__(self, obj, /):
        return self.callback(obj, self.operant)


@_define()
class TernaryOperation(Operation[_T_Co]):
    __abstract__ = True
    operants: tuple[t.Any] = attr.ib(converter=tuple, validator=attr.validators.max_len(2))

    def __call__(self, obj, /, *args):
        return self.callback(obj, *self.operants, *args)


@_define()
class GenericOperation(Operation[_T_Co]):
    __abstract__ = True
    args: tuple[t.Any] = attr.ib(converter=tuple)
    kwargs: tuple[t.Any] = attr.ib(default=FrozenDict(), converter=FrozenDict)
    bind: int | bool = attr.ib(default=True, converter=int, kw_only=True)

    def __call__(self, *a, **kw):
        if bind := a and +self.bind or ():
            bind, args = a[:bind], self.args + a[bind:]
        else:
            args = chain(self.args, a)
        return self.callback(*bind, *args, **self.kwargs | kw)


@_define()
class LazyUnaryOperation(LazyOperation[_T_Co]):
    __abstract__ = True
    source: Operation = attr.ib()

    def __call__(self, obj, /):
        return self.callback(self.source(obj))


@_define()
class LazyBinaryOperation(LazyOperation[_T_Co]):
    __abstract__ = True
    source: Operation = attr.ib()
    operant: Operation = attr.ib()

    def __call__(self, obj, /):
        return self.callback(self.source(obj), self.operant(obj))


@_define()
class LazyTernaryOperation(LazyOperation[_T_Co]):
    __abstract__ = True
    source: Operation = attr.ib()
    operants: tuple[t.Any] = attr.ib(converter=tuple, validator=attr.validators.max_len(2))

    def __call__(self, obj, /, *args):
        return self.callback(self.source(obj), *(op(obj) for op in self.operants), *args)


@_define()
class LazyGenericOperation(LazyOperation[_T_Co]):
    __abstract__ = True
    source: Operation = attr.ib()
    args: tuple[Operation] = attr.ib(converter=tuple)
    kwargs: dict[str, Operation] = attr.ib(default=FrozenDict(), converter=FrozenDict)
    bind: int | bool = attr.ib(default=True, converter=int, kw_only=True)

    def __call__(self, obj, /, *a, **kw):
        src, args = self.source(obj), (op(obj) for op in self.args)
        if bind := +self.bind or ():
            bind, args = (src,), chain(args, a)
        else:
            args = chain(args, (src,), a)

        kwargs = {k: op(obj) for k, op in self.kwargs.items() if k not in kw}
        return self.callback(*bind, *args, **kwargs, **kw)


@_define
class Ref(Operation[_T_Co]):
    obj: _T_Co
    __final__ = True

    def __call__(self, obj=None, /) -> _T_Co:
        return self.obj

    def __add__(self, o: object):
        if isinstance(self.__class__, Ref):
            return o
        return NotImplemented

    def __radd__(self, o: object):
        if isinstance(o, Operation):
            return self
        return NotImplemented

    def deconstruct(self):
        path = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return path, [self.obj], {}


@_define
class Identity(UnaryOperation[_T_Co]):
    __final__ = True

    def __call__(self, obj: _T_Co, /) -> _T_Co:
        return obj

    def __add__(self, o: object):
        if isinstance(o, Operation):
            return +o
        return NotImplemented

    __radd__ = __add__

    def deconstruct(self):
        path = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return path, [], {}


@_define
class LazyIdentity(LazyUnaryOperation[_T_Co]):
    __final__ = True

    def __call__(self, obj: _T_Co, /) -> _T_Co:
        return self.source(obj)


if True:

    class AttrPath(UserTuple[str]):
        __slots__ = ()

        def __new__(cls: type[Self], path: abc.Iterable[str] = ()) -> Self:
            if path.__class__ is cls:
                return path
            elif isinstance(path, str):
                it = path.split(".")
            else:
                it = (
                    at
                    for it in (path or ())
                    for at in (it.split(".") if isinstance(it, str) else (it,))
                )
            return cls.construct(it)

        def __str__(self) -> str:
            return ".".join(map(str, self))

    @_define()
    class Attr(BinaryOperation[_T_Co]):
        path: AttrPath = attr.ib(converter=AttrPath, repr=_repr_str)

        def __str__(self) -> str:
            return f".{self.path}"

        def __call__(self, obj) -> _T_Co:
            __tracebackhide__ = True
            try:
                for name in self.path:
                    obj = getattr(obj, name)
                return obj
            except AttributeError as e:
                raise AttributeSignatureError(self) from e

        def set(self, obj, val):
            __tracebackhide__ = True
            path = self.path
            try:
                for name in path[:-1]:
                    obj = getattr(obj, name)
                setattr(obj, path[-1], val)
            except AttributeError as e:
                raise AttributeSignatureError(self) from e

        def delete(self, obj):
            __tracebackhide__ = True
            path = self.path
            try:
                for name in path[:-1]:
                    obj = getattr(obj, name)
                delattr(obj, path[-1])
            except AttributeError as e:
                raise AttributeSignatureError(self) from e

        def __add__(self, o: Self):
            if not self.__class__ is o.__class__:
                return NotImplemented
            lhs, rhs = self.path, o.path
            path = lhs + rhs
            return self.evolve(path)

    # @_define()
    # class AttrMutator(Signature[_T_Co]):
    #     name: str = attr.ib()
    #     source: Attr = attr.ib(default=None, repr=str)

    #     def __str__(self) -> str:
    #         return f".{'.'.join(map(str, filter(None, (self.source, self.name))))}"

    #     def resolve(self, obj):
    #         return obj

    #     def __attrs_post_init__(self):
    #         if (src := self.source) is not None:
    #             self.__dict__.setdefault('resolve', src)

    #     # def __add__(self, o: Self):
    #     #     if not self.__class__ is o.__class__:
    #     #         return NotImplemented
    #     #     lhs, rhs = self.path, o.path
    #     #     path = lhs + rhs
    #     #     return self.evolve(path)

    # @_define()
    # class AttrGetter(AttrMutator[_T_Co]):
    #     def __call__(self, obj) -> _T_Co:
    #         __tracebackhide__ = True
    #         src = self.resolve(obj)
    #         try:
    #             return getattr(src, self.name)
    #         except AttributeError as e:
    #             raise AttributeSignatureError(self) from e

    # @_define()
    # class AttrSetter(AttrMutator[_T_Co]):
    #     def __call__(self, obj, value) -> _T_Co:
    #         __tracebackhide__ = True

    #         src = self.source
    #         src = src and src(obj)

    #         try:
    #             return getattr(src, self.name)
    #         except AttributeError as e:
    #             raise AttributeSignatureError(self) from e

    # @_define()
    # class AttrDeleter(AttrMutator[_T_Co]):
    #     def __call__(self, obj) -> _T_Co:
    #         __tracebackhide__ = True
    #         path, name = self.path, self.name
    #         try:
    #             for name in path:
    #                 obj = getattr(obj, name)
    #             delattr(obj, name)
    #         except AttributeError as e:
    #             raise AttributeSignatureError(self) from e

    _lookup_errors = {
        LookupError: LookupSignatureError,
        KeyError: KeySignatureError,
        IndexError: IndexSignatureError,
    }

    class SubscriptPath(UserTuple[_T_Key]):
        __slots__ = ()

        def __str__(self) -> str:
            return f"[{']['.join(map(str, self))}]"

    class _SubscriptMixin:
        path: tuple[_T_Key]

        def __str__(self) -> str:
            return f"{self.path}"

        def __call__(self, obj) -> _T_Co:
            __tracebackhide__ = True
            try:
                for arg in self.path:
                    obj = obj[arg]
                return obj
            except LookupError as e:
                raise _lookup_errors.get(e.__class__)(self) from e

        def set(self, obj, val):
            __tracebackhide__ = True
            args = self.path
            try:
                for arg in args[:-1]:
                    obj = obj[arg]
                obj[args[-1]] = val
            except LookupError as e:
                raise _lookup_errors.get(e.__class__)(self) from e

        def delete(self, obj):
            __tracebackhide__ = True
            args = self.path
            try:
                for arg in args[:-1]:
                    obj = obj[arg]
                del obj[args[-1]]
            except LookupError as e:
                raise _lookup_errors.get(e.__class__)(self) from e

        def __add__(self, o: Self):
            if not self.__class__ is o.__class__:
                return NotImplemented
            lhs, rhs = self.path, o.path
            path = lhs + rhs
            return self.evolve(path)

    @_define()
    class Subscript(_SubscriptMixin, Operation[_T_Co]):
        path: SubscriptPath = attr.ib(converter=SubscriptPath, repr=_repr_str)

    _slice_tuple = attrgetter("start", "stop", "step")

    def _to_slice(val):
        __tracebackhide__ = True
        return val if isinstance(val, slice) else slice(*val)

    class SlicePath(UserTuple[slice]):
        __slots__ = ()

        def __new__(cls: type[Self], path: abc.Iterable[str] = ()) -> Self:
            if path.__class__ is cls:
                return path
            return cls.construct(map(_to_slice, path))

        def __str__(self) -> str:
            return "".join(f"[{s.start}:{s.stop}:{s.step}]".replace("None", "") for s in self)

        def astuples(self):
            return tuple(map(_slice_tuple, self))

    @_define()
    class Slice(Subscript[_T_Co]):
        path: SlicePath = attr.ib(converter=SlicePath, repr=_repr_str, cmp=False)
        raw_path: tuple[tuple[_T_Key, _T_Key, _T_Key]] = attr.ib(init=False, repr=False, cmp=True)

        @raw_path.default
        def _init_raw_path(self):
            return self.path.astuples()

        def deconstruct(self):
            path, args, kwargs = super().deconstruct()
            return path, [self.raw_path], {}

    @_define()
    class Call(Operation[_T_Co]):
        args: tuple[t.Any] = attr.ib(default=(), converter=tuple)
        kwargs: FrozenDict[str, t.Any] = attr.ib(default=FrozenDict(), converter=FrozenDict)

        def __str__(self) -> str:
            params = map(repr, self.args), (f"{k}={v!r}" for k, v in self.kwargs.items())
            return f"({', '.join(p for ps in params for p in ps)})"

        def __call__(this, self, /, *a, **kw) -> _T_Co:
            return self(*this.args, *a, **this.kwargs | kw)

    @_define()
    class Callback(Operation[_T_Co]):
        func: abc.Callable = attr.ib()
        args: tuple[t.Any] = attr.ib(default=(), converter=tuple)
        kwargs: FrozenDict[str, t.Any] = attr.ib(default=FrozenDict(), converter=FrozenDict)

        @property
        def __wrapped__(self):
            return self.func

        def __str__(self) -> str:
            params = map(repr, self.args), (f"{k}={v!r}" for k, v in self.kwargs.items())
            return f"{self.func.__name__}({', '.join(p for ps in params for p in ps)})"

        def __call__(self, /, *a, **kw):
            return self.func(*self.args, *a, **self.kwargs | kw)


def _iter_compose(seq: abc.Iterable[Operation], composites: type[abc.Iterable[Operation]]):
    it = iter(seq)
    try:
        pre = next(it)
    except StopIteration:
        return

    for obj in it:
        try:
            pre += obj
        except TypeError:
            if composites and isinstance(pre, composites):
                yield from pre
            else:
                yield pre
            pre = obj

    if composites and isinstance(pre, composites):
        yield from pre
    else:
        yield pre


def _join(a: Operation, b: Operation):
    try:
        return (a + b,)
    except TypeError:
        return a, b


class CompositionPath(UserTuple[Operation]):
    __slots__ = ()

    def __new__(cls: type[Self], path: abc.Iterable[str] = ()) -> Self:
        if path.__class__ is cls:
            return path
        return cls.construct(_iter_compose(path, Composition))

    def first_pack(self, __index=slice(1, None)):
        return self.__tuple_getitem__(0), self.__tuple_getitem__(__index)

    def last_pack(self, __index=slice(-1)):
        return self.__tuple_getitem__(__index), self.__tuple_getitem__(-1)

    def __str__(self) -> str:
        return "".join(map(str, self))

    def __add__(self, other: Self) -> Self:
        if not isinstance(other, abc.Iterable):
            return NotImplemented

        other = self.__class__(other)

        if not (self and other):
            return other or self

        (head, lhs), (rhs, tail) = self.last_pack(), other.first_pack()
        return self.construct(head + _join(lhs, rhs) + tail)

    def __radd__(self, other: tuple) -> Self:
        __tracebackhide__ = True
        if isinstance(other, abc.Iterable):
            return self.__class__(other) + self
        return NotImplemented


@_define()
@Composite.register
class Composition(Operation, abc.Sequence[Operation]):
    path: CompositionPath = attr.ib(repr=_repr_str, converter=CompositionPath)

    size: int = attr.ib(repr=False, init=False)
    size.default(len)

    @singledispatchmethod
    def __call__(this, self=_notset, /, *a, **kw) -> abc.Callable[[t.Any], _T]:
        path, pre = this.path, () if self is _notset else (self,)
        for i in range(len(path) - 1):
            pre = (path[i](*pre),)

        return path[-1](*pre, *a, **kw)

    @__call__.register
    def __call__(self, obj: object, /, *a, **kw) -> abc.Callable[[t.Any], _T]:
        path, pre = self.path, () if obj is _notset else (obj,)
        for i in range(len(path) - 1):
            pre = (path[i](*pre),)

        return path[-1](*pre, *a, **kw)

    def set(this, self, /, val) -> abc.Callable[[t.Any], _T]:
        path = this.path
        for i in range(len(path) - 1):
            self = path[i](self)
        return path[-1].set(self, val)

    def delete(this, self, /) -> abc.Callable[[t.Any], _T]:
        path = this.path
        for i in range(len(path) - 1):
            self = path[i](self)
        return path[-1].delete(self)

    def __pos__(self):
        path, size = self.path, self.size
        return Identity() if size == 0 else +path[0] if size == 1 else self

    def __str__(self) -> str:
        return " | ".join(map(repr, self))

    def __len__(self):
        return len(self.path)

    def __contains__(self, x):
        return x in self.path

    @t.overload
    def __getitem__(self, key: int) -> Operation:
        ...

    @t.overload
    def __getitem__(self, key: slice) -> Self:
        ...

    def __getitem__(self, key):
        val = self.path[key]
        if isinstance(key, slice):
            val = self.evolve(val)
        return val

    def __iter__(self):
        return iter(self.path)

    def __reversed__(self):
        return reversed(self.path)

    def __add__(self, o: TSignatureOrIterable):
        if isinstance(o, Composition):
            path = self.path + o.path
        else:
            return NotImplemented
        return +self.evolve(path)

    def __or__(self, o: TSignatureOrIterable):
        if isinstance(o, Composition):
            path = o.path
        elif isinstance(o, Operation):
            path = (o,)
        elif isinstance(o, abc.Iterable):
            path = o
        else:
            return NotImplemented
        return +self.evolve(self.path + path)

    def __ror__(self, o: TSignatureOrIterable):
        if isinstance(o, abc.Iterable):
            return +self.evolve(o + self.path)
        return NotImplemented

    def deconstruct(self):
        path = f"{__name__}.{compose.__name__}"
        return path, list(self), {}


_T_Op = t.TypeVar("_T_Op", bound=Operation, covariant=True)


@_define
class OperatorInfo(t.Generic[_T_Op]):
    name: str = attr.ib(validator=[attr.validators.instance_of(str), attr.validators.min_len(1)])
    identifier: str = attr.ib(init=False, repr=False, cmp=False)
    type: OperationType = attr.ib(
        converter=OperationType,
        cmp=False,
        repr=attr.converters.pipe(attrgetter("name"), repr),
    )
    callback: abc.Callable = attr.ib(
        cmp=False,
        validator=attr.validators.is_callable(),
        repr=attr.converters.pipe(attrgetter("__name__"), repr),
    )
    reversible: bool = attr.ib(kw_only=True, cmp=False)
    inplace: bool = attr.ib(kw_only=True, cmp=False)
    terminal: bool = attr.ib(default=False, kw_only=True, cmp=False)
    magic_name: str = attr.ib(kw_only=True, cmp=False, alias="magic")
    reverse_magic_name: str = attr.ib(kw_only=True, repr=False, cmp=False)

    _impl: t.Type[_T_Op] = attr.ib(kw_only=True, repr=False, cmp=False, alias="impl")
    _lazy_impl: t.Type[_T_Op] = attr.ib(kw_only=True, repr=False, cmp=False, alias="lazy_impl")

    impl: t.Type[_T_Op] = attr.ib(
        init=False,
        cmp=False,
        repr=attr.converters.optional(attr.converters.pipe(attrgetter("__name__"), repr)),
    )
    lazy_impl: t.Type[_T_Op] = attr.ib(
        init=False,
        cmp=False,
        repr=attr.converters.optional(attr.converters.pipe(attrgetter("__name__"), repr)),
    )

    @magic_name.default
    def _init_magic_name(self):
        if (name := self.name) not in _non_trappable_ops:
            return f"__{name.strip('_')}__"

    @reversible.default
    def _init_reversible(self):
        return self.type is BINARY_OP and f"i{self.name}" in _builtin_binary_ops

    @inplace.default
    def _default_inplace(self):
        nm = self.name
        return self.type is BINARY_OP and nm[:1] == "i" and nm[1:] in _builtin_binary_ops

    @reverse_magic_name.default
    def _init_reverse_magic_name(self):
        if magic := self.reversible and self.magic_name:
            val = f"r{magic.lstrip('_')}"
            return f"{'_' * (1+len(magic)-len(val))}{val}"

    @identifier.default
    def _init_identifier(self):
        if (name := self.name) and name.isidentifier():
            if keyword.iskeyword(name):
                name = f"{name}_"
            return name

    @callback.default
    def _init_callback(self):
        name = self.identifier or self.name
        for mod in _operator_modules:
            if hasattr(mod, name):
                return getattr(mod, name)

    @_impl.default
    def _default_impl(self):
        match self.type:
            case OperationType.UNARY:
                return UnaryOperation
            case OperationType.BINARY:
                return BinaryOperation
            case OperationType.TERNARY:
                return TernaryOperation
            case OperationType.GENERIC:
                return GenericOperation

    @_lazy_impl.default
    def _default_lazy_impl(self):
        match self.type:
            case OperationType.UNARY:
                return LazyUnaryOperation
            case OperationType.BINARY:
                return LazyBinaryOperation
            case OperationType.TERNARY:
                return LazyTernaryOperation
            case OperationType.GENERIC:
                return LazyGenericOperation

    @impl.default
    def _init_impl(self):
        return self.make_impl_class(self._impl)

    @lazy_impl.default
    def _init_lazy_impl(self):
        if base := self._lazy_impl:
            return self.make_impl_class(base)

    def make_impl_class(self, base: t.Type[_T_Op]):
        callback, terminal = staticmethod(self.callback), self.terminal
        if base.__final__:
            assert base.operator is None
            assert terminal or not base.terminal
            base.operator, base.callback, base.terminal = self, base.callback or callback, terminal
            return base

        name = f"{self.identifier or base.__name__}_{'lazy_' if base.lazy else ''}operation"
        name = "".join(map(methodcaller("capitalize"), name.split("_")))
        if getattr(base.__call__, "__isabstractmethod__", False):
            __call__ = callback
        else:
            __call__ = base.__call__

        ns = {
            "__slots__": (),
            "__module__": base.__module__,
            "__name__": name,
            "__qualname__": f"{''.join(base.__qualname__.rpartition('.')[:1])}{name}",
            "__final__": True,
            "__call__": __call__,
            "operator": self,
            "callback": callback,
            "terminal": terminal,
        }

        cls = new_class(name, (base,), None, methodcaller("update", ns))
        _define(cls, auto_attribs=False)
        return cls

    def __str__(self) -> str:
        return self.name

    def define(self, *args, lazy=False, **kwargs) -> _T_Op:
        cls = self.lazy_impl if lazy is True else self.impl
        return cls(*args, **kwargs)

    def trap_method(op, reverse: bool = False):
        match (op.type, not not reverse):
            case (OperationType.UNARY, False):

                def method(self: trap):
                    nonlocal op
                    return self.__class__(*self, op())

            case (OperationType.BINARY, False):

                def method(self: trap, o):
                    nonlocal op

                    if isinstance(o, Composable):
                        return self.__class__(
                            op.define(self.__compose__(), o.__compose__(), lazy=True)
                        )
                    else:
                        return self.__class__(self, op.define(o))

            case (OperationType.BINARY, True):

                def method(self: trap, o):
                    nonlocal op

                    if isinstance(o, Composable):
                        return self.__class__(
                            op.define(o.__compose__(), self.__compose__(), lazy=True)
                        )
                    else:
                        return self.__class__(op.define(o), self)

            case (OperationType.TERNARY, False):

                def method(self: trap, *args):
                    nonlocal op
                    if any(isinstance(o, Composable) for o in args):
                        args = map(ensure_composable, args)
                        return self.__class__(op.define(self.__compose__(), args, lazy=True))
                    else:
                        return self.__class__(self, op.define(args))

            case (OperationType.GENERIC, False):

                def method(self: trap, *args, **kwargs):
                    nonlocal op
                    if any(isinstance(o, Composable) for o in chain(args, kwargs.values())):
                        return self.__class__(
                            op.define(
                                self.__compose__(),
                                map(ensure_composable, args),
                                {k: ensure_composable(v) for k, v in kwargs.items()},
                                lazy=True,
                            )
                        )
                    else:
                        return self.__class__(self, op.define(args, kwargs))

        return method


_builtin_unary_ops = FrozenDict.fromkeys(
    (
        "identity",
        "not",
        "truth",
        "abs",
        "index",
        "invert",
        "neg",
        "pos",
    ),
    UNARY_OP,
)


_builtin_binary_ops = FrozenDict.fromkeys(
    (
        "is",
        "is_not",
        "lt",
        "le",
        "eq",
        "ne",
        "ge",
        "gt",
        "add",
        "and",
        "floordiv",
        "lshift",
        "mod",
        "mul",
        "matmul",
        "or",
        "pow",
        "rshift",
        "sub",
        "truediv",
        "xor",
        "contains",
        "getitem",
        "getattr",
        "iadd",
        "iand",
        "ifloordiv",
        "ilshift",
        "imod",
        "imul",
        "imatmul",
        "ior",
        "ipow",
        "irshift",
        "isub",
        "itruediv",
        "ixor",
        "delattr",
        "delitem",
    ),
    BINARY_OP,
)


_builtin_ternary_ops = FrozenDict.fromkeys(
    [
        "setitem",
        "setattr",
    ],
    TERNARY_OP,
)


_builtin_generic_ops = FrozenDict.fromkeys(
    [
        # "attrgetter",
        # "itemgetter",
        # "methodcaller",
        "call",
        "ref",
    ],
    GENERIC_OP,
)


_non_trappable_ops = {
    "not",
    "truth",
    "index",
    "is",
    "is_not",
    "contains",
    "attrgetter",
    "itemgetter",
    "methodcaller",
}

_terminal_ops = {
    "setattr",
    "setitem",
    "delattr",
    "delitem",
}

_builtin_ops = (
    _builtin_unary_ops | _builtin_binary_ops | _builtin_ternary_ops | _builtin_generic_ops
)


class OperatorRegistry(abc.Mapping[str, OperatorInfo]):
    __slots__ = (
        "__identifiers",
        "__registry",
        "__weakref__",
    )

    __registry: dict[str, OperatorInfo]
    __identifiers: dict[str, OperatorInfo]

    def __init__(self) -> None:
        self.__identifiers = {}
        self.__registry = FallbackDict((), self.__identifiers)

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(cls, key: str):
        return cls.__registry[key]

    def __len__(self) -> int:
        return len(self.__registry)

    def __iter__(self) -> int:
        return iter(self.__registry)

    @t.overload
    def register(self, op: OperatorInfo) -> OperatorInfo:
        ...

    register = t.overload(type[OperatorInfo])

    def register(self, op: str, *args, **kwargs) -> OperatorInfo:
        ...

    def register(self, op, *args, **kwargs):
        if isinstance(op, str):
            op = OperatorInfo(op, *args, **kwargs)
        registry = self.__registry
        assert op.name not in registry or registry[op.name] is op
        assert not op.identifier or registry.get(op.identifier, op) is op
        old = registry.setdefault(op.name, op)
        assert old is op
        if op.identifier:
            old = self.__identifiers.setdefault(op.name, op)
            assert old is op
            self.__identifiers[op.identifier] = op
            self.__class__.__annotations__[op.identifier] = OperatorInfo
        return op

    if t.TYPE_CHECKING:
        register: type[OperatorInfo]


operators = ops = OperatorRegistry()


[
    ops.register("ref", GENERIC_OP, operator.identity, impl=Ref, lazy_impl=None, magic=None),
    ops.register(
        "identity", UNARY_OP, operator.identity, impl=Identity, lazy_impl=LazyIdentity, magic=None
    ),
    *(
        ops.register(name, _builtin_ops[name], terminal=True, inplace=True)
        for name in _terminal_ops
        if name not in ops
    ),
    *(ops.register(name, otp) for name, otp in _builtin_ops.items() if name not in ops),
]


_call_compose = methodcaller("__compose__")


class Composable(Interface, t.Generic[_T_Co], parents=Operation):
    __slots__ = ()

    @abstractmethod
    def __compose__(self) -> "Operation[_T_Co]":
        ...


@Composable.register
class trap(t.Generic[_T_Co]):
    __slots__ = ("__src",)
    __src: tuple[Operation[_T_Co]]

    __class_getitem__ = classmethod(GenericAlias)

    def __new__(cls: type[Self], *src: Operation[_T_Co]) -> _T_Co | Composable[_T_Co]:
        self = _object_new(cls)
        object.__setattr__(self, "_trap__src", src)
        return self

    def __compose__(self):
        if src := self.__src:
            return +Composition(src)
        return Identity()

    # def __getattr__(self, name: str):
    #     if isinstance(name, Composable):
    #         return self.__class__(BinaryOperation("getattr", map(_call_compose, (self, name))))
    #     return self.__class__(*self.__src, Attr(name))

    @singledispatchmethod
    def __getitem__(self, key):
        return self.__class__(*self.__src, Subscript(key))

    @__getitem__.register
    def _(self, key: Composable):
        """Create dynamic trap.
        EXample:
            t = compose(trap().abc(1,2,3)[trap().abc['index']])
            t('foo', 'bar')

        Returns:
            _type_: _description_
        """
        return self.__class__(BinaryOperation("getitem", map(_call_compose, (self, key))))

    @__getitem__.register
    def _(self, key: slice):
        return self.__class__(*self.__src, Slice(key))

    @classmethod
    def __define_operator_method__(cls: type[Self], op: OperatorInfo):
        mro = [b for b in cls.__mro__ if issubclass(b, trap)]
        for magic, reverse in [(op.magic_name, False), (op.reverse_magic_name, True)]:
            if not (magic and all(b.__dict__.get(magic) is None for b in mro)):
                return
            method = op.trap_method(reverse=reverse)

            method.__name__ = magic
            method.__qualname__ = f"{cls.__qualname__}.{magic}"
            method.__module__ = f"{cls.__module__}"

            setattr(cls, magic, method)


[trap.__define_operator_method__(op) for op in operators.values()]

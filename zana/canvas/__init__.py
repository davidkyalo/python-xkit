import builtins
import keyword
import operator
import typing as t
import weakref
from abc import abstractmethod
from collections import abc
from functools import reduce, singledispatch, singledispatchmethod
from itertools import chain
from logging import getLogger
from operator import attrgetter, methodcaller
from types import GenericAlias

import attr
from typing_extensions import Self

from zana.types import Interface
from zana.types.collections import (
    Atomic,
    ChainMap,
    Composite,
    DefaultKeyDict,
    FrozenDict,
    UserTuple,
)
from zana.types.enums import IntEnum
from zana.util import cached_attr

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


class _notset(t.Protocol):
    pass


def _field_transformer(fn=None):
    def hook(cls: type, fields: list[attr.Attribute]):
        fields = fn(cls, fields) if fn else fields
        cls.__positional_attrs__ = {
            f.alias or f.name.lstrip("_"): f.name for f in fields if f.init and not f.kw_only
        }
        return fields

    return hook


_define = attr.s

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


def compose(*signatures: "Operation"):
    return +Composition(map(coerce_signature, signatures))


def coerce_signature(obj=_notset):
    if isinstance(obj, Operation):
        return obj
    elif obj is _notset:
        return Identity()
    elif callable(obj):
        return Callback(obj)
    else:
        return Ref(obj)


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


class TrapLike(Interface, t.Generic[_T_Co], check=False):
    __slots__ = ()

    @abstractmethod
    def __compose__(self) -> "Operation[_T_Co]":
        ...


@TrapLike.register
class Operation(t.Generic[_T_Co]):
    __class_getitem__ = classmethod(GenericAlias)
    __positional_attrs__: t.ClassVar[dict[str, str]] = ...

    name: str = attr.ib(converter=str)

    @cached_attr
    def config(self):
        return operators[self.name]

    @cached_attr
    def operator(self):
        return self.config.function

    def __pos__(self):
        return self

    @abstractmethod
    def __compose__(self) -> "Operation[_T_Co]":
        return +self

    @property
    def atomic(self):
        return (self,)

    # def get(self, obj: _T) -> _T_Co:
    #     return self(obj)

    # def set(self, obj: _T, val: _T_Co):
    #     raise NotImplementedError(f"{self!r} setter not supported")

    # def delete(self, obj: _T):
    #     raise NotImplementedError(f"{self!r} deleter not supported")

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
        path = f"{self.__class__.__module__}.{self.__class__.__name__}"
        fields: list[attr.Attribute] = attr.fields(self.__class__)
        args, kwargs = [], {}

        for field in fields:
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

        return path, args, kwargs


TSignatureOrIterable = Operation | abc.Iterable


@_define()
class GenericOperation(Operation[_T_Co]):
    ...


@_define()
class UnaryOperation(Operation[_T_Co]):
    def __call__(self, obj, /):
        return self.operator(obj)


@_define()
class BinaryOperation(Operation[_T_Co]):
    operants: tuple[t.Any] = attr.ib(converter=tuple)

    def __call__(self, obj, /):
        return reduce(self.operator, self.operants, obj)


@_define()
class TernaryOperation(Operation[_T_Co]):
    operants: tuple[t.Any] = attr.ib(converter=tuple)

    def __call__(self, obj, /):
        return reduce(self.operator, self.operants, obj)


@_define()
class LazyUnaryOperation(UnaryOperation[_T_Co]):
    operant: Operation = attr.ib(default=None)

    def __call__(self, obj, /):
        return self.operator(self.operant(obj))


@_define()
class LazyBinaryOperation(BinaryOperation[_T_Co]):
    operants: tuple[Operation] = attr.ib(converter=tuple)

    def __call__(self, obj, /):
        return reduce(self.operator, (op(obj) for op in self.operants))


@_define
class Ref(GenericOperation[_T_Co]):
    obj: _T_Co

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


@_define
class WeakRef(Ref[_T_Co]):
    obj: weakref.ref[_T_Co] = attr.ib(converter=weakref.ref)

    def __call__(self, obj=None, /) -> _T_Co:
        return self.obj()


@_define
class Identity(UnaryOperation[_T_Co], name="identity"):
    def __call__(self, obj: _T_Co, /) -> _T_Co:
        return obj

    def __add__(self, o: object):
        if isinstance(o, Operation):
            return +o
        return NotImplemented

    __radd__ = __add__


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


@_define()
class DynamicCallback(Callback[_T_Co]):
    func: Operation[abc.Callable] = attr.ib()
    args: tuple[Operation] = attr.ib(default=(), converter=tuple)
    kwargs: FrozenDict[str, Operation] = attr.ib(default=FrozenDict(), converter=FrozenDict)

    def __call__(this, self, /, *a, **kw):
        return this.func(self)(
            *(sig(self) for sig in this.args),
            *a,
            **{k: sig(self) for k, sig in this.kwargs.items() if k not in kw},
            **kw,
        )


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

    @singledispatchmethod
    def __call__(this, self=_notset, /, *a, **kw) -> abc.Callable[[t.Any], _T]:
        path, pre = this.path, () if self is _notset else (self,)
        for i in range(len(path) - 1):
            pre = (path[i](*pre),)

        return path[-1](*pre, *a, **kw)

    @__call__.register
    def __call__(this, self: object, /, *a, **kw) -> abc.Callable[[t.Any], _T]:
        path, pre = this.path, () if self is _notset else (self,)
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
        return +self.path[0] if len(self) == 1 else self

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


class OperatorType(IntEnum):
    PLAIN = 0
    UNARY = 1
    BINARY = 2
    TERNARY = 3
    CALLBACK = 4

    def __bool__(self) -> bool:
        return True


@_define
class OperatorConfig:
    name: str = attr.ib(validator=attr.validators.instance_of(str))
    identifier: str = attr.ib(init=False, repr=False, cmp=False)
    type: OperatorType = attr.ib(converter=OperatorType, cmp=False)
    function: abc.Callable = attr.ib(cmp=False, validator=attr.validators.is_callable())
    magic_name: str = attr.ib(kw_only=True, cmp=False)

    operation_class: t.Type[Operation] = attr.ib(kw_only=True, cmp=False)
    lazy_operation_class: t.Type[Operation] = attr.ib(kw_only=True, cmp=False)

    @magic_name.default
    def _init_magic_name(self):
        if (name := self.name) and name not in _non_trappable_ops:
            return f"__{name.strip('_')}__"

    @identifier.default
    def _init_identifier(self):
        if (name := self.name) and name.isidentifier():
            if keyword.iskeyword(name):
                name = f"{name}_"
            return name

    @function.default
    def _init_function(self):
        for mod in (operator, builtins):
            if hasattr(mod, name := self.identifier or self.name):
                return getattr(mod, name)

    @operation_class.default
    def _init_operation_class(self):
        match self.type:
            case OperatorType.UNARY:
                return UnaryOperation
            case OperatorType.BINARY:
                return BinaryOperation

    @lazy_operation_class.default
    def _init_lazy_operation_class(self):
        match self.type:
            case OperatorType.UNARY:
                return LazyUnaryOperation
            case OperatorType.BINARY:
                return LazyBinaryOperation

    def __str__(self) -> str:
        return self.name


_builtin_unary_ops = {
    "not",
    "abs",
    "index",
    "invert",
    "neg",
    "pos",
}

_builtin_binary_ops = {
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
    "delitem",
    "getitem",
    "iadd",
    "iand",
    "iconcat",
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
    "getattr",
    "delattr",
}

_builtin_ternary_ops = {
    "setitem",
    "setattr",
}

_builtin_callback_ops = {
    "attrgetter",
    "itemgetter",
    "methodcaller",
}


_non_trappable_ops = {
    "not",
    "contains",
    "attrgetter",
    "itemgetter",
    "methodcaller",
}

_builtin_ops = (
    _builtin_unary_ops | _builtin_binary_ops | _builtin_ternary_ops | _builtin_callback_ops
)


class OperatorRegistry(abc.Mapping[str, OperatorConfig]):
    __slots__ = (
        "__registry",
        "__weakref__",
    )

    __registry: ChainMap[str, OperatorConfig]

    def __init__(self) -> None:
        self.__registry = ChainMap({}, {})

    @property
    def __aliases(self) -> abc.Mapping[str, OperatorConfig]:
        return self.__registry.maps[1]

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(cls, key: str):
        return cls.__registry[key]

    def get(self, key, default=None):
        return self.__registry.get(key, default)

    def register(self, op: OperatorConfig):
        registry = self.__registry
        assert op.name not in registry or registry[op.name] is op
        assert not (op.identifier and op.identifier in registry)
        registry[op.name] = op
        if op.identifier:
            assert op.identifier not in registry
            self.__aliases[op.identifier] = op
        return op


operators = OperatorRegistry()

[operators.register(OperatorConfig(name, OperatorType.UNARY)) for name in _builtin_unary_ops]
[operators.register(OperatorConfig(name, OperatorType.BINARY)) for name in _builtin_binary_ops]
[operators.register(OperatorConfig(name, OperatorType.TERNARY)) for name in _builtin_ternary_ops]


_call_compose = methodcaller("__compose__")


@TrapLike.register
class trap(t.Generic[_T, _T_Co]):
    __slots__ = ("_trap__src",)
    __src: tuple[Operation[_T_Co]]

    __class_getitem__ = classmethod(GenericAlias)

    def __new__(cls: type[Self], *src: Operation[_T_Co]) -> _T | TrapLike[_T_Co]:
        self = _object_new(cls)
        object.__setattr__(self, "_trap__src", src)
        return self

    def __compose__(self):
        if src := self.__src:
            return +Composition(src)

    def __getattr__(self, name: str):
        if isinstance(name, TrapLike):
            return self.__class__(BinaryOperation("getattr", map(_call_compose, (self, name))))
        return self.__class__(*self.__src, Attr(name))

    methodname: str = attr.ib(init=False)

    @methodname.default
    def _init_qualname(self):
        return f"__{self.name}__"

    def __getattr__(self, name: str):
        if isinstance(name, TrapLike):
            return self.__class__(BinaryOperation("getattr", map(_call_compose, (self, name))))
        return self.__class__(*self.__src, Attr([name]))

    @singledispatchmethod
    def __getitem__(self, key):
        return self.__class__(*self.__src, Subscript(key))

    @__getitem__.register
    def __getitem__(self, key: TrapLike):
        return self.__class__(BinaryOperation("getitem", map(_call_compose, (self, key))))

    @__getitem__.register
    def __getitem__(self, key: slice):
        return self.__class__(*self.__src, Slice(key))

    def __call__(self, /, *args, **kwargs):
        args, ref_args, ref_kwargs = list(args), [], []
        lazy_args = {
            i: v
            for i, v in enumerate(args)
            if isinstance(v, TrapLike) and (not isinstance(v, Ref) or ref_args.append(i))
        }
        lazy_kwargs = {
            k: v
            for k, v in kwargs.items()
            if isinstance(v, TrapLike) and (not isinstance(v, Ref) or ref_kwargs.append(i))
        }
        for i in ref_args:
            args[i] = args[i].value
        for k in ref_kwargs:
            kwargs[k] = kwargs[k].value

        if lazy_args or lazy_kwargs:
            return DynamicCallback(self.__compose__(), args, kwargs)

        return self.__class__(*self.__src, Call(args, kwargs))

    @classmethod
    def __define_operator_method__(cls: type[Self], op: OperatorConfig):
        name = op.name
        if cls is not None:
            if any(b.__dict__.get(name) is not None for b in cls.__mro__ if issubclass(b, trap)):
                return

        match op.type:
            case OperatorType.UNARY:

                def method(self: Self):
                    return self.__class__(self.__src | UnaryOperation(name))

            case OperatorType.BINARY:

                def method(self: Self, o: Self):
                    if isinstance(o, trap):
                        return self.__class__(LazyBinaryOperation(name, [self.__src, o.__src]))
                    elif isinstance(o, Operation):
                        return self.__class__(LazyBinaryOperation(name, [self.__src, o]))
                    return self.__class__(self.__src | BinaryOperation(name, [o]))

            case OperatorType.TERNARY:

                def method(self: Self):
                    return self.__class__(self.__src | UnaryOperation(name))

        method.__name__ = op.magic_name

        if cls is None:
            return method
        setattr(cls, op, method)


[trap.__define_operator_method__(op) for op in operators.values() if op.magic_name]

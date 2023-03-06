import keyword
import operator as builtin_operator
import typing as t
from abc import abstractmethod
from collections import abc
from itertools import chain
from logging import getLogger
from operator import attrgetter, methodcaller
from types import GenericAlias, new_class

import attr
from typing_extensions import Self

from zana.types import Interface
from zana.types.collections import Composite, FallbackDict, FrozenDict, UserTuple
from zana.types.enums import IntEnum
from zana.util import operator

# __all__ = [
#     "Operation",
#     "operators",
#     "compose",
#     "trap",
#     "operation",
#     "Composition",
# ]

_T = t.TypeVar("_T")
_T_Co = t.TypeVar("_T_Co", covariant=True)


_T_Arg = t.TypeVar("_T_Arg")
_T_Kwarg = t.TypeVar("_T_Kwarg", bound=FrozenDict, covariant=True)
_T_Key = t.TypeVar("_T_Key", slice, int, str, t.SupportsIndex, abc.Hashable)
_T_Fn = t.TypeVar("_T_Fn", bound=abc.Callable)
_T_Attr = t.TypeVar("_T_Attr", bound=str)
_T_Op = t.TypeVar("_T_Op", bound="Operation", covariant=True)


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
    else:
        return +Ref(obj)


def operation(name: str, /, *args, **kwargs):
    return ops[name](*args, **kwargs)


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
    operator: t.ClassVar["OperatorInfo[Self]"] = None
    function: t.ClassVar[abc.Callable] = None
    __abstract__: t.ClassVar[abc.Callable] = True
    __final__: t.ClassVar[abc.Callable] = False
    isterminal: t.ClassVar[bool] = False
    inplace: t.ClassVar[bool] = False

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
        return self.function(obj)


@_define()
class BinaryOperation(Operation[_T_Co]):
    __abstract__ = True
    operant: t.Any = attr.ib()

    def __call__(self, obj, /):
        return self.function(obj, self.operant)


@_define()
class TernaryOperation(Operation[_T_Co]):
    __abstract__ = True
    operants: tuple[t.Any] = attr.ib(converter=tuple, validator=attr.validators.max_len(2))

    def __call__(self, obj, /, *args):
        return self.function(obj, *self.operants, *args)


@_define()
class GenericOperation(Operation[_T_Co]):
    __abstract__ = True
    args: tuple[t.Any] = attr.ib(converter=tuple, default=())
    kwargs: tuple[t.Any] = attr.ib(default=FrozenDict(), converter=FrozenDict)
    bind: int | bool = attr.ib(default=True, converter=int, kw_only=True)

    def __call__(self, *a, **kw):
        if bind := a and +self.bind or ():
            bind, args = a[:bind], self.args + a[bind:]
        else:
            args = chain(self.args, a)
        return self.function(*bind, *args, **self.kwargs | kw)


@_define()
class LazyUnaryOperation(LazyOperation[_T_Co]):
    __abstract__ = True
    source: Operation = attr.ib()

    def __call__(self, obj, /):
        return self.function(self.source(obj))


@_define()
class LazyBinaryOperation(LazyOperation[_T_Co]):
    __abstract__ = True
    source: Operation = attr.ib()
    operant: Operation = attr.ib()

    def __call__(self, obj, /):
        return self.function(self.source(obj), self.operant(obj))


@_define()
class LazyTernaryOperation(LazyOperation[_T_Co]):
    __abstract__ = True
    source: Operation = attr.ib()
    operants: tuple[t.Any] = attr.ib(converter=tuple, validator=attr.validators.max_len(2))

    def __call__(self, obj, /, *args):
        return self.function(self.source(obj), *(op(obj) for op in self.operants), *args)


@_define()
class LazyGenericOperation(LazyOperation[_T_Co]):
    __abstract__ = True
    source: Operation = attr.ib()
    args: tuple[Operation] = attr.ib(converter=tuple, default=())
    kwargs: dict[str, Operation] = attr.ib(default=FrozenDict(), converter=FrozenDict)
    bind: int | bool = attr.ib(default=True, converter=int, kw_only=True)

    def __call__(self, obj, /, *a, **kw):
        src, args = self.source(obj), (op(obj) for op in self.args)
        if bind := +self.bind or ():
            bind, args = (src,), chain(args, a)
        else:
            args = chain(args, (src,), a)

        kwargs = {k: op(obj) for k, op in self.kwargs.items() if k not in kw}
        return self.function(*bind, *args, **kwargs, **kw)


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


def _iter_compose(seq: abc.Iterable[Operation], composites: type[abc.Iterable[Operation]]):
    it = iter(seq)
    for pre in it:
        break
    else:
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


class OperationTuple(UserTuple[Operation]):
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
    operations: OperationTuple = attr.ib(repr=_repr_str, converter=OperationTuple)

    size: int = attr.ib(cmp=False, repr=False, init=False)
    size.default(len)

    @property
    def isterminal(self):
        return (ops := self.operations) and ops[-1].isterminal or False

    @property
    def inplace(self):
        return (ops := self.operations) and ops[-1].inplace or False

    def __call__(self, obj=_notset, /, *a, **kw) -> abc.Callable[[t.Any], _T]:
        ops, pre = self.operations, () if obj is _notset else (obj,)
        i, x = 0, len(ops) - 1
        while x > i:
            pre, i = (ops[i](*pre),), i + 1

        return ops[-1](*pre, *a, **kw)

    def __pos__(self):
        path, size = self.operations, self.size
        return Identity() if size == 0 else +path[0] if size == 1 else self

    def __str__(self) -> str:
        return " | ".join(map(repr, self))

    def __len__(self):
        return len(self.operations)

    def __contains__(self, x):
        return x in self.operations

    @t.overload
    def __getitem__(self, key: int) -> Operation:
        ...

    @t.overload
    def __getitem__(self, key: slice) -> Self:
        ...

    def __getitem__(self, key):
        val = self.operations[key]
        if isinstance(key, slice):
            val = self.evolve(val)
        return val

    def __iter__(self):
        return iter(self.operations)

    def __reversed__(self):
        return reversed(self.operations)

    def __add__(self, o: TSignatureOrIterable):
        if isinstance(o, Composition):
            path = self.operations + o.operations
        else:
            return NotImplemented
        return +self.evolve(path)

    def __or__(self, o: TSignatureOrIterable):
        if isinstance(o, Composition):
            path = o.operations
        elif isinstance(o, Operation):
            path = (o,)
        elif isinstance(o, abc.Iterable):
            path = o
        else:
            return NotImplemented
        return +self.evolve(self.operations + path)

    def __ror__(self, o: TSignatureOrIterable):
        if isinstance(o, abc.Iterable):
            return +self.evolve(o + self.operations)
        return NotImplemented

    def deconstruct(self):
        path = f"{__name__}.{compose.__name__}"
        return path, list(self), {}


class Composable(Interface, t.Generic[_T_Co], parents=Operation):
    __slots__ = ()

    @abstractmethod
    def __compose__(self) -> "Operation[_T_Co]":
        ...


@Composable.register
class trap(t.Generic[_T_Co]):
    __slots__ = ("__src", "__weakref__")
    __src: tuple[Operation[_T_Co]]

    __class_getitem__ = classmethod(GenericAlias)

    def __new__(cls: type[Self], *src: Operation[_T_Co]) -> _T_Co | Composable[_T_Co]:
        self = _object_new(cls)
        object.__setattr__(self, "_trap__src", src)
        return self

    def __compose__(self):
        return compose(*self.__src)

    def __setattr__(self, name, val) -> None:
        raise attr.exceptions.FrozenInstanceError()

    def __setattr__(self, name, val) -> None:
        raise TypeError(f"self.{name} = {val!r} is not trappable")

    def __delattr__(self, name) -> None:
        raise TypeError(f"del self.{name} is not trappable")

    def __setitem__(self, name, val) -> None:
        raise TypeError(f"self[{name!r}] = {val!r} is not trappable")

    def __delitem__(self, name) -> None:
        raise TypeError(f"del self[{name!r}] is not trappable")

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self.__compose__())!r})"

    def __str__(self):
        return f"{self.__compose__()!s}"

    def __len__(self):
        return len(self.__src)

    def __iter__(self):
        return iter(self.__src)

    def __reduce__(self):
        return self.__class__, self.__src

    @classmethod
    def __define_operator_methods__(cls: type[Self], op: "OperatorInfo"):
        mro = [b.__dict__ for b in cls.__mro__ if issubclass(b, trap)]
        for name, reverse in [(op.trap_name, False), (op.reverse_trap_name, True)]:
            if not (name and all(b.get(name) is None for b in mro)):
                return
            elif method := op.trap_method(reverse=reverse):
                method.__name__ = name
                method.__qualname__ = f"{cls.__qualname__}.{name}"
                method.__module__ = f"{cls.__module__}"

                setattr(cls, name, method)


@attr.define(frozen=True, cache_hash=True)
class OperatorInfo(t.Generic[_T_Op]):
    name: str = attr.ib(validator=[attr.validators.instance_of(str), attr.validators.min_len(1)])
    identifier: str = attr.ib(init=False, repr=False, cmp=False)
    type: OperationType = attr.ib(
        converter=OperationType,
        cmp=False,
        repr=attr.converters.pipe(attrgetter("name"), repr),
    )
    function: abc.Callable = attr.ib(
        cmp=False,
        validator=attr.validators.optional(attr.validators.is_callable()),
        repr=attr.converters.pipe(attrgetter("__name__"), repr),
    )
    isterminal: bool = attr.ib(kw_only=True, cmp=False)
    inplace: bool = attr.ib(kw_only=True, cmp=False)
    reverse: bool | str = attr.ib(kw_only=True, cmp=False)
    trap: str | bool = attr.ib(kw_only=True, cmp=False, repr=False)
    trap_name: str = attr.ib(kw_only=True, cmp=False, init=False)
    reverse_trap_name: str = attr.ib(kw_only=True, init=False, repr=False, cmp=False)

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

    @property
    def isbuiltin(self):
        return self.name in _builtin_ops

    @trap.default
    def _default_trap(self):
        return self.isbuiltin and self.name not in _non_trappable_ops

    @trap_name.default
    def _init_trap_name(self):
        if (val := self.trap) is True:
            return f"__{self.name.strip('_')}__"
        elif isinstance(val, str):
            return val

    @isterminal.default
    def _default_isterminal(self):
        return self.name in _terminal_ops or (False if self.isbuiltin else None)

    @inplace.default
    def _default_isinplace(self):
        name = self.name
        return self.isterminal or (
            self.type is BINARY_OP and name[:1] == "i" and name[1:] in _builtin_binary_ops
        )

    @reverse.default
    def _init_reversible(self):
        return self.type is BINARY_OP and f"i{self.name}" in _builtin_binary_ops

    @reverse_trap_name.default
    def _init_reverse_magic_name(self):
        if magic := (rev := self.reverse) is True and self.trap_name:
            val = f"r{magic.lstrip('_')}"
            return f"{'_' * (1+len(magic)-len(val))}{val}"
        elif isinstance(rev, str):
            return rev

    @identifier.default
    def _init_identifier(self):
        if (name := self.name) and name.isidentifier():
            if keyword.iskeyword(name):
                name = f"{name}_"
            return name

    @function.default
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
        function, isterminal, inplace = staticmethod(self.function), self.isterminal, self.inplace
        if base.__final__:
            assert base.operator is None
            assert isterminal or not base.isterminal
            assert inplace or not base.inplace
            base.operator, base.function, base.isterminal, base.inplace = (
                self,
                base.function or function,
                isterminal,
                inplace,
            )
            return base

        name = f"{self.identifier or base.__name__}_{'lazy_' if base.lazy else ''}operation"
        name = "".join(map(methodcaller("capitalize"), name.split("_")))
        if getattr(base.__call__, "__isabstractmethod__", False):
            __call__ = function
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
            "function": function,
            "isterminal": isterminal,
            "inplace": inplace,
        }

        cls = new_class(name, (base,), None, methodcaller("update", ns))
        _define(cls, auto_attribs=False)
        return cls

    def __str__(self) -> str:
        return self.name

    def __call__(self, *args, lazy=False, **kwargs) -> _T_Op:
        cls = self.lazy_impl if lazy is True else self.impl
        return cls(*args, **kwargs)

    def trap_method(op, reverse: bool = False):
        if not op.trap:
            return

        match (op.type, not not reverse):
            case (OperationType.UNARY, False):

                def method(self: trap):
                    nonlocal op
                    return self.__class__(*self, op())

            case (OperationType.BINARY, False):

                def method(self: trap, o):
                    nonlocal op
                    if isinstance(o, Composable):
                        return self.__class__(op(self.__compose__(), o.__compose__(), lazy=True))
                    else:
                        return self.__class__(*self, op(o))

            case (OperationType.BINARY, True):

                def method(self: trap, o):
                    nonlocal op

                    if isinstance(o, Composable):
                        return self.__class__(op(o.__compose__(), self.__compose__(), lazy=True))
                    else:
                        return self.__class__(op(o), *self)

            case (OperationType.TERNARY, False):

                def method(self: trap, *args):
                    nonlocal op
                    if any(isinstance(o, Composable) for o in args):
                        args = map(ensure_composable, args)
                        return self.__class__(op(self.__compose__(), args, lazy=True))
                    else:
                        return self.__class__(*self, op(args))

            case (OperationType.GENERIC, False):

                def method(self: trap, *args, **kwargs):
                    nonlocal op
                    if any(isinstance(o, Composable) for o in chain(args, kwargs.values())):
                        return self.__class__(
                            op(
                                self.__compose__(),
                                map(ensure_composable, args),
                                {k: ensure_composable(v) for k, v in kwargs.items()},
                                lazy=True,
                            )
                        )
                    else:
                        return self.__class__(*self, op(args, kwargs))

        return method


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

    @t.overload
    def __getitem__(cls, key: "_T_OpName") -> OperatorInfo:
        ...

    @t.overload
    def __getitem__(cls, key: str) -> OperatorInfo:
        ...

    def __getitem__(cls, key):
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
        trap.__define_operator_methods__(op)
        return op

    if t.TYPE_CHECKING:
        register: type[OperatorInfo]

        identity: OperatorInfo
        not_: OperatorInfo
        truth: OperatorInfo
        abs: OperatorInfo
        index: OperatorInfo
        invert: OperatorInfo
        neg: OperatorInfo
        pos: OperatorInfo
        is_: OperatorInfo
        is_not: OperatorInfo
        lt: OperatorInfo
        le: OperatorInfo
        eq: OperatorInfo
        ne: OperatorInfo
        ge: OperatorInfo
        gt: OperatorInfo
        add: OperatorInfo
        and_: OperatorInfo
        floordiv: OperatorInfo
        lshift: OperatorInfo
        mod: OperatorInfo
        mul: OperatorInfo
        matmul: OperatorInfo
        or_: OperatorInfo
        pow: OperatorInfo
        rshift: OperatorInfo
        sub: OperatorInfo
        truediv: OperatorInfo
        xor: OperatorInfo
        contains: OperatorInfo
        getitem: OperatorInfo
        getattr: OperatorInfo
        iadd: OperatorInfo
        iand: OperatorInfo
        ifloordiv: OperatorInfo
        ilshift: OperatorInfo
        imod: OperatorInfo
        imul: OperatorInfo
        imatmul: OperatorInfo
        ior: OperatorInfo
        ipow: OperatorInfo
        irshift: OperatorInfo
        isub: OperatorInfo
        itruediv: OperatorInfo
        ixor: OperatorInfo
        delattr: OperatorInfo
        delitem: OperatorInfo
        setitem: OperatorInfo
        setattr: OperatorInfo
        call: OperatorInfo
        ref: OperatorInfo


operators = ops = OperatorRegistry()


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


_terminal_ops = {
    "setattr",
    "setitem",
    "delattr",
    "delitem",
}

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
} | _terminal_ops


_builtin_ops = (
    _builtin_unary_ops | _builtin_binary_ops | _builtin_ternary_ops | _builtin_generic_ops
)


_T_OpName = t.Literal[
    "identity",
    "not_",
    "truth",
    "abs",
    "index",
    "invert",
    "neg",
    "pos",
    "is_",
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
    "setitem",
    "setattr",
    "call",
    "ref",
]


[
    ops.register("ref", GENERIC_OP, operator.none, impl=Ref, lazy_impl=None, trap=None),
    ops.register(
        "identity", UNARY_OP, operator.identity, impl=Identity, lazy_impl=LazyIdentity, trap=None
    ),
    *(ops.register(name, otp) for name, otp in _builtin_ops.items() if name not in ops),
]

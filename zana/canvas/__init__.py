import math
import operator
import typing as t
from abc import abstractmethod
from collections import abc
from functools import reduce
from importlib.resources import path
from itertools import chain
from logging import getLogger
from operator import attrgetter
from types import GenericAlias

import attr
from typing_extensions import Self

from zana.types import Interface
from zana.types.collections import Atomic, ChainMap, Composite, FrozenDict, Subscriptable, UserTuple
from zana.util import cached_attr, pipeline

_T = t.TypeVar("_T")
_T_Co = t.TypeVar("_T_Co", covariant=True)


_T_Arg = t.TypeVar("_T_Arg")
_T_Kwarg = t.TypeVar("_T_Kwarg", bound=FrozenDict, covariant=True)
_T_Key = t.TypeVar("_T_Key", slice, int, str, t.SupportsIndex, abc.Hashable)
_T_Fn = t.TypeVar("_T_Fn", bound=abc.Callable)
_T_Attr = t.TypeVar("_T_Attr", bound=str)

T_ManySignatures = list | tuple | abc.Sequence | abc.Iterator
T_OneOrManySignatures = T_ManySignatures

logger = getLogger(__name__)
_object_new = object.__new__
_empty_dict = FrozenDict()


class _notset(t.Protocol):
    pass


def _field_transformer(fn=None):
    def hook(cls: type, fields: list[attr.Attribute]):
        fields = fn(cls, fields) if fn else fields
        cls.__positional_attrs__ = {
            f.alias or f.name: f.name for f in fields if f.init and not f.kw_only
        }
        return fields

    return hook


_define = attr.s

if not t.TYPE_CHECKING:

    def _define(*a, field_transformer=None, **kw):
        return attr.define(
            *a,
            **dict(
                frozen=True,
                # cmp=True,
                cache_hash=True,
                # collect_by_mro=True,
                field_transformer=_field_transformer(field_transformer),
            )
            | kw,
        )


def compose(*signatures: "Signature"):
    return Composition(
        s if isinstance(s, Signature) else Func(s) if callable(s) else Ref(s) for s in signatures
    )._()


def ensure_signature(obj=_notset):
    if isinstance(obj, Signature):
        return obj
    elif obj is _notset:
        return Return()
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


class Composable(Interface, check=False):
    __slots__ = ()


@Atomic.register
@_define()
class Signature(t.Generic[_T_Co]):
    # __slots__ = "__dict__", "__weakref__"
    __class_getitem__ = classmethod(GenericAlias)
    __positional_attrs__: t.ClassVar[dict[str, str]] = ...

    # __args__: t.Final[tuple[_T_Arg, ...]]
    # __kwargs__: t.Final[_T_Kwarg]

    # _min_args_: t.Final = 0
    # _max_args_: t.Final = math.inf
    # _default_args_: t.Final = ()
    # _default_kwargs_: t.Final = FrozenDict()
    # _required_kwargs_: t.Final = frozenset[str]()
    # _extra_kwargs_: t.Final = False
    # _allowed_kwargs_: t.Final = frozenset[str]()
    # _allows_merging_: t.Final[bool] = True
    # _merge_types_: t.Final[set[type[Self]]]

    # if t.TYPE_CHECKING:
    #     __args__ = __kwargs__ = None

    # def __init_subclass__(
    #     cls,
    #     args=None,
    #     kwargs=None,
    #     required_kwargs=None,
    #     min_args=None,
    #     max_args=None,
    #     extra_kwargs=None,
    #     merge=None,
    #     **kwds,
    # ) -> None:
    #     super().__init_subclass__(**kwds)
    #     if merge is not None:
    #         cls._allows_merging_ = not not merge

    #     if cls._allows_merging_:
    #         cls._merge_types_ = set({cls})
    #     else:
    #         cls._merge_types_ = frozenset()

    #     if min_args is not None:
    #         cls._min_args_ = min_args
    #     if max_args is not None:
    #         cls._max_args_ = max_args

    #     if args is not None:
    #         cls._default_args_ = tuple(args.split() if isinstance(args, str) else args)
    #     if kwargs is not None:
    #         cls._default_kwargs_ = FrozenDict(kwargs)

    #     if required_kwargs is not None:
    #         cls._required_kwargs_ = dict.fromkeys(
    #             args.split() if isinstance(args, str) else args
    #         ).keys()

    #     if extra_kwargs is not None:
    #         cls._extra_kwargs_ = not not extra_kwargs

    #     if cls._extra_kwargs_:
    #         cls._allowed_kwargs_ = cls._default_kwargs_.keys()
    #     else:
    #         cls._allowed_kwargs_ = {}.keys()

    # @classmethod
    # def construct(cls, args=(), kwargs=FrozenDict()):
    #     self: cls = _object_new(cls)
    #     self.__args__, self.__kwargs__ = args, kwargs
    #     return self

    # @classmethod
    # def construct(cls, attrs=()):
    #     self: cls = _object_new(cls)
    #     attrs and self.__dict__.update(attrs)
    #     return self

    # @classmethod
    # def _parse_params_(cls, args, kwargs) -> tuple[tuple[_T_Arg, ...], FrozenDict[str, _T_Kwarg]]:
    #     args, kwargs = tuple(args), FrozenDict(kwargs)

    # def __new__(cls: type[T_Self], *args, **kwargs) -> T_Self:
    #     args, kwargs = tuple(args), FrozenDict(kwargs)

    #     # if not (cls._min_args_ <= len(args) <= cls._max_args_):
    #     #     raise TypeError(
    #     #         f"{cls.__name__} expected "
    #     #         f"{' to '.join(map(str, {cls._min_args_, cls._max_args_}))}"
    #     #         f" arguments but got {len(args)}."
    #     #     )
    #     # elif d_args := cls._default_args_:
    #     #     args = args + d_args[len(args) :]

    #     # kw_keys = kwargs.keys()
    #     # if missed := cls._required_kwargs_ - kw_keys:
    #     #     missed, s = list(map(repr, missed)), "s" if len(missed) > 1 else ""
    #     #     raise TypeError(
    #     #         f"{cls.__name__} is missing required keyword only argument{s} "
    #     #         f"{' and '.join(filter(None, (', '.join(missed[:-1]), missed[-1])))}."
    #     #     )
    #     # elif extra := cls._extra_kwargs_ is False and kw_keys - cls._allowed_kwargs_:
    #     #     extra, s = list(map(repr, extra)), "s" if len(missed) > 1 else ""
    #     #     allowed = list(map(repr, cls._allowed_kwargs_))
    #     #     raise TypeError(
    #     #         f"{cls.__name__} got unexpected keyword only argument{s} "
    #     #         f"{' and '.join(filter(None, (', '.join(extra[:-1]), extra[-1])))}. "
    #     #         f"Allowed "
    #     #         f"{' and '.join(filter(None, (', '.join(allowed[:-1]), allowed and allowed[-1])))}."
    #     #     )
    #     # elif d_kwargs := cls._default_kwargs_:
    #     #     kwargs = d_kwargs | kwargs

    #     return cls.construct(args, FrozenDict(kwargs))

    @property
    def __ident__(self):
        return self.__args__, self.__kwargs__

    def _(self):
        return self

    def evolve(self, *args, **kwds):
        args = dict(zip(self.__positional_attrs__, args)) if args else _empty_dict
        return attr.evolve(self, **args, **kwds)

    if t.TYPE_CHECKING:
        evolve: type[Self]

    @t.overload
    def asdict(
        self,
        *,
        recurse: bool = True,
        filter=None,
        dict_factory=dict,
        retain_collection_types: bool = False,
        value_serializer=None,
    ):
        ...

    def asdict(self, **kwds):
        return attr.asdict(self, **kwds)

    @t.overload
    def astuple(
        self,
        *,
        recurse: bool = True,
        filter=None,
        tuple_factory=tuple,
        retain_collection_types=False,
    ):
        ...

    def astuple(self, **kwds):
        return attr.astuple(self, **kwds)

    # def __repr__(self):
    #     params = map(repr, self.__args__), (f"{k}={v!r}" for k, v in self.__kwargs__.items())
    #     return f"{self.__class__.__name__}({', '.join(p for ps in params for p in ps)})"

    # def __reduce__(self):
    #     return self.__class__.construct, (self.__args__, self.__kwargs__)

    # def __copy__(self):
    #     return self.construct(self.__args__, self.__kwargs__)

    # def __eq__(self, o: T_Self) -> bool:
    #     return o.__class__ is self.__class__ and o.__ident__ == self.__ident__

    # def __ne__(self, o: T_Self) -> bool:
    #     return o.__class__ is not self.__class__ or o.__ident__ != self.__ident__

    # def __hash__(self) -> int:
    #     return hash(self.__ident__)

    @abstractmethod
    def __add__(self, o: T_OneOrManySignatures):
        return NotImplemented

    @abstractmethod
    def __radd__(self, o: T_OneOrManySignatures):
        return NotImplemented

    def __or__(self, o):
        if isinstance(o, Signature):
            return Composition((self, o))._()
        elif isinstance(o, T_ManySignatures):
            return Composition(chain((self,), o))._()
        return NotImplemented

    def __ror__(self, o):
        if isinstance(o, Signature):
            return Composition((o, self))._()
        elif isinstance(o, T_ManySignatures):
            return Composition(chain(o, (self,)))._()
        return NotImplemented

    # def can_merge(self, o: T_Self):
    #     return o.__class__ in self._merge_types_ and o.__kwargs__ == self.__kwargs__

    # def merge(self: T_Self, o: T_Self):
    #     if not self.can_merge(o):
    #         raise TypeError(f"{self!r}  be merged with {o!r}")
    #     return self._merge(o)

    # def _merge(self, o):
    #     return self.construct(self.__args__ + o.__args__, self.__kwargs__ | o.__kwargs__)

    @abstractmethod
    def __call__(self, *a, **kw) -> _T_Co:
        raise NotImplementedError(f"{self!r} getter not supported")

    def get(self, obj: _T) -> _T_Co:
        return self(obj)

    def set(self, obj: _T, val: _T_Co):
        raise NotImplementedError(f"{self!r} setter not supported")

    def delete(self, obj: _T):
        raise NotImplementedError(f"{self!r} deleter not supported")

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


T_OneOrManySignatures = Signature | T_OneOrManySignatures


@_define
class Ref(Signature[_T_Co]):
    value: _T_Co

    def __call__(this, /, self=None) -> _T_Co:
        return this.value

    def __add__(self, o: object):
        if self.__class__ is o.__class__:
            return o
        return NotImplemented

    def __radd__(self, o: object):
        if isinstance(o, Signature):
            return self
        return NotImplemented


@_define
class Return(Signature[_T_Co]):
    def __call__(this, /, self: _T_Co) -> _T_Co:
        return self

    def __add__(self, o: object):
        if self.__class__ is o.__class__:
            return o
        return NotImplemented

    def __radd__(self, o: T_OneOrManySignatures):
        if self.__class__ is o.__class__:
            return self
        return NotImplemented


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
class Attr(Signature[_T_Co]):
    path: AttrPath = attr.ib(converter=AttrPath, repr=str)

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
            print(f"{self.__class__.__name__}.ADD: {self} + {o} = NotImplemented")
            return NotImplemented
        lhs, rhs = self.path, o.path
        path = lhs + rhs
        print(f"{self.__class__.__name__}.ADD: {self} + {o} = {self.evolve(path)}")
        return self.evolve(path)


_lookup_errors = {
    LookupError: LookupSignatureError,
    KeyError: KeySignatureError,
    IndexError: IndexSignatureError,
}


class SubscriptPath(UserTuple[_T_Key]):
    __slots__ = ()

    def __str__(self) -> str:
        return f"[{']['.join(map(str, self))}]"


@_define()
class Subscript(Signature[_T_Co]):
    path: SubscriptPath = attr.ib(converter=SubscriptPath, repr=str)

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
            print(f"{self.__class__.__name__}.ADD: {self} + {o} = NotImplemented")
            return NotImplemented
        lhs, rhs = self.path, o.path
        path = lhs + rhs
        print(f"{self.__class__.__name__}.ADD: {self} + {o} = {self.evolve(path)}")
        return self.evolve(path)


_slice_tuple = attrgetter("start", "stop", "step")


def _to_slice(val):
    __tracebackhide__ = True
    if isinstance(val, slice):
        return val
    elif isinstance(val, int):
        return slice(val, (val + 1) or None)
    else:
        return slice(*val)


class SlicePath(UserTuple[slice]):
    __slots__ = ()

    def __new__(cls: type[Self], path: abc.Iterable[str] = ()) -> Self:
        if path.__class__ is cls:
            return path
        return cls.construct(map(_to_slice, path))

    def __str__(self) -> str:
        return "".join(f"[{s.start}:{s.stop}:{s.step}]".replace("None", "") for s in self)


@_define()
class Slice(Subscript[_T_Co]):
    path: SlicePath = attr.ib(converter=SlicePath, repr=str, cmp=False)
    raw_path: tuple[tuple[_T_Key, _T_Key, _T_Key]] = attr.ib(init=False, repr=False, cmp=True)

    @raw_path.default
    def _init_raw_path(self):
        return tuple(map(_slice_tuple, self.path))

    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        return path, [self.raw_path], {}


@_define()
class Call(Signature[_T_Co]):
    args: tuple[t.Any] = attr.ib(default=(), converter=tuple)
    kwargs: FrozenDict[str, t.Any] = attr.ib(default=FrozenDict(), converter=FrozenDict)

    def __str__(self) -> str:
        params = map(repr, self.args), (f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"({', '.join(p for ps in params for p in ps)})"

    # def __repr__(self):
    #     params = map(repr, self.__args__), (f"{k}={v!r}" for k, v in self.__kwargs__.items())
    #     return f"{self.__class__.__name__}({', '.join(p for ps in params for p in ps)})"

    def __call__(this, /, self, *a, **kw) -> _T_Co:
        return self(*a, *this.args, **this.kwargs | kw)


@_define()
class Func(Signature[_T_Co]):
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
        return self.func(*a, *self.args, **self.kwargs | kw)


def _join(a: list[Signature], b: Signature):
    if not a:
        return [b]
    try:
        a[-1] = a[-1] + b
    except TypeError:
        a.append(b)
    return a


def _reduce(items: T_ManySignatures):
    return tuple(
        i for it in reduce(_join, items, []) for i in (it if isinstance(it, Composite) else (it,))
    )


class CompositionPath(UserTuple[Signature]):
    __slots__ = ()

    def __new__(cls: type[Self], path: abc.Iterable[str] = ()) -> Self:
        if path.__class__ is cls:
            return path
        return cls.construct(
            i
            for it in reduce(_join, path, [])
            for i in (it if isinstance(it, Composite) else (it,))
        )

    def __str__(self) -> str:
        return "".join(map(str, self))


@_define()
class Composition(Signature, abc.Sequence[Signature]):
    path: CompositionPath = attr.ib(default=(), converter=CompositionPath)

    def __call__(this, /, self=_notset, *a, **kw) -> abc.Callable[[t.Any], _T]:
        path, pre = this.path, () if self is _notset else (self,)
        for i in range(len(path) - 1):
            pre = (path[i](*pre),)
        return path[-1](*pre, *a, **kw)

    def set(this, /, self, val) -> abc.Callable[[t.Any], _T]:
        path = this.path
        for i in range(len(path) - 1):
            self = path[i](self)
        return path[-1].set(self, val)

    def delete(this, /, self) -> abc.Callable[[t.Any], _T]:
        path = this.path
        for i in range(len(path) - 1):
            self = path[i](self)
        return path[-1].delete(self)

    def _(self):
        return self.path[0]._() if len(self) == 1 else self

    def __str__(self) -> str:
        return " | ".join(map(repr, self))

    def __len__(self):
        return len(self.path)

    def __contains__(self, x):
        return x in self.path

    @t.overload
    def __getitem__(self, key: int) -> Signature:
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

    def __add__(self, o: T_OneOrManySignatures):
        if not isinstance(o, Signature):
            return NotImplemented

        if isinstance(o, Composition):
            lhs, rhs = self.path, o.path
        else:
            lhs, rhs = self.path, (o,)

        if lhs and rhs:
            path = lhs[:-1] + (*_join([lhs[-1]], rhs[0]),) + rhs[1:]
        else:
            path = lhs or rhs

        return self.evolve(path)._()

    def __radd__(self, o: T_OneOrManySignatures):
        if not isinstance(o, Signature):
            return NotImplemented

        if isinstance(o, Composition):
            lhs, rhs = o.path, self.path
        else:
            lhs, rhs = (o,), self.path

        if lhs and rhs:
            path = lhs[:-1] + (*_join([lhs[-1]], rhs[0]),) + rhs[1:]
        else:
            path = lhs or rhs

        return self.evolve(path)._()

    def __or__(self, o):
        if isinstance(o, Signature):
            return self.__add__(o)
        return super().__or__(o)

    def __ror__(self, o):
        if isinstance(o, Signature):
            return self.__radd__(o)
        return super().__ror__(o)


_builtin_operators = {
    "__lt__": operator.lt,
    "__le__": operator.le,
    "__eq__": operator.eq,
    "__ne__": operator.ne,
    "__ge__": operator.ge,
    "__gt__": operator.gt,
    "__not__": operator.not_,
    "__abs__": operator.abs,
    "__add__": operator.add,
    "__and__": operator.and_,
    "__floordiv__": operator.floordiv,
    "__index__": operator.index,
    "__inv__": operator.inv,
    "__invert__": operator.invert,
    "__lshift__": operator.lshift,
    "__mod__": operator.mod,
    "__mul__": operator.mul,
    "__matmul__": operator.matmul,
    "__neg__": operator.neg,
    "__or__": operator.or_,
    "__pos__": operator.pos,
    "__pow__": operator.pow,
    "__rshift__": operator.rshift,
    "__sub__": operator.sub,
    "__truediv__": operator.truediv,
    "__xor__": operator.xor,
    "__concat__": operator.concat,
    "__contains__": operator.contains,
    "__delitem__": operator.delitem,
    "__getitem__": operator.getitem,
    "__setitem__": operator.setitem,
    "__iadd__": operator.iadd,
    "__iand__": operator.iand,
    "__iconcat__": operator.iconcat,
    "__ifloordiv__": operator.ifloordiv,
    "__ilshift__": operator.ilshift,
    "__imod__": operator.imod,
    "__imul__": operator.imul,
    "__imatmul__": operator.imatmul,
    "__ior__": operator.ior,
    "__ipow__": operator.ipow,
    "__irshift__": operator.irshift,
    "__isub__": operator.isub,
    "__itruediv__": operator.itruediv,
    "__ixor__": operator.ixor,
}


_builtin_operators = FrozenDict(_builtin_operators)
_user_operators, _user_operators_names = {}, {}


class OperationOptions(t.TypedDict):
    operator: str | abc.Callable


class _OperatorDict(ChainMap):
    __slots__ = ()

    def __missing__(self, key):
        if callable(key):
            return key
        raise KeyError(key)


ALL_OPERATORS = _OperatorDict(_builtin_operators, _user_operators)


class Operation(Signature[_T_Co, _T, Signature, OperationOptions]):
    __slots__ = ()

    @t.overload
    def __new__(
        cls: type[Self], operant: Signature, *operants: Signature, operator: str | t.Callable
    ) -> Self:
        ...

    def __new__(cls: type[Self], *operants: Signature, operator: str | t.Callable) -> Self:
        return cls.construct(
            tuple(map(ensure_signature, operants)), FrozenDict(operator=ALL_OPERATORS[operator])
        )

    # @classmethod
    # def _parse_params_(cls, args, kwargs) -> tuple[tuple[_T_Arg, ...], FrozenDict[str, _T_Kwarg]]:
    #     args, kwargs = super()._parse_params_(args, kwargs)
    #     if args[0] not in ALL_OPERATORS:
    #         raise ValueError(f"invalid operator name {args[0]}")
    #     return args, kwargs

    @property
    def operator(self):
        return self.__kwargs__["operator"]

    __wrapped__ = operator

    def __call__(this, /, self):
        items, op = self.__args__, self.operator
        operants = (op(self) for op in items)
        reduce(this.operator, operants)
        return this.operator(*(o(self) for o in this.__args__))


class SupportsSignature(t.Protocol[_T_Co]):
    def _(self) -> Signature[_T_Co]:
        ...


def _var_operator_method(nm, op):
    def method(self: "Var", /, *args):
        return Operation(self.__signature, *args, operator=op)

    method.__name__ = nm
    return method


class Var(t.Generic[_T, _T_Co]):
    __slots__ = ("_Var__signature",)
    __signature: Signature[_T_Co]

    __class_getitem__ = classmethod(GenericAlias)

    for nm, op in _builtin_operators.items():
        vars()[nm] = _var_operator_method(nm, op)
    del nm, op

    def __new__(
        cls: type[Self], sig: Signature[_T_Co] = Composition()
    ) -> _T | SupportsSignature[_T_Co]:
        self = _object_new(cls)
        object.__setattr__(self, "_Var__signature", sig)
        return self

    def _(self):
        return self.__signature

    def __extend__(self, *expr):
        return self.__class__(self.__signature | expr)

    def __getattr__(self, name: str):
        return self.__extend__(Attr(name))

    def __getitem__(self, key):
        cls = Slice if isinstance(key, slice) else Subscript
        return self.__extend__(cls(key))

    def __call__(self, /, *args, **kwargs):
        return self.__extend__(Call(*args, **kwargs))

    def __or__(self, o):
        return self.__extend__(Operation(ensure_signature(o), operator="or_"))

    def __add__(self, o):
        return self.__extend__(Operation(ensure_signature(o), operator="add"))

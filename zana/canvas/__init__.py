import math
import operator
import typing as t
from abc import abstractmethod
from collections import abc
from functools import reduce
from logging import getLogger
from operator import attrgetter
from types import GenericAlias

import attr
from typing_extensions import Self as T_Self

from zana.types import Interface
from zana.types.collections import Atomic, ChainMap, Composite, FrozenDict, Subscriptable
from zana.util import cached_attr, pipeline

_T = t.TypeVar("_T")
_RT = t.TypeVar("_RT")


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


_define = attr.define

if not t.TYPE_CHECKING:

    def _define(*a, **kw):
        return attr.define(*a, **dict(frozen=True, cmp=True, cache_hash=True) | kw)


def compose(*expressions: "Signature"):
    return Composition(*expressions)._()


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
class Signature(t.Generic[_RT, _T, _T_Arg, _T_Kwarg]):
    # __slots__ = "__dict__", "__weakref__"
    __class_getitem__ = classmethod(GenericAlias)

    # __args__: t.Final[tuple[_T_Arg, ...]]
    # __kwargs__: t.Final[_T_Kwarg]

    _min_args_: t.Final = 0
    _max_args_: t.Final = math.inf
    _default_args_: t.Final = ()
    _default_kwargs_: t.Final = FrozenDict()
    _required_kwargs_: t.Final = frozenset[str]()
    _extra_kwargs_: t.Final = False
    _allowed_kwargs_: t.Final = frozenset[str]()
    _allows_merging_: t.Final[bool] = True
    _merge_types_: t.Final[set[type[T_Self]]]

    if t.TYPE_CHECKING:
        __args__ = __kwargs__ = None

    def __init_subclass__(
        cls,
        args=None,
        kwargs=None,
        required_kwargs=None,
        min_args=None,
        max_args=None,
        extra_kwargs=None,
        merge=None,
        **kwds,
    ) -> None:
        super().__init_subclass__(**kwds)
        if merge is not None:
            cls._allows_merging_ = not not merge

        if cls._allows_merging_:
            cls._merge_types_ = set({cls})
        else:
            cls._merge_types_ = frozenset()

        if min_args is not None:
            cls._min_args_ = min_args
        if max_args is not None:
            cls._max_args_ = max_args

        if args is not None:
            cls._default_args_ = tuple(args.split() if isinstance(args, str) else args)
        if kwargs is not None:
            cls._default_kwargs_ = FrozenDict(kwargs)

        if required_kwargs is not None:
            cls._required_kwargs_ = dict.fromkeys(
                args.split() if isinstance(args, str) else args
            ).keys()

        if extra_kwargs is not None:
            cls._extra_kwargs_ = not not extra_kwargs

        if cls._extra_kwargs_:
            cls._allowed_kwargs_ = cls._default_kwargs_.keys()
        else:
            cls._allowed_kwargs_ = {}.keys()

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
        return self.__class__(*self.__args__, *args, **self.__kwargs__ | kwds)

    if not t.TYPE_CHECKING:
        evolve: type[T_Self]

    else:
        evolve: t.ClassVar = attr.evolve

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
            return compose(self, o)
        elif isinstance(o, T_ManySignatures):
            return compose(self, *o)
        return NotImplemented

    def __ror__(self, o):
        if isinstance(o, Signature):
            return compose(o, self)
        elif isinstance(o, T_ManySignatures):
            return compose(*o, self)
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
    def __call__(self, *a, **kw) -> _RT:
        raise NotImplementedError(f"{self!r} getter not supported")

    def get(self, obj: _T) -> _RT:
        return self(obj)

    def set(self, obj: _T, val: _RT):
        raise NotImplementedError(f"{self!r} setter not supported")

    def delete(self, obj: _T):
        raise NotImplementedError(f"{self!r} deleter not supported")

    def deconstruct(self):
        path = f"{self.__class__.__module__}.{self.__class__.__name__}"
        args, kwargs = self.__args__, dict(self.__kwargs__)
        d_args, d_kwargs = self._default_args_, self._default_kwargs_
        if d_args and len(d_args) == len(args):
            for i in range(len(d_args)):
                if args[i:] == d_args[i:]:
                    args = args[:i]
                    break
        for k, v in d_kwargs.items():
            if k in kwargs and kwargs[k] == v:
                del kwargs[k]

        return path, list(args), kwargs


T_OneOrManySignatures = Signature | T_OneOrManySignatures


@_define
class Ref(Signature, t.Generic[_RT]):
    _object: _RT

    def __call__(this, /, self=None) -> _RT:
        return this.__args__[0]

    def __add__(self, o: T_OneOrManySignatures):
        if self.__class__ is o.__class__:
            return o
        return NotImplemented

    def __radd__(self, o: T_OneOrManySignatures):
        if isinstance(o, Signature):
            return self
        return NotImplemented


@_define
class Return(Signature, t.Generic[_RT]):
    def __new__(cls):
        return cls.construct()

    def __call__(this, /, self: _RT) -> _RT:
        return self

    def __add__(self, o: T_OneOrManySignatures):
        if self.__class__ is o.__class__:
            return o
        return NotImplemented

    def __radd__(self, o: T_OneOrManySignatures):
        if self.__class__ is o.__class__:
            return self
        return NotImplemented


class AttrOptions(t.TypedDict, total=False):
    default: _T


class Attr(Signature[_RT, _T, _T_Attr, AttrOptions], min_args=1, kwargs={"default": _notset}):
    __slots__ = ()

    @t.overload
    def __new__(cls, name: _T_Attr, *names: _T_Attr, default=...) -> T_Self:
        ...

    def __new__(cls, *names: _T_Attr, default=_notset):
        return cls.construct(
            tuple(at for a in names for at in (a.split(".") if isinstance(a, str) else (a,))),
            _empty_dict if default is _notset else FrozenDict(default=default),
        )

    @property
    def default(self):
        if "default" in self.__kwargs__:
            return self.__kwargs__["default"]
        raise AttributeError("default")

    @property
    def has_default(self):
        return "default" in self.__kwargs__

    def __call__(self, obj) -> _RT:
        __tracebackhide__ = True
        try:
            for arg in self.__args__:
                obj = getattr(obj, arg)
            return obj
        except AttributeError as e:
            if self.has_default:
                return self.default
            raise AttributeSignatureError(self) from e

    def set(self, obj, val):
        __tracebackhide__ = True
        args = self.__args__
        try:
            for arg in args[:-1]:
                obj = getattr(obj, arg)
            setattr(obj, args[-1], val)
        except AttributeError as e:
            raise AttributeSignatureError(self) from e

    def delete(self, obj):
        __tracebackhide__ = True
        args = self.__args__
        try:
            for arg in args[:-1]:
                obj = getattr(obj, arg)
            delattr(obj, args[-1])
        except AttributeError as e:
            raise AttributeSignatureError(self) from e

    def __add__(self, o: T_OneOrManySignatures):
        cls, args, kwargs = self.__class__, self.__args__, self.__kwargs__
        if not (cls is o.__class__ and kwargs == o.__kwargs__):
            print(f"ADD ++++ {cls.__name__}: {self} + {o} = NotImplemented")
            return NotImplemented
        lhs, rhs = args, o.__args__
        items = lhs + rhs
        print(f"ADD ++++ {cls.__name__}: {self} + {o} = {cls.construct(items)}")

        return cls.construct(items, kwargs.copy())

    def __radd__(self, o: T_OneOrManySignatures):
        cls, args, kwargs = self.__class__, self.__args__, self.__kwargs__
        if not (cls is o.__class__ and kwargs == o.__kwargs__):
            print(f"RADD ++++ {cls.__name__}: {self} + {o} = NotImplemented")
            return NotImplemented
        lhs, rhs = o.__args__, args
        items = lhs + rhs
        print(f"RADD ++++ {cls.__name__}: {self} + {o} = {cls.construct(items)}")
        return cls.construct(items, kwargs.copy())


_lookup_errors = {
    LookupError: LookupSignatureError,
    KeyError: KeySignatureError,
    IndexError: IndexSignatureError,
}


class Item(
    Signature[
        _RT,
        Subscriptable[_T_Arg, _T],
        _T_Arg,
    ],
    min_args=1,
    kwargs={"default": _notset},
):
    __slots__ = ()

    @t.overload
    def __new__(cls, key: _T_Key, *keys: _T_Key, default=...) -> T_Self:
        ...

    def __new__(cls, *keys: _T_Attr, default=_notset):
        return cls.construct(
            keys,
            _empty_dict if default is _notset else FrozenDict(default=default),
        )

    @property
    def default(self):
        __tracebackhide__ = True
        if "default" in self.__kwargs__:
            return self.__kwargs__["default"]
        raise AttributeError("default")

    @property
    def has_default(self):
        return "default" in self.__kwargs__

    def __call__(self, obj) -> _RT:
        __tracebackhide__ = True
        try:
            for arg in self.__args__:
                obj = obj[arg]
            return obj
        except LookupError as e:
            if self.has_default:
                return self.default
            raise _lookup_errors.get(
                e.__class__,
            )(self) from e

    def set(self, obj, val):
        __tracebackhide__ = True
        args = self.__args__
        try:
            for arg in args[:-1]:
                obj = obj[arg]
            obj[args[-1]] = val
        except LookupError as e:
            raise (
                KeySignatureError
                if isinstance(e, KeyError)
                else IndexSignatureError
                if isinstance(e, IndexError)
                else LookupSignatureError
            )(self) from e

    def delete(self, obj):
        __tracebackhide__ = True
        args = self.__args__
        try:
            for arg in args[:-1]:
                obj = obj[arg]
            del obj[args[-1]]
        except LookupError as e:
            raise (
                KeySignatureError
                if isinstance(e, KeyError)
                else IndexSignatureError
                if isinstance(e, IndexError)
                else LookupSignatureError
            )(self) from e

    def __add__(self, o: T_OneOrManySignatures):
        cls, args, kwargs = self.__class__, self.__args__, self.__kwargs__
        if not (cls is o.__class__ and kwargs == o.__kwargs__):
            return NotImplemented
        lhs, rhs = args, o.__args__
        items = lhs + rhs
        return cls.construct(items, kwargs.copy())

    def __radd__(self, o: T_OneOrManySignatures):
        cls, args, kwargs = self.__class__, self.__args__, self.__kwargs__
        if not (cls is o.__class__ and kwargs == o.__kwargs__):
            return NotImplemented
        lhs, rhs = o.__args__, args
        items = lhs + rhs
        return cls.construct(items, kwargs.copy())


_slice_to_tuple = attrgetter("start", "stop", "step")


def _to_slice(val):
    __tracebackhide__ = True
    if isinstance(val, slice):
        return val
    elif isinstance(val, int):
        return slice(val, (val + 1) or None)
    else:
        return slice(*val)


class Slice(Item[_RT, slice], min_args=1):
    __slots__ = ()

    @t.overload
    def __new__(
        cls, key: slice | tuple[int, int, int], *keys: slice | tuple[int, int, int], default=...
    ) -> T_Self:
        ...

    def __new__(cls, *keys: slice | tuple[int, int, int], default=_notset):
        return cls.construct(
            tuple(map(_to_slice, keys)),
            _empty_dict if default is _notset else FrozenDict(default=default),
        )

    # @classmethod
    # def _parse_params_(cls, args, kwargs):
    #     __tracebackhide__ = True
    #     args, kwargs = super()._parse_params_(map(_to_slice, args), kwargs)
    #     if kwargs != cls._default_kwargs_:
    #         names, s = list(map(repr, kwargs)), "s" if len(kwargs) > 1 else ""
    #         names = " and ".join(filter(None, (", ".join(names[:-1]), names[-1])))
    #         raise TypeError(f"{cls.__name__} got unexpected keyword argument{s} {names}")
    #     return args, kwargs

    def __hash__(self) -> int:
        return hash(tuple(map(_slice_to_tuple, self.__args__)))

    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        args = [_slice_to_tuple(s) for s in args]
        return path, args, kwargs


class Call(Signature, merge=False):
    __slots__ = ()

    def __new__(cls: type[T_Self], /, *args, **kwargs) -> T_Self:
        return cls.construct(args, FrozenDict(kwargs))

    def __call__(self, obj, /, *a, **kw):
        args, kwargs = self.__args__, self.__kwargs__
        return obj(*a, *args, **kwargs | kw)


class Func(Signature[_RT, _T_Fn], min_args=1, merge=False):
    __slots__ = ()

    @property
    def __wrapped__(self):
        return self.__args__[0]

    @t.overload
    def __new__(cls: type[T_Self], func: abc.Callable, /, *args, **kwargs) -> T_Self:
        ...

    def __new__(cls: type[T_Self], /, *args, **kwargs) -> T_Self:
        return cls.construct(args, FrozenDict(kwargs))

    def __call__(self, /, *a, **kw):
        args, kwargs = self.__args__, self.__kwargs__
        return args[0](*a, *args[1:], **kwargs | kw)


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


class Operation(Signature[_RT, _T, Signature, OperationOptions]):
    __slots__ = ()

    @t.overload
    def __new__(
        cls: type[T_Self], operant: Signature, *operants: Signature, operator: str | t.Callable
    ) -> T_Self:
        ...

    def __new__(cls: type[T_Self], *operants: Signature, operator: str | t.Callable) -> T_Self:
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


class ChainOptions(t.TypedDict, total=False):
    args: tuple
    kwargs: FrozenDict
    tap: int | tuple[int, int, int]


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


@Composite.register
class Composition(Signature, abc.Sequence[Signature]):
    __slots__ = ()

    @t.overload
    def __new__(cls: type[T_Self], item: abc.Callable, /, *items: abc.Callable) -> T_Self:
        ...

    def __new__(cls: type[T_Self], /, *items: abc.Callable) -> T_Self:
        return cls.construct(_reduce(items))

    @cached_attr
    def __call__(self) -> abc.Callable[[t.Any], _T]:
        return pipeline(self.__args__, tap=-1)

    @cached_attr
    def set(self):
        func = pipeline(self.__args__[:-1] + (self.__args__[-1].set,), tap=-1)
        return func

    @cached_attr
    def delete(self):
        func = pipeline(self.__args__[:-1] + (self.__args__[-1].delete,), tap=-1)
        return func

    def _(self):
        return self.__args__[0]._() if len(self) == 1 else self

    def __str__(self) -> str:
        return " | ".join(map(repr, self))

    def __len__(self):
        return len(self.__args__)

    def __contains__(self, x):
        return x in self.__args__

    @t.overload
    def __getitem__(self, key: int) -> Signature:
        ...

    @t.overload
    def __getitem__(self, key: slice) -> T_Self:
        ...

    def __getitem__(self, key):
        val = self.__args__[key]
        if isinstance(key, slice):
            val = self.construct(val)
        return val

    def __iter__(self):
        return iter(self.__args__)

    def __reversed__(self):
        return reversed(self.__args__)

    def __add__(self, o: T_OneOrManySignatures):
        if not isinstance(o, Signature):
            return NotImplemented

        if isinstance(o, Composition):
            lhs, rhs = self.__args__, o.__args__
        else:
            lhs, rhs = self.__args__, (o,)

        if lhs and rhs:
            items = lhs[:-1] + (*_join([lhs[-1]], rhs[0]),) + rhs[1:]
        else:
            items = lhs or rhs

        return self.construct(items)

    def __radd__(self, o: T_OneOrManySignatures):
        if not isinstance(o, Signature):
            return NotImplemented

        if isinstance(o, Composition):
            lhs, rhs = o.__args__, self.__args__
        else:
            lhs, rhs = (o,), self.__args__

        if lhs and rhs:
            items = lhs[:-1] + (*_join([lhs[-1]], rhs[0]),) + rhs[1:]
        else:
            items = lhs or rhs

        return self.construct(items)

    def __or__(self, o):
        if isinstance(o, Signature):
            return (self + o)._()
        return super().__or__(o)

    def __ror__(self, o):
        if isinstance(o, Signature):
            return (o + self)._()
        return super().__ror__(o)


class SupportsSignature(t.Protocol[_RT]):
    def _(self) -> Signature[_RT]:
        ...


def _var_operator_method(nm, op):
    def method(self: "Var", /, *args):
        return Operation(self.__signature, *args, operator=op)

    method.__name__ = nm
    return method


class Var(t.Generic[_T, _RT]):
    __slots__ = ("_Var__signature",)
    __signature: Signature[_RT]

    __class_getitem__ = classmethod(GenericAlias)

    for nm, op in _builtin_operators.items():
        vars()[nm] = _var_operator_method(nm, op)
    del nm, op

    def __new__(
        cls: type[T_Self], sig: Signature[_RT] = Composition()
    ) -> _T | SupportsSignature[_RT]:
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
        cls = Slice if isinstance(key, slice) else Item
        return self.__extend__(cls(key))

    def __call__(self, /, *args, **kwargs):
        return self.__extend__(Call(*args, **kwargs))

    def __or__(self, o):
        return self.__extend__(Operation(ensure_signature(o), operator="or_"))

    def __add__(self, o):
        return self.__extend__(Operation(ensure_signature(o), operator="add"))

import math
import typing as t
from abc import abstractmethod
from collections import abc
from copy import deepcopy
from functools import cache, reduce
from itertools import chain
from logging import getLogger
from operator import attrgetter
from types import GenericAlias

import attr
from typing_extensions import Self

from zana.types import Interface
from zana.types.collections import FrozenDict
from zana.util import class_property
from zana.util import operator as zana_ops

if t.TYPE_CHECKING:
    from .operator import Operator


_T = t.TypeVar("_T")
_T_Co = t.TypeVar("_T_Co", covariant=True)


_T_Arg = t.TypeVar("_T_Arg")
_T_Kwarg = t.TypeVar("_T_Kwarg", bound=FrozenDict, covariant=True)
_T_Key = t.TypeVar("_T_Key", slice, int, str, t.SupportsIndex, abc.Hashable)
_T_Fn = t.TypeVar("_T_Fn", bound=abc.Callable)
_T_Attr = t.TypeVar("_T_Attr", bound=str)
_T_Expr = t.TypeVar("_T_Expr", bound="Closure")


logger = getLogger(__name__)
_object_new = object.__new__
_object_setattr = object.__setattr__
_empty_dict = FrozenDict()

_repr_str = attr.converters.pipe(str, repr)


_notset = t.TypeVar("_notset")


def _field_transformer(fn=None):
    def hook(cls: type, fields: list[attr.Attribute]):
        fields = fn(cls, fields) if fn else fields
        cls.__positional_attrs__ = {
            f.alias or f.name.strip("_"): f.name for f in fields if f.init and not f.kw_only
        }
        return fields

    return hook


_define = t.overload(attr.s)

if not t.TYPE_CHECKING:

    def _define(*a, no_init=True, auto_attribs=False, field_transformer=None, **kw):
        return attr.define(
            *a,
            **dict(
                frozen=True,
                init=not no_init,
                auto_attribs=auto_attribs,
                field_transformer=_field_transformer(field_transformer),
            )
            | kw,
        )


_TMany = abc.Iterator | abc.Set | abc.Sequence | abc.Mapping


@t.overload
def compose(
    obj: abc.Sequence | abc.Iterator, *, many: t.Literal[True], depth: int = 1
) -> list["Closure"]:
    ...


@t.overload
def compose(obj: abc.Mapping, *, many: t.Literal[True], depth: int = 1) -> dict[t.Any, "Closure"]:
    ...


@t.overload
def compose(obj: abc.Set, *, many: t.Literal[True], depth: int = 1) -> set["Closure"]:
    ...


@t.overload
def compose(obj=..., *, many: t.Literal[False, None] = None) -> "Closure":
    ...


def compose(obj=_notset, **kw):
    if kw.get("many") and kw.setdefault("depth", 1) > 0:
        kw["depth"] -= 1
        if isinstance(obj, (abc.Sequence, abc.Iterator)):
            return [compose(o, **kw) for o in obj]
        elif isinstance(obj, abc.Mapping):
            return {k: compose(v, **kw) for k, v in obj.items()}
        elif isinstance(obj, abc.Set):
            return {compose(v, **kw) for v in obj}
        raise TypeError(f"Expected: {_TMany}. Not {obj.__class__.__name__!r}.")
    elif obj is _notset:
        return Identity()
    elif isinstance(obj, Composable):
        return obj.__zana_compose__()
    else:
        return Val(obj)


class AbcNestedClosure(Interface[_T_Co], parent="Closure"):
    @abstractmethod
    def __call_nested__(self, *a, **kw) -> _T_Co:
        ...


class AbcNestedLazyClosure(Interface[_T_Co], parent="Closure"):
    @abstractmethod
    def __call_nested_lazy__(self, *a, **kw) -> _T_Co:
        raise NotImplementedError


class AbcRootClosure(Interface[_T_Co], parent="Closure"):
    @abstractmethod
    def __call_root__(self, *a, **kw) -> _T_Co:
        raise NotImplementedError


class AbcRootLazyClosure(Interface[_T_Co], parent="Closure"):
    @abstractmethod
    def __call_root_lazy__(self, *a, **kw) -> _T_Co:
        ...


class AbcLazyClosure(Interface[_T_Co], parent="Closure", total=False):
    @abstractmethod
    def __call_nested_lazy__(self, *a, **kw) -> _T_Co:
        raise NotImplementedError

    @abstractmethod
    def __call_root_lazy__(self, *a, **kw) -> _T_Co:
        ...


class Composable(Interface, t.Generic[_T_Co], parent="Closure"):
    __slots__ = ()

    @abstractmethod
    def __zana_compose__(self) -> "Closure[_T_Co]":
        ...

    def _(self, _: ...) -> "Closure[_T_Co]":
        ...


class Closure(t.Generic[_T_Co]):
    __slots__ = ()

    __class_getitem__ = classmethod(GenericAlias)
    __positional_attrs__: t.ClassVar[dict[str, str]] = ...
    __base__: t.ClassVar[type[Self]]
    operator: t.ClassVar["Operator"] = None
    function: t.ClassVar[abc.Callable] = None
    __abstract__: t.ClassVar[abc.Callable] = True
    __final__: t.ClassVar[abc.Callable] = False
    isterminal: t.ClassVar[bool] = False
    inplace: t.ClassVar[bool] = False

    is_root: t.ClassVar[bool] = False
    lazy: t.ClassVar[bool] = False
    source: t.ClassVar[Self | None] = None

    __concrete__: t.ClassVar[type[Self]] = None
    __overload__: t.ClassVar[bool] = False

    _nested_type_: t.ClassVar[type[Self]] = None
    _nested_lazy_type_: t.ClassVar[type[Self]] = None
    _root_type_: t.ClassVar[type[Self]] = None
    _root_lazy_type_: t.ClassVar[type[Self]] = None

    min_nargs: t.ClassVar = 0
    max_nargs: t.ClassVar = 0

    @property
    def name(self):
        return self.operator.name

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.__abstract__ = cls.__dict__.get("__abstract__", False)

        if cls.__abstract__:
            return

        if cls.__overload__:
            for prefix in ("nested_", "root_"):
                for infix in ("", "lazy_"):
                    at = f"_{prefix}{infix}type_"

                    setattr(cls, at, class_property(attrgetter(f"__concrete__.{at}")))

        else:
            ns = {
                "__overload__": True,
                "__module__": cls.__module__,
                "__concrete__": cls,
                "__new__": cls._new_.__func__,
            }

            nsl, nsr = {"lazy": True}, {"is_root": True}

            def overload(__call__, __bases__=(cls,), __name__=cls.__name__, **kwds):
                return _define(type(__name__, __bases__, ns | kwds | {"__call__": __call__}))

            def _nested_type_(self: type[cls]):
                return overload(cls.__call_nested__, source=attr.ib(kw_only=True))

            def _nested_lazy_type_(self: type[cls]):
                return overload(cls.__call_nested_lazy__, source=attr.ib(kw_only=True), **nsl)

            def _root_type_(self: type[cls]):
                return overload(cls.__call_root__, **nsr)

            def _root_lazy_type_(self: type[cls]):
                return overload(cls.__call_root_lazy__, **nsr, **nsl)

            cls.__concrete__ = cls
            cls._nested_type_ = class_property(cache(_nested_type_))
            cls._nested_lazy_type_ = class_property(cache(_nested_lazy_type_))
            cls._root_type_ = class_property(cache(_root_type_))
            cls._root_lazy_type_ = class_property(cache(_root_lazy_type_))
            new_fn = cls._new_.__func__
            if cls.__new__ != Closure.__new__:
                new_fn = cls.__new__
            cls.__new__, cls._new_ = Closure.__new__, classmethod(new_fn)

    def __new__(cls, /, *a, lazy=None, **kw):
        if cls is cls.__concrete__:
            cls = cls.get_dispatch(None is kw.get("source"), lazy)
        return cls._new_(*a, **kw)

    @classmethod
    def get_dispatch(cls, is_root: bool = True, lazy: bool = False):
        match (not not is_root, not not lazy):
            case (True, False):
                klass = cls._root_type_
            case (True, True):
                klass = cls._root_lazy_type_
            case (False, False):
                klass = cls._nested_type_
            case (False, True):
                klass = cls._nested_lazy_type_
        return klass

    @classmethod
    def _new_(cls, /, *a, **kw):
        self = _object_new(cls)
        (a or kw) and self.__attrs_init__(*a, **kw)
        return self

    if t.TYPE_CHECKING:
        _new_: t.Final[type[Self]]

    def __zana_compose__(self) -> Self:
        return self

    @abstractmethod
    def __call__(self, *a, **kw) -> _T_Co:
        raise NotImplementedError(f"{self!r} getter not supported")

    if t.TYPE_CHECKING:

        @abstractmethod
        def __call_root__(self, *a, **kw) -> _T_Co:
            raise NotImplementedError

        @abstractmethod
        def __call_nested__(self, *a, **kw) -> _T_Co:
            raise NotImplementedError

        @abstractmethod
        def __call_nested_lazy__(self, *a, **kw) -> _T_Co:
            raise NotImplementedError

        @abstractmethod
        def __call_root_lazy__(self, *a, **kw) -> _T_Co:
            raise NotImplementedError

    def __or__(self, o):
        if isinstance(o, Closure):
            return o.lift(self)
        return NotImplemented  # pragma: no cover

    def __ror__(self, o):
        if isinstance(o, Closure):
            return self.lift(o)
        return NotImplemented  # pragma: no cover

    def __ior__(self, o):
        return self.__or__(o)

    def pipe(self, op: Self | "Closure", *ops: Self | "Closure"):
        return (
            reduce(zana_ops.or_, map(compose, ops), self | compose(op))
            if ops
            else self | compose(op) | reduce(zana_ops.or_, map(compose, ops), Identity())
        )

    def lift(self, source: Self | "Closure"):
        if not self.is_root:
            source |= self.source

        attrs, kwds = attr.fields(self.__class__), {"source": source}
        for a in attrs:
            if a.init and a.alias not in kwds:
                kwds[a.alias] = getattr(self, a.name)

        return self.get_dispatch(False, self.lazy)(**kwds)

    def deconstruct(self):
        path = f"{__package__}.operator.{self.operator.identifier}"
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

        if lazy and "lazy" not in seen:
            kwargs["lazy"] = True

        return path, args, kwargs


@_define
class Val(Closure[_T_Co]):
    __slots__ = ()
    __final__ = True
    value: _T_Co = attr.ib()

    min_nargs: t.ClassVar = 1
    max_nargs: t.ClassVar = 1

    def __call_root__(self, *a) -> _T_Co:
        return self.value

    __call_nested__ = __call_nested_lazy__ = __call_root_lazy__ = None

    def lift(self, source: Self | "Closure"):
        return self


@_define
class Return(Val[_T_Co]):
    __slots__ = ()
    __final__ = True

    operator: t.Final["Operator"] = None

    def __call_nested__(self, *a) -> _T_Co:
        self.source(*a)
        return self.value

    def lift(self, source: Self | "Closure"):
        if isinstance(source, Val):
            if source.is_root:
                return self
            source = source.source
        return super(Val, self).lift(source)


@_define
class Identity(Closure[_T_Co]):
    __slots__ = ()
    __final__ = True

    def __call_root__(self, obj: _T_Co, /, *a, **kw) -> _T_Co:
        return obj

    __call_nested__ = __call_nested_lazy__ = __call_root_lazy__ = None

    def __or__(self, o: object):
        if isinstance(o, Closure):
            return o
        return NotImplemented

    def lift(self, source: Self | "Closure"):
        return source


@_define
class UnaryClosure(Closure[_T_Co]):
    __slots__ = ()
    __abstract__ = True

    def __call_nested__(self, /, *args):
        return self.function(self.source(*args))

    def __call_root__(self, obj, /):
        return self.function(obj)

    __call_nested_lazy__ = __call_root_lazy__ = None

    @classmethod
    def get_magic_method(cls):
        op = cls.operator

        def method(self: Composer):
            nonlocal op
            return self.__class__(self.__zana_compose__() | op())

        return method


@_define
class BinaryClosure(Closure[_T_Co]):
    __slots__ = ()
    __abstract__ = True

    min_nargs: t.ClassVar = 1
    max_nargs: t.ClassVar = 1

    operant: t.Any = attr.ib()

    def __call_nested__(self, /, *args):
        return self.function(self.source(*args), self.operant)

    def __call_nested_lazy__(self, /, *args):
        return self.function(self.source(*args), self.operant(*args))

    def __call_root__(self, obj, /):
        return self.function(obj, self.operant)

    def __call_root_lazy__(self, obj, /, *args):
        return self.function(obj, self.operant(obj, *args))

    @classmethod
    def get_magic_method(cls):
        op = cls.operator

        def method(self: Composer, o):
            nonlocal op
            if isinstance(o, Composable):
                return self.__class__(self.__zana_compose__() | op(o.__zana_compose__(), lazy=True))
            else:
                return self.__class__(self.__zana_compose__() | op(o))

        return method

    @classmethod
    def get_reverse_magic_method(cls):
        op = cls.operator

        def method(self: Composer, o):
            nonlocal op
            return self.__class__(compose(o) | op(self.__zana_compose__(), lazy=True))

        return method


@_define
class VariantExpression(Closure[_T_Co]):  # pragma: no cover
    __slots__ = ()
    __abstract__ = True
    operants: tuple[t.Any] = attr.ib(converter=tuple)

    def __call_nested__(self, /, *args):
        return self.function(self.source(*args[:1]), *self.operants, *args[1:])

    def __call_nested_lazy__(self, /, *args):
        pre = args[:1]
        return self.function(self.source(*pre), *(op(*pre) for op in self.operants), *args[1:])

    def __call_root__(self, /, *args):
        return self.function(*args[:1], *self.operants, *args[1:])

    def __call_root_lazy__(self, /, *args):
        pre = args[:1]
        return self.function(*pre, *(op(*pre) for op in self.operants), *args[1:])


@_define
class MutationClosure(Closure[_T_Co]):
    __slots__ = ()
    __abstract__ = True

    min_nargs: t.ClassVar = 1
    max_nargs: t.ClassVar = 1

    operant: t.Any = attr.ib()

    def __call_nested__(self, *args):
        if pre := len(args) > 1 or ():
            pre, args = args[:1], args[1:]
        return self.function(self.source(*pre), self.operant, *args)

    def __call_nested_lazy__(self, /, *args):
        if pre := len(args) > 1 or ():
            pre, args = args[:1], args[1:]
        return self.function(self.source(*pre), self.operant(*pre), *args)

    def __call_root__(self, obj, /, *args):
        return self.function(obj, self.operant, *args)

    def __call_root_lazy__(self, obj, /, *args):
        return self.function(obj, self.operant(obj), *args)


@_define
class GenericClosure(Closure[_T_Co]):
    __slots__ = ()
    __abstract__ = True

    min_nargs: t.ClassVar = 0
    max_nargs: t.ClassVar = math.inf

    args: tuple[t.Any] = attr.ib(converter=tuple, default=())
    kwargs: FrozenDict[str, t.Any] = attr.ib(default=FrozenDict(), converter=FrozenDict)
    bind: int | bool = attr.ib(default=1, converter=int)
    offset: int = attr.ib(default=1, converter=int)

    def __call_nested__(self, /, *a, **kw):
        offset, src, args, kwds = self.offset, self.source, self.args, self.kwargs
        a = (src(*a[:offset]),) + a[offset:]
        if bind := a and self.bind or ():
            bind, a = a[:bind], a[bind:]
        return self.function(*bind, *args, *a, **kwds | kw)

    def __call_nested_lazy__(self, /, *a, **kw):
        offset, src, args, kwds = self.offset, self.source, self.args, self.kwargs
        a = (src(*(pre := a[:offset])),) + a[offset:]
        if bind := a and self.bind or ():
            bind, a = a[:bind], a[bind:]

        return self.function(
            *bind,
            *(op(*pre) for op in args),
            *a,
            **{k: op(*pre) for k, op in kwds.items() if k not in kw},
            **kw,
        )

    def __call_root__(self, /, *a, **kw):
        args, kwds = self.args, self.kwargs
        if bind := a and self.bind or ():
            bind, a = a[:bind], a[bind:]
        return self.function(*bind, *args, *a, **kwds | kw)

    def __call_root_lazy__(self, /, *a, **kw):
        offset, args, kwds = self.offset, self.args, self.kwargs
        pre, a = a[:offset], a[max(offset - 1, 0) :]

        if bind := a and self.bind or ():
            bind, a = a[:bind], a[bind:]

        return self.function(
            *bind,
            *(op(*pre) for op in args),
            *a,
            **{k: op(*pre) for k, op in kwds.items() if k not in kw},
            **kw,
        )

    @classmethod
    def get_magic_method(cls):
        op = cls.operator

        def method(self: Composer, *args, **kwds):
            nonlocal op
            if any(isinstance(o, Composable) for o in chain(args, kwds.values())):
                return self.__class__(
                    self.__zana_compose__()
                    | op(
                        args and compose(args, many=True),
                        kwds and compose(kwds, many=True),
                        lazy=True,
                    )
                )
            else:
                return self.__class__(self.__zana_compose__() | op(args, kwds))

        return method


@Composable.register
class Composer(t.Generic[_T_Co]):
    __slots__ = ("__zana_composer_src__", "__weakref__")
    __zana_composer_src__: UnaryClosure[_T_Co] | BinaryClosure[_T_Co] | GenericClosure[
        _T_Co
    ] | Closure[_T_Co]
    __zana_compose_arg__: t.ClassVar = ...
    __zana_compose_attr__: t.ClassVar = "_"
    __class_getitem__ = classmethod(GenericAlias)

    def __new__(cls, src: Closure[_T_Co] = _notset) -> _T_Co | Composable[_T_Co]:
        self = _object_new(cls)
        _object_setattr(self, "__zana_composer_src__", compose(src))
        return self

    def __zana_compose__(self):
        return self.__zana_composer_src__

    def __bool__(self):
        return not isinstance(self.__zana_composer_src__, Identity)

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self.__zana_compose__())!r})"

    def __reduce__(self):
        return type(self), (self.__zana_composer_src__,)

    def __deepcopy__(self, memo):
        return type(self)(deepcopy(self.__zana_composer_src__, memo))

    def __zana_composer_call__(self, *args, **kwds) -> _T_Co | Composable[_T_Co]:
        if not kwds and len(args) == 1 and args[0] is self.__zana_compose_arg__:
            src = self.__zana_composer_src__
            if src.operator is registry.getattr and src.operant == self.__zana_compose_attr__:
                return src.source
        return self.__zana_composer_call__(*args, **kwds)

    @classmethod
    def __zana_composer_define_traps__(cls: type[Self], op: "Operator"):
        mro = [b.__dict__ for b in cls.__mro__ if issubclass(b, Composer)]
        methods = [
            (op.trap_name, getattr(op.impl, "get_magic_method", None)),
            (op.reverse_trap_name, getattr(op.impl, "get_reverse_magic_method", None)),
        ]
        for name, factory in methods:
            if factory and name and all(b.get(name) is None for b in mro):
                method = factory()
                method.__name__ = name
                method.__qualname__ = f"{cls.__qualname__}.{name}"
                method.__module__ = f"{cls.__module__}"

                if name == "__call__":
                    setattr(cls, name, cls.__zana_composer_call__)
                    name = "__zana_composer_call__"
                setattr(cls, name, method)


class magic(Composer[_T_Co]):
    __slots__ = ()

    def forbid(nm):
        def meth(self, *a, **kw) -> None:  # pragma: no cover
            raise TypeError(f"none trappable operation {nm!r}")

        meth.__name__ = meth.__qualname__ = nm
        return meth

    for nm in ("__setattr__", "__delattr__", "__setitem__", "__delitem__"):
        vars()[nm] = forbid(nm)

    del forbid, nm


this = _THIS = magic[_T_Co]()

from .operator import registry

import keyword
import operator as py_operator
import typing as t
from abc import abstractmethod
from collections import abc
from copy import deepcopy
from functools import cache, reduce, update_wrapper
from itertools import chain
from logging import getLogger
from operator import attrgetter, methodcaller
from types import FunctionType, GenericAlias, new_class

import attr
from typing_extensions import Self

from zana.types import Descriptor, Interface
from zana.types.collections import Composite, FallbackDict, FrozenDict, UserTuple
from zana.types.enums import IntEnum
from zana.util import class_property
from zana.util import operator as zana_ops

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
_T_Expr = t.TypeVar("_T_Expr", bound="Closure")


logger = getLogger(__name__)
_object_new = object.__new__
_object_setattr = object.__setattr__
_empty_dict = FrozenDict()

_repr_str = attr.converters.pipe(str, repr)

_operator_modules = (zana_ops, py_operator)

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

    def _define(*a, no_init=True, auto_attribs=False, field_transformer=None, **kw):
        return attr.define(
            *a,
            **dict(
                frozen=True,
                init=not no_init,
                # cache_hash=True,
                auto_attribs=auto_attribs,
                field_transformer=_field_transformer(field_transformer),
            )
            | kw,
        )


_TMany = abc.Iterator | abc.Set | abc.Sequence | abc.Mapping


@t.overload
def compose(obj: abc.Sequence | abc.Iterator, *, many: t.Literal[True]) -> list["Closure"]:
    ...


@t.overload
def compose(obj: abc.Mapping, *, many: t.Literal[True]) -> dict[t.Any, "Closure"]:
    ...


@t.overload
def compose(obj: abc.Set, *, many: t.Literal[True]) -> set["Closure"]:
    ...


@t.overload
def compose(obj=..., *, many: t.Literal[False, None] = None) -> "Closure":
    ...


def compose(obj=_notset, *, many=False):
    if many:
        if isinstance(obj, (abc.Sequence, abc.Iterator)):
            return [compose(o) for o in obj]
        elif isinstance(obj, abc.Mapping):
            return {k: compose(v) for k, v in obj.items()}
        elif isinstance(obj, abc.Set):
            return {compose(v) for v in obj}
        raise TypeError(f"Expected: {_TMany}. Not {obj.__class__.__name__!r}.")
    elif obj is _notset:
        return _IDENTITY
    elif isinstance(obj, Composable):
        return obj.__zana_compose__()
    else:
        return Ref(obj)


class OpType(IntEnum):
    UNARY = 1, "Unary (1 operant)"
    BINARY = 2, "Binary (2 operants)"
    VARIANT = 3, "Variant (* operants)"
    GENERIC = 4, "Generic (allow args and kwargs)"
    _impl_map_: abc.Mapping[Self, type["Closure"]]

    @property
    def implementation(self):
        return self.__class__._impl_map_[self]

    def _register(self, impl):
        assert impl is self.__class__._impl_map_.setdefault(self, impl)
        return impl


OpType._impl_map_ = {}

UNARY_OP = OpType.UNARY
BINARY_OP = OpType.BINARY
VARIANT_OP = OpType.VARIANT
GENERIC_OP = OpType.GENERIC


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
    operator: t.ClassVar["OperatorInfo"] = None
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
            match (kw.get("source") is not None, not lazy):
                case (True, True):
                    cls = cls._nested_type_
                case (True, False):
                    cls = cls._nested_lazy_type_
                case (False, True):
                    cls = cls._root_type_
                case (False, False):
                    cls = cls._root_lazy_type_

        return cls._new_(*a, **kw)

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
        return reduce(zana_ops.or_, ops, self | op) if ops else self | op

    # def rpipe(self, op: Self | "Closure", *ops: Self | "Closure"):
    #     return reduce(zana_ops.or_, ops, op) | self if ops else op | self

    def lift(self, source: Self | "Closure"):
        if not self.is_root:
            source |= self.source

        attrs, kwds = attr.fields(self.__class__), {"source": source, "lazy": self.lazy}
        for a in attrs:
            if a.init and a.alias not in kwds:
                kwds[a.alias] = getattr(self, a.name)

        return self.__concrete__(**kwds)

    def deconstruct(self):
        path = f"{__name__}.operator"
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


@_define
class Ref(Closure[_T_Co]):
    __slots__ = ()
    __final__ = True
    obj: _T_Co = attr.ib()

    def __call_root__(self, obj=None, /) -> _T_Co:
        return self.obj

    __call_nested__ = __call_nested_lazy__ = __call_root_lazy__ = None

    def lift(self, source: Self | "Closure"):
        return self


@_define
class Return(Ref[_T_Co]):
    __slots__ = ()
    __final__ = True
    operator = None

    def __call_nested__(self, *a) -> _T_Co:
        self.source(*a)
        return self.obj

    def lift(self, source: Self | "Closure"):
        if isinstance(source, Ref):
            if source.is_root:
                return self
            source = source.source
        return super(Ref, self).lift(source)


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


@UNARY_OP._register
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


@BINARY_OP._register
@_define
class BinaryOpClosure(Closure[_T_Co]):
    __slots__ = ()
    __abstract__ = True
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


@VARIANT_OP._register
@_define
class VariantOpExpression(Closure[_T_Co]):  # pragma: no cover
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


@GENERIC_OP._register
@_define
class GenericClosure(Closure[_T_Co]):
    __slots__ = ()
    __abstract__ = True
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
    __zana_composer_src__: UnaryClosure[_T_Co] | BinaryOpClosure[_T_Co] | GenericClosure[
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
        return not (self.__zana_composer_src__ is _IDENTITY)

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self.__zana_compose__())!r})"

    def __reduce__(self):
        return type(self), (self.__zana_composer_src__,)

    def __deepcopy__(self, memo):
        return type(self)(deepcopy(self.__zana_composer_src__, memo))

    def __zana_composer_call__(self, *args, **kwds) -> _T_Co | Composable[_T_Co]:
        if not kwds and len(args) == 1 and args[0] == self.__zana_compose_arg__:
            src = self.__zana_composer_src__
            if src.operator is ops.getattr and src.operant == self.__zana_compose_attr__:
                return src.source
        return self.__zana_composer_call__(*args, **kwds)

    @classmethod
    def __zana_composer_define_traps__(cls: type[Self], op: "OperatorInfo"):
        mro = [b.__dict__ for b in cls.__mro__ if issubclass(b, Composer)]
        methods = [
            (op.trap_name, getattr(op.impl, "get_magic_method", ...)),
            (op.reverse_trap_name, getattr(op.impl, "get_reverse_magic_method", ...)),
        ]
        for name, factory in methods:
            if not (name and all(b.get(name) in (None, ...) for b in mro)):
                return
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


@attr.define(frozen=True, cache_hash=True)
class OperatorInfo:
    __slots__ = ()
    name: str = attr.ib(validator=[attr.validators.instance_of(str), attr.validators.min_len(1)])
    identifier: str = attr.ib(init=False, repr=False, cmp=False)
    type: OpType = attr.ib(
        converter=OpType,
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

    _impl: t.Type[Closure] = attr.ib(kw_only=True, repr=False, cmp=False, alias="impl")

    impl: t.Type[Closure] = attr.ib(
        init=False,
        cmp=False,
        repr=attr.converters.optional(attr.converters.pipe(attrgetter("__name__"), repr)),
    )
    __call__: t.Type[Closure] = attr.ib(init=False, cmp=False, repr=False)

    @property
    def isbuiltin(self):
        return self.name in _builtin_ops

    if True:

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
            return self.type.implementation

        @impl.default
        def _init_impl(self):
            return self.make_impl_class(self._impl)

        @__call__.default
        def _init__call__(self):
            return self.impl

    def make_impl_class(self, base: t.Type[_T_Expr]):
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

        name = f"{self.identifier or base.__name__}_operation"
        name = "".join(map(methodcaller("capitalize"), name.split("_")))
        ns = {
            "__slots__": (),
            "__module__": base.__module__,
            "__name__": name,
            "__qualname__": f"{''.join(base.__qualname__.rpartition('.')[:1])}{name}",
            "__final__": True,
            "operator": self,
            "function": function,
            "isterminal": isterminal,
            "inplace": inplace,
        }

        cls = type(name, (base,), ns)
        cls = _define(cls)
        return cls

    def __str__(self) -> str:
        return self.name

    # def get_trap(op, reverse: bool = False):
    #     match (op.type, not not reverse):
    #         case (OpType.UNARY, False):

    #             def method(self: Composer):
    #                 nonlocal op
    #                 return self.__class__(self.__zana_compose__() | op())

    #         case (OpType.BINARY, False):

    #             def method(self: Composer, o):
    #                 nonlocal op
    #                 if isinstance(o, Composable):
    #                     return self.__class__(
    #                         self.__zana_compose__() | op(o.__zana_compose__(), lazy=True)
    #                     )
    #                 else:
    #                     return self.__class__(self.__zana_compose__() | op(o))

    #         case (OpType.BINARY, True):

    #             def method(self: Composer, o):
    #                 nonlocal op
    #                 return self.__class__(compose(o) | op(self.__zana_compose__(), lazy=True))

    #         case (OpType.VARIANT, False):  # pragma: no cover

    #             def method(self: Composer, *args):
    #                 nonlocal op
    #                 if any(isinstance(o, Composable) for o in args):
    #                     return self.__class__(
    #                         self.__zana_compose__() | op(compose(args, many=True), lazy=True)
    #                     )
    #                 else:
    #                     return self.__class__(self.__zana_compose__() | op(args))

    #         case (OpType.GENERIC, False):

    #             def method(self: Composer, *args, **kwds):
    #                 nonlocal op
    #                 if any(isinstance(o, Composable) for o in chain(args, kwds.values())):
    #                     return self.__class__(
    #                         self.__zana_compose__()
    #                         | op(
    #                             args and compose(args, many=True),
    #                             kwds and compose(kwds, many=True),
    #                             lazy=True,
    #                         )
    #                     )
    #                 else:
    #                     return self.__class__(self.__zana_compose__() | op(args, kwds))

    #     return method


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

    def __call__(self, operator: str, /, *args, **kwargs):
        return self[operator](*args, **kwargs)

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
        magic.__zana_composer_define_traps__(op)
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


operator = ops = OperatorRegistry()


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


_builtin_variant_ops = FrozenDict()


_builtin_generic_ops = FrozenDict.fromkeys(
    [
        # "attrgetter",
        # "itemgetter",
        # "methodcaller",
        "setitem",
        "setattr",
        "call",
        "ref",
        "return",
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
    _builtin_unary_ops | _builtin_binary_ops | _builtin_variant_ops | _builtin_generic_ops
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
    ops.register("ref", GENERIC_OP, zana_ops.none, impl=Ref, trap=None),
    ops.register("return", GENERIC_OP, zana_ops.none, impl=Return, trap=None),
    ops.register("identity", UNARY_OP, zana_ops.identity, impl=Identity, trap=None),
    ops.register("setattr", BINARY_OP, zana_ops.setattr, impl=MutationClosure),
    ops.register("setitem", BINARY_OP, zana_ops.setitem, impl=MutationClosure),
    *(ops.register(name, otp) for name, otp in _builtin_ops.items() if name not in ops),
]


_IDENTITY = Identity()

this = _THIS = magic[_T_Co]()

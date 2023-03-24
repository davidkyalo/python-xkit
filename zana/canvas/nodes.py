import keyword
import operator as builtin_operator
import typing as t
from abc import abstractmethod
from collections import abc
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
from zana.util import class_property, operator

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
_T_Expr = t.TypeVar("_T_Expr", bound="Expression")


logger = getLogger(__name__)
_object_new = object.__new__
_object_setattr = object.__setattr__
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


class OpType(IntEnum):
    UNARY = 1, "Unary (1 operant)"
    BINARY = 2, "Binary (2 operants)"
    VARIANT = 3, "Variant (* operants)"
    GENERIC = 4, "Generic (allow args and kwargs)"
    _impl_map_: abc.Mapping[Self, type["Expression"]]

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


def compose(*signatures: "SupportsExpr"):
    return Composition(map(ensure_expr, signatures))


def maybe_expr(obj) -> "Expression":
    return obj.__expr__() if isinstance(obj, SupportsExpr) else obj


def ensure_expr(obj) -> "Expression":
    if isinstance(obj, SupportsExpr):
        return obj.__expr__()
    else:
        return Ref(obj)


def operation(name: str, /, *args, **kwargs):
    return ops[name](*args, **kwargs)


class AbcNestedExpr(Interface[_T_Co], parent="Expression"):
    @abstractmethod
    def __call_nested__(self, *a, **kw) -> _T_Co:
        ...


class AbcNestedLazyExpr(Interface[_T_Co], parent="Expression"):
    @abstractmethod
    def __call_nested_lazy__(self, *a, **kw) -> _T_Co:
        raise NotImplementedError


class AbcRootExpr(Interface[_T_Co], parent="Expression"):
    @abstractmethod
    def __call_root__(self, *a, **kw) -> _T_Co:
        raise NotImplementedError


class AbcRootLazyExpr(Interface[_T_Co], parent="Expression"):
    @abstractmethod
    def __call_root_lazy__(self, *a, **kw) -> _T_Co:
        ...


class AbcLazyExpr(Interface[_T_Co], parent="Expression", total=False):
    @abstractmethod
    def __call_nested_lazy__(self, *a, **kw) -> _T_Co:
        raise NotImplementedError

    @abstractmethod
    def __call_root_lazy__(self, *a, **kw) -> _T_Co:
        ...


class SupportsExpr(Interface, t.Generic[_T_Co], parent="Expression"):
    __slots__ = ()

    @abstractmethod
    def __expr__(self) -> "Expression[_T_Co]":
        ...


class Expression(t.Generic[_T_Co]):
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

    _cs = set()

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
                # if __name__ is None:
                #     if kwds.get("is_root"):
                #         __name__ = "Root"
                #     else:
                #         __name__ = "Nested"

                #     if kwds.get("lazy"):
                #         __name__ += "Lazy"
                #     __name__ += cls.__name__
                if not __call__:
                    raise NotImplementedError
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
            if cls.__new__ != Expression.__new__:
                new_fn = cls.__new__
            cls.__new__, cls._new_ = Expression.__new__, classmethod(new_fn)

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

    # def __pos__(self):
    #     return self

    def __expr__(self) -> "Expression[_T_Co]":
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

    def __add__(self, o) -> Self:
        return NotImplemented

    def __or__(self, o):
        if isinstance(o, Expression):
            return o.lift(self)
        return NotImplemented

    def __ior__(self, o):
        return self.__or__(o)

    def evolve(self, *args, **kwds):
        args = dict(zip(self.__positional_attrs__, args)) if args else _empty_dict
        return attr.evolve(self, **args, **kwds)

    if t.TYPE_CHECKING:
        evolve: t.ClassVar[type[Self]]

    def pipe(self, op: Self | "Expression", *ops: Self | "Expression"):
        return reduce(operator.or_, ops, self | op) if ops else self | op

    def rpipe(self, op: Self | "Expression", *ops: Self | "Expression"):
        return reduce(operator.or_, ops, op) | self if ops else op | self

    def lift(self, source: Self | "Expression"):
        if not self.is_root:
            source |= self.source

        attrs, kwds = attr.fields(self.__class__), {"source": source}
        for a in attrs:
            if a.init and a.alias not in kwds:
                kwds[a.alias] = getattr(self, a.name)
        return self.__concrete__(**kwds)

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


@_define
class Ref(Expression[_T_Co]):
    __slots__ = ()
    __final__ = True
    obj: _T_Co = attr.ib()

    def __call_root__(self, obj=None, /) -> _T_Co:
        return self.obj

    __call_nested__ = __call_nested_lazy__ = __call_root_lazy__ = None

    def __or__(self, o: object):
        if isinstance(o.__class__, Ref):
            return o
        return super().__or__(o)

    def lift(self, source: Self | "Expression"):
        return self


@_define
class Return(Ref[_T_Co]):
    __slots__ = ()
    __final__ = True
    operator = None

    def __call_nested__(self, *a) -> _T_Co:
        self.source(*a)
        return self.obj

    def lift(self, source: Self | "Expression"):
        return super(Ref, self).lift(source)


@_define
class Identity(Expression[_T_Co]):
    __slots__ = ()
    __final__ = True

    def __call_root__(self, obj: _T_Co, /) -> _T_Co:
        return obj

    __call_nested__ = __call_nested_lazy__ = __call_root_lazy__ = None

    def __or__(self, o: object):
        if isinstance(o, Expression):
            return o
        return NotImplemented

    def lift(self, source: Self | "Expression"):
        return source


@UNARY_OP._register
@_define
class UnaryOpExpression(Expression[_T_Co]):
    __slots__ = ()
    __abstract__ = True

    def __call_nested__(self, /, *args):
        return self.function(self.source(*args))

    def __call_root__(self, obj, /):
        return self.function(obj)

    __call_nested_lazy__ = __call_root_lazy__ = None


@BINARY_OP._register
@_define
class BinaryOpExpression(Expression[_T_Co]):
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


@VARIANT_OP._register
@_define
class VariantOpExpression(Expression[_T_Co]):
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


@GENERIC_OP._register
@_define
class GenericOpExpression(Expression[_T_Co]):
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


_default_trap_expr = Identity()


@SupportsExpr.register
class trap(t.Generic[_T_Co]):
    __slots__ = ("__expr", "__weakref__")
    __expr: tuple[Expression[_T_Co]]

    __class_getitem__ = classmethod(GenericAlias)

    def __new__(cls, expr: Expression[_T_Co] = _default_trap_expr) -> _T_Co | SupportsExpr[_T_Co]:
        self = _object_new(cls)
        _object_setattr(self, "_trap__expr", ensure_expr(expr))
        return self

    def __expr__(self):
        return self.__expr

    def __bool__(self):
        return self.__expr is not _default_trap_expr

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self.__expr__())!r})"

    def __reduce__(self):
        return type(self), self.__expr

    def forbid(nm):
        def meth(self, *a, **kw) -> None:
            raise TypeError(f"none trappable operation {nm!r}")

        meth.__name__ = meth.__qualname__ = nm
        return meth

    for nm in ("__setattr__", "__delattr__", "__setitem__", "__delitem__"):
        vars()[nm] = forbid(nm)

    del forbid, nm

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

    _impl: t.Type[Expression] = attr.ib(kw_only=True, repr=False, cmp=False, alias="impl")

    impl: t.Type[Expression] = attr.ib(
        init=False,
        cmp=False,
        repr=attr.converters.optional(attr.converters.pipe(attrgetter("__name__"), repr)),
    )
    __call__: t.Type[Expression] = attr.ib(init=False, cmp=False, repr=False)

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

    def trap_method(op, reverse: bool = False):
        if not op.trap:
            return

        match (op.type, not not reverse):
            case (OpType.UNARY, False):

                def method(self: trap):
                    nonlocal op
                    return self.__class__(self.__expr__() | op())

            case (OpType.BINARY, False):

                def method(self: trap, o):
                    nonlocal op
                    if isinstance(o, SupportsExpr):
                        return self.__class__(self.__expr__() | op(o.__expr__(), lazy=True))
                    else:
                        return self.__class__(self.__expr__() | op(o))

            case (OpType.BINARY, True):

                def method(self: trap, o):
                    nonlocal op
                    return self.__class__(ensure_expr(o) | op(self.__expr__(), lazy=True))

            case (OpType.VARIANT, False):  # pragma: no cover

                def method(self: trap, *args):
                    nonlocal op
                    if any(isinstance(o, SupportsExpr) for o in args):
                        return self.__class__(
                            self.__expr__() | op(map(ensure_expr, args), lazy=True)
                        )
                    else:
                        return self.__class__(self.__expr__() | op(args))

            case (OpType.GENERIC, False):

                def method(self: trap, *args, **kwargs):
                    nonlocal op
                    if any(isinstance(o, SupportsExpr) for o in chain(args, kwargs.values())):
                        return self.__class__(
                            self.__expr__()
                            | op(
                                map(ensure_expr, args),
                                {k: ensure_expr(v) for k, v in kwargs.items()},
                                lazy=True,
                            )
                        )
                    else:
                        return self.__class__(self.__expr__() | op(args, kwargs))

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


_builtin_variant_ops = FrozenDict.fromkeys(
    [
        "setitem",
        "setattr",
    ],
    VARIANT_OP,
)


_builtin_generic_ops = FrozenDict.fromkeys(
    [
        # "attrgetter",
        # "itemgetter",
        # "methodcaller",
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
    ops.register("ref", GENERIC_OP, operator.none, impl=Ref, trap=None),
    ops.register("return", GENERIC_OP, operator.none, impl=Return, trap=None),
    ops.register("identity", UNARY_OP, operator.identity, impl=Identity, trap=None),
    *(ops.register(name, otp) for name, otp in _builtin_ops.items() if name not in ops),
]

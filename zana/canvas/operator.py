import keyword
import operator as py_operator
import typing as t
from collections import abc

import attr

from zana.types.collections import DefaultKeyDict, FallbackDict, FrozenDict
from zana.util import operator as zana_ops

from .core import (
    BinaryClosure,
    Closure,
    GenericClosure,
    Identity,
    MutationClosure,
    Return,
    UnaryClosure,
    Val,
    _define,
    magic,
)

_operator_modules = (zana_ops, py_operator)


@attr.define(frozen=True, cache_hash=True)
class Operator:
    __slots__ = ()
    name: str = attr.ib(validator=[attr.validators.instance_of(str), attr.validators.min_len(1)])
    identifier: str = attr.ib(init=False, repr=False, cmp=False)
    function: abc.Callable = attr.ib(
        cmp=False,
        validator=attr.validators.optional(attr.validators.is_callable()),
        repr=attr.converters.pipe(py_operator.attrgetter("__name__"), repr),
    )
    isterminal: bool = attr.ib(kw_only=True, cmp=False)
    inplace: bool = attr.ib(kw_only=True, cmp=False)
    reverse: bool | str = attr.ib(kw_only=True, cmp=False)
    trap: str | bool = attr.ib(kw_only=True, cmp=False, repr=False)
    trap_name: str = attr.ib(kw_only=True, cmp=False, init=False)
    reverse_trap_name: str = attr.ib(kw_only=True, init=False, repr=False, cmp=False)

    _impl: type[Closure] = attr.ib(kw_only=True, repr=False, cmp=False, alias="impl")

    impl: type[UnaryClosure] | type[BinaryClosure] | type[GenericClosure] = attr.ib(
        init=False,
        cmp=False,
        repr=attr.converters.optional(
            attr.converters.pipe(py_operator.attrgetter("__name__"), repr)
        ),
    )

    _aliases: tuple = attr.ib(
        default=(), repr=False, cmp=False, kw_only=True, converter=tuple, alias="aliases"
    )
    aliases: tuple = attr.ib(init=False, cmp=False)
    __call__: type[UnaryClosure] | type[BinaryClosure] | type[GenericClosure] = attr.ib(
        init=False, cmp=False, repr=False
    )

    if True:

        @property
        def isbuiltin(self):
            return self.name in ALL_OPERATORS

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
                name in BINARY_OPERATORS and name[:1] == "i" and name[1:] in BINARY_OPERATORS
            )

        @reverse.default
        def _init_reversible(self):
            name = self.name
            return all(v in BINARY_OPERATORS for v in (name, f"i{name}"))

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

        @aliases.default
        def _init_aliases(self):
            return tuple(
                dict.fromkeys(
                    k for it in ([self.name, self.identifier], self._aliases) for k in it if k
                )
            )

        @function.default
        def _default_function(self):
            name = self.identifier or self.name
            for mod in _operator_modules:
                if hasattr(mod, name):
                    return zana_ops.getattr(mod, name)

        @impl.default
        def _init_impl(self):
            return self.make_impl_class(self._impl)

        @__call__.default
        def _init__call__(self):
            return self.impl

    def make_impl_class(self, base: type[Closure]):
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

        name = f"{self.identifier or base.__name__}_closure"
        name = "".join(map(py_operator.methodcaller("capitalize"), name.split("_")))
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


class _OperatorDict(FallbackDict[str, Operator]):
    def __missing__(self, key):
        if (aka := self._default[key]) != key:
            return self[aka]
        raise KeyError(key)


class OperatorRegistry(abc.Mapping[str, Operator]):
    __slots__ = ("__aliases", "__identifiers", "__registry", "__weakref__")

    __registry: dict[str, Operator]
    __identifiers: dict[str, Operator]

    def __init__(self) -> None:
        self.__identifiers = {}
        self.__aliases = DefaultKeyDict()
        self.__registry = _OperatorDict((), self.__aliases)

    @property
    def __aliases__(self):
        return self

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    @t.overload
    def __getitem__(cls, key: "_T_OpName") -> Operator:
        ...

    @t.overload
    def __getitem__(cls, key: str) -> Operator:
        ...

    def __getitem__(cls, key):
        return cls.__registry[key]

    def __len__(self) -> int:
        return len(self.__registry)

    def __iter__(self) -> int:
        return iter(self.__registry)

    # def __call__(self, operator: str, /, *args, **kwargs):
    #     return self[operator](*args, **kwargs)

    @t.overload
    def register(self, op: Operator) -> Operator:
        ...

    register = t.overload(type[Operator])

    def register(self, op: str, *args, **kwargs) -> Operator:
        ...

    def register(self, op, *args, **kwargs):
        if isinstance(op, str):
            op = Operator(op, *args, **kwargs)
        registry = self.__registry
        assert op.name not in registry or registry[op.name] is op
        assert not op.identifier or registry.get(op.identifier, op) is op
        old = registry.setdefault(op.name, op)
        assert old is op
        if op.identifier:
            old = self.__identifiers.setdefault(op.identifier, op)
            assert old is op
            self.__class__.__annotations__[op.identifier] = Operator

        aliases, name = self.__aliases, op.name
        for aka in op.aliases:
            old = aliases.setdefault(aka, name)
            assert old == name

        magic.__zana_composer_define_traps__(op)

        return op

    if t.TYPE_CHECKING:
        register: type[Operator]

        identity: Operator
        not_: Operator
        truth: Operator
        abs: Operator
        index: Operator
        invert: Operator
        neg: Operator
        pos: Operator
        is_: Operator
        is_not: Operator
        lt: Operator
        le: Operator
        eq: Operator
        ne: Operator
        ge: Operator
        gt: Operator
        add: Operator
        and_: Operator
        floordiv: Operator
        lshift: Operator
        mod: Operator
        mul: Operator
        matmul: Operator
        or_: Operator
        pow: Operator
        rshift: Operator
        sub: Operator
        truediv: Operator
        xor: Operator
        contains: Operator
        getitem: Operator
        getattr: Operator
        iadd: Operator
        iand: Operator
        ifloordiv: Operator
        ilshift: Operator
        imod: Operator
        imul: Operator
        imatmul: Operator
        ior: Operator
        ipow: Operator
        irshift: Operator
        isub: Operator
        itruediv: Operator
        ixor: Operator
        delattr: Operator
        delitem: Operator
        setitem: Operator
        setattr: Operator
        call: Operator
        val: Operator


registry = ops = OperatorRegistry()


UNARY_OPERATORS = FrozenDict.fromkeys(
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
    FrozenDict({"impl": UnaryClosure}),
)


BINARY_OPERATORS = FrozenDict.fromkeys(
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
    FrozenDict({"impl": BinaryClosure}),
)


MUTATOR_OPERATORS = FrozenDict.fromkeys(
    [
        "setitem",
        "setattr",
    ],
    FrozenDict({"impl": MutationClosure}),
)


GENERIC_OPERATORS = FrozenDict.fromkeys(
    [
        "call",
        "val",
        "return",
    ],
    FrozenDict({"impl": GenericClosure}),
)

ALL_OPERATORS = UNARY_OPERATORS | BINARY_OPERATORS | MUTATOR_OPERATORS | GENERIC_OPERATORS

assert len(ALL_OPERATORS) == sum(
    map(len, (UNARY_OPERATORS, BINARY_OPERATORS, MUTATOR_OPERATORS, GENERIC_OPERATORS))
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
} | _terminal_ops


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
    "val",
]


identity: Operator
not_: Operator
truth: Operator
abs: Operator
index: Operator
invert: Operator
neg: Operator
pos: Operator
is_: Operator
is_not: Operator
lt: Operator
le: Operator
eq: Operator
ne: Operator
ge: Operator
gt: Operator
add: Operator
and_: Operator
floordiv: Operator
lshift: Operator
mod: Operator
mul: Operator
matmul: Operator
or_: Operator
pow: Operator
rshift: Operator
sub: Operator
truediv: Operator
xor: Operator
contains: Operator
getitem: Operator
getattr: Operator
iadd: Operator
iand: Operator
ifloordiv: Operator
ilshift: Operator
imod: Operator
imul: Operator
imatmul: Operator
ior: Operator
ipow: Operator
irshift: Operator
isub: Operator
itruediv: Operator
ixor: Operator
delattr: Operator
delitem: Operator
setitem: Operator
setattr: Operator
call: Operator
val: Operator


[
    ops.register("val", zana_ops.none, impl=Val, trap=None),
    ops.register("return", zana_ops.none, impl=Return, trap=None),
    ops.register("identity", zana_ops.identity, impl=Identity, trap=None),
    ops.register("setattr", zana_ops.setattr, impl=MutationClosure),
    ops.register("setitem", zana_ops.setitem, impl=MutationClosure),
    *(ops.register(name, **kwds) for name, kwds in ALL_OPERATORS.items() if name not in ops),
]

for __name in ALL_OPERATORS:
    if __oid := registry[__name].identifier:
        vars()[__oid] = registry[__name]
del __name, __oid


def __getattr__(nm):
    from .proxy import Proxy

    if registry.get(nm):
        return Proxy(py_operator.itemgetter(nm), registry)
    raise AttributeError(nm)

import typing as t
from abc import abstractmethod
from collections import ChainMap as BaseChainMap
from collections import UserString, abc
from copy import deepcopy
from types import FunctionType, MethodType, NoneType

from typing_extensions import Self

from . import Interface, implements

_T = t.TypeVar("_T")

_FT = t.TypeVar("_FT")
_KT = t.TypeVar("_KT", bound=abc.Hashable)
_VT = t.TypeVar("_VT")
_KT2 = t.TypeVar("_KT2", bound=abc.Hashable)
_T_Mapping = t.TypeVar("_T_Mapping", dict, abc.Mapping)
_VT2 = t.TypeVar("_VT2")
_RT = t.TypeVar("_RT")
_T_Str = t.TypeVar("_T_Str", bound=str)

_T_Func = FunctionType | staticmethod | classmethod | MethodType | type


_object_new = object.__new__
_object_setattr = object.__setattr__
_empty = object()


class Atomic(Interface, members=["__iter__", "__getitem__"], total=False, inverse=True):
    """An `Atomic` object is one that is in it's simplest form and can't be broken
    down into constituting elements.
    """

    __slots__ = ()


Atomic.register(str)
Atomic.register(bytes)
Atomic.register(abc.ByteString)
Atomic.register(UserString)


class Composite(abc.Iterable[_VT], Interface):
    """An object made up of several elemental parts.
    Interaction with these constituting elements is archived through:
        - Iteration: `Composite` objects are iterables of their constituting parts.
        - If `subscriptable`, elemental parts (or subsets of) can be accessed by subscripting.

    Examples:
        list_ = [1,2,3,4,5] #
        list_[2] = 8
        print(list_[1:-1])

    """

    __slots__ = ()

    @classmethod
    def __subclasshook__(self, cls):
        if self is Composite:
            if issubclass(cls, Atomic):
                return False
            elif issubclass(cls, abc.Sequence):
                return cls is not abc.Sequence
            elif implements(cls, ["__iter__"]):
                return True
        return NotImplemented


Composite.register(tuple)
Composite.register(list)
Composite.register(dict)
Composite.register(set)
Composite.register(frozenset)
Composite.register(abc.Iterator)
Composite.register(abc.Generator)
Composite.register(abc.Set)
Composite.register(abc.Mapping)


class Subscriptable(Interface, t.Generic[_KT, _RT]):
    """A `Subscriptable` is an object that supports element access via `__getitem__`.

    This class does not provide concrete generic implementations. Subclasses need to
    implement the __getitem__ methods.
    """

    __slots__ = ()

    @abstractmethod
    def __getitem__(self, key: _KT) -> _RT:
        ...


Subscriptable.register(tuple)
Subscriptable.register(list)
Subscriptable.register(dict)

Subscriptable.register(abc.Sequence)
Subscriptable.register(abc.Mapping)


class Compound(Composite[_VT], abc.Collection[_VT], Interface):
    """A `Composition` is a `Composite` `abc.Collection`.

    This class does not provide any concrete implementations. Subclasses need to
    take care of __contains__, __iter__, and __len__ methods.
    """

    __slots__ = ()

    @classmethod
    def __subclasshook__(self, cls):
        if self is Compound and Composite.__subclasshook__(cls) is True:
            if implements(cls, ["__len__", "__iter__", "__contains__"]):
                return True
        return NotImplemented


Compound.register(tuple)
Compound.register(list)
Compound.register(dict)
Compound.register(set)
Compound.register(frozenset)
Compound.register(abc.Set)
Compound.register(abc.Mapping)
# Compound.register(abc.Iterator)
# Compound.register(abc.Generator)


class UserDict(dict[_KT, _VT]):
    """ """

    __slots__ = ()

    __dict_or__ = dict.__or__
    __dict_ror__ = dict.__ror__
    __dict_repr__ = dict.__repr__

    def __evolve_args__(self, new=()) -> tuple:
        return (new,)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.__dict_repr__()}>"

    def __or__(self, x) -> Self:
        return self.__class__(*self.__evolve_args__(self.__dict_or__(x)))

    def __ror__(self, x) -> Self:
        return self.__class__(*self.__evolve_args__(self.__dict_ror__(x)))

    def copy(self) -> Self:
        return self.__class__(*self.__evolve_args__(self))

    __copy__ = copy

    def __reduce__(self):
        return self.__class__, self.__evolve_args__({**self})


class ReadonlyMapping(abc.Mapping[_KT, _VT]):
    """A readonly `dict` subclass.

    Raises:
        TypeError: on any attempted modification
    """

    __slots__ = ()

    _vars_ = {*vars(), "_vars_"}

    def not_mutable(self, *a, **kw):
        raise TypeError(f"readonly type: {self} ")

    __delitem__ = __setitem__ = setdefault = not_mutable
    clear = pop = popitem = update = __ior__ = not_mutable
    del not_mutable

    @classmethod
    def fromkeys(cls, it: abc.Iterable[_KT], value: _VT = None):
        return cls((k, value) for k in it)

    def __reduce__(self):
        return (self.__class__, (dict(self),))

    def copy(self):
        return self.__class__(self)

    __copy__ = copy

    def __deepcopy__(self, memo=None):
        return self.__class__({k: deepcopy(v, memo) for k, v in self.items()})

    _vars_ = _vars_ ^ {*vars()}
    __local_vars = tuple(_vars_)
    del _vars_

    @classmethod
    def define(self: type[Self], cls):
        for name in self.__local_vars:
            val = self.__dict__[name]
            setattr(cls, name, val)
        self.register(cls)
        return cls


@ReadonlyMapping.define
class ReadonlyDict(ReadonlyMapping[_KT, _VT] if t.TYPE_CHECKING else dict[_KT, _VT]):
    """A readonly `dict` subclass.
    Raises:
        TypeError: on any attempted modification
    """

    __slots__ = ()

    __or = dict[_KT, _VT].__or__

    def __or__(self, o):
        return self.__class__(self.__or(o))


class FrozenDict(ReadonlyDict[_KT, _VT]):
    """An hashable `ReadonlyDict`"""

    __slots__ = ("_hash_value",)

    def __hash__(self):
        try:
            ash = self._hash_value
        except AttributeError:
            ash = None
            items = self._eval_hashable()
            if items is not None:
                try:
                    ash = hash(items)
                except TypeError:
                    pass
            _object_setattr(self, "_hash_value", ash)

        if ash is None:
            raise TypeError(f"un-hashable type: {self.__class__.__name__!r}")

        return ash

    def _eval_hashable(self) -> abc.Hashable:
        return (*((k, self[k]) for k in sorted(self)),)


class EmptyDict(FrozenDict[_KT, NoneType]):
    __slots__ = ()

    def __missing__(self, key):
        return None


empty_dict = EmptyDict()


class ChainMap(BaseChainMap[_KT, _VT]):
    __slots__ = ()

    def _inner_(self: "ChainMap[t.Any, _T]", *a: abc.Mapping[t.Any, _T]):
        return self.__class__(*a)

    def _inner_seq_(self, it: abc.Iterable[_T] = ()):
        return list(it)[::-1]

    def chain(self, key: _KT, default=_empty, type_check: type = None):
        return self._inner_(*self.all(key, default, type_check=type_check or abc.Mapping))

    def list(self, key: _KT, default=_empty, type_check: type = None):
        its: list[abc.Iterable[_T]] = self.all(key, default, type_check=type_check or Composite)
        return self._inner_seq_(i for it in its if it for i in it)

    def all(self, key, default=_empty, *, type_check: type = None):
        if ls := list(self._iter_all(key, type_check=type_check)):
            return ls
        elif default is _empty:
            raise KeyError(key)
        else:
            return default

    def _iter_all(self, key, *, type_check: type = None):
        for m in self.maps:
            try:
                yv = m[key]
            except KeyError:
                pass
            else:
                if not (type_check is None or isinstance(yv, type_check)):
                    raise ValueError(f"expected {type_check.__name__!r} not {type(yv).__name__!r}")
                yield yv

    # def __or__(self, o):
    #     return {**self, **o}

    # def __ror__(self, o):
    #     return {**o, **self}


class DefaultDict(UserDict[_KT, _VT | _FT]):
    __slots__ = ("_default",)

    __dict_init__ = dict.__init__
    __dict_setdefault__ = dict.setdefault

    def __init__(self, arg: Self = (), default: _FT = None, /, **kwargs):
        self.__dict_init__(arg, **kwargs)
        self._default = default

    def __evolve_args__(self, new=()):
        return (new, self._default)

    def __missing__(self, key):
        return self._default

    def setdefault(self, key: _KT, value: _VT = _empty):
        if value is _empty:
            value = self.__missing__(key)
        return self.__dict_setdefault__(key, value)

    def __deepcopy__(self, memo: dict):
        cp = {deepcopy(k, memo): deepcopy(v, memo) for k, v in self.items()}
        return self.__class__(*self.__evolve_args__(cp))


class FallbackDict(DefaultDict[_KT, _VT | _FT]):
    __slots__ = ()

    _default: abc.Mapping[_KT, _FT]

    def __missing__(self, key: _KT):
        return self._default[key]


class DefaultKeyDict(UserDict[_KT, _VT]):
    """A dict that returns the `key` on attempts to retrieve a missing item.
    Example:
        d = DefaultKeyDict(abc=123)
        assert d["abc"] == 123 and d["xyz"] == "xyz" and d[234] == 234
    """

    __slots__ = ()
    __dict_setdefault__ = dict.setdefault

    def __missing__(self, key):
        return key

    def setdefault(self, key: _KT, value: _VT = _empty):
        if value is _empty:
            value = key
        return self.__dict_setdefault__(key, value)


_tuple_new = tuple.__new__


class UserTuple(tuple[_VT]):
    __slots__ = ()

    __tuple_add__ = tuple.__add__
    __tuple_mul__ = tuple.__mul__
    __tuple_rmul__ = tuple.__rmul__
    __tuple_getitem__ = tuple.__getitem__
    __tuple_prototype__ = tuple

    def __init_subclass__(cls) -> None:
        if "__constructor_class__" not in cls.__dict__:
            for b in cls.__mro__:
                if b is UserTuple:
                    break
                if any(n in b.__dict__ for n in ("__new__", "__init__", "construct")):
                    cls.__tuple_prototype__ = b
                    break
        print(f"{cls.__name__} >> {cls.__tuple_prototype__=}")

    def construct(cls, it):
        ...

    if t.TYPE_CHECKING:
        construct: type[Self]
    else:

        def __new__(cls: type[Self], iterable: abc.Iterable[str] = ()) -> Self:
            if iterable.__class__ is cls:
                return iterable
            return _tuple_new(cls, iterable)

        construct = classmethod(__new__)

    def __str__(self) -> str:
        return ", ".join(map(repr, self))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self!s})"

    def __add__(self, o: tuple) -> Self:
        __tracebackhide__ = True
        if not isinstance(o, self.__tuple_prototype__) and isinstance(o, tuple):
            o = self.__class__(o)

        rv = self.__tuple_add__(o)
        return rv if rv is NotImplemented else self.construct(rv)

    def __radd__(self, o: tuple) -> Self:
        __tracebackhide__ = True
        if isinstance(o, tuple):
            return self.__class__(o) + self
        return NotImplemented

    def __mul__(self, o: int) -> Self:
        __tracebackhide__ = True
        rv = self.__tuple_mul__(o)
        return rv if rv is NotImplemented else self.construct(rv)

    def __rmul__(self, o: int) -> Self:
        __tracebackhide__ = True
        rv = self.__tuple_rmul__(o)
        return rv if rv is NotImplemented else self.construct(rv)

    def __getitem__(self, key: int | slice):
        __tracebackhide__ = True
        rv = self.__tuple_getitem__(key)
        return self.construct(rv) if key.__class__ is slice else rv

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo) -> Self:
        return self.construct(deepcopy(o, memo) for o in self)

    def __reduce__(self):
        return self.__class__.construct, (tuple(self),)

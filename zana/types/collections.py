import typing as t
from collections import ChainMap as BaseChainMap
from collections import abc
from copy import deepcopy
from types import NoneType

from typing_extensions import Self

from . import Composite

_T = t.TypeVar("_T")

_TK = t.TypeVar("_TK", bound=abc.Hashable)
_T_Str = t.TypeVar("_T_Str", bound=str)
_TV = t.TypeVar("_TV")

_T_Pk = t.TypeVar("_T_Pk", bound=abc.Hashable)


_object_new = object.__new__
_object_setattr = object.__setattr__
_empty = object()


class ReadonlyMapping(abc.Mapping[_TK, _TV]):
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
    def fromkeys(cls, it: abc.Iterable[_TK], value: _TV = None):
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
    def define(self: type[Self], cls: _T) -> _T | type[Self]:
        print(f"\n{cls.__module__}.{cls.__name__}:")
        for name in self.__local_vars:
            print(" ++", name)
            setattr(cls, name, getattr(self, name))
        self.register(cls)
        return cls


@ReadonlyMapping.define
class ReadonlyDict(ReadonlyMapping[_TK, _TV] if t.TYPE_CHECKING else dict[_TK, _TV]):
    """A readonly `dict` subclass.
    Raises:
        TypeError: on any attempted modification
    """

    __slots__ = ()

    __or = dict[_TK, _TV].__or__

    def __or__(self, o):
        return self.__class__(self.__or(o))


class FrozenDict(ReadonlyDict[_TK, _TV]):
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


class EmptyDict(FrozenDict[_TK, NoneType]):
    __slots__ = ()

    def __missing__(self, key):
        return None


empty_dict = EmptyDict()


class ChainMap(BaseChainMap[_TK, _TV]):
    __slots__ = ()

    def _inner_(self: "ChainMap[t.Any, _T]", *a: abc.Mapping[t.Any, _T]):
        return self.__class__(*a)

    def _inner_seq_(self, it: abc.Iterable[_T] = ()):
        return list(it)[::-1]

    def chain(self, key: _TK, default=_empty, type_check: type = None):
        return self._inner_(*self.all(key, default, type_check=type_check or abc.Mapping))

    def list(self, key: _TK, default=_empty, type_check: type = None):
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

    def __or__(self, o):
        return {**self, **o}

    def __ror__(self, o):
        return {**o, **self}

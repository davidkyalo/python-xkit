import typing as t
from collections import abc

from typing_extensions import Self


class NotSetType:
    __slots__ = ("__token__", "__name__", "__weakref__")

    __self: t.Final[Self] = None

    def __new__(cls: type[Self], token=None) -> Self:
        self = cls.__self
        if self is None:
            self = cls.__self = super().__new__(cls)
        return self

    def __bool__(self):
        return False

    def __copy__(self, *memo):
        return self

    __deepcopy__ = __copy__

    def __reduce__(self):
        return type(self), (self.__token__,)

    def __json__(self):
        return self.__token__

    def __repr__(self):
        return f"NotSet({self.__token__})"

    def __hash__(self) -> int:
        return hash(self.__token__)

    def __eq__(self, other: object) -> int:
        if isinstance(other, NotSetType):
            return other.__token__ == self.__token__
        return NotImplemented

    def __ne__(self, other: object) -> int:
        if isinstance(other, NotSetType):
            return other.__token__ != self.__token__
        return NotImplemented


if t.TYPE_CHECKING:

    class NotSet(NotSetType):
        ...


NotSet = NotSetType()


class Atomic(abc.Iterable):
    __slots__ = ()


Atomic.register(str)
Atomic.register(abc.ByteString)


class Composite(abc.Iterable):
    __slots__ = ()

    @classmethod
    def __subclasshook__(self, sub):
        if self is Composite:
            if not issubclass(sub, Atomic):
                if callable(getattr(sub, "__iter__", None)):
                    return True
        return NotImplemented


Composite.register(tuple)
Composite.register(list)
Composite.register(dict)
Composite.register(set)
Composite.register(frozenset)

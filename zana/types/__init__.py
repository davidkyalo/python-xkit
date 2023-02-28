import typing as t
from abc import ABC, abstractmethod
from collections import abc
from functools import reduce
from operator import or_

from typing_extensions import Self

_T = t.TypeVar("_T")
_RT = t.TypeVar("_RT")
_empty = object()


def iter_implementations(cls: type, methods: abc.Iterable[str], type: tuple[type] | type = None):
    mro, yv = cls.__mro__, _empty
    for method in methods:
        for B in mro:
            if method in B.__dict__:
                im, yv = B.__dict__[method], None
                if im is not None and (type is None or isinstance(im, type)):
                    yv = B
                yield yv
                break
        else:
            yield (yv := None)
    if yv is _empty:
        yield None


def implements(cls: type, methods: abc.Iterable[str], type: tuple[type] | type = None):
    return all(iter_implementations(cls, methods, type))


def implements_any(cls: type, methods: abc.Iterable[str], type: tuple[type] | type = None):
    return any(iter_implementations(cls, methods, type))


class Intersect:
    """Intersection of a couple of Protocols. Allows to specify a type that
    conforms to multiple protocols without defining a separate class.
    Even though it doesn't derive Generic, mypy treats it as such when used
    with the typing_protocol_intersection plugin.
    Reads best when imported as `Has`.
    Example usage:
        >>> from typing_extensions import Protocol
        >>> from typing_protocol_intersection import And as Has
        >>> class X(Protocol): ...
        >>> class Y(Protocol): ...
        >>> class Z(Protocol): ...
        >>> def foo(bar: Has[X, Y, Z]) -> None:
        ...     pass
    See package's README or tests for more advanced examples.
    """

    def __class_getitem__(cls, _item) -> type["Intersect"]:
        return cls


class Interface(ABC):
    def __init_subclass__(
        cls,
        total: bool = None,
        members: abc.Iterable[str] = None,
        check: tuple[type] | type | bool = None,
        parents: tuple[type] | type = None,
        forbidden: tuple[type] | type = None,
        inverse: bool = None,
        strict: bool = None,
    ) -> None:
        super().__init_subclass__()
        if "__subclasshook__" not in cls.__dict__ and check is not False:
            predicate = implements_any if total is False else implements
            expected = False if inverse else True
            d_type = abc.Callable | Descriptor
            if check is True or check is None:
                check = d_type
            elif not strict:
                check = reduce(or_, check if isinstance(check, abc.Sequence) else (check,), d_type)

            if members is not None:
                members = tuple(members)

            if parents is not None:
                if not isinstance(parents, (list | tuple)):
                    parents = (parents,)

            if forbidden is not None:
                if not isinstance(forbidden, (list | tuple)):
                    forbidden = (forbidden,)

            @classmethod
            def __subclasshook__(self, sub: type[Self]):
                if self is cls:
                    names = self.__abstractmethods__ if members is None else members
                    print(
                        f"issubclass({sub.__name__},  {cls.__name__}) {parents = }, {forbidden = }\n -->"
                        f""
                        f"{predicate.__name__}({sub.__name__}, {[*names]}, {check}) is "
                        f"{expected} = {predicate(sub, names, check) is expected}\n"
                    )

                    if parents and any(not issubclass(sub, p) for p in parents):
                        return NotImplemented

                    if forbidden and any(issubclass(sub, p) for p in forbidden):
                        return NotImplemented

                    if names and predicate(sub, names, check) is expected:
                        return True
                return NotImplemented

            cls.__subclasshook__ = __subclasshook__


class Descriptor(Interface, t.Generic[_T, _RT], total=False):
    __slots__ = ()

    @classmethod
    def __subclasshook__(self, sub: type[Self]):
        if self is Descriptor:
            if implements_any(sub, self.__abstractmethods__):
                return True
        return NotImplemented

    @abstractmethod
    def __get__(self, obj: _T, cls: type[_T]) -> _RT:
        ...

    @abstractmethod
    def __set__(self, obj: _T, val: _RT) -> _RT:
        ...

    @abstractmethod
    def __delete__(self, obj: _T) -> _RT:
        ...


@Descriptor.register
class GetDescriptor(Interface, t.Generic[_T, _RT]):
    __slots__ = ()

    @abstractmethod
    def __get__(self, obj: _T, cls: type[_T]) -> _RT:
        ...


@Descriptor.register
class SetDescriptor(Interface, t.Generic[_T, _RT]):
    __slots__ = ()

    @abstractmethod
    def __set__(self, obj: _T, val: _RT) -> _RT:
        ...


@Descriptor.register
class DelDescriptor(Interface, t.Generic[_T, _RT]):
    __slots__ = ()

    @abstractmethod
    def __delete__(self, obj: _T) -> _RT:
        ...


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

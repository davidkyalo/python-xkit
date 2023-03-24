import typing as t
from abc import ABC, abstractmethod
from collections import abc
from functools import reduce
from importlib import import_module
from logging import getLogger
from operator import or_
from types import GenericAlias

from typing_extensions import Self

_T = t.TypeVar("_T")
_RT = t.TypeVar("_RT")
_empty = object()

logger = getLogger(__name__)


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
    __class_getitem__ = classmethod(GenericAlias)

    def __init_subclass__(
        cls,
        total: bool = None,
        members: abc.Iterable[str] = None,
        check: tuple[type] | type | bool = None,
        parent: tuple[type] | type = None,
        forbidden: tuple[type] | type = None,
        inverse: bool = None,
        strict: bool = None,
        predicate=None,
    ) -> None:
        super().__init_subclass__()
        if "__subclasshook__" not in cls.__dict__ and check is not False:
            if predicate is None:
                predicate = implements_any if total is False else implements
            expected = False if inverse else True
            d_type = abc.Callable | Descriptor
            f_members = f_parents = f_check = f_forbidden = lambda: None

            _module = None

            def mod():
                nonlocal _module
                if _module is None:
                    _module = import_module(cls.__module__)
                return vars(_module)

            _refs = {}
            fwd = (
                lambda v: v
                if not isinstance(v, str)
                else _refs[v]
                if v in _refs
                else _refs.setdefault(v, g[v])
                if v in (g := mod())
                else v
            )

            if check is True or check is None:
                f_check = lambda: d_type
            else:
                init = () if strict else (d_type,)
                f_check = lambda: reduce(
                    or_, v if isinstance(v := fwd(check), abc.Sequence) else (v,), *init
                )

            if members is not None:
                f_members = lambda: tuple(fwd(members))

            if parent is None:
                f_parents = lambda: ()
            elif isinstance(parent, (list | tuple)):
                f_parents = lambda: [fwd(p) for p in parent]
            else:
                f_parents = lambda: (fwd(parent),)

            if forbidden is not None:
                if not isinstance(forbidden, (list | tuple)):
                    f_forbidden = lambda: (fwd(forbidden),)
                else:
                    f_forbidden = lambda: forbidden

            @classmethod
            def __subclasshook__(self, sub: type[Self]):
                if self is cls:
                    parents, forbidden, check, members = (
                        f_parents(),
                        f_forbidden(),
                        f_check(),
                        f_members(),
                    )
                    names = self.__abstractmethods__ if members is None else members
                    msg = (
                        f"issubclass({sub.__name__},  {cls.__name__}) {parents = }, {forbidden = }\n -->"
                        f""
                        f"{predicate.__name__}({sub.__name__}, {[*names]}, {check}) is "
                        f"{expected} = {predicate(sub, names, check) is expected}\n"
                    )
                    logger.error(msg)

                    if all(issubclass(sub, p) for p in parents):
                        if not (forbidden and issubclass(sub, forbidden)):
                            if predicate(sub, names, check) is expected:
                                return True
                return NotImplemented

            cls.__subclasshook__ = __subclasshook__

            def __new__(cls: type[Self], *args, **kwargs) -> Self:
                raise TypeError(f"interfaces cannot be instantiated. {cls} called.")

            cls.__init__ = cls.__new__ = __new__


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

import typing as t

from importlib import import_module
from typing_extensions import Self


class NotSetType:
    __slots__ = ()
    __self: t.Final[Self] = None

    def __new__(cls: type[Self]) -> Self:
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
        return f"{__name__}.NotSet"

    def __json__(self):
        return

    def __repr__(self):
        return "NotSet"


if t.TYPE_CHECKING:

    class NotSet(NotSetType):
        ...


NotSet = NotSetType()

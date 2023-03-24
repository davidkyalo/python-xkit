from unittest.mock import MagicMock, Mock, PropertyMock


class StaticMixin:
    __slots__ = ()

    def __copy__(self, *memo):
        return self

    __deepcopy__ = __copy__


class StaticMock(StaticMixin, Mock):
    ...


class StaticMagicMock(StaticMixin, MagicMock):
    ...


class StaticPropertyMock(StaticMixin, PropertyMock):
    ...

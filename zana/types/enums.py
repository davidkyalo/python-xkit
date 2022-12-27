import enum
import typing as t
from functools import reduce
from operator import or_
from types import DynamicClassAttribute

from typing_extensions import Self

__all__ = ["Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"]

try:
    from django.db.models.enums import Choices as Enum
    from django.db.models.enums import ChoicesMeta as EnumMeta
    from django.db.models.enums import IntegerChoices as IntEnum
    from django.db.models.enums import TextChoices as StrEnum
except ImportError:

    class EnumMeta(enum.EnumMeta):
        """A metaclass for creating a enum choices."""

        def __new__(metacls, classname, bases, classdict, **kwds):
            labels = []
            for key in classdict._member_names:
                value = classdict[key]
                if (
                    isinstance(value, (list, tuple))
                    and len(value) > 1
                    and isinstance(value[-1], str)
                ):
                    *value, label = value
                    value = tuple(value)
                else:
                    label = key.replace("_", " ").title()
                labels.append(label)
                # Use dict.__setitem__() to suppress defenses against double
                # assignment in enum's classdict.
                dict.__setitem__(classdict, key, value)
            cls = super().__new__(metacls, classname, bases, classdict, **kwds)
            for member, label in zip(cls.__members__.values(), labels):
                member._label_ = label
            return enum.unique(cls)

        def __contains__(cls, member):
            if not isinstance(member, enum.Enum):
                # Allow non-enums to match against member values.
                return any(x.value == member for x in cls)
            return super().__contains__(member)

        @property
        def names(cls):
            empty = ["__empty__"] if hasattr(cls, "__empty__") else []
            return empty + [member.name for member in cls]

        @property
        def choices(cls):
            empty = [(None, cls.__empty__)] if hasattr(cls, "__empty__") else []
            return empty + [(member.value, member.label) for member in cls]

        @property
        def labels(cls):
            return [label for _, label in cls.choices]

        @property
        def values(cls):
            return [value for value, _ in cls.choices]

    class Enum(enum.Enum, metaclass=EnumMeta):
        """Class for creating enumerated choices."""

        @DynamicClassAttribute
        def label(self):
            return self._label_

        @property
        def do_not_call_in_templates(self):
            return True

        def __str__(self):
            """
            Use value when cast to str, so that Choices set as model instance
            attributes are rendered as expected in templates and similar contexts.
            """
            return str(self.value)

        # A similar format was proposed for Python 3.10.
        def __repr__(self):
            return f"{self.__class__.__qualname__}.{self._name_}"


    class IntEnum(int, Enum):
        """Class for creating enumerated integer choices."""
        pass


    class StrEnum(str, Enum):
        """Class for creating enumerated string choices."""

        def _generate_next_value_(name, start, count, last_values):
            return name



class FlagEnumMeta(EnumMeta):
   
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._member_map_:
            self._all_ = None
    
    @property
    def all(self) -> Self:
        if self._all_ is None:
            if union := reduce(or_, map(int, self._member_map_.values()), 0):
                self._all_ = self(union)
        return self._all_



class Flag(Enum, enum.Flag, metaclass=FlagEnumMeta):
    
    all: Self
    _all_: Self

    def __iter__(self):
        return (m for m in self.__class__ if m in self)

    # def __contains__(self: Self, other: Self) -> bool:
    #     return not not (self & other)

    def __len__(self) -> int:
        return f'{self:b}'.count('1')


class IntFlagChoices(Flag, enum.IntFlag):
    """Class for creating enumerated integer flag choices."""

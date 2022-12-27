from importlib import import_module

import pytest

from zana.common import try_import
from zana.types.enums import Enum, EnumMeta


class test_Enum:

    def test_basic(self):

        class Choices(Enum):
            one = object()
            two = object()
            three = object()
        
        assert isinstance(Choices, EnumMeta)


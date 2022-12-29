from cProfile import label
from enum import auto
from importlib import import_module
from re import A

import pytest

from zana.common import try_import
from zana.types.enums import Flag, StrEnum


class test_Enum:

    def test_basic(self):

        class Foo(StrEnum):
            __empty__ = None
            one = '111', 'Moja'
            two = auto()
            three = '333', 'Tatu'

        print(f'{Foo}', *[f'{m!s}: {m!r}' for m in Foo], sep='\n  - ')

        assert Foo.choices == [(None, None), *((m._value_, m.label) for m in Foo)]
        assert Foo.one._value_ == '111'
        assert Foo.two._value_ == 'two'
        assert Foo.one.label == 'Moja'
        assert Foo.three.label == 'Tatu'
        
        assert Foo.names == ['__empty__'] + [m.name for m in Foo]
        assert Foo.labels == [label for _, label in Foo.choices]
        assert Foo.values == [value for value, _ in Foo.choices]
        assert all(m in Foo and m._value_ in Foo for m in Foo)


class test_Flag:

    def test_basic(self):

        class Foo(Flag):
            one = auto()
            two = auto()
            three = auto()

        assert Foo.one in Foo.two | Foo.one
        assert not Foo.three in Foo.two | Foo.one
            
        for m in Foo:
            assert m and m in Foo.all

        assert Foo.all == Foo.one | Foo.two | Foo.three

        assert len(Foo.all) == 3
        assert len(Foo.one) == 1
        assert len(Foo.one | Foo.two) == 2
        assert list(Foo.two | Foo.one) == [Foo.one, Foo.two]
        


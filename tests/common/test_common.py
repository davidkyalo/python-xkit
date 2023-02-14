import typing as t
from functools import partial
from importlib import import_module
from operator import attrgetter, itemgetter
from types import MappingProxyType, SimpleNamespace
from unittest.mock import CallableMixin, Mock

import pytest

from zana.common import (
    cached_attr,
    class_property,
    kw_apply,
    pipe,
    pipeline,
    try_import,
)


class ExampleType:
    __slots__ = ()

    mk_foo: Mock
    foo: t.Union[property, Mock]
    has_mutators: bool


_T = t.TypeVar("_T", bound=ExampleType)


class test_class_property:
    @pytest.fixture()
    def Type(self, TypeWithDict):
        return TypeWithDict

    @pytest.fixture()
    def TypeWithDict(self):
        class Type:
            mk_foo = Mock()

            @class_property().getter
            def foo(cls):
                assert cls is Type
                return cls.mk_foo(cls)

        return Type

    @pytest.fixture()
    def TypeWithDictSlots(self):
        class Type:
            __slots__ = ("__dict__",)

            mk_foo = Mock()

            @class_property
            def foo(cls):
                assert cls is Type
                return cls.mk_foo(cls)

        return Type

    @pytest.fixture()
    def TypeWithNoDict(self):
        class Type:
            __slots__ = ()

            mk_foo = Mock()

            @class_property
            def foo(cls):
                assert cls is Type
                return cls.mk_foo(cls)

        return Type

    @pytest.fixture()
    def TypeWithImmutableDict(self):
        class Type:
            __slots__ = ()

            mk_foo = Mock()

            @property
            def __dict__(self):
                return MappingProxyType({})

            @class_property
            def foo(cls):
                assert cls is Type
                return cls.mk_foo(cls)

        return Type

    def test_basic(self, Type: type[_T]):
        obj = Type()
        assert obj.foo is Type.foo is Type.mk_foo.return_value

    def test_set_name(self, Type):
        attr = class_property()

        class Bar:
            x = attr

        attr.__set_name__(Bar, "x")
        pytest.raises(TypeError, attr.__set_name__, Bar, "y")
        pytest.xfail("name mismatch")

    @pytest.mark.parametrize(
        "Type",
        [
            pytest.lazy_fixture("TypeWithDict"),
            pytest.lazy_fixture("TypeWithDictSlots"),
        ],
    )
    def test_with_instance_attribute(self, Type: type[_T]):
        mk_two = Mock()

        obj = Type()
        assert obj.foo is Type.foo is Type.mk_foo.return_value
        del obj.foo
        obj.foo = mk_two
        assert (obj.foo, Type.foo) == (mk_two, Type.mk_foo.return_value)
        del obj.foo
        assert obj.foo is Type.foo is Type.mk_foo.return_value

    @pytest.mark.parametrize(
        "Type",
        [
            pytest.lazy_fixture("TypeWithNoDict"),
            pytest.lazy_fixture("TypeWithImmutableDict"),
        ],
    )
    def test_set_immutable_instance_attribute(self, Type: type[_T]):
        obj = Type()
        assert obj.foo is Type.foo is Type.mk_foo.return_value
        with pytest.raises(TypeError):
            obj.foo = Mock()
        pytest.xfail("is immutable")

    @pytest.mark.parametrize(
        "Type",
        [
            pytest.lazy_fixture("TypeWithNoDict"),
            pytest.lazy_fixture("TypeWithImmutableDict"),
        ],
    )
    def test_del_immutable_instance_attribute(self, Type: type[_T]):
        obj = Type()
        assert obj.foo is Type.foo is Type.mk_foo.return_value
        with pytest.raises(TypeError):
            del obj.foo
        pytest.xfail("is immutable")


class test_cached_attr:
    @pytest.fixture()
    def Type(self, TypeWithDict):
        return TypeWithDict

    @pytest.fixture()
    def TypeWithDict(self):
        class Type:
            mk_foo = Mock()
            has_mutators = False

            @cached_attr().getter
            def foo(self):
                assert self.__class__ is Type
                return self.mk_foo(self)

        return Type

    @pytest.fixture()
    def TypeWithDictSlots(self):
        class Type:
            __slots__ = ("__dict__",)

            mk_foo = Mock()
            has_mutators = False

            @cached_attr
            def foo(self):
                assert self.__class__ is Type
                return self.mk_foo(self)

        return Type

    @pytest.fixture()
    def TypeWithNoDict(self):
        class Type:
            __slots__ = ()

            has_mutators = False
            mk_foo = Mock()

            @cached_attr
            def foo(self):
                assert self.__class__ is Type
                return self.mk_foo(self)

        return Type

    @pytest.fixture()
    def TypeWithImmutableDict(self):
        class Type:
            __slots__ = ()

            mk_foo = Mock()
            has_mutators = False

            @property
            def __dict__(self):
                return object.__dict__

            @cached_attr
            def foo(self):
                assert self.__class__ is Type
                return self.mk_foo(self)

        return Type

    @pytest.fixture()
    def TypeWithMutators(self):
        class Type:
            mk_foo = Mock()
            has_mutators = True

            @cached_attr
            def foo(self):
                assert self.__class__ is Type
                return self.mk_foo(self)

            @foo.setter
            def foo(self, val):
                assert self.__class__ is Type
                self.__dict__["foo"] = val
                return self.mk_foo(self, val)

            @foo.deleter
            def foo(self):
                assert self.__class__ is Type
                self.__dict__.pop("foo", ...)
                return self.mk_foo(self, ...)

        return Type

    def test_basic(self, Type: type[_T]):
        obj = Type()
        assert isinstance(Type.foo, cached_attr)
        assert obj.foo is obj.mk_foo.return_value is obj.foo
        Type.mk_foo.assert_called_once_with(obj)

    def test_set_name(self, Type: type[_T]):
        attr = cached_attr()

        class Bar:
            x = attr

        attr.__set_name__(Bar, "x")
        pytest.raises(TypeError, attr.__set_name__, Bar, "y")
        pytest.xfail("name mismatch")

    @pytest.mark.parametrize(
        "Type",
        [
            pytest.lazy_fixture("TypeWithDict"),
            pytest.lazy_fixture("TypeWithDictSlots"),
            pytest.lazy_fixture("TypeWithMutators"),
        ],
    )
    def test_with_instance_attribute(self, Type: type[_T]):
        mk_two = Mock()

        obj = Type()
        assert obj.foo is obj.mk_foo.return_value is obj.foo
        obj.mk_foo.reset_mock()
        obj.foo = mk_two
        Type.has_mutators and obj.mk_foo.assert_called_once_with(obj, mk_two)
        assert obj.foo is mk_two
        obj.mk_foo.reset_mock()
        del obj.foo
        Type.has_mutators and obj.mk_foo.assert_called_once_with(obj, ...)
        del obj.foo

        assert obj.foo is obj.mk_foo.return_value is obj.foo

    @pytest.mark.parametrize(
        "Type",
        [
            pytest.lazy_fixture("TypeWithNoDict"),
            pytest.lazy_fixture("TypeWithImmutableDict"),
        ],
    )
    def test_get_immutable_instance_attribute(self, Type: type[_T]):
        obj = Type()
        with pytest.raises(TypeError):
            obj.foo
        pytest.xfail("cannot cache")

    @pytest.mark.parametrize(
        "Type",
        [
            pytest.lazy_fixture("TypeWithNoDict"),
            pytest.lazy_fixture("TypeWithImmutableDict"),
        ],
    )
    def test_set_immutable_instance_attribute(self, Type: type[_T]):
        obj = Type()
        with pytest.raises(TypeError):
            obj.foo = Mock()
        pytest.xfail("is immutable")

    @pytest.mark.parametrize(
        "Type",
        [
            pytest.lazy_fixture("TypeWithNoDict"),
            pytest.lazy_fixture("TypeWithImmutableDict"),
        ],
    )
    def test_del_immutable_instance_attribute(self, Type: type[_T]):
        obj = Type()
        with pytest.raises(TypeError):
            del obj.foo
        pytest.xfail("is immutable")


class test_try_import:
    class Inner:
        obj = object()

    def test_module_import(self):
        assert try_import(__name__) is import_module(__name__)

    def test_module_object_import(self):
        assert try_import(f"{__name__}.test_try_import") is test_try_import
        assert try_import(f"{__name__}", "test_try_import.Inner.obj") is test_try_import.Inner.obj
        assert try_import(f"{__name__}:test_try_import.Inner.obj") is test_try_import.Inner.obj

    def test_with_default(self):
        default = object()
        assert try_import(default, default=default) is default
        assert try_import(f"{__name__}:asdwqewq", default=default) is default
        assert try_import("foobar.awqewqe.asdwqewq", default=default) is default

    @pytest.mark.parametrize(
        "path",
        [
            pytest.param(object(), marks=pytest.mark.xfail(raises=TypeError)),
            pytest.param(
                f"{__name__}:a_fake_member", marks=pytest.mark.xfail(raises=AttributeError)
            ),
            pytest.param(
                "a_fake_package.a_fake_module", marks=pytest.mark.xfail(raises=ImportError)
            ),
        ],
    )
    def test_without_default(self, path):
        try_import(path)


class test_pipeline:
    @pytest.mark.parametrize(
        "expected, pipes, args, kwargs, call_args, call_kwargs",
        [
            (
                "object(**attrs).foo['bar']",
                [SimpleNamespace, attrgetter("foo"), itemgetter("bar")],
                (),
                {},
                (),
                {"foo": {"bar": "object(**attrs).foo['bar']"}},
            )
        ],
    )
    def test(self, expected, pipes, args, kwargs, call_args, call_kwargs):
        pp = pipeline(pipes, *(args or ()), **(kwargs or {}))
        result = pp(*(call_args or ()), **(call_kwargs or {}))
        if callable(expected) and not isinstance(expected, CallableMixin):
            expected(result)
        else:
            assert result == expected


class test_pipe:
    @pytest.mark.parametrize(
        "expected, pipes, args, kwargs",
        [
            (
                "object(**attrs).foo['bar']",
                [partial(kw_apply, SimpleNamespace), attrgetter("foo"), itemgetter("bar")],
                ({"foo": {"bar": "object(**attrs).foo['bar']"}},),
                {},
            )
        ],
    )
    def test(self, expected, pipes, args, kwargs):
        result = pipe(pipes, *(args or ()), **(kwargs or {}))
        if callable(expected) and not isinstance(expected, CallableMixin):
            expected(result)
        else:
            assert result == expected

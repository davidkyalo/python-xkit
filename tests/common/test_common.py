import pytest

from importlib import import_module
from zana.common import try_import


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
        assert try_import("foobar.awqewqe.asdwqewq", default=default) is default

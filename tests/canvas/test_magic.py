from operator import methodcaller

import pytest as pyt

from tests.canvas.conftest import T_ExprMode, T_ExprType
from zana.canvas import (
    ALL_OPERATORS,
    AbcLazyClosure,
    BinaryClosure,
    Closure,
    Operator,
    Val,
    compose,
    magic,
)
from zana.testing.mock import StaticMagicMock, StaticMock


class test_magic:
    @pyt.fixture(params=[*ALL_OPERATORS])
    def op_name(self, request: pyt.FixtureRequest):
        return request.param

    def test(self, op: Operator, expr_type: T_ExprType, expr_mode: T_ExprMode):
        if not op.trap:
            pyt.skip(f"Operator {op!s} not trappable")
        is_lazy, is_root = expr_mode == "lazy", expr_type == "root"
        src = None
        if not is_root:
            src = Val(StaticMock)

        obj = magic(*([] if src is None else [src]))
        if is_root:
            assert not obj
        else:
            assert compose(src) is obj.__zana_compose__() is obj(...)

        args = tuple(f"var_{i}" for i in range(min(5, op.impl.max_nargs)))
        if is_lazy:
            args = tuple(map(Val, args))

        method = methodcaller(op.trap_name, *args)

        sub = method(obj)

        print("", str(sub), repr(sub), "\n")
        assert isinstance(sub, magic)

        exp = sub.__zana_compose__()
        assert isinstance(exp, Closure)
        assert exp.source == (None if is_root else obj.__zana_compose__())
        assert isinstance(exp, op.impl)

        if op.reverse and op.reverse_trap_name:
            rmethod = methodcaller(op.reverse_trap_name, *args)
            sub_r = rmethod(obj)
            exp_r: BinaryClosure = sub_r.__expr__()
            assert isinstance(exp, AbcLazyClosure)
            assert exp_r.source == compose(exp.operant)
            assert exp.source is None if is_root else exp_r.operant == exp.source

from operator import methodcaller

import pytest as pyt

from tests.canvas.conftest import T_ExprMode, T_ExprType
from zana.canvas.nodes import (
    AbcLazyExpr,
    BinaryOpExpression,
    Expression,
    GenericOpExpression,
    OperatorInfo,
    OpType,
    Ref,
    VariantOpExpression,
    _builtin_ops,
    ensure_expr,
    trap,
)
from zana.testing.mock import StaticMock


class test_trap:
    @pyt.fixture(params=[*_builtin_ops])
    def op_name(self, request: pyt.FixtureRequest):
        return request.param

    def test(self, op: OperatorInfo, expr_type: T_ExprType, expr_mode: T_ExprMode):
        if not op.trap:
            pyt.skip(f"Operator {op!s} not trappable")
        is_lazy, is_root = expr_mode == "lazy", expr_type == "root"
        src = None if is_root else StaticMock()
        obj = trap(*([] if src is None else [src]))
        assert not obj if is_root else ensure_expr(src) == obj.__expr__()

        args = tuple(f"var_{i}" for i in range(+op.type - 1))
        if is_lazy:
            args = tuple(map(Ref, args))

        method = methodcaller(op.trap_name, *args)

        sub = method(obj)

        print("", str(sub), repr(sub), "\n")
        assert isinstance(sub, trap)

        exp = sub.__expr__()
        assert isinstance(exp, Expression)
        assert exp.source == (None if is_root else obj.__expr__())
        assert isinstance(exp, op.impl)

        match op.type:
            case OpType.UNARY:
                assert not args
            case OpType.BINARY:
                assert not isinstance(exp, BinaryOpExpression) or (exp.operant,) == args
            case OpType.VARIANT:
                assert not isinstance(exp, VariantOpExpression) or exp.operants == args
            case OpType.GENERIC:
                assert not isinstance(exp, GenericOpExpression) or exp.args == args

        if op.reverse and op.reverse_trap_name:
            rmethod = methodcaller(op.reverse_trap_name, *args)
            sub_r = rmethod(obj)
            exp_r: BinaryOpExpression = sub_r.__expr__()
            assert isinstance(exp, AbcLazyExpr)
            assert exp_r.source == ensure_expr(exp.operant)
            assert exp.source is None if is_root else exp_r.operant == exp.source

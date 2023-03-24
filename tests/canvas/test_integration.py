from copy import copy, deepcopy
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
    operators,
    to_expr,
    trap,
)
from zana.testing.mock import StaticMagicMock, StaticMock, StaticPropertyMock


class test_integration:
    def test_deep_nesting(self):
        mk = StaticMagicMock()
        mk_bar = StaticPropertyMock()
        mk_bar_rv = mk_bar.return_value
        mk_foo: StaticMagicMock = mk.abc["xyz"].foo

        class FooMock(StaticMock):
            bar = mk_bar

        mk_foo.return_value = FooMock()
        args = trap().xyz(), Ref(StaticMock()), StaticMock()
        foo = trap().abc["xyz"].foo
        foo_rv = foo(*args)
        src = +foo_rv.bar
        sub = src == +mk_bar_rv
        subr = +mk_bar_rv == src

        expr, exprr = to_expr(sub), to_expr(subr)
        exprl = to_expr(src) | operators.is_(to_expr(src), lazy=True)

        print("VARS", *(f" - {k:16}: {v and v or v!r}" for k, v in vars().items()), "", sep="\n ")
        cp_s, dcp_s = to_expr(copy(sub)), to_expr(deepcopy(sub))
        cp_e, dcp_e = copy(expr), deepcopy(expr)
        assert to_expr(src)(mk) is +mk_bar.return_value

        mk_foo.assert_called_once_with(mk.xyz.return_value, args[1](mk), args[2])
        mk.xyz.assert_called_once_with()

        assert expr(mk) is exprr(mk) is exprl(mk) is True
        assert cp_e(mk) is dcp_e(mk) is cp_s(mk) is dcp_s(mk) is True
        assert to_expr(src != mk_bar)(mk) is to_expr(mk_foo != src)(mk) is True
        assert to_expr(src != mk_foo)(mk) is True

        mk_bar.reset_mock()
        op = to_expr(foo_rv) | operators.setattr(("bar",))
        mk_val = StaticMock()
        op(mk, mk_val)
        mk_bar.assert_called_once_with(mk_val)

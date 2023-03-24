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


class test_integration:
    def test(self):
        assert 0

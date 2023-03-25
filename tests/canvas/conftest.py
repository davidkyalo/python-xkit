import typing as t

import pytest as pyt

from zana.canvas import (
    AbcNestedClosure,
    AbcNestedLazyClosure,
    AbcRootClosure,
    AbcRootLazyClosure,
    operators,
)
from zana.testing.mock import StaticMagicMock
from zana.util.operator import identity

T_ExprType = t.Literal["nested", "root"]
T_ExprMode: type[t.Literal["eager", "lazy"]] = t.Literal["eager", "lazy"]


@pyt.fixture
def op_name():
    return


@pyt.fixture(params=["nested", "root"])
def expr_type(request: pyt.FixtureRequest, op_name):
    return request.param


@pyt.fixture(params=["eager", "lazy"])
def expr_mode(request: pyt.FixtureRequest, op_name):
    return request.param


@pyt.fixture()
def expr_source():
    return StaticMagicMock(identity, wraps=identity)


@pyt.fixture()
def expr_args():
    return ()


@pyt.fixture()
def expr_kwargs(
    expr_source, expr_type: t.Literal["nested", "root"], expr_mode: t.Literal["eager", "lazy"]
):
    kwds = {}
    if expr_type == "nested":
        kwds["source"] = expr_source
    if expr_mode == "lazy":
        kwds["lazy"] = True
    return kwds


@pyt.fixture()
def op_abc_map():
    return {
        ("nested", "eager"): AbcNestedClosure,
        ("nested", "lazy"): AbcNestedLazyClosure,
        ("root", "eager"): AbcRootClosure,
        ("root", "lazy"): AbcRootLazyClosure,
    }


@pyt.fixture()
def op_abc(op_abc_map: dict, expr_type, expr_mode):
    return op_abc_map[expr_type, expr_mode]


@pyt.fixture()
def op(
    request: pyt.FixtureRequest,
    op_name: str,
    op_abc: type,
    expr_type: t.Literal["nested", "root"],
    expr_mode: t.Literal["eager", "lazy"],
):
    op = operators[op_name]

    if not (op_abc and issubclass(op.impl, op_abc)):
        pyt.skip(
            f"{op_name!r} doesn't support {expr_type!r} operations in {expr_mode!r} expr_mode."
        )
    return op

import operator as py_operator
from collections import Counter
from copy import copy, deepcopy
from functools import reduce
from itertools import repeat
from operator import or_
from types import SimpleNamespace
from unittest.mock import Mock

import pytest as pyt

from zana.canvas import (
    ALL_OPERATORS,
    AbcNestedClosure,
    AbcNestedLazyClosure,
    AbcRootClosure,
    AbcRootLazyClosure,
    Closure,
    Identity,
    Operator,
    Return,
    Val,
    compose,
    operator,
)
from zana.testing.mock import StaticMock
from zana.util import try_import


@pyt.fixture()
def expr(op: Operator, expr_args: tuple, expr_kwargs: dict, make_expr):
    return make_expr(op, *expr_args, **expr_kwargs)


@pyt.fixture()
def make_expr():
    def make(op: Operator, *a, **kw):
        return op(*a, **kw)

    return make


@pyt.fixture
def op_params(request, op_name):
    return deepcopy(request.instance.operators[op_name])


_py_ops = {
    n: d
    for n, v in vars(py_operator).items()
    if (d := f"__{n.strip('_')}__") != n and getattr(py_operator, d, ...) is v
}


class test_Registry:
    def test(self):
        builtin = ALL_OPERATORS | _py_ops
        available = {*operator} & {*builtin}
        print(operator)
        assert len(builtin) >= 51 <= min(len(available), len(operator))

    def test_compose(self):
        abc = StaticMock(Closure)
        xyz = StaticMock()
        dct = compose(dict(abc=abc, xyz=xyz), many=True)
        st = compose({abc, xyz}, many=True)
        ls = compose([abc, xyz], many=True)
        assert dct == dict(abc=abc.__zana_compose__.return_value, xyz=Val(xyz))
        assert st == {abc.__zana_compose__.return_value, Val(xyz)}
        assert ls == [abc.__zana_compose__.return_value, Val(xyz)]

        with pyt.raises(TypeError):
            compose(1, many=True)


class test_Identity:
    def test(self):
        obj = Identity()
        mk = StaticMock()
        other = StaticMock(Closure)

        assert obj.is_root is True
        assert obj.lazy is False

        assert isinstance(obj, AbcRootClosure)
        assert not isinstance(obj, AbcNestedClosure | AbcNestedLazyClosure | AbcRootLazyClosure)

        cp, dcp = copy(obj), deepcopy(obj)
        assert obj == cp == dcp
        assert obj(mk) is mk
        assert obj | other is other
        assert other | obj is other

        with pyt.raises(TypeError):
            obj | StaticMock()


#


class test_Ref:
    def test(self):
        mk = StaticMock()
        obj = Val(mk)
        other_e, other_o = StaticMock(Closure), StaticMock(Val)

        assert obj.is_root is True
        assert obj.lazy is False

        assert isinstance(obj, AbcRootClosure)
        assert not isinstance(obj, AbcNestedClosure | AbcNestedLazyClosure | AbcRootLazyClosure)

        cp, dcp = copy(obj), deepcopy(obj)
        assert obj == cp == dcp
        assert obj(Mock()) is mk is obj()
        assert obj | other_e is other_e.lift.return_value
        assert other_e | obj is obj
        assert other_o | obj is obj

        other_e.lift.assert_called_once_with(obj)


class test_Return:
    def test(self):
        mk, mk_src = StaticMock(), StaticMock(Closure)
        obj = Return(mk)
        arg = Mock()
        other_e, other_or, other_on = StaticMock(Closure), StaticMock(Return), StaticMock(Return)
        other_or.is_root = True
        other_on.is_root = False
        other_on.source = mk_src

        assert obj.is_root is True
        assert obj.lazy is False

        assert isinstance(obj, AbcRootClosure | AbcNestedClosure)
        assert not isinstance(obj, AbcNestedLazyClosure | AbcRootLazyClosure)

        cp, dcp = copy(obj), deepcopy(obj)
        assert obj == cp == dcp
        assert obj(arg) is mk is obj()

        assert Val(678) | obj is obj
        assert other_or | obj is obj
        assert obj | other_e is other_e.lift.return_value
        other_e.lift.assert_called_once_with(obj)

        nested = other_e | obj
        assert nested(arg) is mk
        other_e.assert_called_once_with(arg)

        d_nested = other_on | obj
        assert d_nested(arg) is mk
        mk_src.assert_called_once_with(arg)
        assert d_nested() is mk


#
class test_UnaryOpExpression:
    operators = {
        "not": ([], True),
        "truth": ([123], True),
        "abs": (-234, 234),
        "index": (22, 22),
        "identity": ((o := StaticMock()), o),
        "invert": (2, -3),
        "neg": (565, -565),
        "pos": (Counter(a=-2, b=-1, x=3, y=1, z=0), Counter(x=3, y=1)),
    }

    @pyt.fixture(params=[*operators])
    def op_name(self, request: pyt.FixtureRequest):
        return request.param

    def test(self, expr: Closure, op_params, expr_type):
        this, expected = op_params
        print(expr.name, str(expr), repr(expr), repr(expr.operator), sep="\n  -")

        d_path, d_args, d_kwds = expr.deconstruct()
        cp, dcp = copy(expr), deepcopy(expr)

        assert expr == cp == dcp == try_import(d_path)(*d_args, **d_kwds)
        obj = expr(this)
        assert obj == expected
        if expr_type == "nested":
            mk: Mock = expr.source
            mk.assert_called_once_with(this)

            object.__setattr__(expr, "source", Identity())

        exp_or = reduce(or_, repeat(expr, 3), expr)
        exp_pipe = expr.pipe(*repeat(expr, 3))
        assert exp_or == exp_pipe
        if isinstance(expr, Identity):
            assert exp_pipe == expr
        else:
            assert exp_pipe.source.source.source == expr


class test_BinaryOpExpression:
    operators = {
        "is": ((o := StaticMock()), o, True),
        "is_not": (o, StaticMock(), True),
        "lt": (234, 432, True),
        "le": ((123,), (123, 456), True),
        "eq": ("abc", "abc", True),
        "ne": ("abc", "xyz", True),
        "ge": ("zzz", "aaa", True),
        "gt": ("zzz", "aaa", True),
        "add": ([1, 2, 3], [4, 5], [1, 2, 3] + [4, 5]),
        "and": ({1, 2, 3, 4}, {1, 2, 5, 7}, {1, 2}),
        "floordiv": (33, 5, 33 // 5),
        "lshift": (1, 3, 1 << 3),
        "mod": (33, 5, 33 % 5),
        "mul": (3, 5, 3 * 5),
        "matmul": None,
        "or": ({1, 2, 3, 4}, {1, 2, 5, 6}, {1, 2, 3, 4, 5, 6}),
        "pow": (5, 3, 5**3),
        "rshift": (16, 3, 16 >> 3),
        "sub": (60, 25, 60 - 25),
        "truediv": (60, 15, 60 / 15),
        "xor": ({1, 2, 3}, {3, 4, 5}, {1, 2, 3} ^ {3, 4, 5}),
        "contains": ({1, 2, 3}, 2, True),
        "iadd": ([1, 2, 3], [4, 5], [1, 2, 3] + [4, 5]),
        "iand": ({1, 2, 3, 4}, {1, 2, 5, 7}, {1, 2}),
        "ifloordiv": (33, 5, 33 // 5),
        "ilshift": (1, 3, 1 << 3),
        "imod": (33, 5, 33 % 5),
        "imul": (3, 5, 3 * 5),
        "imatmul": None,
        "ior": ({1, 2, 3, 4}, {1, 2, 5, 6}, {1, 2, 3, 4, 5, 6}),
        "ipow": (5, 3, 5**3),
        "irshift": (16, 3, 16 >> 3),
        "isub": (60, 25, 60 - 25),
        "itruediv": (60, 15, 60 / 15),
        "ixor": ({1, 2, 3}, {3, 4, 5}, {1, 2, 3} ^ {3, 4, 5}),
        "getattr": (SimpleNamespace(abc=(mk := StaticMock())), "abc", mk),
        "getitem": ([1, 2, 3], 1, 2),
        "delattr": (SimpleNamespace(abc=StaticMock()), "abc", SimpleNamespace()),
        "delitem": ({"a": 1, "b": 2}, "a", {"b": 2}),
    }

    @pyt.fixture(params=[*operators])
    def op_name(self, request: pyt.FixtureRequest):
        name = request.param
        if self.operators[name] is None:
            pyt.skip(f"install implementation")
        return name

    @pyt.fixture()
    def expr_args(self, op_params: tuple, expr_mode):
        args = op_params[1:2]
        if expr_mode == "lazy":
            args = compose(args, many=True)
        return tuple(args)

    def test(self, expr: Closure, op_params, expr_type):
        this, other, expected = op_params
        print(expr.name, str(expr), repr(expr), repr(expr.operator), sep="\n  -")

        d_path, d_args, d_kwds = expr.deconstruct()
        cp, dcp = copy(expr), deepcopy(expr)

        assert expr == cp == dcp == try_import(d_path)(*d_args, **d_kwds)
        obj = expr(this)
        if expr.isterminal:
            assert obj is None
            obj = this
        assert obj == expected
        if expr.inplace:
            immutable = isinstance(this, str | int | float | tuple)
            assert (immutable and this != obj) or (not immutable and this is obj)

        if expr_type == "nested":
            mk: Mock = expr.source
            mk.assert_called_once_with(this)

            object.__setattr__(expr, "source", Identity())

        exp_or = reduce(or_, repeat(expr, 3), expr)
        exp_pipe = expr.pipe(*repeat(expr, 3))
        assert exp_or == exp_pipe
        assert exp_pipe.source.source.source == expr


class test_MutationClosure:
    operators = {
        "setattr": (SimpleNamespace(), "abc", (mk := StaticMock()), SimpleNamespace(abc=mk)),
        "setitem": ({"a": 1, "b": 2}, "x", 123, {"a": 1, "b": 2, "x": 123}),
    }

    @pyt.fixture(params=[*operators])
    def op_name(self, request: pyt.FixtureRequest):
        return request.param

    @pyt.fixture()
    def expr_args(self, op_params: tuple, expr_mode):
        args = op_params[1:2]
        if expr_mode == "lazy":
            args = compose(args, many=True)
        return tuple(args)

    def test(self, expr: Closure, op_params, expr_type):
        this, _, arg, expected = op_params
        print(expr.name, str(expr), repr(expr), repr(expr.operator), sep="\n  -")

        d_path, d_args, d_kwds = expr.deconstruct()
        cp, dcp = copy(expr), deepcopy(expr)

        assert expr == cp == dcp == try_import(d_path)(*d_args, **d_kwds)
        obj = expr(this, arg)
        if expr.isterminal:
            assert obj is None
            obj = this
        assert obj == expected
        if expr.inplace:
            immutable = isinstance(this, str | int | float | tuple)
            assert (immutable and this != obj) or (not immutable and this is obj)

        if expr_type == "nested":
            mk: Mock = expr.source
            mk.assert_called_once_with(this)

            object.__setattr__(expr, "source", Identity())

        exp_or = reduce(or_, repeat(expr, 3), expr)
        exp_pipe = expr.pipe(*repeat(expr, 3))
        assert exp_or == exp_pipe
        assert exp_pipe.source.source.source == expr


class test_GenericOpExpression:
    operators = {
        "call": (
            (mk := StaticMock()),
            tuple(map(StaticMock, repeat(int, 3))),
            {f"kw_{i}": StaticMock() for i in range(3)},
            tuple(map(StaticMock, repeat(int, 3))),
            {f"kw_{i}": StaticMock() for i in range(2, 5)},
            mk.return_value,
        )
    }

    @pyt.fixture(params=[*operators])
    def op_name(self, request: pyt.FixtureRequest):
        return request.param

    @pyt.fixture()
    def expr_args(self, op_params: tuple, expr_mode):
        args = op_params[1:3]
        if expr_mode == "lazy":
            args = compose(args, many=True, depth=2)
        return tuple(args)

    def test(self, expr: Closure, op_params, expr_type):
        conf = self.operators[expr.name]
        this, *_, args, kwds, expected = op_params
        print(expr.name, str(expr), repr(expr), repr(expr.operator), sep="\n  -")

        d_path, d_args, d_kwds = expr.deconstruct()
        cp, dcp = copy(expr), deepcopy(expr)

        assert expr == cp == dcp == try_import(d_path)(*d_args, **d_kwds)
        obj = expr(this, *args, **kwds)
        if expr.isterminal:
            assert obj is None
            obj = this
        assert obj == expected
        if expr.inplace:
            immutable = isinstance(this, str | int | float | tuple)
            assert (immutable and this != obj) or (not immutable and this is obj)

        if expr_type == "nested":
            mk: Mock = expr.source
            mk.assert_called_once_with(this)

            object.__setattr__(expr, "source", Identity())

        exp_or = reduce(or_, repeat(expr, 3), expr)
        exp_pipe = expr.pipe(*repeat(expr, 3))
        assert exp_or == exp_pipe
        assert exp_pipe.source.source.source == expr

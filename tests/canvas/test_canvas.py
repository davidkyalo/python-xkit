from collections import Counter
from copy import copy, deepcopy
from itertools import repeat
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest as pyt

from zana.canvas import (
    BinaryOperation,
    GenericOperation,
    Identity,
    LazyBinaryOperation,
    LazyGenericOperation,
    LazyTernaryOperation,
    LazyUnaryOperation,
    Ref,
    TernaryOperation,
    UnaryOperation,
    ops,
)
from zana.util import try_import


#
class test_UnaryOperation:
    @pyt.mark.parametrize(
        "op_name, input, expected",
        [
            ("not", [], True),
            ("truth", [123], True),
            ("abs", -234, 234),
            ("index", 22, 22),
            ("identity", (o := Mock()), o),
            ("invert", 2, -3),
            ("neg", 565, -565),
            ("pos", Counter(a=-2, b=-1, x=3, y=1, z=0), Counter(x=3, y=1)),
        ],
    )
    def test_eager(self, op_name: str, input, expected):
        op = ops[op_name]
        sub = op.define()
        print(op.name, "\n", str(op), repr(op))
        print("", str(sub), repr(sub), "\n")

        assert isinstance(sub, UnaryOperation)

        d_path, d_args, d_kwds = sub.deconstruct()
        assert sub == copy(sub) == deepcopy(sub) == try_import(d_path)(*d_args, **d_kwds)
        obj = sub(input)
        assert obj == expected

    @pyt.mark.parametrize(
        "op_name, input, expected",
        [
            ("not", [], True),
            ("truth", [123], True),
            ("abs", -234, 234),
            ("index", 22, 22),
            ("identity", (o := Mock()), o),
            ("invert", 2, -3),
            ("neg", 565, -565),
            ("pos", Counter(a=-2, b=-1, x=3, y=1, z=0), Counter(x=3, y=1)),
        ],
    )
    def test_lazy(self, op_name: str, input, expected):
        op = ops[op_name]
        sub = op.define(Ref(input), lazy=True)
        print(op.name, "\n", str(op), repr(op))
        print("", str(sub), repr(sub), "\n")

        assert isinstance(sub, LazyUnaryOperation)

        d_path, d_args, d_kwds = sub.deconstruct()
        assert sub == copy(sub) == try_import(d_path)(*d_args, **d_kwds)
        obj = sub(object())
        assert obj == expected


class test_BinaryOperation:
    @pyt.mark.parametrize(
        "op_name, input, other, expected",
        [
            ("is", (o := Mock()), o, True),
            ("is_not", o, Mock(), True),
            ("lt", 234, 432, True),
            ("le", (123,), (123, 456), True),
            ("eq", "abc", "abc", True),
            ("ne", "abc", "xyz", True),
            ("ge", "zzz", "aaa", True),
            ("gt", "zzz", "aaa", True),
            ("add", [1, 2, 3], [4, 5], [1, 2, 3] + [4, 5]),
            ("and", {1, 2, 3, 4}, {1, 2, 5, 7}, {1, 2}),
            ("floordiv", 33, 5, 33 // 5),
            ("lshift", 1, 3, 1 << 3),
            ("mod", 33, 5, 33 % 5),
            ("mul", 3, 5, 3 * 5),
            pyt.param("matmul", None, None, None, marks=pyt.mark.skip("install implementation")),
            ("or", {1, 2, 3, 4}, {1, 2, 5, 6}, {1, 2, 3, 4, 5, 6}),
            ("pow", 5, 3, 5**3),
            ("rshift", 16, 3, 16 >> 3),
            ("sub", 60, 25, 60 - 25),
            ("truediv", 60, 15, 60 / 15),
            ("xor", {1, 2, 3}, {3, 4, 5}, {1, 2, 3} ^ {3, 4, 5}),
            ("contains", {1, 2, 3}, 2, True),
            ("getitem", [1, 2, 3], 1, 2),
            ("iadd", [1, 2, 3], [4, 5], [1, 2, 3] + [4, 5]),
            ("iand", {1, 2, 3, 4}, {1, 2, 5, 7}, {1, 2}),
            ("ifloordiv", 33, 5, 33 // 5),
            ("ilshift", 1, 3, 1 << 3),
            ("imod", 33, 5, 33 % 5),
            ("imul", 3, 5, 3 * 5),
            pyt.param("imatmul", None, None, None, marks=pyt.mark.skip("install implementation")),
            ("ior", {1, 2, 3, 4}, {1, 2, 5, 6}, {1, 2, 3, 4, 5, 6}),
            ("ipow", 5, 3, 5**3),
            ("irshift", 16, 3, 16 >> 3),
            ("isub", 60, 25, 60 - 25),
            ("itruediv", 60, 15, 60 / 15),
            ("ixor", {1, 2, 3}, {3, 4, 5}, {1, 2, 3} ^ {3, 4, 5}),
            ("getattr", SimpleNamespace(abc=(mk := Mock())), "abc", mk),
            ("delattr", SimpleNamespace(abc=Mock()), "abc", SimpleNamespace()),
            ("delitem", {"a": 1, "b": 2}, "a", {"b": 2}),
        ],
    )
    def test_eager(self, op_name: str, input, other, expected):
        op = ops[op_name]
        sub = op.define(other)
        print(op.name, "\n", str(op), repr(op))
        print("", str(sub), repr(sub), "\n")

        assert isinstance(sub, BinaryOperation)

        d_path, d_args, d_kwds = sub.deconstruct()
        assert sub == copy(sub) == try_import(d_path)(*d_args, **d_kwds)
        dcp = deepcopy(sub)
        obj = sub(input)
        if op.terminal:
            assert obj is None
            obj = input
        assert obj == expected
        if op.inplace:
            immutable = isinstance(input, str | int | float | tuple)
            assert (immutable and input != obj) or (not immutable and input is obj)

    @pyt.mark.parametrize(
        "op_name, input, other, expected",
        [
            ("is", (o := Mock()), o, True),
            ("is_not", o, Mock(), True),
            ("lt", 234, 432, True),
            ("le", (123,), (123, 456), True),
            ("eq", "abc", "abc", True),
            ("ne", "abc", "xyz", True),
            ("ge", "zzz", "aaa", True),
            ("gt", "zzz", "aaa", True),
            ("add", [1, 2, 3], [4, 5], [1, 2, 3] + [4, 5]),
            ("and", {1, 2, 3, 4}, {1, 2, 5, 7}, {1, 2}),
            ("floordiv", 33, 5, 33 // 5),
            ("lshift", 1, 3, 1 << 3),
            ("mod", 33, 5, 33 % 5),
            ("mul", 3, 5, 3 * 5),
            pyt.param("matmul", None, None, None, marks=pyt.mark.skip("install implementation")),
            ("or", {1, 2, 3, 4}, {1, 2, 5, 6}, {1, 2, 3, 4, 5, 6}),
            ("pow", 5, 3, 5**3),
            ("rshift", 16, 3, 16 >> 3),
            ("sub", 60, 25, 60 - 25),
            ("truediv", 60, 15, 60 / 15),
            ("xor", {1, 2, 3}, {3, 4, 5}, {1, 2, 3} ^ {3, 4, 5}),
            ("contains", {1, 2, 3}, 2, True),
            ("getitem", [1, 2, 3], 1, 2),
            ("iadd", [1, 2, 3], [4, 5], [1, 2, 3] + [4, 5]),
            ("iand", {1, 2, 3, 4}, {1, 2, 5, 7}, {1, 2}),
            ("ifloordiv", 33, 5, 33 // 5),
            ("ilshift", 1, 3, 1 << 3),
            ("imod", 33, 5, 33 % 5),
            ("imul", 3, 5, 3 * 5),
            pyt.param("imatmul", None, None, None, marks=pyt.mark.skip("install implementation")),
            ("ior", {1, 2, 3, 4}, {1, 2, 5, 6}, {1, 2, 3, 4, 5, 6}),
            ("ipow", 5, 3, 5**3),
            ("irshift", 16, 3, 16 >> 3),
            ("isub", 60, 25, 60 - 25),
            ("itruediv", 60, 15, 60 / 15),
            ("ixor", {1, 2, 3}, {3, 4, 5}, {1, 2, 3} ^ {3, 4, 5}),
            ("getattr", SimpleNamespace(abc=(mk := Mock())), "abc", mk),
            ("delattr", SimpleNamespace(abc=Mock()), "abc", SimpleNamespace()),
            ("delitem", {"a": 1, "b": 2}, "a", {"b": 2}),
        ],
    )
    def test_lazy(self, op_name: str, input, other, expected):
        op = ops[op_name]
        sub = op.define(Identity(), Ref(other), lazy=True)
        print(op.name, "\n", str(op), repr(op))
        print("", str(sub), repr(sub), "\n")

        assert isinstance(sub, LazyBinaryOperation)

        d_path, d_args, d_kwds = sub.deconstruct()
        assert sub == copy(sub) == try_import(d_path)(*d_args, **d_kwds)
        dcp = deepcopy(sub)
        obj = sub(input)
        if op.terminal:
            assert obj is None
            obj = input
        assert obj == expected
        if op.inplace:
            immutable = isinstance(input, str | int | float | tuple)
            assert (immutable and input != obj) or (not immutable and input == obj)


class test_TernaryOperation:
    @pyt.mark.parametrize(
        "op_name, input, args, c_args, expected",
        [
            ("setattr", SimpleNamespace(), ("abc", (mk := Mock())), (), SimpleNamespace(abc=mk)),
            ("setattr", SimpleNamespace(), ("abc",), ((mk := Mock()),), SimpleNamespace(abc=mk)),
            ("setattr", SimpleNamespace(), (), ("abc", (mk := Mock())), SimpleNamespace(abc=mk)),
            ("setitem", {"a": 1, "b": 2}, ("x", 123), (), {"a": 1, "b": 2, "x": 123}),
            ("setitem", {"a": 1, "b": 2}, ("x",), (123,), {"a": 1, "b": 2, "x": 123}),
            ("setitem", {"a": 1, "b": 2}, (), ("x", 123), {"a": 1, "b": 2, "x": 123}),
        ],
    )
    def test_eager(self, op_name, input, args, c_args, expected):
        op = ops[op_name]
        sub = op.define(args)
        print(op.name, "\n", str(op), repr(op))
        print("", str(sub), repr(sub), "\n")

        assert isinstance(sub, TernaryOperation)

        d_path, d_args, d_kwds = sub.deconstruct()
        assert sub == copy(sub) == try_import(d_path)(*d_args, **d_kwds)
        dcp = deepcopy(sub)
        obj = sub(input, *c_args)
        if op.terminal:
            assert obj is None
            obj = input
        assert obj == expected
        if op.inplace:
            immutable = isinstance(input, str | int | float | tuple)
            assert (immutable and input != obj) or (not immutable and input is obj)

    @pyt.mark.parametrize(
        "op_name, input, args, c_args, expected",
        [
            ("setattr", SimpleNamespace(), ("abc", (mk := Mock())), (), SimpleNamespace(abc=mk)),
            ("setattr", SimpleNamespace(), ("abc",), ((mk := Mock()),), SimpleNamespace(abc=mk)),
            ("setattr", SimpleNamespace(), (), ("abc", (mk := Mock())), SimpleNamespace(abc=mk)),
            ("setitem", {"a": 1, "b": 2}, ("x", 123), (), {"a": 1, "b": 2, "x": 123}),
            ("setitem", {"a": 1, "b": 2}, ("x",), (123,), {"a": 1, "b": 2, "x": 123}),
            ("setitem", {"a": 1, "b": 2}, (), ("x", 123), {"a": 1, "b": 2, "x": 123}),
        ],
    )
    def test_lazy(self, op_name: str, input, args, c_args, expected):
        op = ops[op_name]
        sub = op.define(Identity(), map(Ref, args), lazy=True)
        print(op.name, "\n", str(op), repr(op))
        print("", str(sub), repr(sub), "\n")

        assert isinstance(sub, LazyTernaryOperation)

        d_path, d_args, d_kwds = sub.deconstruct()
        assert sub == copy(sub) == try_import(d_path)(*d_args, **d_kwds)
        dcp = deepcopy(sub)
        obj = sub(input, *c_args)
        if op.terminal:
            assert obj is None
            obj = input
        assert obj == expected
        if op.inplace:
            immutable = isinstance(input, str | int | float | tuple)
            assert (immutable and input != obj) or (not immutable and input == obj)


class test_GenericOperation:
    @pyt.mark.parametrize(
        "op_name, input, args, kwds, c_args, c_kwds, expected",
        [
            (
                "call",
                (mk := MagicMock()),
                tuple(map(Mock, repeat(int, 3))),
                {f"kw_{i}": Mock() for i in range(3)},
                tuple(map(Mock, repeat(int, 3))),
                {f"kw_{i}": Mock() for i in range(2, 5)},
                mk.return_value,
            )
        ],
    )
    def test_eager(self, op_name, input: Mock, args, kwds, c_args, c_kwds, expected):
        op = ops[op_name]
        sub = op.define(args, kwds)
        print(op.name, "\n", str(op), repr(op))
        print("", str(sub), repr(sub), "\n")

        assert isinstance(sub, GenericOperation)

        d_path, d_args, d_kwds = sub.deconstruct()
        assert sub == copy(sub) == try_import(d_path)(*d_args, **d_kwds)
        dcp = deepcopy(sub)
        obj = sub(input, *c_args, **c_kwds)
        if op.terminal:
            assert obj is None
            obj = input
        if isinstance(input, Mock) and expected is input.return_value:
            input.assert_called_once_with(*args, *c_args, **kwds | c_kwds)
        assert obj == expected

        if op.inplace:
            immutable = isinstance(input, str | int | float | tuple)
            assert (immutable and input != obj) or (not immutable and input is obj)

    @pyt.mark.parametrize(
        "op_name, input, args, kwds, c_args, c_kwds, expected",
        [
            (
                "call",
                (mk := MagicMock()),
                tuple(map(Mock, repeat(int, 3))),
                {f"kw_{i}": Mock() for i in range(3)},
                tuple(map(Mock, repeat(int, 3))),
                {f"kw_{i}": Mock() for i in range(2, 5)},
                mk.return_value,
            )
        ],
    )
    def test_lazy(self, op_name, input: Mock, args, kwds, c_args, c_kwds, expected):
        op = ops[op_name]
        sub = op.define(Identity(), map(Ref, args), {k: Ref(v) for k, v in kwds.items()}, lazy=True)
        print(op.name, "\n", str(op), repr(op))
        print("", str(sub), repr(sub), "\n")

        assert isinstance(sub, LazyGenericOperation)

        d_path, d_args, d_kwds = sub.deconstruct()
        assert sub == copy(sub) == try_import(d_path)(*d_args, **d_kwds)
        dcp = deepcopy(sub)
        obj = sub(input, *c_args, **c_kwds)
        if op.terminal:
            assert obj is None
            obj = input
        if isinstance(input, Mock) and expected is input.return_value:
            input.assert_called_once_with(*args, *c_args, **kwds | c_kwds)
        assert obj == expected

        if op.inplace:
            immutable = isinstance(input, str | int | float | tuple)
            assert (immutable and input != obj) or (not immutable and input is obj)

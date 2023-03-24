import pytest as pyt

from zana.canvas import (
    UNARY_OP,
    BinaryOperation,
    Composition,
    GenericOperation,
    LazyBinaryOperation,
    LazyGenericOperation,
    LazyTernaryOperation,
    OperationType,
    Ref,
    TernaryOperation,
    _builtin_ops,
    operators,
    trap,
)


class test_trap:
    @pyt.mark.parametrize(
        "op_name",
        [*_builtin_ops],
    )
    def test_eager(self, op_name):
        op = operators[op_name]
        if not op.trap:
            pyt.skip(f"Operator {op!s} not trappable")
        src_ops = [operators.getattr("abc"), operators.getitem(123)]
        src = trap(*src_ops)

        args = tuple(f"var_{i}" for i in range(+op.type - 1))
        sub = getattr(trap, op.trap_name)(src, *args)
        print("", str(sub), repr(sub), "\n")
        assert isinstance(sub, trap)

        obj = sub.__compose__()
        assert isinstance(obj, Composition)
        *base, tip = obj
        assert Composition(base) == src.__compose__()
        assert isinstance(tip, op.impl)

        match op.type:
            case OperationType.UNARY:
                assert not args
            case OperationType.BINARY:
                assert not isinstance(tip, BinaryOperation) or (tip.operant,) == args
            case OperationType.TERNARY:
                assert not isinstance(tip, TernaryOperation) or tip.operants == args
            case OperationType.GENERIC:
                assert not isinstance(tip, GenericOperation) or tip.args == args

        if op.reverse and op.reverse_trap_name:
            rev = getattr(trap, op.reverse_trap_name)(src, *args)
            assert rev.__compose__()[0] == tip.__compose__()

    @pyt.mark.parametrize(
        "op_name",
        [*_builtin_ops],
    )
    def test_lazy(self, op_name):
        op = operators[op_name]
        if not op.trap:
            pyt.skip(f"Operator {op!s} not trappable")
        src_ops = [operators.getattr("abc"), operators.getitem(123)]
        src = trap(*src_ops)
        args = tuple(Ref(f"var_{i}") for i in range(+op.type - 1))
        sub = getattr(trap, op.trap_name)(src, *args)

        print("", str(sub), repr(sub), "\n")
        assert isinstance(sub, trap)

        obj = sub.__compose__()
        assert op.type == UNARY_OP or isinstance(obj, op.lazy_impl)
        tip = obj

        match op.type:
            case OperationType.UNARY:
                assert not args
            case OperationType.BINARY:
                assert not isinstance(tip, LazyBinaryOperation) or (tip.operant,) == args
            case OperationType.TERNARY:
                assert not isinstance(tip, LazyTernaryOperation) or tip.operants == args
            case OperationType.GENERIC:
                assert not isinstance(tip, LazyGenericOperation) or tip.args == args

        if op.reverse and op.reverse_trap_name:
            rev = getattr(trap, op.reverse_trap_name)(src, *args)
            # assert rev.__compose__() == tip.__compose__()

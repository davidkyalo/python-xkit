import sys
from inspect import getmembers, isfunction
from itertools import groupby, repeat

import attr

from zana.canvas.nodes import (
    GenericOperatorNode,
    Identity,
    LazyGenericOperatorNode,
    Node,
    OperatorNode,
    Ref,
    operators,
)


def dump_attrs(cls: type, pre=None, sep="\n  ", indent=""):
    if pre is None and cls:
        pre = f"{indent}{cls.__module__}.{cls.__qualname__}: attrs"
    it = [] if pre is None else [pre]
    if hasattr(cls, "__attrs_attrs__"):
        it += (
            f"{indent}- {f.name:16}: {f.alias = !s:6} {f.init = !s:6} {f.kw_only = !s:6} {f.inherited = !s:6} {f.default=}"
            for f in attr.fields(cls)
        )
    return it if sep is None else sep.join(it)


def run():
    print(dump_attrs(Ref))
    print(dump_attrs(Identity))
    print(dump_attrs(OperatorNode))
    print(dump_attrs(GenericOperatorNode))
    print(dump_attrs(LazyGenericOperatorNode))

    print("\n")

    for i, (tp, ops) in enumerate(
        groupby(operators.values(), lambda o: o if o.name in ("ref", "identity") else o.type), 1
    ):
        op, *_ = ops
        o, l = op.impl, op.lazy_impl
        print(f"\n=> {op!s:<12}: {op!r}")
        print(dump_attrs(op.impl, sep="\n   ", indent=" " * 6))
        print(dump_attrs(op.lazy_impl, sep="\n   ", indent=" " * 6))

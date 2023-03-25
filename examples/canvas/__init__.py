from itertools import groupby

import attr

from zana.canvas import (
    BinaryOpClosure,
    Closure,
    GenericClosure,
    Identity,
    OperatorInfo,
    Ref,
    UnaryClosure,
    operators,
)
from zana.util import subclasses


def get_fields(cls):
    try:
        return attr.fields(cls)
    except Exception:
        return ()


def dump_attrs(cls: type, title="attrs", pre=None, sep="\n  ", indent=""):
    if pre is None and cls:
        pre = f"{indent}{cls.__module__}.{cls.__qualname__}: {title}"
    it = [] if pre is None else [pre]
    # # it += [f"{indent}{b.__module__}.{b.__qualname__}" for b in cls.mro()]
    # it += (
    #     f"{indent}- {f.name:16}: {f.alias = !s:6} {f.init = !s:6} {f.kw_only = !s:6} {f.inherited = !s:6} {f.default=}"
    #     for f in get_fields(cls)
    # )
    # it.append("")
    return it if sep is None else sep.join(it)


def run():
    print(dump_attrs(Ref))
    print(dump_attrs(Identity))
    print(dump_attrs(Closure))
    print(dump_attrs(UnaryClosure))
    print(dump_attrs(BinaryOpClosure))
    print(dump_attrs(GenericClosure))

    print("\n", "-" * 40, "\n")
    op: OperatorInfo[Closure]
    for i, (tp, ops) in enumerate(
        groupby(operators.values(), lambda o: o if o.name in ("ref", "identity") else o.type), 1
    ):
        op, *_ = ops
        indent = " " * 4
        print(f"\n=> {op!s:<12}:\n    {op!r}")
        print(dump_attrs(op.impl, "BASE", indent=indent))
        print(dump_attrs(op.impl._nested_type_, "NESTED", indent=indent))
        print(dump_attrs(op.impl._nested_lazy_type_, "NESTED/LAZY", indent=indent))
        print(dump_attrs(op.impl._root_type_, "ROOT", indent=indent))
        print(dump_attrs(op.impl._root_lazy_type_, "ROOT/LAZY", indent=indent))
        print("")

    # for sub in subclasses(Expression):
    #     print(f"{sub}")

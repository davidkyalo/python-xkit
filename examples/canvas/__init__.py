import sys
from inspect import getmembers, isfunction
from itertools import repeat

from zana.canvas import operators, trap

attr_str = (
    lambda f, i=1: f"{' '.join(repeat('', i))}- {f.name:16}: {f.init = !s:6} {f.kw_only = !s:6} {f.inherited = !s:6} {f.default=}"
)
for i, op in enumerate(operators.values(), 1):
    o, l = op.impl, op.lazy_impl
    print(
        f" - {i:3} : {op.name:16} {op.type.name:8} --> '{op.function.__module__.lstrip('_')}.{op.function.__name__}'",
        f" -> EAGER: {o.__name__:16} {o.lazy = !s:6}",
        *(
            f"   - {f.name:16}: {f.init = !s:6} {f.kw_only = !s:6} {f.inherited = !s:6} {f.default=}"
            for f in o.__attrs_attrs__
        ),
        f" -> LAZY: {l and l.__name__ or '':16} {getattr(l, 'lazy', '???') = !s:6}",
        *(
            f"   - {f.name:16}: {f.init = !s:6} {f.kw_only = !s:6} {f.inherited = !s:6} {f.default=}"
            for f in (l and l.__attrs_attrs__ or ())
        ),
        sep="\n      ",
        end="\n\n",
    )


print(
    "TRAP",
    *(f" - {i}: {n} {v}" for i, (n, v) in enumerate(getmembers(trap, isfunction), 1)),
    sep="\n  ",
)

print(f"{sys.getrecursionlimit() = }")
sys.setrecursionlimit(999)

print(trap().abc + trap().xyz)

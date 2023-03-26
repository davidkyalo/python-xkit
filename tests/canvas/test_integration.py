from copy import copy, deepcopy

from zana.canvas import Ref, compose, magic, operator
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
        args = magic().xyz(), Ref(StaticMock()), StaticMock()
        foo = magic().abc["xyz"].foo
        foo_rv = foo(*args)
        src = +foo_rv.bar
        sub = src == +mk_bar_rv
        subr = +mk_bar_rv == src

        expr, exprr = compose(sub), compose(subr)
        exprl = compose(src) | operator.is_(compose(src), lazy=True)

        print("VARS", *(f" - {k:16}: {v and v or v!r}" for k, v in vars().items()), "", sep="\n ")
        cp_s, dcp_s = compose(copy(sub)), compose(deepcopy(sub))
        cp_e, dcp_e = copy(expr), deepcopy(expr)
        assert compose(src)(mk) is +mk_bar.return_value

        mk_foo.assert_called_once_with(mk.xyz.return_value, args[1](mk), args[2])
        mk.xyz.assert_called_once_with()

        assert expr(mk) is exprr(mk) is exprl(mk) is True
        assert cp_e(mk) is dcp_e(mk) is cp_s(mk) is dcp_s(mk) is True
        assert compose(src != mk_bar)(mk) is compose(mk_foo != src)(mk) is True
        assert compose(src != mk_foo)(mk) is True

        mk_bar.reset_mock()
        op = compose(foo_rv) | operator.setattr("bar")
        mk_val = StaticMock()
        op(mk, mk_val)
        mk_bar.assert_called_once_with(mk_val)

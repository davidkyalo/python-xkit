from copy import copy, deepcopy
from itertools import chain, repeat
from operator import attrgetter, itemgetter
from types import SimpleNamespace
from unittest.mock import CallableMixin, Mock

import pytest

from zana.util.operator import call, callback


class test_callback:
    @pytest.mark.parametrize(
        "n_args, n_kwargs",
        [
            (0, 0),
            (0, 3),
            (3, 0),
            (3, 3),
        ],
    )
    def test(self, n_args, n_kwargs):
        p_args, c_args = [f"arg_{i}" for i in range(n_args)], [Mock() for _ in range(n_args)]
        p_kwargs, c_kwargs = {f"kw_p_{i}": f"val_p_{i}" for i in range(n_args)}, {
            f"kw_c_{i}": Mock() for i in range(n_args)
        }
        o_kwargs = {"kw_p_0": Mock()} if n_kwargs > 1 else {}

        mk = Mock()

        obj = callback(call, *p_args, **p_kwargs)
        print(f"{obj!s} --> {obj!r}")
        assert obj == copy(obj) == deepcopy(obj)

        res = obj(mk, *c_args, **c_kwargs, **o_kwargs)
        assert res == mk.return_value

        mk.assert_called_once_with(*c_args, *p_args, **c_kwargs, **p_kwargs | o_kwargs)

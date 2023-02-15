from operator import attrgetter, itemgetter
from types import SimpleNamespace
from unittest.mock import CallableMixin

import pytest

from zana.common import call, pipe, pipeline


class test_pipeline:
    @pytest.mark.parametrize(
        "expected, pipes, args, kwargs, tap, call_args, call_kwargs",
        [
            (
                "object(**attrs).foo['bar']",
                [SimpleNamespace, attrgetter("foo"), itemgetter("bar")],
                (),
                {},
                None,
                (),
                {"foo": {"bar": "object(**attrs).foo['bar']"}},
            ),
            (
                "[None, object][::-1][0](**attrs).foo['bar']",
                [itemgetter(1, 0), itemgetter(0), call, attrgetter("foo"), itemgetter("bar")],
                (),
                {},
                2,
                ((None, SimpleNamespace),),
                {"foo": {"bar": "[None, object][::-1][0](**attrs).foo['bar']"}},
            ),
            (
                "[None, object][::-1][0](**attrs).foo['bar']",
                [itemgetter(1, 0), itemgetter(0), call, attrgetter("foo"), itemgetter("bar")],
                (),
                {},
                -3,
                ((None, SimpleNamespace),),
                {"foo": {"bar": "[None, object][::-1][0](**attrs).foo['bar']"}},
            ),
            (
                "[None, object][::-1][0](**attrs).foo['bar']",
                [itemgetter(1, 0), itemgetter(0), call, attrgetter("foo"), itemgetter("bar")],
                (),
                {},
                (-3, -2),
                ((None, SimpleNamespace),),
                {"foo": {"bar": "[None, object][::-1][0](**attrs).foo['bar']"}},
            ),
        ],
    )
    def test(self, expected, pipes, args, kwargs, tap, call_args, call_kwargs):
        pp = pipeline(pipes, args, kwargs, *(() if tap is None else (tap,)))
        result = pp(*call_args, **call_kwargs)
        if tap is None:
            p_result = pipe(pipes, *call_args, *args, **kwargs | call_kwargs)
            assert p_result == result
        if callable(expected) and not isinstance(expected, CallableMixin):
            expected(result)
        else:
            assert result == expected

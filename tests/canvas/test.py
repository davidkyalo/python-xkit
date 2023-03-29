import pytest as pyt

from zana.canvas import Composable, maybe_compose
from zana.testing.mock import StaticMock


@pyt.mark.parametrize(
    "value, expected, kwargs",
    [
        (123, 123, None),
        (123, 123, {"many": True}),
        ([123], [123], None),
        ([123], [123], {"many": True}),
        ([(ma := StaticMock(Composable))], [ma], None),
        ({(ma := StaticMock(Composable))}, {ma}, None),
        ({"k": (ma := StaticMock(Composable))}, {"k": ma}, None),
        ((ma := StaticMock(Composable)), ma.__zana_compose__(), None),
        ([(ma := StaticMock(Composable))], [ma.__zana_compose__()], {"many": True}),
        (iter([(ma := StaticMock(Composable))]), [ma.__zana_compose__()], {"many": True}),
        ({(ma := StaticMock(Composable))}, {ma.__zana_compose__()}, {"many": True}),
        ({"k": (ma := StaticMock(Composable))}, {"k": ma.__zana_compose__()}, {"many": True}),
    ],
)
def test_maybe_compose(value, expected, kwargs):
    res = maybe_compose(value, **(kwargs or {}))
    assert res == expected

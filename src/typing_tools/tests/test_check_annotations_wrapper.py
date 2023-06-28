import pytest
from typing import Annotated

from ..annotations import check_annotations
from ..annotation_checkers import ValueRange

func1_a1 = ValueRange(0, 10)
func1_b1 = ValueRange(5, 10)


@check_annotations
def func1(a: Annotated[int, func1_a1], b: Annotated[int, func1_b1]):
    """
    A basic function for testing check_annotations
    """
    return a, b


def test_args_func1():
    a = -1
    b = 0
    with pytest.raises(ValueError):
        func1_a1.check(a)
    with pytest.raises(ValueError):
        func1_b1.check(b)
    with pytest.raises(ValueError):
        func1(a=-1, b=0)
    with pytest.raises(ValueError):
        func1(-1, 0)
    with pytest.raises(ValueError):
        func1(-1, b=0)


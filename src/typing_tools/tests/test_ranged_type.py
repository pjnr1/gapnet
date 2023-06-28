import argparse

import pytest

from ..argparsers import ranged_type


def test_base_usage():
    checker = ranged_type(float, 0.0, 1.0)
    with pytest.raises(argparse.ArgumentTypeError):
        checker('-1')
    with pytest.raises(argparse.ArgumentTypeError):
        checker('1.1')
    with pytest.raises(argparse.ArgumentTypeError):
        checker('-0.1')
    assert(checker('0.0') == 0.0)
    assert(checker('-0.0') == 0.0)
    assert(checker('0.5') == 0.5)
    assert(checker('1.0') == 1.0)
    assert(checker('0.000000000') == 0.0)
    assert(checker('1.000000000') == 1.0)
    assert(checker('0.0000000000000001') == 0.0000000000000001)


def test_non_inclusive_min():
    checker = ranged_type(float, 0.0, 1.0, min_inclusive=False)
    with pytest.raises(argparse.ArgumentTypeError):
        checker('0.0')
    assert(checker('0.0000000000000001') == 0.0000000000000001)
    assert(checker('1.0') == 1.0)
    checker = ranged_type(float, -10.0, 5.0, min_inclusive=False)
    with pytest.raises(argparse.ArgumentTypeError):
        checker('-10.0')
    assert(checker('-9.999999999999999') == -9.999999999999999)
    assert(checker('5.0') == 5.0)


def test_non_inclusive_max():
    checker = ranged_type(float, 0.0, 1.0, max_inclusive=False)
    with pytest.raises(argparse.ArgumentTypeError):
        checker('1.0')
    checker = ranged_type(float, 0.0, 10.0, max_inclusive=False)
    with pytest.raises(argparse.ArgumentTypeError):
        checker('10.0')


def test_non_inclusive_min_and_max():
    checker = ranged_type(float, 0.0, 1.0, min_inclusive=False, max_inclusive=False)
    with pytest.raises(argparse.ArgumentTypeError):
        checker('0.0')
    with pytest.raises(argparse.ArgumentTypeError):
        checker('1.0')

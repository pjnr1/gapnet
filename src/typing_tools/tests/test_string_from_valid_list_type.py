import argparse

import pytest

from ..argparsers import string_from_valid_list_type


def test_short_valid_list():
    valid_values = ['a', 'b', 'c', 'd', 'e']
    _check = string_from_valid_list_type(valid_values)
    with pytest.raises(argparse.ArgumentTypeError):
        _check('f')
    with pytest.raises(argparse.ArgumentTypeError):
        _check('aa')
    with pytest.raises(argparse.ArgumentTypeError):
        _check('bb')

    for c in valid_values:
        assert(_check(c) == c)


def test_long_valid_list():
    valid_values = ['abe', 'bet', 'ced', 'dat', 'egf']
    _check = string_from_valid_list_type(valid_values)
    with pytest.raises(argparse.ArgumentTypeError):
        _check('gf')
    with pytest.raises(argparse.ArgumentTypeError):
        _check('da')
    with pytest.raises(argparse.ArgumentTypeError):
        _check('ab')

    for c in valid_values:
        assert(_check(c) == c)


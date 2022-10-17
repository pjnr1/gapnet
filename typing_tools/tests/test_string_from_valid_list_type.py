import argparse

import pytest

from ..argparsers import string_from_valid_list_type


def test_short_valid_list():
    valid_values = ['a', 'b', 'c', 'd', 'e']
    check = string_from_valid_list_type(valid_values)
    with pytest.raises(argparse.ArgumentTypeError):
        check('f')
    with pytest.raises(argparse.ArgumentTypeError):
        check('aa')
    with pytest.raises(argparse.ArgumentTypeError):
        check('bb')

    for c in valid_values:
        assert(check(c) == c)


def test_long_valid_list():
    valid_values = ['abe', 'bet', 'ced', 'dat', 'egf']
    check = string_from_valid_list_type(valid_values)
    with pytest.raises(argparse.ArgumentTypeError):
        check('gf')
    with pytest.raises(argparse.ArgumentTypeError):
        check('da')
    with pytest.raises(argparse.ArgumentTypeError):
        check('ab')

    for c in valid_values:
        assert(check(c) == c)


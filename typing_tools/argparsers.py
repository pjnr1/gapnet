from __future__ import annotations

import argparse
import operator
from typing import Callable, Iterable, TypeVar

T = TypeVar('T')


def ranged_type(value_type: type,
                min_value: T | None = None,
                max_value: T | None = None,
                min_inclusive: bool = True,
                max_inclusive: bool = True) -> Callable[[str], T]:
    """
    Return function handle of an argument type function for ArgumentParser checking a range::
        min_value <= arg <= max_value

    Example:
        >>> ranged_type(float, 0.0, 1.0) # Checks the range ]0, 1[
        >>> ranged_type(float, 0.0, 1.0, False, False) # Checks the range [0, 1]

    @param value_type:
        value-type to convert arg to
    @param min_value:
        minimum acceptable argument value, can be None
    @param max_value:
        maximum acceptable argument value, can be none
    @param min_inclusive:
        boolean flag for including min_value as valid value
    @param max_inclusive:
        boolean flag for including max_value as valid value
    @return:
        function handle of an argument type function for ArgumentParser
    @raise ValueError:
        if both min_value and max_value is None. In this case, simply use value_type directly instead of range_type(...)
    """
    if min_value is None and max_value is None:
        raise ValueError('either min_value or max_value should be different from None')

    min_operator = operator.lt if min_inclusive else operator.le
    max_operator = operator.gt if max_inclusive else operator.ge

    def range_checker(arg: str) -> T:
        try:
            x = value_type(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f'must be a valid {value_type}')
        if min_value is not None and min_operator(x, min_value):
            raise argparse.ArgumentTypeError(f'must be within [{min_value},{max_value}]')
        if max_value is not None and max_operator(x, max_value):
            raise argparse.ArgumentTypeError(f'must be within [{min_value},{max_value}]')
        return x

    # Return function handle to checking function
    return range_checker


def string_from_valid_list_type(valid_values: Iterable[str]) -> Callable[[str], str]:
    """
    Return function handle that checks whether input argument matches any of the C{valid_values}

    Example:
        >>> string_from_valid_list_type(['some_string', 'another'])

    @param valid_values:
        List of accepted string values

    @return:
        function handle of an argument type function for ArgumentParser
    """

    def value_checker(arg: str) -> str:
        if arg in valid_values:
            return arg
        else:
            raise argparse.ArgumentTypeError(f'must match a value from the list [{valid_values}]')

    # Return function handle to checking function
    return value_checker

import operator
import os
from typing import Any, Sized


class AnnotatedTypeChecker:
    """
    Parent class required for any annotation checker
    """

    def check(self, arg: Any) -> None:
        """
        Takes any argument and raises a ValueError if it doesn't check out
        """
        raise NotImplementedError()


class MaxLen(AnnotatedTypeChecker):
    def __init__(self, max_length, inclusive: bool = True):
        self.max_length = max_length
        """
        Maximum allowed length of the iterable
        """
        self.inclusive = inclusive
        """
        Whether to include max_length as valid value or not
        """

    def check(self, arg: Sized):
        op = operator.gt if self.inclusive else operator.ge
        if op(len(arg), self.max_length):
            raise ValueError(f'argument exceeds type-hinted length {len(arg)} {op} {self.max_length}')


class MinLen(AnnotatedTypeChecker):
    def __init__(self, min_length, inclusive: bool = True):
        self.min_length = min_length
        """
        Maximum allowed length of the iterable
        """
        self.inclusive = inclusive
        """
        Whether to include min_length as valid value or not
        """

    def check(self, arg: Sized):
        op = operator.lt if self.inclusive else operator.le
        if op(len(arg), self.min_length):
            raise ValueError(f'argument exceeds type-hinted length {len(arg)} {op} {self.min_length}')


class ExactLen(AnnotatedTypeChecker):
    def __init__(self, length):
        self.length = length
        """
        Maximum allowed length of the iterable
        """

    def check(self, arg: Sized):
        if len(arg) != self.length:
            raise ValueError(f'length of argument didn\'t match type-hinted length ({len(arg)} != {self.length})')


class ValueRange(AnnotatedTypeChecker):
    def __init__(self, min_value, max_value, min_inclusive: bool = True, max_inclusive: bool = True):
        self.min_value = min_value
        """
        Minimum allowed value
        """
        self.max_value = max_value
        """
        Maximum allowed value
        """
        self.min_inclusive = min_inclusive
        """
        Whether to include min_value as valid value or not
        """
        self.max_inclusive = max_inclusive
        """
        Whether to include max_value as valid value or not
        """

    def check(self, arg):
        min_operator = operator.lt if self.min_inclusive else operator.le
        max_operator = operator.gt if self.max_inclusive else operator.ge

        if min_operator(arg, self.min_value) or max_operator(arg, self.max_value):
            mi = '[' if self.min_inclusive else ']'
            ma = '[' if self.min_inclusive else ']'
            range_string = f'{mi}{self.min_value}, {self.max_value}{ma}'
            raise ValueError(f'argument was {arg}, but type-hinted in range {range_string}')


class PathExists(AnnotatedTypeChecker):
    def check(self, arg):
        if not os.path.exists(arg):
            raise ValueError('argument was type-hinted to exist, but os.path.exists returned False')

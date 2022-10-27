"""
Helpers for checking Annotated type-hints
"""
from functools import wraps
from typing import get_type_hints, get_args
from .annotation_checkers import AnnotatedTypeChecker


def _check_hint_args(hint):
    _args = get_args(hint)
    if _args is None or len(_args) == 0:
        return None, []
    else:
        return _args


def check_annotations(func):
    """
    Function decorator to check Annotated types in function arguments

    Note: default values are not checked

    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        # Convert *args to **kwargs
        _kwargs = kwargs
        # perform runtime annotation checking
        # first, get type hints from function
        type_hints = get_type_hints(func, include_extras=True)
        for i, (param, hint) in zip(range(len(type_hints)), type_hints.items()):
            # get base type and additional arguments
            hint_type, *hint_args = _check_hint_args(hint)
            if hint_type is None:
                continue
            for arg in hint_args:
                if not isinstance(arg, AnnotatedTypeChecker):
                    continue
                if param not in kwargs.keys():
                    if len(args) <= i:
                        continue
                    arg.check(args[i])
                else:
                    arg.check(kwargs[param])
        # execute function once all checks passed
        return func(*args, **kwargs)

    return wrapped

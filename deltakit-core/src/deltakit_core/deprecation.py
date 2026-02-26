import functools
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar, overload

from packaging.version import Version

P = ParamSpec("P")
RetType = TypeVar("RetType")


@overload
def deprecated(f: Callable[P, RetType] | None, /) -> Callable[P, RetType]:
    pass  # pragma: no cover


@overload
def deprecated(
    *,
    reason: str | None = None,
    replaced_by: str | None = None,
    removed_in_version: Version | None = None,
) -> Callable[[Callable[P, RetType]], Callable[P, RetType]]:
    pass  # pragma: no cover


def deprecated(
    f: Callable[P, RetType] | None = None,
    /,
    *,
    reason: str | None = None,
    replaced_by: str | None = None,
    removed_in_version: Version | None = None,
) -> Callable[[Callable[P, RetType]], Callable[P, RetType]] | Callable[P, RetType]:
    """Deprecation decorator.

    Args:
        f: function to deprecate, allow this function to be used as a decorator.
        reason: rationale behind the deprecation.
        replaced_by: a description of the new method that should be used instead of the
            deprecated function.
        removed_in_version: minimum version in which the deprecated function might be
            removed. The deprecated function should not be removed earlier than this
            version.

    Returns:
        Depending on the provided arguments, either a wrapper around the provided ``f``
        that raises a deprecation warning or a decorator that can deprecate a function.
    """
    if all(p is None for p in (f, reason, replaced_by, removed_in_version)):
        msg = (
            "Expected at least one of f, reason, replaced_by or removed_in_version to "
            "be provided. All of them are None."
        )
        raise ValueError(msg)
    if f is not None:
        msg = f"{f.__name__} is deprecated and will eventually be removed."

        @functools.wraps(f)
        def deprecated_func(*args: P.args, **kwargs: P.kwargs) -> RetType:
            warnings.warn(msg, DeprecationWarning, stacklevel=1)
            return f(*args, **kwargs)

        return deprecated_func

    # else
    def deprecation_decorator(
        func_to_deprecate: Callable[P, RetType],
    ) -> Callable[P, RetType]:
        msg = f"{func_to_deprecate.__name__} is deprecated and will "
        msg += (
            f"be removed in version {removed_in_version}."
            if removed_in_version is not None
            else "eventually be removed."
        )
        if reason is not None:
            msg += f" Reason for deprecation: '{reason}'."
        if replaced_by is not None:
            msg += f" Consider using '{replaced_by}' instead."

        @functools.wraps(func_to_deprecate)
        def deprecated_func(*args: P.args, **kwargs: P.kwargs) -> RetType:
            warnings.warn(msg, DeprecationWarning, stacklevel=1)
            return func_to_deprecate(*args, **kwargs)

        return deprecated_func

    return deprecation_decorator

import pytest
from packaging.version import Version

from deltakit_core.deprecation import deprecated


def non_deprecated_add(a: float, b: float, ndigits: int | None = None) -> float:
    return round(a + b, ndigits)


@deprecated
def default_deprecated_add(a: float, b: float, ndigits: int | None = None) -> float:
    return round(a + b, ndigits)


@deprecated(
    reason="Testing purposes",
    replaced_by="nothing",
    removed_in_version=Version("0.2.0"),
)
def custom_deprecated_add(a: float, b: float, ndigits: int | None = None) -> float:
    return round(a + b, ndigits)


def test_default_deprecated() -> None:
    msg = "^default_deprecated_add is deprecated and will eventually be removed.$"
    with pytest.warns(DeprecationWarning, match=msg):
        assert default_deprecated_add(2.00000001, 3.0000000003, ndigits=3) == 5


def test_custom_deprecated() -> None:
    msg = (
        "^custom_deprecated_add is deprecated and will be removed in version 0.2.0. "
        "Reason for deprecation: 'Testing purposes'. Consider using 'nothing' instead."
    )
    with pytest.warns(DeprecationWarning, match=msg):
        assert custom_deprecated_add(2.00000001, 3.0000000003, ndigits=3) == 5


def test_direct_calling_no_argument() -> None:
    msg = (
        "Expected at least one of f, reason, replaced_by or removed_in_version to "
        "be provided. All of them are None."
    )
    with pytest.raises(ValueError, match=msg):
        deprecated()


def test_direct_calling_with_none_callable() -> None:
    msg = (
        "Expected at least one of f, reason, replaced_by or removed_in_version to "
        "be provided. All of them are None."
    )
    with pytest.raises(ValueError, match=msg):
        deprecated(None)


def test_direct_calling_all_kwargs_none() -> None:
    msg = (
        "Expected at least one of f, reason, replaced_by or removed_in_version to "
        "be provided. All of them are None."
    )
    with pytest.raises(ValueError, match=msg):
        deprecated(reason=None, replaced_by=None, removed_in_version=None)

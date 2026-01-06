import warnings
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version


def _get_stim_version() -> Version:
    try:
        return Version(version("stim"))
    except PackageNotFoundError:
        warnings.warn(
            "Could not get the current version of 'stim'. Assuming 0.0.1.", stacklevel=1
        )
        return Version("0.0.1")


_INSTALLED_STIM_VERSION = _get_stim_version()
_LOWEST_STIM_VERSION_WITH_TAG_FEATURE = Version("1.15")


def is_stim_tag_feature_available() -> bool:
    return _INSTALLED_STIM_VERSION >= _LOWEST_STIM_VERSION_WITH_TAG_FEATURE

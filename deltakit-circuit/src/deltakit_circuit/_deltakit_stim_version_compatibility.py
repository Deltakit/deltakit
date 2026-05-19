from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version


def _get_deltakit_stim_version() -> Version | None:
    try:
        return Version(version("deltakit_stim"))
    except PackageNotFoundError:
        return None


_INSTALLED_DELTAKIT_STIM_VERSION = _get_deltakit_stim_version()
_DELTAKIT_STIM_VERSION_WITH_TAG = Version("0.1.2")


def is_deltakit_stim_tag_feature_available() -> bool:
    if _INSTALLED_DELTAKIT_STIM_VERSION is not None:
        return _INSTALLED_DELTAKIT_STIM_VERSION >= _DELTAKIT_STIM_VERSION_WITH_TAG
    return False

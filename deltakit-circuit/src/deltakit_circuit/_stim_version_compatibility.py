from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

_LOWEST_STIM_VERSION_WITH_TAG_FEATURE = Version("0.1.2")


def _get_stim_version() -> Version | None:
    try:
        return Version(version("deltakit_stim"))
    except PackageNotFoundError:
        return None


def is_stim_tag_feature_available() -> bool:
    installed = _get_stim_version()
    return installed is not None and installed >= _LOWEST_STIM_VERSION_WITH_TAG_FEATURE

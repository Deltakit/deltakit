import importlib.metadata
import warnings
import importlib
import semver


def _get_stim_version() -> semver.Version:
    try:
        return semver.Version.parse(importlib.metadata.version("stim"))
    except AttributeError:
        warnings.warn(
            "Could not get the current version of 'stim'. Assuming 0.0.1.", stacklevel=1
        )
        return semver.Version(0, 0, 1)


_INSTALLED_STIM_VERSION = _get_stim_version()
_LOWEST_STIM_VERSION_WITH_TAG_FEATURE = semver.Version(1, 15)


def is_stim_tag_feature_available() -> bool:
    return _INSTALLED_STIM_VERSION >= _LOWEST_STIM_VERSION_WITH_TAG_FEATURE

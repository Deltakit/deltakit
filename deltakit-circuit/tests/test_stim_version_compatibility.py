import deltakit_stim
from packaging.version import Version

from deltakit_circuit._deltakit_stim_version_compatibility import (
    is_deltakit_stim_tag_feature_available,
)


def test_deltakit_stim_version_compatibility() -> None:
    current_deltakit_stim_version = Version(deltakit_stim.__version__)
    if current_deltakit_stim_version >= Version("0.1.2"):
        assert is_deltakit_stim_tag_feature_available()
    else:
        assert not is_deltakit_stim_tag_feature_available()

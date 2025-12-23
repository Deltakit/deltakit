import stim
from packaging.version import Version

from deltakit_circuit._stim_version_compatibility import is_stim_tag_feature_available


def test_stim_version_compatibility() -> None:
    current_stim_version = Version(stim.__version__)
    if current_stim_version >= Version("1.15"):
        assert is_stim_tag_feature_available()
    else:
        assert not is_stim_tag_feature_available()

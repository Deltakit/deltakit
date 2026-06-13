# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import os
import random

import pytest

from deltakit_explorer._api import _auth
from deltakit_explorer._utils import _utils as utils


def test_set_token(mocker):
    randint = random.randint(100000, 999999)
    mocker.patch(
        "deltakit_explorer._utils._utils.APP_NAME", f"qec-testplorer-{randint}"
    )
    token = "2134"  # nosec B105
    _auth.set_token(token)
    assert _auth.get_token() == token


def test_if_no_token_raises(mocker):
    randint = random.randint(100000, 999999)
    mocker.patch(
        "deltakit_explorer._utils._utils.APP_NAME", f"qec-testplorer-{randint}"
    )
    utils.override_persisted_variables({}, utils.get_config_file_path())
    os.environ.pop(_auth.TOKEN_VARIABLE)
    with pytest.raises(RuntimeError, match=r"^Token could not be found neither"):
        _auth.get_token()


def test_http_verification_is_set():
    # clear SSL verify variable
    os.environ.pop(_auth.TLS_DISABLE_CHECK_VARIABLE, None)
    # sets _auth.TLS_DISABLE_CHECK_VARIABLE to 'false'
    _auth.set_https_verification(True)
    val = os.environ.get(_auth.TLS_DISABLE_CHECK_VARIABLE)
    assert val not in ["1", "yes", "true"]


def test_https_verification_not_disabled_by_default():
    # Assert SSH verification is disabled by default
    assert _auth.https_verification_disabled() is False

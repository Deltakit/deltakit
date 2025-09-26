# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import os
import random
from contextlib import suppress

import pytest
import requests
from deltakit_explorer._api import _auth
from deltakit_explorer._utils import _utils as utils


def test_set_token(mocker):
    randint = random.randint(100000, 999999)
    mocker.patch("deltakit_explorer._utils._utils.APP_NAME", f"qec-testplorer-{randint}")
    token = "2134"  # nosec B105
    _auth.set_token(token)
    assert _auth.get_token() == token


def test_if_no_token_raises(mocker):
    randint = random.randint(100000, 999999)
    mocker.patch("deltakit_explorer._utils._utils.APP_NAME", f"qec-testplorer-{randint}")
    utils.override_persisted_variables({}, utils.get_config_file_path())
    os.environ.pop(_auth.TOKEN_VARIABLE)
    with pytest.raises(RuntimeError, match=r"^Token could not be found neither"):
        _auth.get_token()


def test_http_verification_is_set():
    _auth.set_https_verification(True)
    # wrong host certificate
    url = "https://wrong.host.badssl.com/"
    with pytest.raises(requests.exceptions.SSLError):
        requests.get(
            url,
            verify=not _auth.https_verification_disabled(),
            timeout=5,
        )


@pytest.mark.filterwarnings('ignore:Unverified HTTPS')
def test_http_verification_is_unset():
    # Related to `test_http_verification_is_set above`, but this time
    # we don't require HTTPS verification, so we should *not* get an `SSLError`.
    # An "Unverified HTTPS" warning is OK.
    # (Sometimes we get a `ConnectionError`, though, and we don't want to
    # fail because of that, so suppress it.)
    _auth.set_https_verification(False)
    # wrong host certificate
    url = "https://wrong.host.badssl.com/"
    with suppress(requests.exceptions.ConnectionError):
        requests.get(
            url,
            verify=not _auth.https_verification_disabled(),
            timeout=5,
        )

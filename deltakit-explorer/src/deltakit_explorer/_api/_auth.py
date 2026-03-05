# (c) Copyright Riverlane 2020-2025.
"""Authentication-related functions and variables."""

from __future__ import annotations

import os

from deltakit_explorer._utils import _utils as utils

TOKEN_VARIABLE = "DELTAKIT_TOKEN"  # nosec B105
TLS_DISABLE_CHECK_VARIABLE = "DELTAKIT_DISABLE_TLS_CHECK"
# pylint: disable=fixme
# TODO: domain name for QEC portal.
RIVERLANE_PORTAL = "https://deltakit.rivelane.com/dashboard/token"


def get_token() -> str:
    """
    Load the token from drive (if any is persisted) to env.
    If there is no token on the drive, read environment variable.
    Return the token.

    Returns:
        str: an auth token.

    Raises:
        RuntimeError: if token is not present at all.
    """
    utils.load_environment_variables_from_drive()
    token = os.environ.get(TOKEN_VARIABLE)
    if token is None:
        file = utils.get_config_file_path()
        msg = (
            f"Token could not be found neither in ({file}) "
            f"nor environment variable ({TOKEN_VARIABLE}). "
            f"Please obtain your token at {RIVERLANE_PORTAL} "
            "and use `Client.set_token` function to "
            "register it."
        )
        raise RuntimeError(msg)
    return token


def set_token(token: str):
    """
    Set a new token as env variable.

    Args:
        token (str): a string auth token.
    """
    update_dict = {TOKEN_VARIABLE: token}
    utils.set_variables(update_dict, True)
    utils.merge_variables(update_dict, utils.get_config_file_path())


def https_verification_disabled() -> bool:
    """
    In the debug mode you may switch off TLS certificate validation.

    Returns:
        bool: True, if config file DELTAKIT_DISABLE_TLS_CHECK=1|yes|true
    """
    val = os.environ.get(TLS_DISABLE_CHECK_VARIABLE, "0").lower()
    return val in ["1", "yes", "true"]


def set_https_verification(enabled: bool):
    """
    In the debug mode you may switch off TLS certificate validation.

    Args:
        enabled (bool): if set to False, disables HTTPS check.
    """
    val = str(not enabled).lower()
    os.environ[TLS_DISABLE_CHECK_VARIABLE] = val
    utils.set_variables({TLS_DISABLE_CHECK_VARIABLE: val})

# (c) Copyright Riverlane 2020-2025.

import os
import pytest
import random
from pathlib import Path
from tests.helpers._utils import FakeResponse
from deltakit_explorer._api import _auth
from deltakit_explorer._api._client import Client
from deltakit_explorer._api._gql_client import GQLClient
from deltakit_explorer._utils import _utils as utils
from deltakit_explorer.types._exceptions import ServerException


class TestGQLClientTokenManipulations:

    @pytest.mark.parametrize("api_version", [1, 2])
    def test_set_token_raises_on_connection(self, api_version, mocker):
        old_server = os.environ.pop(utils.DELTAKIT_SERVER_URL_ENV, default="")
        client = Client("http://localhorse:81/", api_version=api_version)
        mocker.patch(
            "deltakit_explorer._api._client.Client.get_instance",
            return_value=client,
        )
        os.environ[utils.DELTAKIT_SERVER_URL_ENV] = "http://localhorse:81/"
        with pytest.raises(ServerException, match="^Could not validate token"):
            Client.set_token("abc", validate=True)
        if old_server:
            os.environ[utils.DELTAKIT_SERVER_URL_ENV] = old_server

    def test_set_token_raises_on_server_v1_error(self):
        old_server = os.environ.pop(utils.DELTAKIT_SERVER_URL_ENV, default="")
        os.environ[utils.DELTAKIT_SERVER_URL_ENV] = "https://deltakit.riverlane.com/proxy"
        with pytest.raises(ServerException, match="^Token failed validation: Status 401"):
            GQLClient("https://deltakit.riverlane.com/proxy").set_token("abcdefghijklmnopqrstuvwxyzabcdef", validate=True)
            # Token should be a 32-character string
        if old_server:
            os.environ[utils.DELTAKIT_SERVER_URL_ENV] = old_server

    @pytest.mark.parametrize("api_version", [1, 2])
    def test_set_token_works(self, api_version, mocker):
        client = Client("http://localhorse:81/", api_version=api_version)
        mocker.patch(
            "deltakit_explorer._api._client.Client.get_instance",
            return_value=client,
        )
        randint = random.randint(1000, 9999)
        token = f"abc-{randint}"
        Client.set_token(token, validate=False)
        assert _auth.get_token() == token

    @pytest.mark.parametrize("api_version", [1, 2])
    def test_set_token_works_with_no_token_before(self, api_version, mocker):
        client = Client("http://localhorse:81/", api_version=api_version)
        mocker.patch(
            "deltakit_explorer._api._client.Client.get_instance",
            return_value=client,
        )
        Path.unlink(utils.get_config_file_path())
        randint = random.randint(1000, 9999)
        token = f"abc-{randint}"
        GQLClient("http://localhost").set_token(token, validate=False)
        assert _auth.get_token() == token

    def test_set_token_works_with_404_on_v2(self, mocker):
        client = Client("https://localhost/", api_version=2)
        mocker.patch(
            "deltakit_explorer._api._client.Client.get_instance",
            return_value=client,
        )
        mocker.patch.object(client._api._request_session, "get", return_value=FakeResponse(404))
        Path.unlink(utils.get_config_file_path())
        randint = random.randint(1000, 9999)
        token = f"abc-{randint}"
        Client.set_token(token, validate=True)
        assert _auth.get_token() == token

    def test_set_token_v2_fails_on_401(self, mocker):
        client = Client("https://localhost/", api_version=2)
        mocker.patch(
            "deltakit_explorer._api._client.Client.get_instance",
            return_value=client,
        )
        mocker.patch.object(client._api._request_session, "get", return_value=FakeResponse(401))
        Path.unlink(utils.get_config_file_path())
        with pytest.raises(ServerException):
            Client.set_token("abcdefghijklmnopqrstuvwxyzabcdef", validate=True) # Token should be a 32-character string

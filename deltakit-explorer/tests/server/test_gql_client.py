# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import random
import re

import gql
import gql.transport.exceptions
import graphql
import pytest
import requests
from deltakit_explorer._api._client import Client
from deltakit_explorer._api._gql_client import GQLClient
from deltakit_explorer.types._exceptions import ServerException


# This tests GQLClient, which is the underlying client for v1 of the API.
# Due to this, all tests must specify `api_version=1` when creating a client to test.
class TestGQLClient:

    def test_execute_query_rethrows_on_error(self, mocker):
        mocker.patch(
            "gql.client.SyncClientSession.execute",
            side_effect=gql.transport.exceptions.TransportQueryError(
                "",
                errors=[{"message": "some message"}],
            )
        )
        Client.set_token("123", validate=False)
        client = Client("http://localhost/", api_version=1)
        with pytest.raises(ServerException, match="some message"):
            client._api.execute_query("mutation Copy { copy {location, uid}}", {}, "")
        gql.client.SyncClientSession.execute.assert_called_once()


    def test_execute_query_throws_on_execution_error(self, mocker):
        mocker.patch(
            "gql.client.SyncClientSession.execute",
            return_value=graphql.ExecutionResult(
                {},
                errors=[graphql.GraphQLError("some message")]
            )
        )
        randint = random.randint(100000, 999999)
        mocker.patch("deltakit_explorer._utils._utils.APP_NAME", f"deltakit-testplorer-{randint}")

        client = Client("http://localhost/", api_version=1)
        Client.set_token("123", validate=False)
        with pytest.raises(ServerException, match="some message"):
            client._api.execute_query("mutation Copy { copy {location, uid}}", {}, "")
        gql.client.SyncClientSession.execute.assert_called_once()

    def test_execute_query_kills_on_interrupt(self, mocker):
        mocker.patch(
            "gql.client.SyncClientSession.execute",
            side_effect=KeyboardInterrupt,
        )
        randint = random.randint(100000, 999999)
        mocker.patch(
            "deltakit_explorer._utils._utils.APP_NAME",
            f"deltakit-testplorer-{randint}"
        )
        client = Client("http://riverlane.com/", api_version=1)
        mocker.patch.object(client._api, "kill", return_value=1)
        Client.set_token("123", validate=False)
        with pytest.raises(KeyboardInterrupt):
            client._api.execute_query("mutation Copy { copy {location, uid}}", {}, "qid")
        gql.client.SyncClientSession.execute.assert_called_once()
        # called kill() for with query id provided
        client._api.kill.assert_called_once_with("qid")

    def test_get_query_200(self, mocker):
        resp = requests.Response()
        resp.status_code = 200
        resp._content = b"some text"
        mocker.patch("requests.Session.get", return_value=resp)
        client = Client("http://localhost/", api_version=1)
        assert client._api._get_query("123", "") == "some text"
        requests.Session.get.assert_called_once()

    @pytest.mark.parametrize(("response_text", "message"), [
        (b"", "Status 400 (Error #400): "),
        (b"{\"detail\": \"123\", \"error_code\": 543}", "Status 400 (Error #543): 123"),
        (b"[1, 2, 3]", "Status 400 (Error #400): [1, 2, 3]"),
    ])
    def test_get_query_gave_other_than_200(self, mocker, response_text, message):
        resp = requests.Response()
        resp.status_code = 400
        resp._content = response_text
        mocker.patch("requests.Session.get", return_value=resp)
        client = Client("http://localhost/", api_version=1)
        with pytest.raises(ServerException, match=re.escape(message)):
            client._api._get_query("123", "")
        requests.Session.get.assert_called_once()

    @pytest.mark.parametrize(
        ("code", "message"),
        [
            (503, "Service unavailable"),
            (404, "URL not found"),
            (621, ""),
        ]
    )
    def test_get_message(self, code, message):
        assert GQLClient._get_message(code) == message

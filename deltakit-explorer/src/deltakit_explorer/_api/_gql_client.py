# (c) Copyright Riverlane 2020-2025.
"""This file contains implementation of an API for accessing
the Deltakit service.
"""

from __future__ import annotations

import json
from typing import Any, cast, TYPE_CHECKING
from typing_extensions import override
from urllib.parse import urljoin

import numpy as np
import numpy.typing as npt
import requests
import requests.adapters
if TYPE_CHECKING:
    import stim
from deltakit_explorer._api._api_client import APIClient
from deltakit_explorer._api._auth import (get_token,
                                          https_verification_disabled,
                                          set_token)
from deltakit_explorer._utils._logging import Logging
from deltakit_explorer.enums._api_enums import DataFormat, APIEndpoints
from deltakit_explorer.types import (DataString, Decoder, DecodingResult,
                                     DetectionEvents, LeakageFlags, Measurements,
                                     NoiseModel, ObservableFlips,
                                     QubitCoordinateToDetectorMapping)
from deltakit_explorer.types._exceptions import ServerException
from deltakit_explorer.types._experiment_types import QECExperimentDefinition
from gql import Client, gql
from gql.client import SyncClientSession
from gql.transport.exceptions import TransportQueryError
from gql.transport.requests import RequestsHTTPTransport
from graphql import ExecutionResult


# pylint: disable=too-many-instance-attributes
class GQLClient(APIClient):
    """This is a parent class which contains methods
    to GQL server functionality.
    """

    def __init__(
        self,
        base_url: str,
        request_timeout: int = 60000,
    ):
        """
        The constructor accepts and endpoint and a timeout.
        `base_url` is a URL of the service.
        - `/api/graphql/` accepts GraphQL POST-queries,
        - `/api/data/` allows GET-requests of content.

        Args:
            base_url (str): Service URL.
            request_timeout (int):
                request timeout (seconds). Default is 60_000 seconds.

        Examples:
            This is how an instance of the child class is created::

                client = Client(
                    base_url="http://deltakit.rivelane.com/proxy"
                )

        """
        super().__init__(base_url, request_timeout)
        self.content_endpoint = f"{self.base_url}/api/data/"
        self.graphql_endpoint = f"{self.base_url}/api/graphql"

        self.auth_headers: dict[str, str | bytes] = {}
        self.transport = self._get_transport()
        self.client = self._get_client()
        self._session: SyncClientSession = self.client.connect_sync()
        self._request_session = requests.Session()
        retries = requests.adapters.Retry(
                total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504],
        )
        self._request_session.mount(
            "http://",
            requests.adapters.HTTPAdapter(max_retries=retries),
        )
        self._request_session.mount(
            "https://",
            requests.adapters.HTTPAdapter(max_retries=retries),
        )
        self._request_session.headers = self.auth_headers

    def _update_headers(self):
        self.auth_headers = {
            "Authorization": "Bearer " + get_token()
        }

    def _get_transport(self):
        self._update_headers()
        return RequestsHTTPTransport(
            url=self.graphql_endpoint,
            headers=self.auth_headers,
            timeout=self.request_timeout,
            verify=not https_verification_disabled(),
            stream=True,
        )

    def _get_client(self):
        return Client(
            transport=self.transport,
            fetch_schema_from_transport=False,
            execute_timeout=self.request_timeout,
        )

    @staticmethod
    def _get_message(code: int) -> str:
        if code == 503:
            return "Service unavailable"
        if code == 404:
            return "URL not found"
        return ""

    def _get_query(self, name: str, request_id: str) -> str:
        """Downloads a GraphQL query given a name,
        and substitutes provided named arguments.

        Args:
            name (str): query name.
            request_id (str): request_id for tracking in logs.

        Returns:
            str: GraphQL parametric query.
        """
        url = (
            urljoin(self.content_endpoint + "/query/", name)
            + f"?request_id={request_id}"
        )
        resp = self._request_session.get(
            url,
            headers=self.auth_headers,
            verify=not https_verification_disabled(),
            timeout=self.client.execute_timeout,
        )
        if resp.status_code != 200:
            # gateway responds in json,
            # but it may also forward errors from the server
            try:
                error_dict = json.loads(resp.text)
                if isinstance(error_dict, dict):
                    # HTTP error from the app
                    server_message = error_dict.get("detail", resp.text)
                    # Auth error from the gateway
                    gateway_message = error_dict.get("message", server_message)
                    error_code = error_dict.get("error_code", resp.status_code)
                else:
                    # is not a dict
                    error_code = resp.status_code
                    gateway_message = resp.text
            except json.JSONDecodeError:
                # could not parse as JSON
                gateway_message = resp.text
                error_code = resp.status_code
            if not gateway_message:
                gateway_message = self._get_message(error_code)
            # probably, html content
            if len(gateway_message) > 160:
                gateway_message = gateway_message[:160] + "..."
            msg = f"Status {resp.status_code} (Error #{error_code}): {gateway_message}"
            raise ServerException(
                msg
            )
        return resp.text

    def execute_query(
        self, query: str, variable_values: dict, query_id: str
    ) -> dict[str, Any]:
        """Executes arbitrary GraphQL query and
        returns server response as a python dicr.

        Args:
            query (str): GraphQL query text.
            variable_values (Dict): arguments.
            query_id (str): identifier of the request.

        Returns:
            Dict[str, Any]: Server response.

        Raises:
            TransportQueryError
        """
        request = gql(query)
        try:
            result = self._session.execute(request, variable_values=variable_values)
            if isinstance(result, ExecutionResult) and result.errors:
                raise ServerException(str(result.errors[0].message))
            return cast(dict[str, Any], result)
        except TransportQueryError as err:
            Logging.error(err, query_id)
            message = str(err.errors[0].get("message") if err.errors else err)
            raise ServerException(message) from err
        except KeyboardInterrupt:
            count = self.kill(query_id)
            Logging.warn(f"Cancelled a task, {count} worker(s) cleared.", query_id)
            raise

    @override
    def execute(
        self,
        query_name: APIEndpoints,
        variable_values: dict,
        request_id: str,
    ) -> dict[str, Any]:
        query = self._get_query(query_name.value, request_id)
        return self.execute_query(query, variable_values, request_id)

    def kill(self, request_id: str) -> int:
        resp = self._request_session.get(
            f"{self.content_endpoint}kill/{request_id}",
            headers=self.auth_headers,
            verify=not https_verification_disabled(),
            timeout=10,
        )
        return int(resp.text)

    @override
    def set_token(self, token: str, validate: bool = True):
        try:
            old_token = get_token()
        except RuntimeError as err:  # pragma: nocover
            Logging.warn(f"No token was set previously. {err.args[0]}", "set_token")
            old_token = ""  # nosec B105

        set_token(token)
        self._update_headers()
        self._request_session.headers = self.auth_headers
        if validate:
            try:
                # test a content endpoint
                self._get_query("decode", "set_token")
            except ServerException as exc:
                Logging.error(exc, "set_token")
                set_token(old_token)
                msg = f"Token failed validation: {exc.message}."
                raise ServerException(
                    msg
                ) from exc
            except requests.exceptions.ConnectionError as exc_coonection:
                message = "Could not validate token: cannot reach client."
                set_token(old_token)
                Logging.error(exc_coonection, "set_token")
                raise ServerException(message) from exc_coonection
        msg = (
            "Token successfully validated and "
            "will be used automatically in future sessions."
        )
        Logging.info(msg, "set_token")

    @override
    def generate_circuit(self, experiment_definition: QECExperimentDefinition, request_id: str) -> str:
        result = self.execute(  # pragma: nocover
            query_name=APIEndpoints.GENERATE_CIRCUIT,
            variable_values={
                "expType": experiment_definition.experiment_type.value,
                "codeType": experiment_definition.code_type.value,
                "observableBasis":
                    experiment_definition.observable_basis.value,
                "basisGates": experiment_definition.basis_gates,
                "rounds": experiment_definition.num_rounds,
                "outputCircuitLocation": "",
                "outputCircuitName": DataString.empty,
                "generateCircuitParameters":
                    experiment_definition.get_parameters_gql_string(),
                "requestId": request_id,
            },
            request_id=request_id,
        )
        return DataString.from_data_string(
            result["generateCircuit"]["uid"]).to_string()

    @override
    def simulate_circuit(self, stim_circuit: str | stim.Circuit, shots: int, request_id: str) -> tuple[Measurements, LeakageFlags | None]:
        output_format = DataFormat.F01
        result = self.execute(
            query_name=APIEndpoints.SIMULATE_CIRCUIT,
            variable_values={
                "circuit": str(DataString(str(stim_circuit))),
                "location": "",
                "leakageLocation": "",
                "outputData": DataString.empty,
                "outputLeakageData": DataString.empty,
                "outputFormat": output_format.value,
                "outputLeakageFormat": output_format.value,
                "shots": shots,
                "requestId": request_id,
            },
            request_id=request_id,
        )
        uids = [
            DataString.from_data_string(uri["uid"])
            for uri in result["simulateWithStim"]
        ]
        if len(uids) == 1:  # no leakage returned
            return (Measurements(uids[0], data_format=output_format), None)
        if len(uids) == 2:
            return (
                Measurements(uids[0], data_format=output_format),
                LeakageFlags(uids[1], data_format=output_format),
            )
        msg = f"Expected 1 or 2 result files, but got {len(uids)}"
        raise ServerException(msg)

    @override
    def add_noise(self, stim_circuit: str | stim.Circuit, noise_model: NoiseModel, request_id: str) -> str:
        variables = {
            "circuit": str(DataString(str(stim_circuit))),
            "result_file": DataString.empty,
            "requestId": request_id,
            "result_folder": "",
            **noise_model.__dict__,
        }
        if noise_model.ENDPOINT is None:
            raise NotImplementedError(f"Noise addition for {type(noise_model)} is not implemented.")
        result = self.execute(
            query_name=noise_model.ENDPOINT, variable_values=variables, request_id=request_id)
        dstring = DataString.from_data_string(result[noise_model.ENDPOINT_RESULT_FIELDNAME]["uid"])
        return dstring.to_string()

    @override
    def decode(self, detectors: DetectionEvents,
        observables: ObservableFlips,
        decoder: Decoder,
        noisy_stim_circuit: str | stim.Circuit,
        leakage_flags: LeakageFlags | None,
        request_id: str,
    ) -> DecodingResult:
        parameters = []
        if decoder.parameters is not None:
            for key, value in decoder.parameters.items():
                parameters.append([key, str(value)])
        use_format = DataFormat.B8
        # used to extract size
        predictions_format = DataFormat.F01
        variable_values = {
            "contentType": use_format.value,
            "decoder": decoder.decoder_type.value,
            "detFile": detectors.as_data_string(use_format),
            "obsFile": observables.as_data_string(use_format),
            "useExperimentalGraph": decoder.use_experimental_graph,
            "noisyStimFile": str(DataString(str(noisy_stim_circuit))),
            "parameters": parameters,
            "requestId": request_id,
            "resultFileLocation": "",
            "resultFile": DataString.empty,
            "resultContentType": predictions_format.value,
            "threads": decoder.parallel_jobs,
        }
        if leakage_flags is not None:
            query_name = APIEndpoints.DECODE_LEAKAGE
            variable_values["leakageFile"] = leakage_flags.as_data_string(use_format)
        else:
            query_name = APIEndpoints.DECODE
        Logging.info(
            f"Decoding request {request_id} has been sent to a client.",
            request_id
        )
        result = self.execute(query_name, variable_values, request_id)
        return DecodingResult(
            predictions_format=predictions_format,
            **result["decode"],
        )

    @override
    def defect_rates(
        self,
        detectors: DetectionEvents,
        stim_circuit: str | stim.Circuit,
        request_id: str,
    ) -> dict[tuple[float, ...], list[float]]:
        use_format = DataFormat.B8
        result = self.execute(
            query_name=APIEndpoints.DEFECT_RATES,
            variable_values={
                "contentType": use_format.value,
                "detFile": detectors.as_data_string(use_format),
                "stimFile": str(DataString(str(stim_circuit))),
                "coordinates": None,
                "requestId": request_id,
            },
            request_id=request_id,
        )
        rates = result["defectRates"]
        return {
            tuple(map(float, pair["key"])): pair["value"]
            for pair in rates["items"]
        }

    @override
    def get_correlation_matrix_for_trimmed_data(
        self,
        detectors: DetectionEvents,
        noise_floor_circuit: str | stim.Circuit,
        use_default_noise_model_edges: bool,
        request_id: str,
    ) -> tuple[npt.NDArray[np.float64], QubitCoordinateToDetectorMapping]:
        use_format = DataFormat.B8
        result = self.execute(
            query_name=APIEndpoints.CORRELATION_MATRIX,
            variable_values={
                "contentType": use_format.value,
                "detFile": detectors.as_data_string(use_format),
                "stimFile": str(DataString(str(noise_floor_circuit))),
                "useStimGraph": use_default_noise_model_edges,
                "requestId": request_id,
            },
            request_id=request_id,
        )
        matrix = result["correlationMatrix"]["matrix"]
        qubit_to_detectors = result["correlationMatrix"]["qubitToDetectionEvents"]
        return np.array(matrix), QubitCoordinateToDetectorMapping({
            tuple(qubit_det["qubit"]): qubit_det["detectors"] for qubit_det in qubit_to_detectors
        })

    @override
    def trim_circuit_and_detectors(
        self,
        stim_circuit: str | stim.Circuit,
        detectors: DetectionEvents,
        request_id: str
    ) -> tuple[str, DetectionEvents]:
        use_format = DataFormat.B8
        result = self.execute(
            query_name=APIEndpoints.TRIM_CIRCUIT_AND_DETECTORS,
            variable_values={
                "contentType": detectors.data.data_format.value,
                "data": detectors.as_data_string(use_format),
                "circuit": str(DataString(str(stim_circuit))),
                "trimmedCircuit": DataString.empty,
                "trimmedDetectorData": DataString.empty,
                "location": "",
                "outputFormat": use_format.value,
                "requestId": request_id,
            },
            request_id=request_id,
        )
        response = result["trimCircuitAndDetectionEvents"]
        return (
            DataString.from_data_string(response["circuit"]["uid"]).to_string(),
            DetectionEvents(
                data=DataString.from_data_string(response["detectors"]["uid"]),
                data_format=use_format,
            )
        )

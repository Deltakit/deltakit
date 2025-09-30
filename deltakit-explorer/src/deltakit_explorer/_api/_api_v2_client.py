from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, cast, TYPE_CHECKING
from typing_extensions import override
from urllib.parse import urljoin

import numpy as np
import numpy.typing as npt
import requests
import requests.adapters
if TYPE_CHECKING:
    import stim
from deltakit_explorer._api._api_client import APIClient, APIEndpoints
from deltakit_explorer._api._auth import (get_token,
                                          https_verification_disabled,
                                          set_token)
from deltakit_explorer._utils._logging import Logging
from deltakit_explorer.enums._api_enums import DataFormat
from deltakit_explorer.types._exceptions import ServerException
from deltakit_explorer.types._experiment_types import QECExperimentDefinition
from deltakit_explorer.types._types import (BinaryDataType, DataString,
                                            Decoder, DecodingResult, DetectionEvents,
                                            LeakageFlags, Measurements,
                                            NoiseModel, ObservableFlips,
                                            QubitCoordinateToDetectorMapping)


class JobStatus(Enum):
    """Job statuses."""

    SUBMITTED = "SUBMITTED"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Job:
    """Job record, returned by the server."""

    status: str
    request_id: str
    type: APIEndpoints
    error: Optional[str] = None
    workload: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)

    def raise_on_error(self):
        if self.error is not None:
            raise ServerException(self.error)


class APIv2Client(APIClient):
    """
    APIv2 implementation replaces long-living requests with job-based approach.
    Client library submits the job and polls the server to obtain the result of
    computations. This makes an HTTP request to the server a lightweight
    operation, completed within milliseconds time.
    """

    STATUS_CHECK_DELAY = 2

    def __init__(self, base_url: str, request_timeout: int = 100):
        """
        Constructor of the APIv2 client instance.

        Args:
            base_url (str): URL of the service endpoint.
            request_timeout (int):
                Timeout (seconds) on the transport level.
                Matches Cloudflare 100s timeout by default.
        """
        super().__init__(base_url, request_timeout)
        self.add_task_endpoint = f"{self.base_url}/api/v2/tasks/add/"
        self.status_endpoint = f"{self.base_url}/api/v2/tasks/get/"
        self.delete_endpoint = f"{self.base_url}/api/v2/tasks/kill/"
        self.request_timeout = request_timeout

        self.auth_headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {get_token()}",
        }
        self._request_session = requests.Session()
        retries = requests.adapters.Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        self._request_session.mount(
            "http://",
            requests.adapters.HTTPAdapter(max_retries=retries),
        )
        self._request_session.mount(
            "https://",
            requests.adapters.HTTPAdapter(max_retries=retries),
        )
        self._request_session.headers.update(self.auth_headers)

    def _update_headers(self):
        self.auth_headers = {
            "Authorization": "Bearer " + get_token()
        }

    def _submit_task(
        self,
        task_type: APIEndpoints,
        payload: dict,
        request_id: str,
    ) -> Job:
        headers = self.auth_headers.copy()
        headers["X-Request-ID"] = request_id
        resp = self._request_session.post(
            urljoin(self.add_task_endpoint, task_type.value),
            timeout=self.request_timeout,
            headers=headers,
            json=payload,
            verify=not https_verification_disabled(),
        )
        if resp.ok:
            return Job(**resp.json())
        else:
            raise ServerException(f"[{resp.status_code}] Job not submitted: {resp.text}")

    def _get_job_status(self, request_id: str) -> Job:
        headers = self.auth_headers.copy()
        headers["X-Request-ID"] = request_id
        resp = self._request_session.get(
            self.status_endpoint,
            timeout=self.request_timeout,
            headers=headers,
            params={"request_id": request_id},
            verify=not https_verification_disabled(),
        )
        if resp.status_code == 404:
            raise KeyError(f"Request {request_id} not found.")
        if not resp.ok:
            raise ServerException(f"[{resp.status_code}] {resp.text}")
        return Job(**resp.json())

    @override
    def set_token(self, token: str, validate: bool = True):
        """Persist the token, set it to env var and a file.
        Validate it.

        Args:
            token (str): auth token.
            validate (bool): you may switch off validation explicitly.
        """
        try:
            old_token = get_token()
        except RuntimeError as err:  # pragma: nocover
            Logging.warn(f"No token was set previously. {err.args[0]}", "set_token")
            old_token = ""  # nosec B105

        set_token(token)
        self._update_headers()
        self._request_session.headers.update(self.auth_headers)
        if validate:
            try:
                # try to obtain status of some job
                self._get_job_status("")
            except KeyError:
                # request is validated, even the key is missing
                pass
            except ServerException as exc:
                Logging.error(exc, "set_token")
                set_token(old_token)
                msg = f"Token failed validation: {exc.message}."
                raise ServerException(msg) from exc
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
    def execute(
        self,
        query_name: APIEndpoints,
        variable_values: dict,
        request_id: str,
    ) -> dict[str, Any]:
        job = self._submit_task(query_name, variable_values, request_id)
        try:
            # A client request ID is going to be different from
            # a server-generated one.
            Logging.info(f"Server created the job {job.request_id}", request_id)
            while job.status in [JobStatus.SUBMITTED.value, JobStatus.IN_PROGRESS.value]:
                time.sleep(APIv2Client.STATUS_CHECK_DELAY)
                Logging.info(
                    f"Job ({job.type}, {job.request_id}), status = {job.status}",
                    request_id
                )
                job = self._get_job_status(job.request_id)
            job.raise_on_error()
            Logging.info(
                f"Job ({job.type}, {job.request_id}) completed, status = {job.status}",
                request_id
            )
            return job.result
        except KeyboardInterrupt:
            count = self.kill(job.request_id)
            raise InterruptedError(
                f"Cancelled job {job.request_id} ({count} worker(s))."
            )

    @override
    def kill(self, request_id: str) -> int:
        headers = self.auth_headers.copy()
        headers["X-Request-ID"] = request_id
        resp = self._request_session.delete(
            self.delete_endpoint,
            params={"request_id": request_id},
            headers=headers,
        )
        if resp.text and resp.text.isnumeric():
            return int(resp.text)
        return 0

    @override
    def generate_circuit(self, experiment_definition: QECExperimentDefinition, request_id: str) -> str:
        result = self.execute(
            query_name=APIEndpoints.GENERATE_CIRCUIT,
            variable_values=experiment_definition.to_json(),
            request_id=request_id,
        )
        return DataString.from_data_string(str(result.get("circuit"))).to_string()

    @override
    def simulate_circuit(self, stim_circuit: str | stim.Circuit, shots: int, request_id: str) -> tuple[Measurements, LeakageFlags | None]:
        result = self.execute(
            query_name=APIEndpoints.SIMULATE_CIRCUIT,
            variable_values={
                "circuit": str(DataString(str(stim_circuit))),
                "shots": shots,
            },
            request_id=request_id,
        )
        mmts = Measurements(
            DataString.from_data_string(str(result.get("measurements"))),
            data_format=DataFormat.F01,
        )
        leakage = None
        if result.get("leakage") is not None:
            leakage = LeakageFlags(
                DataString.from_data_string(str(result.get("leakage"))),
                DataFormat.F01,
            )
        return mmts, leakage

    @override
    def add_noise(self, stim_circuit: str | stim.Circuit, noise_model: NoiseModel, request_id: str) -> str:
        if noise_model.ENDPOINT is None:
            raise NotImplementedError(f"Noise addition for {type(noise_model)} is not implemented.")
        result = self.execute(
            query_name=noise_model.ENDPOINT,
            variable_values={
                "circuit": str(DataString(str(stim_circuit))),
                "noise_model": noise_model.__dict__,
            },
            request_id=request_id,
        )
        return DataString.from_data_string(str(result.get("circuit"))).to_string()

    @override
    def decode(
        self,
        detectors: DetectionEvents,
        observables: ObservableFlips,
        decoder: Decoder,
        noisy_stim_circuit: str | stim.Circuit,
        leakage_flags: LeakageFlags | None,
        request_id: str,
    ) -> DecodingResult:
        query_name = APIEndpoints.DECODE if leakage_flags is None else APIEndpoints.DECODE_LEAKAGE
        job_result = self.execute(
            query_name=query_name,
            variable_values={
                "circuit": str(DataString(str(noisy_stim_circuit))),
                "detectors": detectors.as_data_string(DataFormat.B8),
                "observables": observables.as_data_string(DataFormat.B8),
                "leakage_flags": leakage_flags.as_data_string(DataFormat.B8) if leakage_flags else None,
                "decoder_type": decoder.decoder_type.value,
                "decoder_parameters": decoder.parameters,
                "decoder_jobs": decoder.parallel_jobs,
                "decoder_expgraph": decoder.use_experimental_graph,
            },
            request_id=request_id,
        )
        predictions = job_result.pop("predictions")
        job_result.pop("indices")
        result = DecodingResult(
            predictions_format=DataFormat.F01,
            **job_result,
        )
        result.predictions = BinaryDataType(predictions)
        return result

    @override
    def defect_rates(self,
        detectors: DetectionEvents,
        stim_circuit: str | stim.Circuit,
        request_id: str,
    ) -> dict[tuple[float, ...], list[float]]:
        query_name = APIEndpoints.DEFECT_RATES
        result = self.execute(
            query_name=query_name,
            variable_values={
                "circuit": str(DataString(str(stim_circuit))),
                "detectors": detectors.as_data_string(DataFormat.B8),
            },
            request_id=request_id,
        )
        rates = result["defect_rates"]
        return {
            tuple(map(float, pair["key"])): pair["value"]
            for pair in rates
        }

    @override
    def get_correlation_matrix_for_trimmed_data(
        self,
        detectors: DetectionEvents,
        noise_floor_circuit: str | stim.Circuit,
        use_default_noise_model_edges: bool,
        request_id: str,
    ) -> tuple[npt.NDArray[np.float64], QubitCoordinateToDetectorMapping]:
        query_name = APIEndpoints.CORRELATION_MATRIX
        result = self.execute(
            query_name=query_name,
            variable_values={
                "circuit": str(DataString(str(noise_floor_circuit))),
                "detectors": detectors.as_data_string(DataFormat.B8),
                "use_stim_graph": use_default_noise_model_edges,
            },
            request_id=request_id,
        )
        matrix = result.get("correlation_matrix")
        qubit_to_detectors = cast(list, result.get("mapping", []))
        mapping = QubitCoordinateToDetectorMapping({
            tuple(qubit_det["qubit"]): qubit_det["detectors"] for qubit_det in qubit_to_detectors
        })
        return np.array(matrix, dtype=np.float64), mapping

    @override
    def trim_circuit_and_detectors(
        self,
        stim_circuit: str | stim.Circuit,
        detectors: DetectionEvents,
        request_id: str
    ) -> tuple[str, DetectionEvents]:
        query_name = APIEndpoints.TRIM_CIRCUIT_AND_DETECTORS
        result = self.execute(
            query_name=query_name,
            variable_values={
                "circuit": str(DataString(str(stim_circuit))),
                "detectors": detectors.as_data_string(DataFormat.B8),
            },
            request_id=request_id,
        )
        circuit = str(result.get("circuit"))
        dets_datastring = str(result.get("detectors"))
        detectors = DetectionEvents(DataString.from_data_string(dets_datastring), DataFormat.F01)
        return DataString.from_data_string(circuit).to_string(), detectors

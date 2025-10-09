# (c) Copyright Riverlane 2020-2025.

import pytest
from tests.helpers._utils import FakeResponse
from deltakit_explorer._api._api_client import APIEndpoints
from deltakit_explorer._api._api_v2_client import APIv2Client, Job, JobStatus
from deltakit_explorer.enums._api_enums import DecoderType
from deltakit_explorer.types._exceptions import ServerException
from deltakit_explorer.types._experiment_types import QECExperimentDefinition
from deltakit_explorer.types._types import (Decoder, DetectionEvents, ObservableFlips,
                                            PhysicalNoiseModel,
                                            SI1000NoiseModel)


class TestExecute:

    def test_execute_success(self, mocker):
        client = APIv2Client("http://localhost")
        job_submit = Job(
            status=JobStatus.SUBMITTED.value,
            request_id="id",
            type=APIEndpoints.GENERATE_CIRCUIT,
            result={"foo": "bar"},
        )
        job_success = Job(
            status=JobStatus.SUCCESS.value,
            request_id="id",
            type=APIEndpoints.GENERATE_CIRCUIT,
            result={"foo": "bar"},
        )
        mocker.patch.object(client, "_submit_task", return_value=job_submit)
        mocker.patch.object(client, "_get_job_status", return_value=job_success)
        result = client.execute(APIEndpoints.GENERATE_CIRCUIT, {}, "id")
        assert result == {"foo": "bar"}

    def test_execute_success_in_a_loop(self, mocker):
        client = APIv2Client("http://localhost")
        job_submit = Job(
            status=JobStatus.SUBMITTED.value,
            request_id="id",
            type=APIEndpoints.GENERATE_CIRCUIT,
            result={"foo": "bar"},
        )
        job_success = Job(
            status=JobStatus.SUCCESS.value,
            request_id="id",
            type=APIEndpoints.GENERATE_CIRCUIT,
            result={"foo": "bar"},
        )
        mocker.patch.object(client, "_submit_task", return_value=job_submit)
        mocker.patch.object(
            client, "_get_job_status", side_effect=[job_submit, job_success]
        )
        result = client.execute(APIEndpoints.GENERATE_CIRCUIT, {}, "id")
        assert result == {"foo": "bar"}

    def test_execute_job_error(self, mocker):
        client = APIv2Client("http://localhost")
        job = Job(
            status=JobStatus.FAILED.value,
            request_id="id",
            type=APIEndpoints.GENERATE_CIRCUIT,
            error="fail",
        )
        mocker.patch.object(client, "_submit_task", return_value=job)
        mocker.patch.object(client, "_get_job_status", return_value=job)
        with pytest.raises(ServerException, match="fail"):
            client.execute(APIEndpoints.GENERATE_CIRCUIT, {}, "id")


def test_generate_circuit_calls_execute(mocker):
    client = APIv2Client("http://localhost")
    mocker.patch.object(client, "execute", return_value={"circuit": "duck://0d0a"})
    result = client.generate_circuit(
        QECExperimentDefinition.get_repetition_z_quantum_memory(1, 1), "id"
    )
    assert isinstance(result, str)


class TestAPICalls:

    def test_simulate_circuit_leakage(self, mocker):
        client = APIv2Client("http://localhost")
        mocker.patch.object(
            client,
            "execute",
            return_value={"measurements": "duck://00", "leakage": "duck://00"},
        )
        mmts, leakage = client.simulate_circuit("circuit", 1, "id")
        assert mmts is not None
        assert leakage is not None

    def test_add_noise(self, mocker):
        client = APIv2Client("http://localhost")
        mocker.patch.object(client, "execute", return_value={"circuit": "duck://0a0b"})
        result = client.add_noise(
            "circ", PhysicalNoiseModel.get_superconducting_noise(), "id"
        )
        assert isinstance(result, str)

    def test_add_noise_si1000(self, mocker):
        client = APIv2Client("http://localhost")
        mocker.patch.object(client, "execute", return_value={"circuit": "duck://0a0b"})
        result = client.add_noise("circ", SI1000NoiseModel(0.1, 0.1), "id")
        assert isinstance(result, str)

    def test_decode_calls_execute(self, mocker):
        client = APIv2Client("http://localhost")
        mocker.patch.object(
            client,
            "execute",
            return_value={
                "fails": 1,
                "shots": 2,
                "indices": None,
                "times": [1.0],
                "counts": [2],
                "predictions": [[0], [1]],
            },
        )
        result = client.decode(
            DetectionEvents([[0]]),
            ObservableFlips([[0]]),
            Decoder(DecoderType.AC),
            "stim circuit",
            None,
            "id",
        )
        assert result.fails == 1
        assert result.shots == 2

    def test_defect_rates_returns_dict(self, mocker):
        client = APIv2Client("http://localhost")
        mocker.patch.object(
            client,
            "execute",
            return_value={"defect_rates": [{"key": [1.0], "value": [0.1]}]},
        )
        result = client.defect_rates(DetectionEvents([[0]]), "circ", "id")
        assert (1.0,) in result

    @pytest.mark.parametrize("use_graph", [False, True])
    def test_correlation_matrix_returns_matrix(self, mocker, use_graph):
        client = APIv2Client("http://localhost")
        mocker.patch.object(
            client,
            "execute",
            return_value={
                "correlation_matrix": [[0, 0], [0, 0]],
                "mapping": [
                    {"qubit": [1, 2], "detectors": [3, 4]},
                    {"qubit": [1, 2], "detectors": [3, 4]},
                ],
            },
        )
        _, mapping = client.get_correlation_matrix_for_trimmed_data(
            DetectionEvents([[0]]), "circ", use_graph, "id"
        )
        assert (1.0, 2.0) in mapping.detector_map

    def test_trim_circuit_and_detectors(self, mocker):
        client = APIv2Client("http://localhost")
        mocker.patch.object(
            client,
            "execute",
            return_value={"circuit": "duck://0b0c", "detectors": "duck://0c0d"},
        )
        circuit, dets = client.trim_circuit_and_detectors(
            "circ", DetectionEvents([[0]]), "id"
        )
        assert isinstance(circuit, str)
        assert isinstance(dets, DetectionEvents)

    def test_get_job_status_not_found(self, mocker):
        url = "https://unknown/url"
        client = APIv2Client(url)
        mocker.patch.object(
            client._request_session, "get", return_value=FakeResponse(404)
        )
        with pytest.raises(KeyError):
            client._get_job_status("123")

    def test_get_job_status_other_failure(self, mocker):
        url = "https://unknown/url"
        client = APIv2Client(url)
        mocker.patch.object(
            client._request_session, "get", return_value=FakeResponse(401)
        )
        with pytest.raises(ServerException, match=r"\[401\] BODY text"):
            client._get_job_status("123")


class TestServiceCalls:

    def test_get_job_status_ok(self, mocker):
        url = "https://unknown/url"
        client = APIv2Client(url)
        mocker.patch.object(
            client._request_session, "get", return_value=FakeResponse(200)
        )
        assert client._get_job_status("123").status == "SUBMITTED"

    def test_kill(self, mocker):
        url = "https://unknown/url"
        client = APIv2Client(url)
        mocker.patch.object(
            client._request_session, "delete", return_value=FakeResponse(200, text="12")
        )
        assert client.kill("1243") == 12

    def test_kill_non_number(self, mocker):
        url = "https://unknown/url"
        client = APIv2Client(url)
        mocker.patch.object(
            client._request_session,
            "delete",
            return_value=FakeResponse(200, text="abf"),
        )
        assert client.kill("1243") == 0

    def test_submit_task(self, mocker):
        client = APIv2Client("http://localhost")
        mocker.patch.object(
            client._request_session, "post", return_value=FakeResponse()
        )
        job = client._submit_task(APIEndpoints.SIMULATE_CIRCUIT, {}, "some")
        assert job.request_id == "some_id"
        assert job.status == "SUBMITTED"
        assert job.type == "simulate"

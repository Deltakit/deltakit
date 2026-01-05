# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import requests
from deltakit_circuit.gates import PauliBasis
from pytest_mock import MockerFixture

from deltakit_explorer import Client
from deltakit_explorer._utils._utils import (
    DELTAKIT_SERVER_DEFAULT_URL_ENV,
    DELTAKIT_SERVER_URL_ENV,
)
from deltakit_explorer.enums import (
    DataFormat,
    DecoderType,
    QECECodeType,
    QECExperimentType,
)
from deltakit_explorer.types import (
    CircuitParameters,
    DataString,
    Decoder,
    DecodingResult,
    DetectionEvents,
    LeakageFlags,
    Measurements,
    NoiseModel,
    ObservableFlips,
    PhysicalNoiseModel,
    QECExperiment,
    QECExperimentDefinition,
    QubitCoordinateToDetectorMapping,
    SI1000NoiseModel,
)


@pytest.fixture
def mock_client(request, mocker):
    """Server object which is used to replace real server calls."""
    version = request.param
    mocker.patch("deltakit_explorer._utils._utils.APP_NAME", "deltakit-testplorer")
    Client.set_token("1234", validate=False)
    return Client("", 5000, version)


class NewNoiseModel(NoiseModel):
    """User-defined noise model"""
    ENDPOINT: ClassVar = None
    ENDPOINT_RESULT_FIELDNAME: ClassVar = None


class TestClient:

    @pytest.mark.parametrize(
        ("method", "args", "server_return_value", "server_expected_result"),
        [
            (
                Client.add_noise,
                ("X 0 1", PhysicalNoiseModel.get_floor_superconducting_noise()),
                {"addNoiseToStimCircuit": {"uid": "duck://20"}},
                " ",
            ),
            (
                Client.decode,
                (
                        DetectionEvents([[0, 1]], DataFormat.F01),
                        ObservableFlips([[1]], DataFormat.F01),
                        Decoder(DecoderType.AC, False, parameters={"ac_kappa_proportion": 0.2}),
                    "X 0 1 ~~~",
                ),
                {
                    "decode":
                    {
                        "shots": 0, "fails": 1,
                        "times": [0.4], "counts": [1],
                        "predictionsFile": None,
                    }
                },
                DecodingResult(
                    fails=1, shots=0,
                    times=[0.4], counts=[1],
                    predictionsFile=None,
                )
            ),
            (
                Client.decode,
                (
                    DetectionEvents([[0, 1]], DataFormat.F01),
                    ObservableFlips([[1]], DataFormat.F01),
                    Decoder(DecoderType.LCD, False, parameters={"weighted": True}),
                    "X 0 1 ~~~",
                    LeakageFlags([[0, 1]], DataFormat.F01),
                ),
                {
                    "decode":
                    {
                        "shots": 0, "fails": 1,
                        "times": [0.4], "counts": [1],
                        "predictionsFile": None,
                    }
                },
                DecodingResult(
                    fails=1, shots=0,
                    times=[0.4], counts=[1],
                    predictionsFile=None,
                )
            ),
            (
                Client.defect_rates,
                (DetectionEvents([[0, 1]], DataFormat.F01), "X 0 1 ~~~"),
                {"defectRates": {"items": [{"key": (), "value": []}]}},
                {(): []},
            ),
            (
                Client.get_correlation_matrix_for_trimmed_data,
                (DetectionEvents([[0, 1]], DataFormat.F01), "X 0 1 ~~~~", False),
                {
                    "correlationMatrix": {
                        "matrix": [],
                        "qubitToDetectionEvents": [{"qubit": {}, "detectors": {}}],
                    }
                },
                (np.array([]), QubitCoordinateToDetectorMapping({(): {}})),  # type: ignore[dict-item]
            ),
            (
                Client.trim_circuit_and_detectors,
                ("X 0 1 ~~~", DetectionEvents([[0, 1]], DataFormat.F01)),
                {"trimCircuitAndDetectionEvents":
                 {"circuit":
                  {"uid": str(DataString("X 0"))}, "detectors": {"uid": "duck://20"}}},
                ("X 0", DetectionEvents(DataString(" "), DataFormat.B8)),
            ),
        ],
    )
    @pytest.mark.parametrize("mock_client", [1], indirect=True)
    def test_functions_calls_server_methods_correctly_v1(
        self, mocker: MockerFixture,
        method, args,
        server_return_value, server_expected_result, mock_client
    ):
        # setup mocks
        # use patch.object as we are import Client in this test module
        mock_execute = mocker.patch.object(
            mock_client._api, "execute", return_value=server_return_value
        )
        # run, pass server as self
        result = method(mock_client, *args)
        # asserts
        mock_execute.assert_called_once()
        assert str(result) == str(server_expected_result)

    @pytest.mark.parametrize("mock_client", [1], indirect=True)
    def test_get_correlation_matrix_success_v1(self, mocker, mock_client):
        mock_get_query = mocker.patch.object(
            mock_client._api, "_get_query", return_value="test"
        )
        mock_generate_noisy_stim_circuit = mocker.patch.object(
            mock_client, "add_noise", return_value="X 0 1\n~~~",
        )
        mock_gql = mocker.patch(
            "deltakit_explorer._api._gql_client.gql", return_value="test2"
        )
        mock_client_execute = mocker.patch(
            "gql.client.SyncClientSession.execute",
            spec=True,
            side_effect=[
                {
                    "trimCircuitAndDetectionEvents": {
                        "detectors": {"uid": "duck://abcd"},
                        "circuit": {"uid": "duck://2020"},
                    }
                },
                {
                    "correlationMatrix": {
                        "matrix": [],
                        "qubitToDetectionEvents": [{"qubit": {}, "detectors": {}}],
                    }
                },
            ],
        )
        result = mock_client.get_correlation_matrix(
            DetectionEvents([[0, 1]], DataFormat.F01), "X 0 1", False,
        )
        expected_result = (np.array([]), QubitCoordinateToDetectorMapping({(): {}}))

        # asserts
        assert mock_get_query.call_count == 2
        assert mock_gql.call_count == 2
        assert mock_client_execute.call_count == 2
        mock_generate_noisy_stim_circuit.assert_called_once()
        assert str(result) == str(expected_result)

    @pytest.mark.parametrize("mock_client", [1, 2], indirect=True)
    def test_decode_measurements_success(self, mocker, mock_client):
        mock_generate_noisy_stim_circuit = mocker.patch.object(
            Client, "add_noise", return_value=DataString("01"),
        )
        mock_convert_meas_to_dets_and_obs = mocker.patch.object(
            Measurements,
            "to_detectors_and_observables",
            return_value=(
                DetectionEvents([[0, 1]], DataFormat.F01),
                ObservableFlips([[0]], DataFormat.F01),
            ),
        )
        mock_decode = mocker.patch.object(Client, "decode")

        mock_client.decode_measurements(
            Measurements([[0, 1]], DataFormat.F01),
            decoder=Decoder(DecoderType.BP_OSD),
            ideal_stim_circuit="",
            noise_model=PhysicalNoiseModel.get_floor_superconducting_noise(),
            leakage_flags=LeakageFlags([[]], DataFormat.F01),
        )
        # asserts
        mock_generate_noisy_stim_circuit.assert_called_once()
        mock_convert_meas_to_dets_and_obs.assert_called_once()
        mock_decode.assert_called_once()

    @pytest.mark.parametrize("mock_client", [1, 2], indirect=True)
    def test_get_defect_rates_for_defect_diagram_success(self, mocker, mock_client):
        mock_convert_meas_to_dets_and_obs = mocker.patch.object(
            Measurements,
            "to_detectors_and_observables",
            return_value=(DetectionEvents([[0, 1]]), ObservableFlips([[0]])),
        )
        mock_decode = mocker.patch.object(
            Client,
            "trim_circuit_and_detectors",
            return_value=("X 0 1", DetectionEvents([0, 1])),
        )
        mock_defect_rate = mocker.patch.object(
            Client, "defect_rates", return_value={}
        )
        result = mock_client.get_experiment_detectors_and_defect_rates(
            QECExperiment(noisy_circuit="", measurements=Measurements([0, 1]))
        )
        mock_convert_meas_to_dets_and_obs.assert_called_once()
        mock_decode.assert_called_once()
        mock_defect_rate.assert_called_once()
        assert result == ({}, {})

    @pytest.mark.parametrize(
        "data_format", [DataFormat.F01, DataFormat.B8])
    @pytest.mark.parametrize("mock_client", [1, 2], indirect=True)
    def test_decode_measurements_success_with_format(self, mocker, data_format, mock_client):
        detectors = DetectionEvents([[0, 1]])
        observables = ObservableFlips([[0]])
        mocker.patch.object(
            Measurements,
            "to_detectors_and_observables",
            return_value=(detectors, observables),
        )
        mocker.patch.object(
            Client, "add_noise", return_value="X 0 1 ~~~",
        )
        mock_decode = mocker.patch.object(
            mock_client, "decode",
        )
        measurements = Measurements(
            Path(f"meas_file.{data_format}"), data_format=data_format)
        decoder = Decoder(
            decoder_type=DecoderType.CC,
            use_experimental_graph=False,
        )
        mock_client.decode_measurements(
            measurements=measurements,
            decoder=decoder,
            ideal_stim_circuit="X 0 1",
            noise_model=PhysicalNoiseModel.get_floor_superconducting_noise(),
        )
        # asserts
        mock_decode.assert_called_once_with(
            detectors=detectors,
            decoder=decoder,
            observables=observables,
            noisy_stim_circuit="X 0 1 ~~~",
            leakage_flags=None,
        )

    @pytest.mark.parametrize("mock_client", [1, 2], indirect=True)
    def test_decode_measurements_success_with_leakage_with_format(self, mocker, mock_client):
        mocker.patch.object(
            Client,
            "simulate_stim_circuit",
            return_value=(Measurements([[0, 1]]), LeakageFlags([[0, 1]])),
        )
        mocker.patch.object(
            Measurements,
            "to_detectors_and_observables",
            return_value=(DetectionEvents([[0, 1]]), ObservableFlips([[0]])),
        )
        mocker.patch.object(
            Client, "add_noise", return_value="X 2 3",
        )
        mock_decode = mocker.patch.object(mock_client, "decode")
        decoder = Decoder(DecoderType.AC, use_experimental_graph=False)
        leakage = LeakageFlags([[0, 1]])
        mock_client.decode_measurements(
            measurements=Measurements([[0, 1]]),
            decoder=decoder,
            ideal_stim_circuit="",
            noise_model=PhysicalNoiseModel.get_floor_superconducting_noise(),
            leakage_flags=leakage,
        )

        # asserts
        mock_decode.assert_called_once_with(
            detectors=DetectionEvents([[0, 1]]),
            observables=ObservableFlips([[0]]),
            decoder=decoder,
            noisy_stim_circuit="X 2 3",
            leakage_flags=leakage,
        )

    @pytest.mark.parametrize(
        "leakage_source", [LeakageFlags([[0, 1]], DataFormat.F01), None]
    )
    @pytest.mark.parametrize("decoder", [DecoderType.AC, DecoderType.LCD])
    @pytest.mark.parametrize("add_detectors", [False, True])
    @pytest.mark.parametrize("mock_client", [1], indirect=True)
    def test_decode_gets_and_executes_correct_query_v1(
        self, mocker, leakage_source, mock_client, decoder, add_detectors
    ):
        detectors_width = 2
        shots = 12_000
        call_count = 4
        mock_get_query = mocker.patch.object(mock_client._api, "_get_query")
        mock_execute_query = mocker.patch.object(
            mock_client._api,
            "execute_query",
            return_value={
                "decode":
                {"shots": 1, "fails": 0, "times": [],
                 "counts": [1], "predictionsFile": None}
            }
        )
        circuit = (
            "OBSERVABLE_INCLUDE\n" +
            ("HERALD_LEAKAGE_EVENT\n" if leakage_source is not None else "\n")
        ) * 2600
        if add_detectors:
            circuit += "DETECTOR(2, 3) rec[-1]\n" * 100000
        mock_client.decode(
            decoder=Decoder(decoder),
            detectors=DetectionEvents([[0] * detectors_width] * shots, DataFormat.F01),
            observables=ObservableFlips([[0]] * shots, DataFormat.F01),
            noisy_stim_circuit=circuit,
            leakage_flags=leakage_source,
        )
        mult = 1
        # more detectors - additional sharding happens
        # to reduce submitted data size
        if add_detectors:
            mult = 4
        # there as sharding or requests happening
        assert call_count * mult == mock_get_query.call_count
        assert call_count * mult == mock_execute_query.call_count

    @pytest.mark.parametrize("mock_client", [1, 2], indirect=True)
    def test_decode_measurements_throws_exception_if_interior_method_fails(
        self, mocker, mock_client
    ):
        mocker.patch.object(
            Client, "add_noise",
            side_effect=Exception("something went wrong"),
        )
        with pytest.raises(Exception, match="something went wrong"):
            mock_client.decode_measurements(
                measurements=Measurements([[0, 1]]),
                decoder=Decoder(DecoderType.BELIEF_MATCHING),
                ideal_stim_circuit="",
                noise_model=PhysicalNoiseModel.get_floor_superconducting_noise(),
            )

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            (Client.add_noise,
             ("", PhysicalNoiseModel.get_floor_superconducting_noise())),
            (Client.add_noise,
             ("", SI1000NoiseModel(0.1, 0.2))),
            (Client.decode, (
                    DetectionEvents([[]], DataFormat.F01),
                    ObservableFlips([[]], DataFormat.F01),
                    Decoder(DecoderType.CC), "",
            )),
            (Client.defect_rates, (DetectionEvents([[]], DataFormat.F01), "")),
            (Client.get_correlation_matrix, (DetectionEvents([[]], DataFormat.F01), "", False)),
            (Client.trim_circuit_and_detectors, ("", DetectionEvents([[]], DataFormat.F01))),
            (Client.get_correlation_matrix_for_trimmed_data, (
                    DetectionEvents([[]], DataFormat.F01), "", False
            )),
        ],
    )
    @pytest.mark.parametrize("mock_client", [1], indirect=True)
    def test_function_throws_exception_if_inner_methods_fail_v1(
        self, mocker: MockerFixture, method, args, mock_client
    ):
        mocker.patch.object(
            mock_client._api, "_get_query",
            side_effect=Exception("_get_query failed for some reason"),
        )

        with pytest.raises(Exception, match="_get_query failed for some reason"):
            # passing mock server as self
            method(mock_client, *args)

    @pytest.mark.parametrize("mock_client", [1, 2], indirect=True)
    def test_get_defect_rates_for_defect_diagram_fail(self, mocker: MockerFixture, mock_client):
        mocker.patch.object(
            Measurements,
            "to_detectors_and_observables",
            side_effect=Exception("failed for some reason"),
        )
        with pytest.raises(Exception, match=r"failed for some reason"):
            mock_client.get_experiment_detectors_and_defect_rates(
                QECExperiment(noisy_circuit="", measurements=Measurements([[]], DataFormat.F01)))

    def test_preconfigured_constructor_other_machine_v1(self):
        os.environ.pop(DELTAKIT_SERVER_URL_ENV, "")
        client = Client.get_instance(api_version=1)
        assert client._api.content_endpoint == f"{DELTAKIT_SERVER_DEFAULT_URL_ENV}/api/data/"
        assert client._api.graphql_endpoint == f"{DELTAKIT_SERVER_DEFAULT_URL_ENV}/api/graphql"

    def test_kill_request_v1(self, mocker):
        # fix the auth token
        mocker.patch("deltakit_explorer._utils._utils.APP_NAME", "deltakit-testplorer")
        Client.set_token("1234qwerty", False)
        os.environ.pop(DELTAKIT_SERVER_URL_ENV, "")
        client = Client.get_instance(api_version=1)

        resp = requests.Response()
        resp._content = b"7"

        kill_mock = mocker.patch.object(
            client._api._request_session, "get", spec=requests.get,
            return_value=resp
        )
        number = client.kill("abcdefg")
        assert number == 7
        kill_mock.assert_called_once_with(
            url=f"{DELTAKIT_SERVER_DEFAULT_URL_ENV}/api/data/kill/abcdefg",
            headers={"Authorization": "Bearer 1234qwerty"},
            verify=True,
            timeout=10,
        )

    @pytest.mark.parametrize("use_default_noise_model_edges", [True, False])
    def test_get_correlation_matrix_for_trimmed_data_v1(self, mocker, use_default_noise_model_edges):

        client = Client.get_instance(api_version=1)
        det = DetectionEvents([[0, 0, 0, 1]], DataFormat.F01)
        circuit = "SOME STIM CIRCUIT"

        mocker.patch.object(client._api, "_get_query", return_value="")
        mocker.patch.object(
            client._api,
            "execute_query",
            return_value={
                "correlationMatrix": {
                    "matrix": [
                        [0.0, 0.0, 0.1, 0.2],
                        [0.0, 0.0, 0.1, 0.2],
                        [0.0, 0.0, 0.1, 0.2],
                        [0.0, 0.0, 0.1, 0.2]
                    ],
                    "qubitToDetectionEvents": [
                        {"qubit": (1.0, 0.0), "detectors": [4, 5, 6, 7,]},
                        {"qubit": (3.0, 0.0), "detectors": [8, 9, 10, 11]},
                    ]
                }
            }
        )
        corr, mapping = client.get_correlation_matrix_for_trimmed_data(
            det, circuit, use_default_noise_model_edges)
        assert corr.shape == (4, 4)
        assert len(mapping.detector_map) == 2

    @pytest.mark.parametrize("version", [1, 2])
    def test_get_experiment_detectors_and_defect_rates_fails_with_empty_experiment(self, version):
        client = Client.get_instance(api_version=version)
        with pytest.raises(ValueError, match=r"^Experiment object should"):
            client.get_experiment_detectors_and_defect_rates(QECExperiment(""))


    @pytest.mark.parametrize("version", [1, 2])
    def test_add_noise_unsupported_noise_model_raises(self, version):
        client = Client.get_instance(api_version=version)
        with pytest.raises(NotImplementedError):
            client.add_noise("circuit", NewNoiseModel())

    def test_generate_circuit_v1(self, mocker):
        client = Client.get_instance(api_version=1)
        mocker.patch.object(client._api, "_get_query", return_value="")
        mocker.patch.object(
            client._api,
            "execute_query",
            return_value={"generateCircuit": {"uid": "duck://30313233"}}
        )
        text = client.generate_circuit(QECExperimentDefinition.get_repetition_z_quantum_memory(3, 3))
        assert text == "0123"


    def test_generate_circuit_heavy_bb_runs_check_v1(self, mocker):
        client = Client.get_instance(api_version=1)
        mocker.patch.object(client._api, "_get_query", return_value="")
        mocker.patch.object(
            client._api,
            "execute_query",
            return_value={"generateCircuit": {"uid": "duck://30313233"}}
        )
        text = client.generate_circuit(QECExperimentDefinition(
            experiment_type=QECExperimentType.QUANTUM_MEMORY,
            code_type=QECECodeType.BIVARIATE_BICYCLE,
            observable_basis=PauliBasis.Z,
            num_rounds=123,
            parameters=CircuitParameters.from_matrix_specification(
                21, 18, [3, 1, 2], [3, 1, 2]
            ),
            basis_gates=["CX", "H"],
        ))
        assert text == "0123"

    def test_generate_circuit_heavy_planar_runs_check_v1(self, mocker):
        client = Client.get_instance(api_version=1)
        mocker.patch.object(client._api, "_get_query", return_value="")
        mocker.patch.object(
            client._api,
            "execute_query",
            return_value={"generateCircuit": {"uid": "duck://30313233"}}
        )
        text = client.generate_circuit(QECExperimentDefinition(
            experiment_type=QECExperimentType.QUANTUM_MEMORY,
            code_type=QECECodeType.UNROTATED_PLANAR,
            observable_basis=PauliBasis.Z,
            num_rounds=123,
            parameters=CircuitParameters.from_sizes([31, 31]),
            basis_gates=["CX", "H"],
        ))
        assert text == "0123"


class TestAPIVersions:

    @pytest.mark.parametrize("version", [1, 2])
    def test_version_implemented(self, version):
        client = Client("url", api_version=version)
        assert client._api_version == version

    @pytest.mark.parametrize("version", [0, 3])
    def test_version_not_implemented(self, version):
        with pytest.raises(NotImplementedError):
            Client("url", api_version=version)

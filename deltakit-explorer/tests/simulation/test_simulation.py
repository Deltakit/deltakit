# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

from pathlib import Path

import pytest
import stim

from deltakit_explorer import simulation
from deltakit_explorer._api._client import Client
from deltakit_explorer.types import LeakageFlags, Measurements


class TestStimSimulation:

    @pytest.mark.parametrize("use_file", [False, True])
    @pytest.mark.parametrize(
        ("shots", "filesize"),
        [
            (1, 2),
            (10, 20),
            (1000000, 2000000),
        ],
    )
    def test_stim_simulates_locally_ok(self, use_file, shots, filesize, tmp_path):
        stim_file = Path(__file__).parent / "../resources/rep_code_mutated_default_noise_data.stim"
        measurements, _ = simulation.simulate_with_stim(
            stim_circuit=str(stim.Circuit.from_file(stim_file)),
            shots=shots,
            result_file=tmp_path / "tst.b8" if use_file else None,
        )
        assert len(measurements.as_b8_bytes()) == filesize

    def test_stim_simulates_locally_ok_with_stim_circuit(self):
        stim_file = Path(__file__).parent / "../resources/rep_code_mutated_default_noise_data.stim"
        simulation.simulate_with_stim(
            stim_circuit=stim.Circuit.from_file(stim_file),
            shots=1,
        )

    def test_stim_fails_locally_wrong_stim_format(self):
        stim_file = Path(__file__).parent / "../resources/rep_code_noisy_stim_dets.01"
        with pytest.raises(ValueError, match="Gate not found"), \
            Path.open(Path(stim_file)) as file:
                simulation.simulate_with_stim(
                    stim_circuit=file.read(),
                    shots=10,
                )

    def test_stim_fails_locally_no_destination_folder(self, tmp_path):
        stim_file = Path(__file__).parent / "../resources/rep_code_mutated_default_noise_data.stim"
        (Path(tmp_path) / "test-folder").mkdir(exist_ok=True)
        with pytest.raises(ValueError, match="Failed to open"):
            simulation.simulate_with_stim(
                stim_circuit=str(stim.Circuit.from_file(stim_file)),
                shots=10,
                result_file=Path(tmp_path) / "test-folder",
            )

    def test_stim_fails_negative_shots(self):
        stim_file = Path(__file__).parent / "../resources/rep_code_mutated_default_noise_data.stim"
        with pytest.raises(ValueError, match="non-negative"):
            simulation.simulate_with_stim(
                stim_circuit=str(stim.Circuit.from_file(stim_file)),
                shots=-2,
            )

    @pytest.mark.parametrize(
            ("circuit", "result"),
            [
                (
                    "MZ 1\nLEAKAGE(0.0) 1",
                    (Measurements([[0], [0]]), LeakageFlags([[0], [0]]))
                ),  # with leakage
                (
                    "MZ(0.0) 1",
                    (Measurements([[0], [0]]), None)
                )  # only measurement
            ]
    )
    def test_simulate_on_server(self, circuit, result, mocker):
        mock_client = Client("http://localhost")
        mocker.patch.object(mock_client, "simulate_stim_circuit", return_value=result)
        meas, _ = simulation.simulate_with_stim(
            stim_circuit=circuit,
            shots=2,
            client=mock_client,
        )
        assert meas.as_01_string() == "0\n0\n"
        mock_client.simulate_stim_circuit.assert_called_once_with(circuit, shots=2)


    def test_simulate_on_server_cannot_save_to_disk(self):
        client = Client("http://localhost")
        with pytest.raises(NotImplementedError):
            simulation.simulate_with_stim(
                stim_circuit="123", shots=2, result_file="somefile", client=client,
            )

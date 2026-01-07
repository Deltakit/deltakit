# (c) Copyright Riverlane 2020-2025.
"""Abstract class to define base functionality of API implementation."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
import stim

from deltakit_explorer.enums._api_enums import APIEndpoints
from deltakit_explorer.types import (
    Decoder,
    DecodingResult,
    DetectionEvents,
    LeakageFlags,
    Measurements,
    NoiseModel,
    ObservableFlips,
    QubitCoordinateToDetectorMapping,
)
from deltakit_explorer.types._experiment_types import QECExperimentDefinition


class APIClient(ABC):  # pragma: nocover
    """Abstract class which presents and interface
    for an API"""

    def __init__(self, base_url: str, request_timeout: int):
        super().__init__()
        self.request_timeout = request_timeout
        if base_url and base_url[-1] == "/":
            base_url = base_url[:-1]
        self.base_url = base_url

    @abstractmethod
    def set_token(self, token: str, validate: bool = True):
        """Persist the token, set it to env var and a file.
        Validate it.

        Args:
            token (str): auth token.
            validate (bool): you may switch off validation explicitly.
        """
        raise NotImplementedError()

    @abstractmethod
    def execute(
        self,
        query_name: APIEndpoints,
        variable_values: dict,
        request_id: str,
    ) -> dict[str, Any]:
        """
        Executes a query by its name and
        returns server response as a python dict.

        Args:
            query_name (APIEndpoints): name of the query to execute.
            variable_values (dict): arguments for the query.
            request_id (str): identifier of the request.

        Returns:
            dict[str, Any]: Server response.
        """
        raise NotImplementedError()

    @abstractmethod
    def kill(self, request_id: str) -> int:
        """Kill a decoding task submitted to a client.
        Decoding methods report a request ID. Calling this
        method with an ID tries to kill all CPU
        workers associated with the task. Method returns the number
        of CPU workers it could terminate. If the task is already finished,
        the method should return 0.

        Args:
            request_id (str):
                ID of the requests which started the decoding task.

        Returns:
            int: number of workers killed
        """
        raise NotImplementedError()

    @abstractmethod
    def generate_circuit(
        self,
        experiment_definition: QECExperimentDefinition,
        request_id: str
    ) -> str:
        """Generate a circuit based on the experiment definition.

        Args:
            experiment_definition (QECExperimentDefinition):
                Definition of the quantum error correction experiment.
            request_id (str): Identifier of the request.

        Returns:
            str: Generated circuit in stim format."""
        raise NotImplementedError()

    @abstractmethod
    def simulate_circuit(
        self,
        stim_circuit: str | stim.Circuit,
        shots: int,
        request_id: str
    ) -> tuple[Measurements, LeakageFlags | None]:
        """Simulate a circuit and return measurements and leakage flags.

        Args:
            stim_circuit (str | stim.Circuit): Circuit to simulate.
            shots (int): Number of shots for the simulation.
            request_id (str): Identifier of the request.

        Returns:
            tuple[Measurements, LeakageFlags | None]: Tuple containing measurements and optional leakage flags."""
        raise NotImplementedError()

    @abstractmethod
    def add_noise(self, stim_circuit: str | stim.Circuit, noise_model: NoiseModel, request_id: str) -> str:
        """Add noise to a circuit based on the provided noise model.

        Args:
            stim_circuit (str | stim.Circuit): Circuit to which noise will be added.
            noise_model (NoiseModel): Noise model to apply.
            request_id (str): Identifier of the request.

        Returns:
            str: Noisy circuit in stim format. May contain leakage (not stim-compatible), if SI1000NoiseModel is used."""
        raise NotImplementedError()

    @abstractmethod
    def decode(
        self,
        detectors: DetectionEvents,
        observables: ObservableFlips,
        decoder: Decoder,
        noisy_stim_circuit: str | stim.Circuit,
        leakage_flags: LeakageFlags | None,
        request_id: str,
    ) -> DecodingResult:
        """Decode a noisy circuit with given detectors and observables.

        Args:
            detectors (DetectionEvents): DetectionEvents to use for decoding.
            observables (ObservableFlips): ObservableFlips to use for decoding.
            decoder (Decoder): Decoder to use.
            noisy_stim_circuit (str | stim.Circuit): Noisy circuit to decode.
            leakage_flags (LeakageFlags | None): Optional leakage flags.
            request_id (str): Identifier of the request.

        Returns:
            DecodingResult:
                Result of the decoding, which includes predictions,
                fails count, and CPU time spent on decoding.

        """
        raise NotImplementedError()

    @abstractmethod
    def defect_rates(
        self,
        detectors: DetectionEvents,
        stim_circuit: str | stim.Circuit,
        request_id: str,
    ) -> dict[tuple[float, ...], list[float]]:
        """Get defect rates for detectors.

        Args:
            detectors (DetectionEvents):
                DetectionEvents values from simulations to compute defect rates.
            stim_circuit (str | stim.Circuit): Circuit to analyse.
            request_id (str): Identifier of the request."""
        raise NotImplementedError()

    @abstractmethod
    def get_correlation_matrix_for_trimmed_data(
        self,
        detectors: DetectionEvents,
        noise_floor_circuit: str | stim.Circuit,
        use_default_noise_model_edges: bool,
        request_id: str,
    ) -> tuple[npt.NDArray[np.float64], QubitCoordinateToDetectorMapping]:
        """Get correlation matrix for detectors.

        Args:
            detectors (DetectionEvents): Detector data to analyse.
            noise_floor_circuit (str | stim.Circuit): Circuit with the minimal noise.
            use_default_noise_model_edges (bool): Whether to use default noise model edges.
            request_id (str): Identifier of the request.

        Returns:
            tuple[npt.NDArray[np.float64], QubitCoordinateToDetectorMapping]:
                Tuple containing the correlation matrix and a mapping of qubit coordinates to detectors.
        """
        raise NotImplementedError()

    @abstractmethod
    def trim_circuit_and_detectors(
        self,
        stim_circuit: str | stim.Circuit,
        detectors: DetectionEvents,
        request_id: str
    ) -> tuple[str, DetectionEvents]:
        """Trim a circuit and detectors to remove qubits and detectors irrelevant to decoding problem.

        Args:
            stim_circuit (str | stim.Circuit): Circuit to trim.
            detectors (DetectionEvents): DetectionEvents to trim.
            request_id (str): Identifier of the request.

        Returns:
            tuple[str, DetectionEvents]: Tuple containing the trimmed circuit in stim format and the trimmed detectors
        """
        raise NotImplementedError()

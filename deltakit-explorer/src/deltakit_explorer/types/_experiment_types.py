# (c) Copyright Riverlane 2020-2025.
"""This file contains types to support QEC experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import stim
from deltakit_circuit.gates import PauliBasis

from deltakit_explorer.enums._api_enums import (
    DataFormat,
    QECECodeType,
    QECExperimentType,
)
from deltakit_explorer.types._types import (
    BinaryDataType,
    CircuitParameters,
    DetectionEvents,
    JSONable,
    LeakageFlags,
    Measurements,
    ObservableFlips,
)


@dataclass
class QECExperiment:
    """
    Represents a core of simulated experiment:
     - a noisy circuit which is used for simulation;
     - sweep bits, if used for initialisation;
     - obtained measurements, detector, observable and leakage values.
    """

    noisy_circuit: str
    sweep_bits: BinaryDataType | None = None

    measurements: Measurements | None = None

    detectors: DetectionEvents | None = None
    observables: ObservableFlips | None = None
    leakage_flags: LeakageFlags | None = None

    @staticmethod
    def from_circuit_and_measurements(
        stim_path: Path,
        measurements_path: Path,
        measurements_format: DataFormat,
        sweep_path: Path | None = None,
        sweep_format: DataFormat | None = None,
    ) -> QECExperiment:
        """Build an experiment object from 2 files.

        Args:
            stim_path (Path): location of a circuit.
            measurements_path (Path):
                location of a measurements file.
            measurements_format (DataFormat):
                format of measurement data.
            sweep_path (Optional[Path]):
                location of sweep bits, optional.
            sweep_format (Optional[DataFormat]):
                format of a sweep bits file.

        Returns:
            QECExperiment: experiment object.

        Raises:
            ValueError:
                If only sweep_path or sweep_format provided.
        """
        if (sweep_format is None) ^ (sweep_path is None):
            msg = (
                "sweep_path and sweep_format should be both provided, "
                "or should both be None."
            )
            raise ValueError(
                msg
            )
        circuit = stim.Circuit.from_file(stim_path)
        measurements = Measurements(
            measurements_path,
            measurements_format,
            circuit.num_measurements,
        )
        sweeps = None
        if sweep_path and sweep_format:
            sweeps = BinaryDataType(
                sweep_path,
                sweep_format,
                circuit.num_sweep_bits,
            )
        return QECExperiment(
            noisy_circuit=str(circuit),
            measurements=measurements,
            sweep_bits=sweeps,
        )

    def compute_detectors_and_observables(self):
        """If measurements and a circuit are provided, ensure that
        detectors and observables are computed."""
        if (
            (self.detectors is None or self.observables is None)
            and self.measurements is not None
        ):
            dets, obs = (
                self.measurements.to_detectors_and_observables(
                    stim_circuit=self.noisy_circuit,
                    sweep_bits=self.sweep_bits,
                )
            )
            if self.detectors is None:
                self.detectors = dets
            if self.observables is None:
                self.observables = obs


@dataclass
class QECExperimentDefinition(JSONable):
    """Definition, essential to generate a QEC experiment circuit."""

    experiment_type: QECExperimentType
    """Type of experiment, e.g. Quantum Memory or Stability."""

    code_type: QECECodeType
    """QEC code, e.g. Rotated Planar Code."""

    observable_basis: PauliBasis
    """Basis, in which observable is measured. E.g. X or Z."""

    num_rounds: int
    """Number of experiment rounds."""

    basis_gates: list[str] | None = None
    """If circuit is generated with a specific basis gate set."""

    parameters: CircuitParameters | None = None
    """Parameters of circuit generation."""

    def get_parameters_gql_string(self) -> dict[str, Any] | None:
        """
        Prepare graphql parameters to be inserted
        into GraphQL query.
        """
        if self.parameters is not None:
            return self.parameters.to_gql()
        return None

    @staticmethod
    def get_repetition_z_quantum_memory(
        distance: int,
        num_rounds: int,
        basis_gates: list[str] | None = None
    ) -> QECExperimentDefinition:
        """Convenience method to quickly define a Z-memory
        repetition code experiment."""
        return QECExperimentDefinition(
            experiment_type=QECExperimentType.QUANTUM_MEMORY,
            code_type=QECECodeType.REPETITION,
            observable_basis=PauliBasis.Z,
            num_rounds=num_rounds,
            basis_gates=basis_gates,
            parameters=CircuitParameters.from_sizes([distance]),
        )

    @staticmethod
    def get_rotated_planar_z_quantum_memory(
        distance: int,
        num_rounds: int,
        basis_gates: list[str] | None = None
    ) -> QECExperimentDefinition:
        """Convenience method to quickly define a Z-memory
        (d x d)-rotated planar code experiment."""
        return QECExperimentDefinition(
            experiment_type=QECExperimentType.QUANTUM_MEMORY,
            code_type=QECECodeType.ROTATED_PLANAR,
            observable_basis=PauliBasis.Z,
            num_rounds=num_rounds,
            basis_gates=basis_gates,
            parameters=CircuitParameters.from_sizes([distance, distance]),
        )

    def to_json(self):
        dump = super().to_json()
        dump["generate_circuit_parameters"] = dump.pop("parameters")
        dump["qe_code_type"] = dump.pop("code_type")
        dump["rounds"] = dump.pop("num_rounds")
        dump["basis"] = dump.pop("observable_basis")
        dump["basis_gate_set"] = dump.pop("basis_gates")
        return dump

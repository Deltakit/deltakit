# (c) Copyright Riverlane 2020-2025.
"""
Check-redundancy (meta-check) detectors for CSS memory circuits.

When a parity-check matrix has more rows than its GF(2) rank, syndrome bits
satisfy linear dependencies. This module emits optional DETECTOR annotations
for those dependency products.
"""

from __future__ import annotations

from collections.abc import Sequence

import galois
import numpy as np
from deltakit_circuit import Circuit, Detector, MeasurementRecord, PauliX, PauliZ
from deltakit_circuit.gates import PauliBasis
from numpy.typing import NDArray

from deltakit_explorer.codes._bivariate_bicycle_code import BivariateBicycleCode
from deltakit_explorer.codes._css._css_code import CSSCode
from deltakit_explorer.codes._css._css_stage import CSSStage
from deltakit_explorer.codes._css._stabiliser_code import StabiliserCode
from deltakit_explorer.codes._stabiliser import Stabiliser


def measurement_gate_index(num_measurements: int, record: MeasurementRecord) -> int:
    """
    Chronological index of a ``MeasurementRecord`` in ``measurement_gates``.

    Args:
        num_measurements: Total measurements in the circuit.
        record: Lookback record (``rec[-1]`` is the most recent measurement).

    Returns:
        Zero-based index into ``circuit.measurement_gates``.
    """
    return num_measurements + record.lookback_index


def detector_parity_matrix(
    detectors: Sequence[Detector], num_measurements: int
) -> NDArray[np.uint8]:
    """
    Build a GF(2) parity-check matrix for detector annotations.

    Each row is one DETECTOR; columns follow chronological measurement order
    (column 0 is the first ``measurement_gates`` entry).

    Args:
        detectors: Detectors to encode as rows.
        num_measurements: Total number of measurements in the circuit.

    Returns:
        Binary matrix with shape ``(len(detectors), num_measurements)``.
    """
    if not detectors:
        return np.zeros((0, num_measurements), dtype=np.uint8)
    rows = np.zeros((len(detectors), num_measurements), dtype=np.uint8)
    for row_index, detector in enumerate(detectors):
        for record in detector.measurements:
            rows[row_index, measurement_gate_index(num_measurements, record)] = 1
    return rows


def detector_annotation_rank(circuit: Circuit) -> int:
    """
    GF(2) rank of all DETECTOR parities in ``circuit``.

    Args:
        circuit: Circuit whose nested detectors are analysed.

    Returns:
        Rank of the detector parity matrix over GF(2).
    """
    detectors = circuit.detectors(include_nested=True)
    parity_matrix = detector_parity_matrix(detectors, len(circuit.measurement_gates))
    if parity_matrix.size == 0:
        return 0
    return int(np.linalg.matrix_rank(galois.GF2(parity_matrix)))


def check_rank_deficit(
    code: StabiliserCode,
    logical_basis: PauliBasis,
    *,
    random_basis_side: bool = True,
) -> int:
    """
    Number of independent check redundancies on a syndrome side.

    Args:
        code: CSS stabiliser code.
        logical_basis: Z- or X-memory basis.
        random_basis_side: If True, use the random-basis syndrome side
            (``H_x`` for Z-memory). If False, use the deterministic side.

    Returns:
        ``num_check_rows - rank(H)`` over GF(2).
    """
    check_matrix = check_matrix_for_memory_basis(
        code, logical_basis, random_basis_side=random_basis_side
    )
    if check_matrix.size == 0:
        return 0
    rank = int(np.linalg.matrix_rank(galois.GF2(check_matrix)))
    return check_matrix.shape[0] - rank


def left_nullspace_basis(H: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Row basis of the left nullspace of ``H`` over GF(2).

    Each returned row ``λ`` satisfies ``λ @ H = 0 (mod 2)``.

    Args:
        H: Parity-check matrix.

    Returns:
        Row basis with shape ``(num_dependencies, num_rows)``.
    """
    if H.size == 0:
        return np.zeros((0, H.shape[0]), dtype=np.uint8)
    H_gf2 = galois.GF2(H)
    rank = int(np.linalg.matrix_rank(H_gf2))
    num_rows = H.shape[0]
    if rank == num_rows:
        return np.zeros((0, num_rows), dtype=np.uint8)
    null_space = H_gf2.T.null_space()
    return np.array(null_space, dtype=np.uint8)


def _is_x_stabiliser(stabiliser: Stabiliser) -> bool:
    operators = stabiliser.operator_repr
    return bool(operators) and all(isinstance(pauli, PauliX) for pauli in operators)


def _is_z_stabiliser(stabiliser: Stabiliser) -> bool:
    operators = stabiliser.operator_repr
    return bool(operators) and all(isinstance(pauli, PauliZ) for pauli in operators)


def check_matrix_for_memory_basis(
    code: StabiliserCode,
    logical_basis: PauliBasis,
    *,
    random_basis_side: bool = True,
) -> NDArray[np.uint8]:
    """
    Parity-check matrix for a syndrome side of a memory experiment.

    Z-memory uses ``H_x`` on the random side and ``H_z`` on the deterministic
    side; X-memory mirrors this.

    Args:
        code: CSS stabiliser code.
        logical_basis: Z- or X-memory basis.
        random_basis_side: If True, return the random-basis check matrix.
            If False, return the deterministic-basis matrix.

    Returns:
        Parity-check matrix for the requested syndrome side.

    Raises:
        NotImplementedError: If ``code`` is not a supported CSS code type.
    """
    if isinstance(code, BivariateBicycleCode):
        if logical_basis == PauliBasis.Z:
            matrix = code.m_Hx if random_basis_side else code.m_Hz
        else:
            matrix = code.m_Hz if random_basis_side else code.m_Hx
        return np.asarray(matrix, dtype=np.uint8)
    if isinstance(code, CSSCode):
        hx, hz = code.parity_check_matrices
        if logical_basis == PauliBasis.Z:
            matrix = hx if random_basis_side else hz
        else:
            matrix = hz if random_basis_side else hx
        return np.asarray(matrix, dtype=np.uint8)
    msg = (
        "Check-redundancy emission is only implemented for CSSCode and "
        f"BivariateBicycleCode, not {type(code).__name__}."
    )
    raise NotImplementedError(msg)


def stabiliser_row_indices_for_memory_basis(
    stabiliser_stage: CSSStage,
    logical_basis: PauliBasis,
    *,
    random_basis_side: bool = True,
) -> list[int]:
    """
    Ordered stabiliser indices that correspond to rows of ``check_matrix``.

    Args:
        stabiliser_stage: Syndrome measurement stage for one round.
        logical_basis: Z- or X-memory basis.
        random_basis_side: If True, map the random-basis stabiliser side.

    Returns:
        Stabiliser indices in ``ordered_stabilisers`` order.
    """
    indices: list[int] = []
    for index, stabiliser in enumerate(stabiliser_stage.ordered_stabilisers):
        on_random_x_side = logical_basis == PauliBasis.Z and _is_x_stabiliser(
            stabiliser
        )
        on_random_z_side = logical_basis == PauliBasis.X and _is_z_stabiliser(
            stabiliser
        )
        on_deterministic_z_side = logical_basis == PauliBasis.Z and _is_z_stabiliser(
            stabiliser
        )
        on_deterministic_x_side = logical_basis == PauliBasis.X and _is_x_stabiliser(
            stabiliser
        )
        if (random_basis_side and (on_random_x_side or on_random_z_side)) or (
            not random_basis_side
            and (on_deterministic_z_side or on_deterministic_x_side)
        ):
            indices.append(index)
    return indices


def first_syndrome_round_measurement_records(
    circuit: Circuit, num_stabilisers: int
) -> list[MeasurementRecord]:
    """
    Measurement records for the first full syndrome round.

    The memory pipeline performs no measurements before the first syndrome
    round, so the first ``num_stabilisers`` measurements in the circuit are
    the first-round ancilla outcomes (in ``ordered_stabilisers`` order).

    Args:
        circuit: Full memory circuit before meta-checks are appended.
        num_stabilisers: Number of stabilisers measured per round.

    Returns:
        First-round syndrome measurement records.
    """
    total_measurements = len(circuit.measurement_gates)
    return [
        MeasurementRecord(index - total_measurements)
        for index in range(num_stabilisers)
    ]


def build_check_redundancy_detectors(
    code: StabiliserCode,
    circuit: Circuit,
    logical_basis: PauliBasis,
    num_rounds: int,
    *,
    random_basis_side: bool = True,
) -> list[Detector]:
    """
    Build meta-check DETECTORs for check-matrix dependencies on a syndrome side.

    Args:
        code: CSS stabiliser code.
        circuit: Full merged memory circuit before meta-checks are appended.
        logical_basis: Z- or X-memory basis.
        num_rounds: Number of syndrome rounds in the circuit.
        random_basis_side: If True, emit dependencies on the random-basis side.
            Set False only for regression testing of the deterministic side.

    Returns:
        One detector per independent check redundancy. Empty if none exist.

    Raises:
        ValueError: If the check matrix row count does not match stabiliser count.
    """
    del num_rounds  # first-round representatives; valid mod existing detectors
    stabiliser_stage = code.measure_stabilisers(num_rounds=1)
    ordered_stabilisers = stabiliser_stage.ordered_stabilisers
    row_to_stabiliser_index = stabiliser_row_indices_for_memory_basis(
        stabiliser_stage, logical_basis, random_basis_side=random_basis_side
    )
    if not row_to_stabiliser_index:
        return []

    check_matrix = check_matrix_for_memory_basis(
        code, logical_basis, random_basis_side=random_basis_side
    )
    if check_matrix.shape[0] != len(row_to_stabiliser_index):
        msg = (
            "Check matrix row count does not match the number of stabilisers on "
            f"the random-basis side ({check_matrix.shape[0]} vs "
            f"{len(row_to_stabiliser_index)})."
        )
        raise ValueError(msg)

    dependency_rows = left_nullspace_basis(check_matrix)
    if dependency_rows.shape[0] == 0:
        return []

    first_round_records = first_syndrome_round_measurement_records(
        circuit, len(ordered_stabilisers)
    )
    stabiliser_to_record = {
        index: first_round_records[index] for index in range(len(ordered_stabilisers))
    }
    detector_coordinates = stabiliser_stage.detector_coordinates

    detectors: list[Detector] = []
    for dependency in dependency_rows:
        participating = [
            row_to_stabiliser_index[row_index]
            for row_index, bit in enumerate(dependency)
            if bit
        ]
        if len(participating) < 2:
            continue
        measurements = [stabiliser_to_record[index] for index in participating]
        detectors.append(
            Detector(
                measurements=measurements,
                coordinate=detector_coordinates[participating[0]],
                tag="check_redundancy",
            )
        )
    return detectors


def append_check_redundancy_detectors(
    circuit: Circuit,
    code: StabiliserCode,
    logical_basis: PauliBasis,
    num_rounds: int,
) -> Circuit:
    """
    Append optional check-redundancy DETECTOR layers to ``circuit``.

    Args:
        circuit: Memory circuit to extend in place.
        code: CSS stabiliser code.
        logical_basis: Z- or X-memory basis.
        num_rounds: Number of syndrome rounds in the circuit.

    Returns:
        The same circuit, with meta-check detectors appended when present.
    """
    meta_detectors = build_check_redundancy_detectors(
        code, circuit, logical_basis, num_rounds
    )
    if meta_detectors:
        circuit.append_layers(meta_detectors)
    return circuit


def check_redundancy_record_sets(
    code: StabiliserCode,
    circuit: Circuit,
    logical_basis: PauliBasis,
    num_rounds: int,
) -> tuple[frozenset[MeasurementRecord], ...]:
    """
    Return meta-check measurement-record sets without mutating a circuit.

    This accessor exposes the same dependency products as
    ``include_check_redundancies=True``, for decoder tooling that consumes
    record sets directly rather than re-parsing Stim output.

    Args:
        code: CSS stabiliser code.
        circuit: Base memory circuit before meta-checks are appended.
        logical_basis: Z- or X-memory basis.
        num_rounds: Number of syndrome rounds in the circuit.

    Returns:
        One frozenset of first-round ``MeasurementRecord`` indices per
        independent check redundancy on the random-basis side.
    """
    return tuple(
        frozenset(detector.measurements)
        for detector in build_check_redundancy_detectors(
            code, circuit, logical_basis, num_rounds
        )
    )

# (c) Copyright Riverlane 2020-2025.
"""
This module stores an implementation of a function to construct a
circuit for a quantum memory experiment with a Calderbank-Shor-Steane
(CSS) quantum error correction code.
"""

from __future__ import annotations

from typing import Optional

import deltakit_explorer
import stim
from deltakit_circuit import Circuit
from deltakit_circuit.gates import PauliBasis
from deltakit_explorer.codes._bivariate_bicycle_code import \
    BivariateBicycleCode
from deltakit_explorer.codes._css._experiment_circuit import experiment_circuit
from deltakit_explorer.codes._css._stabiliser_code import StabiliserCode
from deltakit_explorer.codes._planar_code._planar_code import PlanarCode
from deltakit_explorer.codes._planar_code._rotated_planar_code import \
    RotatedPlanarCode
from deltakit_explorer.codes._planar_code._unrotated_planar_code import \
    UnrotatedPlanarCode
from deltakit_explorer.codes._planar_code._unrotated_toric_code import \
    UnrotatedToricCode
from deltakit_explorer.codes._repetition_code import RepetitionCode


def css_code_memory_circuit(
    css_code: StabiliserCode,
    num_rounds: int,
    logical_basis: PauliBasis,
    client: Optional[deltakit_explorer.Client] = None,
    use_iswap_gates: bool = False,
) -> Circuit:
    """
    Return a noiseless `deltakit.circuit.Circuit` for an X or Z quantum memory experiment
    for CSS codes.

    Parameters
    ----------
    css_code : StabiliserCode
        Stabiliser code.
    num_rounds : int
        Number of rounds to measure the stabilisers for.
    logical_basis : PauliBasis
        PauliBasis instance specifying whether to perform an X or Z quantum memory
        experiment.
    client : Optional[deltakit_explorer.Client]
        The `client` used to perform the calculation.
    use_iswap_gates : bool
        If you generate using cloud, you may request using
        iSWAP native gate set. Default is False.

    Returns
    -------
    Circuit
        Noiseless circuit for quantum memory.

    Raises
    ------
    ValueError
        If num_rounds is not positive.
    ValueError
        If logical_basis is neither PauliBasis.X nor PauliBasis.Z.
    """
    if num_rounds < 1:
        raise ValueError("Invalid num_rounds, it has to be positive.")
    if logical_basis not in [PauliBasis.X, PauliBasis.Z]:
        raise ValueError(
            "Invalid logical_basis, it has to be PauliBasis.X or PauliBasis.Z"
        )
    if use_iswap_gates and client is None:
        raise NotImplementedError(
            "`use_iswap_gates == True` is only supported when a `client` object is provided."
        )
    if client is not None:
        return _cloud_css_code_experiment_circuit(deltakit_explorer.enums.QECExperimentType.QUANTUM_MEMORY,
                                              css_code, num_rounds, logical_basis, client, use_iswap_gates)
    data_qubit_init_stage = (
        css_code.encode_logical_zeroes()
        if logical_basis == PauliBasis.Z
        else css_code.encode_logical_pluses()
    )
    stabiliser_meas_stage = css_code.measure_stabilisers(num_rounds=num_rounds)
    data_qubit_meas_stage = (
        css_code.measure_z_logicals()
        if logical_basis == PauliBasis.Z
        else css_code.measure_x_logicals()
    )
    return experiment_circuit(
        experiment=[data_qubit_init_stage, stabiliser_meas_stage, data_qubit_meas_stage]
    )


def _cloud_css_code_experiment_circuit(
    experiment_type: deltakit_explorer.enums.QECExperimentType,
    css_code: StabiliserCode,
    num_rounds: int,
    logical_basis: PauliBasis,
    client: Optional[deltakit_explorer.Client] = None,
    use_iswap_gates: bool = False,
) -> Circuit:
    """
    Return a noiseless `deltakit.circuit.Circuit` for an X or Z quantum stability experiment
    for CSS codes.

    Parameters
    ----------
    experiment_type: QECExperimentType
        Type of experiment.
    css_code : StabiliserCode
        Stabiliser code.
    num_rounds : int
        Number of rounds to measure the stabilisers for.
    logical_basis : PauliBasis
        PauliBasis instance specifying whether to perform an X or Z quantum memory
        experiment.
    client : Optional[deltakit_explorer.Client]
        The `client` used to perform the calculation.
    use_iswap_gates : bool
        If you generate using cloud, you may request using
        iSWAP native gate set. Default is False.

    Returns
    -------
    Circuit
        Noiseless circuit for a quantum stability experiment.

    Raises
    ------
    NotImplementedError :
        If `client` is not provided.
    ValueError :
        If `css_code` is not of a valid type.
        If `use_iswap_gates` is used without `client`.
    """
    if use_iswap_gates and client is None:
        raise NotImplementedError(
            "`use_iswap_gates == True` is only supported when a `client` object is provided."
        )
    if client is None:
        raise NotImplementedError("A `client` is required to obtain a stability circuit.")

    code_types = {
        RotatedPlanarCode: deltakit_explorer.enums.QECECodeType.ROTATED_PLANAR,
        UnrotatedPlanarCode: deltakit_explorer.enums.QECECodeType.UNROTATED_PLANAR,
        UnrotatedToricCode: deltakit_explorer.enums.QECECodeType.UNROTATED_TORIC,
        RepetitionCode: deltakit_explorer.enums.QECECodeType.REPETITION,
        BivariateBicycleCode: deltakit_explorer.enums.QECECodeType.BIVARIATE_BICYCLE,
    }
    code_type = code_types[css_code.__class__]

    if isinstance(css_code, PlanarCode):
        parameters = deltakit_explorer.types.CircuitParameters.from_sizes(
            (css_code.width, css_code.height)
        )
    elif isinstance(css_code, RepetitionCode):
        parameters = deltakit_explorer.types.CircuitParameters.from_sizes(
            (css_code.distance,)
        )
    elif isinstance(css_code, BivariateBicycleCode):
        parameters=deltakit_explorer.types.CircuitParameters.from_matrix_specification(
            param_l=css_code.param_l,
            param_m=css_code.param_m,
            m_A_powers=css_code.m_A_powers,
            m_B_powers=css_code.m_B_powers,
        )
    else:
        raise ValueError("Unrecognised `css_code` type.")
    basis_gates: Optional[list[str]] = None
    if use_iswap_gates:
        basis_gates = [
            "ISWAP",
            "MZ", "RZ",
            "X", "Y", "Z", "H",
            "S", "S_DAG", "SQRT_X", "SQRT_X_DAG",
        ]

    circuit = client.generate_circuit(
        deltakit_explorer.types.QECExperimentDefinition(
            experiment_type=experiment_type,
            code_type=code_type,
            observable_basis=logical_basis,
            num_rounds=num_rounds,
            parameters=parameters,
            basis_gates=basis_gates,
        )
    )

    return Circuit.from_stim_circuit(stim.Circuit(circuit))


def css_code_stability_circuit(
    css_code: StabiliserCode,
    num_rounds: int,
    logical_basis: PauliBasis,
    client: Optional[deltakit_explorer.Client] = None,
    use_iswap_gates: bool = False,
) -> Circuit:
    """
    Return a noiseless `deltakit.circuit.Circuit` for an X or Z quantum stability experiment
    for CSS codes.

    Parameters
    ----------
    css_code : StabiliserCode
        Stabiliser code.
    num_rounds : int
        Number of rounds to measure the stabilisers for.
    logical_basis : PauliBasis
        PauliBasis instance specifying whether to perform an X or Z quantum memory
        experiment.
    client : Optional[deltakit_explorer.Client]
        The `client` used to perform the calculation.
    use_iswap_gates : bool
        If you generate using cloud, you may request using
        iSWAP native gate set. Default is False.

    Returns
    -------
    Circuit
        Noiseless circuit for a quantum stability experiment.

    Raises
    ------
    NotImplementedError :
        If `client` is not provided.
    ValueError :
        If `css_code` is not of a valid type.
    """
    return _cloud_css_code_experiment_circuit(deltakit_explorer.enums.QECExperimentType.STABILITY,
                                              css_code, num_rounds, logical_basis, client, use_iswap_gates)

# (c) Copyright Riverlane 2020-2025.
"""
This module contains standalone functions related to observables used to generate
stabiliser code circuits.
"""
from collections.abc import Iterable, Mapping, Sequence

from deltakit_circuit import MeasurementRecord, Observable, Qubit
from deltakit_circuit.gates import ONE_QUBIT_MEASUREMENT_GATES, _MeasurementGate


def _construct_observables(
    observable_definitions: Mapping[int, Iterable[_MeasurementGate | Qubit]],
    measurements: Sequence[_MeasurementGate],
) -> list[Observable]:
    """
    Constructs Observable objects from a set of observable definitions and
    a list of lookback measurements. For each observable definition,
    finds the lookback index of the latest measurement matching each element in the
    definition.

    Parameters
    ----------
    observable_definitions : Mapping[int, Iterable[_MeasurementGate | Qubit]]
        Dictionary describing the observable definitions from which to construct
        the Observable objects. Its keys are observable indices and its values
        specify the set of measurements that should make up the corresponding
        observable to construct. These can either be (1) a qubit, in which case any
        measurement involving that qubit will get matched, or (2) a measurement,
        in which case only a measurement in that specific basis will get matched.
    measurements : Sequence[_MeasurementGate]
        The list of all measurements that have previously occurred ordered in the same
        way than they occur in the circuit. These must go far enough in the past such
        that all elements in the observable definitions are covered. Will raise a
        ValueError if no match is found.

    Returns
    -------
    List[Observable]
        Constructed list of observables.
    """
    observables: list[Observable] = []
    num_measurements = len(measurements)
    measurement_qubits = [
        measurement.qubit
        if isinstance(measurement, tuple(ONE_QUBIT_MEASUREMENT_GATES))
        else None
        for measurement in measurements
    ]
    for (
        observable_ind,
        observable_measurements,
    ) in observable_definitions.items():
        measurement_records = []
        for observable_measurement in observable_measurements:
            meas_ind = -num_measurements
            try:
                if isinstance(observable_measurement, Qubit):
                    meas_ind += measurement_qubits.index(observable_measurement)
                else:
                    meas_ind += measurements.index(observable_measurement)
            except ValueError as ve:
                msg = (
                    f"{observable_measurement} has not been measured and "
                    "thus its measurement result cannot be included in a "
                    "logical observable."
                )
                raise ValueError(msg) from ve
            measurement_records.append(MeasurementRecord(meas_ind))

        observables.append(Observable(observable_ind, measurement_records))

    return observables

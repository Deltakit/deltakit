# (c) Copyright Riverlane 2020-2025.
"""
This module contains an implementation of a QPU simulator class.
The class provides native gate compilation, noise addition and
execution time computation features. This is a fully-functional class.
"""
# pylint: disable=too-many-branches
from collections.abc import Iterable
from copy import deepcopy
from typing import NamedTuple

from deltakit_circuit import Circuit, GateLayer, NoiseLayer, Qubit
from deltakit_circuit.gates import I

from deltakit_explorer.qpu._circuits import (
    compile_circuit_to_native_gates,
    remove_identities,
)
from deltakit_explorer.qpu._native_gate_set import (
    ExhaustiveGateSet,
    NativeGateSetAndTimes,
)
from deltakit_explorer.qpu._noise import NoiseParameters, PhenomenologicalNoise


class CircuitSchedule(NamedTuple):
    active_times_list: list[dict[Qubit, float]]
    previous_layer_times: list[float]



class QPU:
    """
    Class capturing properties and functionality of a QPU. The class can
    compile a circuit to the device's native gates and add noise to a
    circuit corresponding to the device's noise model.

    Parameters
    ----------
    qubits : Set[Qubit]
        The device qubits.
    native_gates_and_times : NativeGateSetAndTimes
        The native gates (including measurement and reset) and the time they
        take to apply.
    noise_model : NoiseParameters
        The noise associated with operations on the QPU.
    maximise_parallelism : bool, optional
        Whether to do as many gates in parallel as possible (True) or only to
        perform gates in parallel if they are in the same GateLayer (False).
        By default, True.
    """

    def __init__(
        self,
        qubits: Iterable[Qubit],
        native_gates_and_times: NativeGateSetAndTimes = NativeGateSetAndTimes(),
        noise_model: NoiseParameters = NoiseParameters(),
        maximise_parallelism: bool = True,
    ):
        self.qubits = set(qubits)
        self.native_gates_and_times = native_gates_and_times
        self.noise_model = noise_model
        self._maximise_parallelism = maximise_parallelism

        if isinstance(noise_model, PhenomenologicalNoise):
            self.native_gates_and_times.add_gate(I, 0.0)

        self.full_native_gates_and_times = {
            **self.native_gates_and_times.one_qubit_gates,
            **self.native_gates_and_times.two_qubit_gates,
            **self.native_gates_and_times.reset_gates,
            **self.native_gates_and_times.measurement_gates,
        }

    def _get_circuit_schedule(self, circuit: Circuit) -> CircuitSchedule:
        if not circuit.qubits.issubset(self.qubits):
            qubit_ids = {q.unique_identifier for q in circuit.qubits - self.qubits}
            msg = f"Qubits {qubit_ids} in the circuit are not present on the QPU."
            raise ValueError(msg)

        nonnative_gateset = set()
        active_times_list, previous_layer_times = [], []
        active_times = dict.fromkeys(self.qubits, 0.0)
        previous_layer_time = 0.0

        for layer in circuit.layers:
            if not isinstance(layer, (GateLayer, Circuit)):
                # these numbers don't change with non-gate layers
                previous_layer_times.append(previous_layer_time)
                active_times_list.append(active_times.copy())
                continue
            if isinstance(layer, Circuit):
                # Finish previous layer before entering repeat block
                previous_layer_time = max(active_times.values())
            # Calculate previous layer time based on qubits acted upon in current layer
            elif self._maximise_parallelism:
                previous_layer_time = max(
                    (
                        time
                        for (qubit, time) in active_times.items()
                        if qubit in layer.qubits
                    ),
                    default=0,
                )
            else:
                previous_layer_time = max(
                    active_times.values(),
                    default=0,
                )

            # here we record values for noise computation
            # noise according these values will be added
            # before the layer.
            previous_layer_times.append(previous_layer_time)
            active_times_list.append(active_times.copy())

            # Update qubit active times
            active_times = {
                qubit: max(0.0, time - previous_layer_time)
                for (qubit, time) in active_times.items()
            }
            if isinstance(layer, GateLayer):
                # for a gate layer add active time to qubits
                for gate in layer.gates:
                    for qubit in gate.qubits:
                        # check that the gate is a native gate
                        if type(gate) not in self.full_native_gates_and_times:
                            nonnative_gateset.add(type(gate).__name__)
                        # add execution time to the gate
                        active_times[qubit] += self.full_native_gates_and_times[
                            type(gate)
                        ]
            elif isinstance(layer, Circuit):
                # NB: for a circuit layer add circuit time
                # to all qpu (!) qubits
                circuit_time = self.get_circuit_execution_time(layer)
                for qubit in self.qubits:
                    active_times[qubit] = circuit_time

        # the final layer
        previous_layer_time = max(active_times.values())
        previous_layer_times.append(previous_layer_time)
        active_times_list.append(active_times.copy())

        if nonnative_gateset:
            msg = (
                f"Gate(s) {nonnative_gateset} present in the circuit "
                "do not belong to the native gate set."
            )
            raise ValueError(msg)
        return CircuitSchedule(active_times_list, previous_layer_times)

    def _apply_idle_noise(self, circuit: Circuit) -> Circuit:
        """
        Add idle noise to the input circuit.

        Parameters
        ----------
        circuit : Circuit
            Circuit to which to add idle noise.

        Returns
        -------
        Circuit
            Circuit with idle noise added.
        """

        def _get_idle_noise_channels(
            previous_layer_time: float, active_times: dict[Qubit, float]
        ):
            if (
                previous_layer_time > 0.0
                and (idle_noise := self.noise_model.idle_noise) is not None
            ):
                idle_noise_channels = [
                    idle_noise(qubit, previous_layer_time - active_times[qubit])
                    for qubit in self.qubits
                    if previous_layer_time - active_times[qubit] > 0.0
                ]
            else:
                idle_noise_channels = []
            return idle_noise_channels

        layers_with_noise = []
        iterations = circuit.iterations
        qubit_times, layer_times = self._get_circuit_schedule(circuit)

        for layer, active_times, previous_layer_time in zip(
            circuit.layers, qubit_times, layer_times
        ):
            if not isinstance(layer, (GateLayer, Circuit)):
                layers_with_noise.append(layer)
                continue
            if isinstance(layer, Circuit):
                # Add layers from repeat block
                layer_to_add = self._apply_idle_noise(layer)
            else:
                layer_to_add = GateLayer(layer.gates)

            # Apply idle noise associated with previous layer
            idle_noise_channels = _get_idle_noise_channels(
                previous_layer_time, active_times
            )
            if len(idle_noise_channels) > 0:
                layers_with_noise.append(NoiseLayer(idle_noise_channels))

            # Append current layer and update previous layer
            layers_with_noise.append(layer_to_add)

        previous_layer_time, active_times = layer_times[-1], qubit_times[-1]
        idle_noise_channels = _get_idle_noise_channels(
            previous_layer_time, active_times
        )
        if len(idle_noise_channels) > 0:
            layers_with_noise.append(NoiseLayer(idle_noise_channels))

        return Circuit(layers_with_noise, iterations)

    def get_circuit_execution_time(self, circuit: Circuit) -> float:
        """
        Compute the total circuit execution time (in seconds), given a
        compiled circuit. This provides an estimate of the runtime of the
        circuit on a real QPU.

        Parameters
        ----------
        circuit : Circuit
            Circuit to be assessed.

        Returns
        -------
        float
            Time (in seconds) of total circuit execution.

        Raises
        ------
        ValueError
            If circuit contains non-native gates for the QPU.
            If circuit uses qubits not present on the QPU.
        """
        total_time = 0.0
        schedule = self._get_circuit_schedule(circuit)
        for previous_layer_time, layer in zip(
            schedule.previous_layer_times, circuit.layers
        ):
            if not isinstance(layer, (GateLayer, Circuit)):
                continue
            total_time += previous_layer_time
        # Apply time from final layer
        total_time += schedule.previous_layer_times[-1]
        return total_time * circuit.iterations

    def add_noise_to_circuit(self, circuit: Circuit) -> Circuit:
        """
        Add noise to the supplied circuit based on the QPU's noise model.

        Parameters
        ----------
        circuit : Circuit
            Noiseless circuit to which to add noise.

        Returns
        -------
        Circuit
            Circuit with noise added.
        """
        noisy_circuit = Circuit(deepcopy(circuit.layers))

        # apply measurement flip noise
        noisy_circuit.replace_gates(self.noise_model.measurement_flip)

        noisy_circuit.apply_gate_noise(
            self.noise_model.as_noise_profile_after_gate(), Circuit.LayerAdjacency.AFTER
        )

        # add idle noise
        noisy_circuit = self._apply_idle_noise(noisy_circuit)

        if len(self.noise_model.measurement_noise_before) != 0:
            # add noise which occurs before gates
            noisy_circuit.apply_gate_noise(
                self.noise_model.measurement_noise_before,
                Circuit.LayerAdjacency.BEFORE,
            )

        return noisy_circuit

    def _get_valid_qubit_mapping(self, circuit: Circuit) -> dict[Qubit, Qubit]:
        """
        Get a mapping from qubits in the supplied circuit to qubits in the QPU.
        Currently, this can return only the trivial mapping and thus will raise
        an error if any qubit present in the input circuit is not present in the
        QPU.

        Parameters
        ----------
        circuit : Circuit
            The circuit for which to obtain a qubit mapping.

        Returns
        -------
        Dict[Qubit, Qubit]
            A dictionary which gives a mapping from qubits in the input circuit
            to qubits in the QPU.
        """
        mapping = {}
        for qubit in circuit.qubits:
            if qubit not in self.qubits:
                msg = (
                    "A valid circuit-to-QPU qubit mapping could not be found "
                    f"because qubit {qubit} is not present in the QPU."
                )
                raise ValueError(msg)
            mapping[qubit] = qubit

        return mapping

    def compile_circuit(
        self,
        circuit: Circuit,
        remove_paulis: bool = True,
    ) -> Circuit:
        """
        Compile the supplied circuit so that it uses the QPU's native gates.

        Parameters
        ----------
        circuit : Circuit
            Circuit to be compiled.
        remove_paulis : bool, optional
            Whether to remove Pauli gates from the circuit. This should be set
            to False if, for example, there is a desire to prepare a specific
            initial state which requires Pauli gates. By default, True.

        Returns
        -------
        Circuit
            Compiled circuit.
        """
        return compile_circuit_to_native_gates(
            circuit, self.native_gates_and_times, up_to_paulis=remove_paulis
        )

    def compile_and_add_noise_to_circuit(
        self,
        circuit: Circuit,
        remove_paulis: bool = True,
    ) -> Circuit:
        """
        Compile the supplied circuit so that it uses the QPU's native gates
        and add noise based on the QPU's noise model.

        Parameters
        ----------
        circuit : Circuit
            Circuit to be compiled and to which to add noise.
        remove_paulis : bool, optional
            Whether to remove Pauli gates from the circuit. This should be set
            to False if, for example, there is a desire to prepare a specific
            initial state which requires Pauli gates. By default, True.

        Returns
        -------
        Circuit
            Compiled circuit with noise added.
        """
        return remove_identities(
            self.add_noise_to_circuit(
                self.compile_circuit(circuit, remove_paulis)
            )
        )

    @staticmethod
    def from_circuit_compile_and_add_noise(
        circuit: Circuit,
        native_gates_and_times: NativeGateSetAndTimes = ExhaustiveGateSet(),
        noise_model: NoiseParameters = NoiseParameters(),
        remove_paulis: bool = True,
        maximise_parallelism: bool = True,
    ) -> Circuit:
        """
        Compile the supplied circuit with a given noise model on an unconstrained QPU
        which, by default, supports all gates natively.

        Parameters
        ----------
        circuit : Circuit
            Circuit to be compiled and to which to add noise.
        native_gates_and_times : NativeGateSetAndTimes
            The native gates (including measurement and reset) and the time they
            take to apply.
        noise_model : NoiseParameters
            The noise associated with operations on the QPU.
        remove_paulis : bool, optional
            Whether to remove Pauli gates from the circuit. This should be set
            to False if, for example, there is a desire to prepare a specific
            initial state which requires Pauli gates. By default, True.
        maximise_parallelism : bool, optional
            Whether to do as many gates in parallel as possible (True) or only to
            perform gates in parallel if they are in the same GateLayer (False).
            By default, True.

        Returns
        -------
        Circuit
            Compiled circuit with noise added.
        """
        qpu = QPU(
            qubits=deepcopy(circuit.qubits),
            native_gates_and_times=native_gates_and_times,
            noise_model=noise_model,
            maximise_parallelism=maximise_parallelism,
        )
        return qpu.compile_and_add_noise_to_circuit(
            circuit=circuit,
            remove_paulis=remove_paulis,
        )

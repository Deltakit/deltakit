# (c) Copyright Riverlane 2020-2025.
"""
This module implements a repetition code for quantum memory and stability experiments.
"""

from collections.abc import Callable

from deltakit_circuit import PauliX, PauliZ, Qubit
from deltakit_circuit._basic_maps import BASIS_TO_PAULI
from deltakit_circuit._basic_types import Coord2D, Coord2DDelta
from deltakit_circuit._qubit_identifiers import PauliGate
from deltakit_circuit.gates import PauliBasis

from deltakit_explorer.codes._css._css_code import CSSCode
from deltakit_explorer.codes._stabiliser import Stabiliser


class RepetitionCode(CSSCode):
    """
    Class representing an n-qubit repetition code
    for memory and stability experiments.
    Qubits are laid out in a line alternating between data and ancilla.
    Connectivity is only required between neighbouring pairs of qubits.
    See graph below for a distance 3 code.

    User can specify whether they want to implement a bit-flip code
    (with Z stabilisers) or a phase-flip code (with X stabilisers)

    This class also contains methods that help set up circuits for memory
    and stability experiments.

    .. code-block:: text

        0├    o --- Z/X --- o --- Z/X --- o
         |
         └----┴------┴------┴------┴------┴
              0      1      2      3      4

    Parameters
    ----------
    distance : int
        Minimum weight for a logical operator.
    stabiliser_type : PauliBasis
        Sets the stabiliser type for the repetition code e.g. PauliBasis.Z for a
        bit-flip code.
    use_ancilla_qubits : bool, optional
        Sets whether stabilisers have ancilla qubits for full circuit simulations.
        Setting this to False will compile the output circuit with MPP measurements
        for phenomenological noise. By default, True.
    use_looping_stabiliser : bool, optional
        Sets whether a looping stabiliser (between first and last data qubits)
        is defined. By default, False.
    default_schedule : bool, optional
        Whether to use the default schedule to use for syndrome extraction. If True,
        the syndrome extraction is performed on the qubit to the left of the ancilla
        first, and then on the qubit to the right. If False, it is the opposite way
        round. By default, True.
    odd_data_qubit_coords : bool, optional
        Whether the data qubits should have odd x-coordinates (True) or even (False).
        By default, False.
    """

    def __init__(
        self,
        distance: int,
        stabiliser_type: PauliBasis = PauliBasis.Z,
        use_ancilla_qubits: bool = True,
        use_looping_stabiliser: bool = False,
        default_schedule: bool = True,
        odd_data_qubit_coords: bool = False,
    ):
        self.distance = distance
        self._odd_data_qubit_coords = odd_data_qubit_coords
        self._check_stabiliser_type_is_valid(stabiliser_type)
        self._check_distance_at_least_two(self.distance)

        if default_schedule:
            self._schedule = (Coord2DDelta(-1, 0), Coord2DDelta(1, 0))
        else:
            self._schedule = (Coord2DDelta(1, 0), Coord2DDelta(-1, 0))

        self._data_qubits = self._calculate_data_qubits()
        self._stabiliser_ancilla_qubits = self._calculate_stabiliser_ancilla_qubits(
            use_looping_stabiliser
        )

        self._stabiliser_pauli = BASIS_TO_PAULI[stabiliser_type]
        stabilisers = self._calculate_stabilisers(use_ancilla_qubits)
        x_logical, z_logical = self._calculate_logical_operators()

        super().__init__(
            stabilisers=stabilisers,
            x_logical_operators=x_logical,
            z_logical_operators=z_logical,
            use_ancilla_qubits=use_ancilla_qubits,
        )

    @staticmethod
    def _check_stabiliser_type_is_valid(stabiliser_type: PauliBasis):
        """
        Checks to ensure a valid stabiliser type is inputted into the code.
        """
        if stabiliser_type not in (PauliBasis.X, PauliBasis.Z):
            msg = (
                f"{stabiliser_type} is unsupported, only PauliBasis.X and "
                "PauliBasis.Z are allowed."
            )
            raise ValueError(msg)

    @staticmethod
    def _check_distance_at_least_two(distance: int):
        """
        Check the repetition code has a distance that is at least two (for error
        detection).
        """
        if distance < 2:
            msg = "Code distance must be at least 2."
            raise ValueError(msg)

    def _calculate_data_qubits(self) -> set[Qubit]:
        """
        Calculate data qubits for the code.

        Returns
        -------
        Set[Qubit]
            Data qubits of the code.
        """
        delta = 1 if self._odd_data_qubit_coords else 0
        return {Qubit(Coord2D(2 * x + delta, 0)) for x in range(self.distance)}

    def _calculate_stabiliser_ancilla_qubits(
        self, use_looping_stabiliser
    ) -> set[Qubit]:
        """
        Calculate ancilla qubits used for constructing the code stabilisers.

        Returns
        -------
        Set[Qubit]
            Ancilla qubits used for constructing code stabilisers.
        """
        stabiliser_ancillas = {
            Qubit(Coord2D(2 * x + 1 + self._odd_data_qubit_coords, 0))
            for x in range(self.distance - 1)
        }
        if use_looping_stabiliser:
            x_coord = 0 if self._odd_data_qubit_coords else 2 * self.distance - 1
            stabiliser_ancillas.add(Qubit(Coord2D(x_coord, 0)))
        return stabiliser_ancillas

    def _calculate_stabilisers(self, use_ancilla_qubits) -> list[list[Stabiliser]]:
        """
        Get a full list of stabilisers to measure for the code.
        Stabilisers are pairs of ZZ (for the bit-flip code) or XX (phase-flip code)
        operators acting on data qubits adjacent to each ancilla.
        """

        stabilisers_first: list[Stabiliser] = []
        stabilisers_second: list[Stabiliser] = []

        for ancilla in self._stabiliser_ancilla_qubits:
            paulis = []
            for delta in self._schedule:
                (x, y) = ancilla.unique_identifier + delta
                qubit = Qubit(Coord2D(x % (2 * self.distance), y))
                paulis.append(self._stabiliser_pauli(qubit))
            stabiliser = Stabiliser(paulis=paulis, ancilla_qubit=ancilla)

            if use_ancilla_qubits or ((ancilla.unique_identifier[0] - 1) // 2) % 2 == 0:
                stabilisers_first.append(stabiliser)
            else:
                stabilisers_second.append(stabiliser)

        _sort_fun: Callable[[Stabiliser], int] = lambda stab: (  # noqa: E731
            stab.ancilla_qubit.unique_identifier.x + 1
        ) % (2 * self.distance)

        stabilisers_first.sort(key=_sort_fun)
        stabilisers_second.sort(key=_sort_fun)
        return [stabilisers_first + stabilisers_second]

    def _calculate_logical_operators(
        self,
    ) -> tuple[tuple[set[PauliGate]], tuple[set[PauliGate]]]:
        """Return logical operators for the repetition code."""
        if self._stabiliser_pauli == PauliZ:
            x_logical = {PauliX(qubit) for qubit in self._data_qubits}
            z_logical = {
                PauliZ(Qubit(Coord2D(1, 0)))
                if self._odd_data_qubit_coords
                else PauliZ(Qubit(Coord2D(0, 0)))
            }
        else:
            x_logical = {
                PauliX(Qubit(Coord2D(1, 0)))
                if self._odd_data_qubit_coords
                else PauliX(Qubit(Coord2D(0, 0)))
            }
            z_logical = {PauliZ(qubit) for qubit in self._data_qubits}

        return (x_logical,), (z_logical,)

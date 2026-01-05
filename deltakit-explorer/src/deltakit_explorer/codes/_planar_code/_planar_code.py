# (c) Copyright Riverlane 2020-2025.
"""
This module contains common implementation parts for planar codes.
Other planar code classes derive from PlanarCode.
"""
import itertools
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from deltakit_circuit import PauliX, PauliZ, Qubit
from deltakit_circuit._basic_types import Coord2D, Coord2DDelta
from deltakit_circuit._qubit_identifiers import PauliGate

from deltakit_explorer.codes._css._css_code import CSSCode
from deltakit_explorer.codes._stabiliser import Stabiliser
from deltakit_explorer.enums._basic_enums import DrawingColours


class ScheduleType(Enum):
    """
    Enum for specifying the schedule type for measuring stabilisers for a planar
    code.
    There are five options available:

    - SIMULTANEOUS, meaning all the stabilisers will be measured at the same
      time step.
    - Z_FIRST, meaning all the Z stabilisers will be measured in the first
      time step and the X stabilisers in the second time step.
    - X_FIRST, meaning all the X stabilisers will be measured in the first
      time step and the Z stabilisers in the second time step.
    - Z_ONLY, meaning only the Z stabilisers will be measured.
    - X_ONLY, meaning only the X stabilisers will be measured.

    """

    SIMULTANEOUS = "simultaneous"
    Z_FIRST = "z_first"
    X_FIRST = "x_first"
    Z_ONLY = "z_only"
    X_ONLY = "x_only"


class PlanarCode(CSSCode, ABC):
    """
    Class representing an abstract planar CSS code.

    This class also contains methods that help set up circuits for memory
    experiments.

    Parameters
    ----------
    width: int
        Width of the code patch.
    height: int
        Height of the code patch.
    schedule_type : Optional[ScheduleType]
        The schedule type for measuring stabilisers. By default,
        ScheduleType.SIMULTANEOUS.
    use_ancilla_qubits: bool, optional
        Specifies whether the stabilisers should be defined with (True) or without
        (False) ancilla qubits. By default, True.
    shift : Coord2DDelta, optional
        A vector by which to shift all of the qubit coordinates, by adding it to each
        qubit coordinate.
        By default, the 0 vector: Coord2DDelta(0, 0).
    """

    def __init__(
        self,
        width: int,
        height: int,
        untransformed_x_schedule: tuple[Coord2DDelta, ...],
        untransformed_z_schedule: tuple[Coord2DDelta, ...],
        schedule_type: ScheduleType = ScheduleType.SIMULTANEOUS,
        use_ancilla_qubits: bool = True,
        shift: Coord2DDelta = Coord2DDelta(0, 0)
    ):
        self._shift = shift
        self.linear_tr = np.array([[1, 0], [0, 1]])
        if np.linalg.det(self.linear_tr) == 0.0:
            msg = "Determinant of linear transformation matrix cannot be 0"
            raise ValueError(msg)

        self._dimensions_at_least_2(width, height)

        self.width = width
        self.height = height

        self.schedule_type = schedule_type
        self._x_schedule = tuple(
            Coord2DDelta(*(self.linear_tr @ schedule_delta))
            for schedule_delta in untransformed_x_schedule
        )
        self._z_schedule = tuple(
            Coord2DDelta(*(self.linear_tr @ schedule_delta))
            for schedule_delta in untransformed_z_schedule
        )

        (
            self._untransformed_x_ancilla_qubits,
            self._untransformed_z_ancilla_qubits,
            self._untransformed_data_qubits,
        ) = self._calculate_untransformed_all_qubits()

        self._data_qubits = self._calculate_data_qubits()
        (
            self._x_ancilla_qubits,
            self._z_ancilla_qubits,
        ) = self._calculate_x_and_z_ancilla_qubits()

        stabilisers = self._calculate_stabilisers()
        (
            x_logicals,
            z_logicals,
        ) = self._calculate_logical_operators()

        self._perform_css_checks = getattr(self, "_perform_css_checks", True)

        super().__init__(
            stabilisers=stabilisers,
            x_logical_operators=x_logicals,
            z_logical_operators=z_logicals,
            use_ancilla_qubits=use_ancilla_qubits,
        )

    @staticmethod
    def _dimensions_at_least_2(width: int, height: int):
        """
        Verify both dimensions are greater than or equal to 2.

        Parameters
        ----------
        width: int
            Width of the code patch.
        height: int
            Height of the code patch.

        Raises
        ------
        ValueError
            Raises an error if either width or height is smaller than 2.
        """
        if width < 2 or height < 2:
            msg = "Width and height need to be greater than or equal to 2."
            raise ValueError(msg)

    @abstractmethod
    def _calculate_untransformed_logical_operators(
        self,
    ) -> tuple[tuple[set[PauliGate], ...], tuple[set[PauliGate], ...]]:
        """
        Calculate logical operators before the linear transformation and
        shift vector are applied.

        Returns
        -------
        Tuple[Tuple[Set[PauliGate], ...], Tuple[Set[PauliGate], ...]]
            Logical X operators, each consisting of X operators defining the
            logical and logical Z operators, each consisting of Z operators
            defining the logical.
        """

    def _calculate_logical_operators(
        self,
    ) -> tuple[tuple[set[PauliGate], ...], tuple[set[PauliGate], ...]]:
        """
        Calculate logical operators, applying a linear transformation
        and shift vector to each qubit coordinate.

        Returns
        -------
        Tuple[Tuple[Set[PauliGate], ...], Tuple[Set[PauliGate], ...]]
            Logical X operators, each consisting of X operators defining the
            logical and logical Z operators, each consisting of Z operators
            defining the logical.
        """
        (
            untransformed_x_logicals,
            untransformed_z_logicals,
        ) = self._calculate_untransformed_logical_operators()
        return tuple(
            set(
                {
                    PauliX(self._update_qubit_coordinates(pauli.qubit))
                    for pauli in logical
                }
            )
            for logical in untransformed_x_logicals
        ), tuple(
            set(
                {
                    PauliZ(self._update_qubit_coordinates(pauli.qubit))
                    for pauli in logical
                }
            )
            for logical in untransformed_z_logicals
        )

    @abstractmethod
    def _calculate_untransformed_all_qubits(
        self,
    ) -> tuple[set[Qubit], set[Qubit], set[Qubit]]:
        """
        Calculate X and Z-type stabiliser ancilla qubits and data qubits before
        the linear transformation and shift vector are applied.

        Returns
        -------
        Tuple[Set[Qubit], Set[Qubit], Set[Qubit]]
            Ancilla qubits used to measure X-type stabilisers, ancilla qubits
            used to measure Z-type stabilisers and data qubits.

        """

    def _calculate_x_and_z_ancilla_qubits(self) -> tuple[set[Qubit], set[Qubit]]:
        """
        Calculate X and Z-type stabiliser ancilla qubits, applying a linear transformation
        and shift vector to each qubit coordinate.

        Returns
        -------
        Tuple[Set[Qubit], Set[Qubit]]
            Ancilla qubits used to measure X-type stabilisers and ancilla
            qubits used to measure Z-type stabilisers.
        """
        return {
            self._update_qubit_coordinates(x_anc)
            for x_anc in self._untransformed_x_ancilla_qubits
        }, {
            self._update_qubit_coordinates(z_anc)
            for z_anc in self._untransformed_z_ancilla_qubits
        }

    def _calculate_data_qubits(self) -> set[Qubit]:
        """
        Calculate all data qubits and return a set of them, applying a
        linear transformation and shift vector to all the qubit coordinates.

        Returns
        -------
        Set[Qubit]
            Set of all data qubits.
        """
        return {
            self._update_qubit_coordinates(data_q)
            for data_q in self._untransformed_data_qubits
        }

    def _update_qubit_coordinates(self, qubit: Qubit) -> Qubit:
        """
        Given a qubit, a linear transformation and a shift vector, return a new qubit
        with coordinates acquired by adding the shift vector to the original qubit coordinates
        and then applying the linear transformation.

        Parameters
        ----------
        qubit : Qubit
            The qubit we wish to apply the transformations to.
        """
        qubit_coords_as_np = np.array(
            [qubit.unique_identifier.x, qubit.unique_identifier.y]
        )
        return Qubit(Coord2D(*(self.linear_tr @ qubit_coords_as_np)) + self._shift)

    def _calculate_ancilla_qubits(self) -> set[Qubit]:
        if self._use_ancilla_qubits:
            return self._x_ancilla_qubits.union(self._z_ancilla_qubits)
        return set()

    def _calculate_stabilisers(self) -> tuple[tuple[Stabiliser, ...], ...]:
        """
        Calculate stabilisers defining the code.

        Returns
        -------
        stabilisers: Tuple[Tuple[Stabiliser, ...], ...]
            Stabilisers defining the code. The ith element of stabilisers
            contains a tuple of stabilisers measured during the ith time step.
            Stabilisers measured during the same time step are also sorted from
            ones that are finished measuring first (meaning more Nones at the
            end of the tuple) to ones that are finished measuring last (meaning
            fewer Nones at the end of the tuple).
        """
        x_stabilisers = self._calculate_single_type_stabilisers(
            ancilla_qubits=self._x_ancilla_qubits,
            schedule=self._x_schedule,
            gate=PauliX,
        )
        z_stabilisers = self._calculate_single_type_stabilisers(
            ancilla_qubits=self._z_ancilla_qubits,
            schedule=self._z_schedule,
            gate=PauliZ,
        )

        stabilisers = []
        if self.schedule_type == ScheduleType.SIMULTANEOUS:
            stabilisers.append(self._sort_stabilisers(x_stabilisers + z_stabilisers))
        elif self.schedule_type == ScheduleType.Z_FIRST:
            stabilisers.append(self._sort_stabilisers(z_stabilisers))
            stabilisers.append(self._sort_stabilisers(x_stabilisers))
        elif self.schedule_type == ScheduleType.X_FIRST:
            stabilisers.append(self._sort_stabilisers(x_stabilisers))
            stabilisers.append(self._sort_stabilisers(z_stabilisers))
        elif self.schedule_type == ScheduleType.Z_ONLY:
            stabilisers.append(self._sort_stabilisers(z_stabilisers))
        elif self.schedule_type == ScheduleType.X_ONLY:
            stabilisers.append(self._sort_stabilisers(x_stabilisers))
        else:
            msg = f"Unknown schedule type {self.schedule_type} encountered."
            raise ValueError(msg)

        return tuple(stabilisers)

    def _calculate_single_type_stabilisers(
        self,
        ancilla_qubits: set[Qubit],
        schedule: tuple[Coord2DDelta, ...],
        gate: type[PauliGate],
    ) -> tuple[Stabiliser, ...]:
        """
        Calculate stabilisers, which are products of a single type of gate.

        Parameters
        ----------
        ancilla_qubits : Set[Qubit]
            Ancilla qubits to use for measuring stabilisers.
        schedule : Tuple[Coord2DDelta]
            Defines order in which controlled Paulis are applied between
            ancilla qubits and data qubits in the syndrome extraction circuit.
        gate : Type[PauliGate]
            Stabilisers are products of gates of type gate.

        Returns
        -------
        stabilisers: Tuple[Stabiliser, ...]
            Stabilisers, which are products of the input gate.
        """
        stabilisers = []
        for ancilla in ancilla_qubits:
            paulis = []
            for delta in schedule:
                qubit = Qubit(Coord2D(*ancilla.unique_identifier) + delta)
                if qubit in self._data_qubits:
                    paulis.append(gate(qubit))
                else:
                    paulis.append(None)

            stabilisers.append(Stabiliser(paulis=paulis, ancilla_qubit=ancilla))

        return tuple(stabilisers)

    @staticmethod
    def _sort_stabilisers(
        stabilisers: tuple[Stabiliser, ...],
    ) -> tuple[Stabiliser, ...]:
        """
        Sort a tuple of Stabilisers starting with stabilisers whose measurements
        can be finished earliest.

        Parameters
        ----------
        stabilisers : Tuple[Stabiliser, ...]
            Stabilisers to sort.

        Returns
        -------
        Tuple[Stabiliser, ...]
            Stabilisers sorted starting with stabilisers whose measurements
            can be finished earliest.
        """

        def _count_nones_at_the_end(stabiliser: Stabiliser) -> int:
            """Count the number of Nones at the end of stabiliser.paulis."""

            counter = 0
            for pauli in stabiliser.paulis[::-1]:
                if pauli is None:
                    counter += 1
                else:
                    break

            return counter

        def _contains_x(stabiliser: Stabiliser) -> int:
            """Return 1 if there are any PauliXs in the stabiliser and 0 otherwise."""
            for pauli in stabiliser.paulis:
                if isinstance(pauli, PauliX):
                    return 1
            return 0

        return tuple(
            sorted(
                sorted(stabilisers, key=_contains_x, reverse=True),
                key=_count_nones_at_the_end,
                reverse=True,
            )
        )

    def draw_patch(self, filename: str | None = None, unrotated_code: bool = False) -> None:
        """
        Draw a picture of the planar code, optionally saving it to a .png file.

        Parameters
        ----------
        filename: str, optional
            Path to the file where to save the pictorial representation of the
            planar code stored in this class.
        """
        all_qubit_x_coords = [qubit.unique_identifier.x for qubit in self.qubits]
        all_qubit_y_coords = [qubit.unique_identifier.y for qubit in self.qubits]
        diff_from_max_coord_to_margin_no_ancilla = (
            2 if not unrotated_code or not (self.linear_tr == np.eye(2)).all() else 1
        )
        if self._use_ancilla_qubits:
            min_x, max_x = min(all_qubit_x_coords) - 1, max(all_qubit_x_coords) + 1
            min_y, max_y = min(all_qubit_y_coords) - 1, max(all_qubit_y_coords) + 1
        else:
            min_x, max_x = (
                min(all_qubit_x_coords) - diff_from_max_coord_to_margin_no_ancilla,
                max(all_qubit_x_coords) + diff_from_max_coord_to_margin_no_ancilla,
            )
            min_y, max_y = (
                min(all_qubit_y_coords) - diff_from_max_coord_to_margin_no_ancilla,
                max(all_qubit_y_coords) + diff_from_max_coord_to_margin_no_ancilla,
            )
        x_lim = (min_x, max_x)
        y_lim = (min_y, max_y)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        stabilisers = tuple(itertools.chain.from_iterable(self._stabilisers))
        stabilisers = self._sort_stabilisers(stabilisers)

        # Draw stabiliser plaquettes
        for stabiliser in stabilisers:
            data_qubit_x_coords = [
                pauli.qubit.unique_identifier[0]
                for pauli in stabiliser.paulis
                if pauli is not None
            ]
            data_qubit_y_coords = [
                pauli.qubit.unique_identifier[1]
                for pauli in stabiliser.paulis
                if pauli is not None
            ]

            paulis = [pauli for pauli in stabiliser.paulis if pauli is not None]

            if len(paulis) == 2:
                ancilla_coord = stabiliser.ancilla_qubit.unique_identifier
                data_qubit_x_coords.append(ancilla_coord[0])
                data_qubit_y_coords.append(ancilla_coord[1])
            elif len(paulis) == 4:
                data_qubit_x_coords[2], data_qubit_x_coords[3] = (
                    data_qubit_x_coords[3],
                    data_qubit_x_coords[2],
                )
                data_qubit_y_coords[2], data_qubit_y_coords[3] = (
                    data_qubit_y_coords[3],
                    data_qubit_y_coords[2],
                )

            if isinstance(paulis[0], PauliX):
                ax.fill(
                    data_qubit_x_coords,
                    data_qubit_y_coords,
                    color=DrawingColours.X_COLOUR.value,
                    alpha=1,
                )
            else:
                ax.fill(
                    data_qubit_x_coords,
                    data_qubit_y_coords,
                    color=DrawingColours.Z_COLOUR.value,
                    alpha=1,
                )

        # Draw data qubits
        for qubit in self._data_qubits:
            cc = plt.Circle(
                qubit.unique_identifier,
                0.2,
                color=DrawingColours.DATA_QUBIT_COLOUR.value,
                alpha=1,
            )
            ax.set_aspect(1)
            ax.add_artist(cc)

        if self._use_ancilla_qubits:
            # Draw X stabiliser ancilla qubits
            for qubit in self._x_ancilla_qubits:
                cc = plt.Circle(
                    qubit.unique_identifier,
                    0.2,
                    color=DrawingColours.ANCILLA_QUBIT_COLOUR.value,
                    alpha=1,
                )
                ax.set_aspect(1)
                ax.add_artist(cc)

            # Draw Z stabiliser ancilla qubits
            for qubit in self._z_ancilla_qubits:
                cc = plt.Circle(
                    qubit.unique_identifier,
                    0.2,
                    color=DrawingColours.ANCILLA_QUBIT_COLOUR.value,
                    alpha=1,
                )
                ax.set_aspect(1)
                ax.add_artist(cc)

        # Save the file
        if filename:
            output_directory = Path(filename)
            if not output_directory.exists():
                output_directory.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(filename)
            plt.close(fig)

    @cached_property
    def x0(self) -> int:
        """
        Across the data qubits of the patch, get the smallest x-coordinate value.

        Returns
        -------
        int
            The smallest x-coordinate value across all the data qubits in the patch.
        """
        return min(
            self.data_qubits, key=lambda qubit: qubit.unique_identifier.x
        ).unique_identifier.x

    @cached_property
    def x1(self) -> int:
        """
        Across the data qubits of the patch, get the largest x-coordinate value.

        Returns
        -------
        int
            The largest x-coordinate value across all the data qubits in the patch.
        """
        return max(
            self.data_qubits, key=lambda qubit: qubit.unique_identifier.x
        ).unique_identifier.x

    @cached_property
    def y0(self) -> int:
        """
        Across the data qubits of the patch, get the smallest y-coordinate value.

        Returns
        -------
        int
            The smallest y-coordinate value across all the data qubits in the patch.
        """
        return min(
            self.data_qubits, key=lambda qubit: qubit.unique_identifier.y
        ).unique_identifier.y

    @cached_property
    def y1(self) -> int:
        """
        Across the data qubits of the patch, get the largest y-coordinate value.

        Returns
        -------
        int
            The largest y-coordinate value across all the data qubits in the patch.
        """
        return max(
            self.data_qubits, key=lambda qubit: qubit.unique_identifier.y
        ).unique_identifier.y

    def get_shifted_logicals(
        self, shift_z: bool, delta: Coord2DDelta
    ) -> tuple[frozenset[PauliGate], ...]:
        """
        Given a boolean specifying whether to modify X or Z logicals, and a Coord2DDelta,
        modify the relevant logical(s) by adding the Coord2DDelta to the coordinates of
        each qubit making up the logical.
        A check is performed to ensure the shifted logical still lies on the data qubits
        of the patch. If this is not the case, a ValueError is thrown.

        Parameters
        ----------
        shift_z : bool
            Boolean specifying whether to shift the Z-logicals (True) or X-logicals
            (False).
        delta : Coord2DDelta
            The delta to add to each qubit coordinate.

        Returns
        -------
        tuple[frozenset[PauliGate], ...]
            A tuple of frozensets of PauliX or PauliZ operators, depending on the
            value of shift_z being False or True respectively, which will be the
            logicals of the code modified by adding the delta vector to each qubit
            coordinate.
        """
        if shift_z:
            logicals = self.z_logical_operators
            pauli_op = PauliZ
        else:
            logicals = self.x_logical_operators
            pauli_op = PauliX
        new_logicals = []
        for old_logical in logicals:
            new_logical = frozenset(
                pauli_op(Qubit(pauli.qubit.unique_identifier + delta))
                for pauli in old_logical
            )
            # Verify that the new logical lies within the patch, if we are not the
            # UnrotatedToricCode class. Do it with __name__ to avoid circular import
            if type(self).__name__ != "UnrotatedToricCode":
                for pauli in new_logical:
                    if pauli.qubit not in self.data_qubits:
                        msg = f"Pauli operator on qubit {pauli.qubit} is not contained in the patch data qubits after shifting by {tuple(delta)}"
                        raise ValueError(msg)
            else:
                warnings.warn(
                    "Logical shifting validation is disabled for the UnrotatedToricCode; new logicals may be invalid"
                )
            new_logicals.append(new_logical)
        return tuple(new_logicals)

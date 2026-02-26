# (c) Copyright Riverlane 2020-2025.
"""
This module stores an implementation of the unrotated planar code.
"""

import itertools

from deltakit_circuit import PauliX, PauliZ, Qubit
from deltakit_circuit._basic_types import Coord2D, Coord2DDelta

from deltakit_explorer.codes._planar_code._planar_code import PlanarCode, ScheduleType
from deltakit_explorer.codes._schedules import (
    ScheduleOrder,
    UnrotatedPlanarCodeSchedules,
    get_x_and_z_schedules,
)


class UnrotatedPlanarCode(PlanarCode):
    """
    Class representing the unrotated planar code. By default, Z logical is vertical
    (i.e. Z boundaries are horizontal) and X logical is horizontal (i.e. X boundaries
    are vertical). See the graph below for the default 3x3 unrotated code.

    This class also contains methods that help set up circuits for memory
    experiments.

    .. code-block:: text

        4├    ○ ---- Z ---- ○ ---- Z ---- ○
         │    |      |      |      |      |
        3├    X ---- ○ ---- X ---- ○ ---- X
         │    |      |      |      |      |
        2├    ○ ---- Z ---- ○ ---- Z ---- ○
         │    |      |      |      |      |
        1├    X ---- ○ ---- X ---- ○ ---- X
         │    |      |      |      |      |
        0├    ○ ---- Z ---- ○ ---- Z ---- ○
         │
         └----┴------┴------┴------┴------┴
              0      1      2      3      4

    Parameters
    ----------
    width : int
        The maximum number of data qubits in a single row.
    height : int
        The maximum number of data qubits in a single column.
    schedule_type : ScheduleType, optional
        The type of syndrome extraction schedule to use. By default,
        ScheduleType.SIMULTANEOUS.
    schedule_order : ScheduleOrder, optional
        The order of the syndrome extraction rounds, i.e, in which order to perform
        the consecutive entangling gates to measure the stabilier of a plaquette.
        By default, ScheduleOrder.STANDARD.
    use_ancilla_qubits : bool, optional
        Whether or not to use ancilla qubits for the stabilisers. By default, True.
    horizontal_stabiliser_with_top_left_is_z : bool, optional
        Whether the leftmost stabiliser in the top row is a Z stabiliser. By default,
        True.
    shift : Coord2DDelta, optional
        A vector by which to shift all of the qubit coordinates, by adding it to each
        qubit coordinate.
        By default, the 0 vector: Coord2DDelta(0, 0).
    """

    def __init__(
        self,
        width: int,
        height: int,
        schedule_type: ScheduleType = ScheduleType.SIMULTANEOUS,
        schedule_order: ScheduleOrder = ScheduleOrder.STANDARD,
        use_ancilla_qubits: bool = True,
        horizontal_stabiliser_with_top_left_is_z: bool = True,
        shift: Coord2DDelta = Coord2DDelta(0, 0),
    ):
        # coordinates of the bottom left vertex of the rectangle
        (self._x0, self._y0) = (0, 0)
        x_schedule, z_schedule = get_x_and_z_schedules(UnrotatedPlanarCodeSchedules, schedule_order)

        self._perform_css_checks = False

        self._horizontal_stabiliser_with_top_left_is_z = (
            horizontal_stabiliser_with_top_left_is_z
        )
        if self._horizontal_stabiliser_with_top_left_is_z:
            self.x_distance = width
            self.z_distance = height
        else:
            self.x_distance = height
            self.z_distance = width

        super().__init__(
            width=width,
            height=height,
            untransformed_x_schedule=x_schedule,
            untransformed_z_schedule=z_schedule,
            schedule_type=schedule_type,
            use_ancilla_qubits=use_ancilla_qubits,
            shift=shift,
        )

    def _calculate_untransformed_all_qubits(
        self,
    ) -> tuple[set[Qubit], set[Qubit], set[Qubit]]:
        # horizontal and vertical total length of the rectangle in which the data and ancillary qubits are.
        x_length = 2 * self.width - 1
        y_length = 2 * self.height - 1

        all_qubits = {
            Qubit(
                Coord2D(
                    x,
                    y,
                )
            )
            for x, y in itertools.product(
                range(self._x0, self._x0 + x_length),
                range(self._y0, self._y0 + y_length),
            )
        }

        # calculate all at the same time, so we only loop over all_qubits once
        x_ancilla, z_ancilla, data_qubits = set(), set(), set()

        # Calculate the parity of the x-coordinate of the left-most ancilla in the top row. This is of Z type in the default horizontal_stabiliser_with_top_left_is_z == True case, hence a reference.
        top_left_anc_x_mod_2 = (self._x0 + 1) % 2

        for qubit in all_qubits:
            x_coord_mod_2 = qubit.unique_identifier.x % 2
            y_coord_mod_2 = qubit.unique_identifier.y % 2
            # Data qubits are at coordinates of same parity
            if x_coord_mod_2 == y_coord_mod_2:
                data_qubits.add(qubit)
            # The rest of the qubits are all ancillas. If top row stabilisers are of Z type, then Z-ancillas's x-coord matches the reference, otherwise it doesn't.
            elif (
                self._horizontal_stabiliser_with_top_left_is_z
                and (x_coord_mod_2 == top_left_anc_x_mod_2)
            ) or (
                (not self._horizontal_stabiliser_with_top_left_is_z)
                and (x_coord_mod_2 != top_left_anc_x_mod_2)
            ):
                z_ancilla.add(qubit)
            else:
                x_ancilla.add(qubit)

        return x_ancilla, z_ancilla, data_qubits

    def _calculate_untransformed_logical_operators(
        self,
    ) -> tuple[tuple[set[PauliX], ...], tuple[set[PauliZ], ...]]:
        horizontal_logical_qubits = {
            Qubit(
                Coord2D(
                    x,
                    2 * self.height - 2,
                )
            )
            for x in range(0, 2 * self.width - 1, 2)
        }
        vertical_logical_qubits = {
            Qubit(Coord2D(0, y)) for y in range(0, 2 * self.height - 1, 2)
        }
        if self._horizontal_stabiliser_with_top_left_is_z:
            x_logical = {PauliX(qubit) for qubit in horizontal_logical_qubits}
            z_logical = {PauliZ(qubit) for qubit in vertical_logical_qubits}
        else:
            x_logical = {PauliX(qubit) for qubit in vertical_logical_qubits}
            z_logical = {PauliZ(qubit) for qubit in horizontal_logical_qubits}

        return (x_logical,), (z_logical,)

    def draw_patch(self, filename: str | None = None, unrotated_code: bool = True) -> None:
        return super().draw_patch(filename, unrotated_code)

# (c) Copyright Riverlane 2020-2025.
"""
This module stores an implementation of the rotated planar code.
"""
# pylint: disable=too-many-branches, too-many-boolean-expressions
import itertools

from deltakit_circuit import PauliX, PauliZ, Qubit
from deltakit_circuit._basic_types import Coord2D, Coord2DDelta

from deltakit_explorer.codes._planar_code._planar_code import PlanarCode, ScheduleType
from deltakit_explorer.codes._schedules._rotated_planar_code_schedules import (
    RotatedPlanarCodeSchedules,
)
from deltakit_explorer.codes._schedules._schedule_order import (
    ScheduleOrder,
    get_x_and_z_schedules,
)


class RotatedPlanarCode(PlanarCode):
    r"""
    Class representing the default rotated planar code. For the default rotated
    planar code, Z logical is vertical (i.e. Z boundaries are horizontal) and X
    logical is horizontal (i.e. X boundaries are vertical). The default patch
    has a weight 2 horizontal Z-stabiliser on the left top corner. See the graph
    below showing a default 3x3 rotated code.

    This class also contains methods that help set up circuits for memory
    experiments.

    By default, the rotated planar code will have the following layout:

    .. code-block:: text

       6├              /   Z   \
        │            /           \
       5├           ○ ----------- ○ ----------- ○ \
        │           │             │             │   \
       4├           │      X      │      Z      │     X
        │           │             │             │   /
       3├         / ○ ----------- ○ ----------- ○ /
        │       /   │             │             │
       2├    X      │      Z      │      X      │
        │       \   │             │             │
       1├         \ ○ ----------- ○ ----------- ○
        │                          \           /
       0├                            \   Z   /
        │
        └----┴------┴------┴------┴------┴------┴------┴
             0      1      2      3      4      5      6

    Parameters
    ----------
    width : int
        Has to be >= 2. The patch is built on a rectangular grid of points
        of size width x height.
    height : int
        Has to be >= 2. The patch is built on a rectangular grid of points
        of size width x height.
    schedule_type : ScheduleType, optional
        The scheduling type to measure stabilisers.
        By default, ScheduleType.SIMULTANEOUS.
    schedule_order : ScheduleOrder, optional
        Enum specifying the order of the schedule, from a number of available
        options. By default, ScheduleOrder.HORIZONTALLY_REFLECTED, which,
        in case `top_bumps_are_z == True`, will perform an N-shape schedule on
        the X stabilisers and a Z-shape schedule on the Z stabilisers, both
        starting in the top left and finishing in the bottom right. In the other
        case `top_bumps_are_z == False`, it will do N-shape on Z and Z-shape on
        X stabilisers.
    use_ancilla_qubits : bool
        Whether or not to use ancilla qubits for the stabilisers. By default, True.
    shift : Coord2DDelta, optional
        A vector by which to shift all of the qubit coordinates, by adding it to each
        qubit coordinate.
        By default, the 0 vector: Coord2DDelta(0, 0).
    horizontal_bump_with_top_left: bool, optional.
        A bump is a weight-2 stabiliser.
        True if and only if the patch is stabilised by a weight-2
        plaquette supported on top_left_vertex and
        top_left_vertex + Coord2DDelta(2, 0). By default, True.
    top_bumps_are_z: bool, optional.
        Specifies whether the weight-2 stabilisers on the top side of
        the patch are of Z type or not. This fixes the direction of the logical operators;
        if True then logical Z is vertical and logical X is horizontal.
        By default, True.
    """

    def __init__(
        self,
        width: int,
        height: int,
        schedule_type: ScheduleType = ScheduleType.SIMULTANEOUS,
        schedule_order: ScheduleOrder = ScheduleOrder.HORIZONTALLY_REFLECTED,
        use_ancilla_qubits: bool = True,
        shift: Coord2DDelta = Coord2DDelta(0, 0),
        horizontal_bump_with_top_left: bool = True,
        top_bumps_are_z: bool = True,
    ):
        self._horizontal_bump_with_top_left = horizontal_bump_with_top_left
        self._top_bumps_are_z = top_bumps_are_z
        self._x_type_has_N_shape = (  # pylint: disable=invalid-name
            top_bumps_are_z
        )
        x_schedule, z_schedule = get_x_and_z_schedules(
            RotatedPlanarCodeSchedules,
            schedule_order,
            x_type_has_N_shape=self._x_type_has_N_shape,
        )

        self._perform_css_checks = False

        if self._top_bumps_are_z:
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

    def _calculate_default_patch_data_qubits(self) -> set[Qubit]:
        """Calculates the untransformed data qubits for a default rotated
        code patch.

        Returns
        -------
        Set[Qubit]
            Data qubits for a standard rotated code patch.
        """
        return {
            Qubit(Coord2D(x, y))
            for (x, y) in itertools.product(
                range(1, 2 * self.width, 2),
                range(1, 2 * self.height, 2),
            )
        }

    def _calculate_default_patch_ancilla_qubits(
        self,
    ) -> tuple[set[Qubit], set[Qubit]]:
        """Calculates the untransformed ancilla qubits for a default rotated
        code patch.

        Returns
        -------
        Tuple[Set[Qubit], Set[Qubit]]
            All ancilla qubits of opposite type from the top weight-2
            stabilizers for a standard rotated code patch. All ancilla qubits
            of same type as the top weight-2 stabilizers for a standard rotated
            code patch.
        """
        # Collect all qubits where an X-ancilla could be placed in the default
        # `top_bumps_are_z == True` case.
        possible_x_ancilla_qubits = {
            Qubit(Coord2D(x, y))
            for x, y in itertools.product(
                range(0, 2 * self.width + 1, 2),
                range(2, 2 * self.height - 1, 2),
            )
        }
        default_x_ancilla_qubits = {
            anc
            for anc in possible_x_ancilla_qubits
            if (anc.unique_identifier.x + anc.unique_identifier.y - 2 * self.height) % 4
            == 0
        }

        # Collect all qubits where a Z-ancilla could be placed in the default
        # `top_bumps_are_z == True` case.
        possible_z_ancilla_qubits = {
            Qubit(Coord2D(x, y))
            for x, y in itertools.product(
                range(2, 2 * self.width - 1, 2),
                range(0, 2 * self.height + 1, 2),
            )
        }
        default_z_ancilla_qubits = {
            anc
            for anc in possible_z_ancilla_qubits
            if (anc.unique_identifier.x + anc.unique_identifier.y - 2 * self.height) % 4
            == 2
        }

        if self._horizontal_bump_with_top_left:
            x_ancilla_qubits = default_x_ancilla_qubits
            z_ancilla_qubits = default_z_ancilla_qubits
        else:
            x_ancilla_qubits = possible_x_ancilla_qubits.difference(
                default_x_ancilla_qubits
            )
            z_ancilla_qubits = possible_z_ancilla_qubits.difference(
                default_z_ancilla_qubits
            )

        return x_ancilla_qubits, z_ancilla_qubits

    def _calculate_untransformed_all_qubits(
        self,
    ) -> tuple[set[Qubit], set[Qubit], set[Qubit]]:
        data_qubits = self._calculate_default_patch_data_qubits()
        (
            x_ancilla_qubits,
            z_ancilla_qubits,
        ) = self._calculate_default_patch_ancilla_qubits()

        return (
            (x_ancilla_qubits, z_ancilla_qubits, data_qubits)
            if self._top_bumps_are_z
            else (z_ancilla_qubits, x_ancilla_qubits, data_qubits)
        )

    def _calculate_untransformed_logical_operators(
        self,
    ) -> tuple[tuple[set[PauliX], ...], tuple[set[PauliZ], ...]]:
        horiz_op_pauli, vert_op_pauli = (
            (PauliX, PauliZ) if self._top_bumps_are_z else (PauliZ, PauliX)
        )
        vert_op = {
            vert_op_pauli(Qubit(Coord2D(1, y))) for y in range(1, 2 * self.height, 2)
        }
        horiz_op = {
            horiz_op_pauli(Qubit(Coord2D(x, 2 * self.height - 1)))
            for x in range(1, 2 * self.width, 2)
        }

        return (
            ((horiz_op,), (vert_op,))
            if self._top_bumps_are_z
            else ((vert_op,), (horiz_op,))
        )

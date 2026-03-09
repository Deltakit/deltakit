# (c) Copyright Riverlane 2020-2025.
"""
This module includes implementation of the unrotated toric code. The code
represents two logical qubits.
"""

import itertools
from pathlib import Path
from typing import Literal

from deltakit_circuit import PauliX, PauliZ, Qubit
from deltakit_circuit._basic_types import Coord2D, Coord2DDelta
from deltakit_circuit._qubit_identifiers import PauliGate
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing_extensions import override

from deltakit_explorer.codes._planar_code._planar_code import PlanarCode, ScheduleType
from deltakit_explorer.codes._schedules._schedule_order import (
    ScheduleOrder,
    get_x_and_z_schedules,
)
from deltakit_explorer.codes._schedules._unrotated_planar_code_schedules import (
    UnrotatedPlanarCodeSchedules,
)
from deltakit_explorer.codes._stabiliser import Stabiliser


class UnrotatedToricCode(PlanarCode):
    """
    Class representing the default unrotated toric code. The code has a periodic
    boundary and encodes two logical qubits. Logical operators are formed of loops around
    the torus. See the graph below showing a default 3x3 toric code. Open edges are
    connected to qubits on opposite sides. The distance of the code is given by the
    min(horizontal_distance, vertical_distance).

    This class also contains methods that help set up circuits for memory
    experiments.


    .. code-block:: text

        6├
        |    |      |      |      |      |      |
        5├    X ---- o ---- X ---- o ---- X ---- o ----
        |    |      |      |      |      |      |
        4├    ○ ---- Z ---- ○ ---- Z ---- ○ ---- Z ----
        │    |      |      |      |      |      |
        3├    X ---- ○ ---- X ---- ○ ---- X ---- o ----
        │    |      |      |      |      |      |
        2├    ○ ---- Z ---- ○ ---- Z ---- ○ ---- Z ----
        │    |      |      |      |      |      |
        1├    X ---- ○ ---- X ---- ○ ---- X ---- o ----
        │    |      |      |      |      |      |
        0├    ○ ---- Z ---- ○ ---- Z ---- ○ ---- Z ----
        │
        └----┴------┴------┴------┴------┴------┴------┴
            0      1      2      3      4      5      6

    Parameters
    ----------
    horizontal_distance: int
        The width of the toric code patch, which defines the distance for the
        horizontal logical operators X1 and Z2.
    vertical_distance: int
        The height of the toric code patch, which defines the distance for the
        vertical logical operators X2 and Z1.
    """

    def __init__(
        self,
        horizontal_distance: int,
        vertical_distance: int,
        schedule_type: ScheduleType = ScheduleType.SIMULTANEOUS,
        schedule_order: ScheduleOrder = ScheduleOrder.STANDARD,
        use_ancilla_qubits: bool = True,
    ):
        x_schedule, z_schedule = get_x_and_z_schedules(
            UnrotatedPlanarCodeSchedules, schedule_order
        )

        self._perform_css_checks = False

        super().__init__(
            width=horizontal_distance,
            height=vertical_distance,
            untransformed_x_schedule=x_schedule,
            untransformed_z_schedule=z_schedule,
            schedule_type=schedule_type,
            use_ancilla_qubits=use_ancilla_qubits,
        )

    def _calculate_untransformed_all_qubits(
        self,
    ) -> tuple[set[Qubit], set[Qubit], set[Qubit]]:
        data_qubits = {
            Qubit(Coord2D(x, y))
            for (x, y) in itertools.product(
                range(0, 2 * self.width - 1, 2),
                range(0, 2 * self.height - 1, 2),
            )
        }.union(
            {
                Qubit(Coord2D(x, y))
                for (x, y) in itertools.product(
                    range(1, 2 * self.width, 2),
                    range(1, 2 * self.height, 2),
                )
            }
        )

        x_ancilla_qubits = {
            Qubit(Coord2D(x, y))
            for (x, y) in itertools.product(
                range(0, 2 * self.width - 1, 2),
                range(1, 2 * self.height, 2),
            )
        }

        z_ancilla_qubits = {
            Qubit(Coord2D(x, y))
            for (x, y) in itertools.product(
                range(1, 2 * self.width, 2),
                range(0, 2 * self.height - 1, 2),
            )
        }

        return x_ancilla_qubits, z_ancilla_qubits, data_qubits

    def _calculate_single_type_stabilisers(
        self,
        ancilla_qubits: set[Qubit],
        schedule: tuple[Coord2DDelta, ...],
        gate: type[PauliGate],
    ) -> tuple[Stabiliser, ...]:
        stabilisers = []
        for ancilla in ancilla_qubits:
            paulis = []
            for delta in schedule:
                coordinate = Coord2D(*ancilla.unique_identifier) + delta
                x_mod = coordinate[0] % (2 * self.width)
                z_mod = coordinate[1] % (2 * self.height)
                qubit = Qubit(Coord2D(x_mod, z_mod))
                paulis.append(gate(qubit))

            stabilisers.append(Stabiliser(paulis=paulis, ancilla_qubit=ancilla))

        return tuple(stabilisers)

    def _calculate_untransformed_logical_operators(
        self,
    ) -> tuple[tuple[set[PauliGate], ...], tuple[set[PauliGate], ...]]:
        x_logicals = (
            {PauliX(Qubit(Coord2D(x, 0))) for x in range(0, 2 * self.width - 1, 2)},
            {PauliX(Qubit(Coord2D(1, y))) for y in range(1, 2 * self.height, 2)},
        )
        z_logicals = (
            {PauliZ(Qubit(Coord2D(0, y))) for y in range(0, 2 * self.height - 1, 2)},
            {PauliZ(Qubit(Coord2D(x, 1))) for x in range(1, 2 * self.width, 2)},
        )

        return (x_logicals, z_logicals)

    @override
    def draw_patch(
        self,
        filename: Path | None = None,
        unrotated_code: bool = True,
        backend: Literal["matplotlib", "svg", "pgf"] = "matplotlib",
    ) -> tuple[Figure, Axes]:
        return super().draw_patch(filename, unrotated_code, backend)

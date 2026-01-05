# (c) Copyright Riverlane 2020-2025.
"""
This module includes implementation of the unrotated toric code. The code
represents two logical qubits.
"""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy.typing as npt
from deltakit_circuit import PauliX, PauliZ, Qubit
from deltakit_circuit._basic_types import Coord2D, Coord2DDelta
from deltakit_circuit._qubit_identifiers import PauliGate
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy import arctan2, argsort, array, int8

from deltakit_explorer.codes._planar_code._planar_code import PlanarCode, ScheduleType
from deltakit_explorer.codes._schedules._schedule_order import (
    ScheduleOrder,
    get_x_and_z_schedules,
)
from deltakit_explorer.codes._schedules._unrotated_planar_code_schedules import (
    UnrotatedPlanarCodeSchedules,
)
from deltakit_explorer.codes._stabiliser import Stabiliser
from deltakit_explorer.enums._basic_enums import DrawingColours


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

    def draw_patch(self, filename: str | None = None, unrotated_code: bool = False) -> None:  # noqa: ARG002
        fig, ax = plt.subplots(nrows=1, ncols=1)
        all_qubit_x_coords = [qubit.unique_identifier.x for qubit in self.qubits]
        all_qubit_y_coords = [qubit.unique_identifier.y for qubit in self.qubits]
        min_x, max_x = min(all_qubit_x_coords) - 2, max(all_qubit_x_coords) + 2
        min_y, max_y = min(all_qubit_y_coords) - 2, max(all_qubit_y_coords) + 2
        ax.set_xlim((min_x, max_x))
        ax.set_ylim((min_y, max_y))

        stabilisers: tuple[Stabiliser, ...] = tuple(
            itertools.chain.from_iterable(self._stabilisers)
        )

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

            # Wrap boundary stabilisers on the X axis
            if 0 in data_qubit_x_coords and 2 * self.width - 1 in data_qubit_x_coords:
                if data_qubit_x_coords.count(0) > data_qubit_x_coords.count(
                    2 * self.width - 1
                ):
                    data_qubit_x_coords = [
                        -1 if x == 2 * self.width - 1 else x
                        for x in data_qubit_x_coords
                    ]
                else:
                    data_qubit_x_coords = [
                        2 * self.width if x == 0 else x for x in data_qubit_x_coords
                    ]

            # Wrap boundary stabilisers on the Y axis
            if 0 in data_qubit_y_coords and 2 * self.height - 1 in data_qubit_y_coords:
                if data_qubit_y_coords.count(0) > data_qubit_y_coords.count(
                    2 * self.height - 1
                ):
                    data_qubit_y_coords = [
                        -1 if y == 2 * self.height - 1 else y
                        for y in data_qubit_y_coords
                    ]
                else:
                    data_qubit_y_coords = [
                        2 * self.height if y == 0 else y for y in data_qubit_y_coords
                    ]

            # Re order data qubit coords for polygon drawing
            ordered_data_qubit_y_coords: npt.NDArray[int8] = array(data_qubit_y_coords)
            ordered_data_qubit_x_coords: npt.NDArray[int8] = array(data_qubit_x_coords)

            order = argsort(
                arctan2(
                    ordered_data_qubit_y_coords - ordered_data_qubit_y_coords.mean(),
                    ordered_data_qubit_x_coords - ordered_data_qubit_x_coords.mean(),
                )
            )

            paulis = [pauli for pauli in stabiliser.paulis if pauli is not None]
            ax.fill(
                ordered_data_qubit_x_coords[order],
                ordered_data_qubit_y_coords[order],
                color=(
                    DrawingColours.X_COLOUR.value
                    if isinstance(paulis[0], PauliX)
                    else DrawingColours.Z_COLOUR.value
                ),
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

        # Draw stabiliser ancilla qubits
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

            # Draw X stabiliser ancilla qubits
            for qubit in self._z_ancilla_qubits:
                cc = plt.Circle(
                    qubit.unique_identifier,
                    0.2,
                    color=DrawingColours.ANCILLA_QUBIT_COLOUR.value,
                    alpha=1,
                )
                ax.set_aspect(1)
                ax.add_artist(cc)

        # Create a legend
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Data Qubit",
                markerfacecolor=DrawingColours.DATA_QUBIT_COLOUR.value,
                markersize=15,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Ancilla Qubit",
                markerfacecolor=DrawingColours.ANCILLA_QUBIT_COLOUR.value,
                markersize=15,
            ),
            Patch(facecolor=DrawingColours.X_COLOUR.value, label="X Stabiliser"),
            Patch(facecolor=DrawingColours.Z_COLOUR.value, label="Z Stabiliser"),
        ]
        legend = ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=2,
        )

        if filename is not None:
            # Save the file
            output_directory = Path(filename)
            if not output_directory.exists():
                output_directory.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(filename, bbox_extra_artists=(legend,), bbox_inches="tight")
            plt.close(fig)

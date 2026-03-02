from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from deltakit_circuit import PauliX
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

if TYPE_CHECKING:
    from deltakit_explorer.codes._planar_code import PlanarCode
from deltakit_explorer.enums._basic_enums import DrawingColours


def _draw_code(
    code: PlanarCode,
    filename: Path | None = None,
    unrotated_code: bool = False,
    backend: Literal["matplotlib", "svg", "pgf"] = "matplotlib",
) -> tuple[Figure, Axes]:
    """Function for drawing the Planar codes.

    Args:
        code: Planar Error Correction Code.
        filename: The filename where to store the figure.
        unrotated_code: Boolean indicator is the code unrotated or not.
        backend: The backend or file format that will be used for storing the plot.

    Returns:
        figure and axes of plot.
    """
    if backend == "pgf":
        mpl.use("pgf")
        mpl.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
            }
        )
    fig, ax = plt.subplots(nrows=1, ncols=1)

    all_qubit_x_coords = [qubit.unique_identifier.x for qubit in code.qubits]
    all_qubit_y_coords = [qubit.unique_identifier.y for qubit in code.qubits]

    _use_ancilla_qubits = (
        code._use_ancilla_qubits if (code._use_ancilla_qubits is not None) else False
    )
    if _use_ancilla_qubits and not unrotated_code:
        min_x, max_x = min(all_qubit_x_coords) - 1, max(all_qubit_x_coords) + 1
        min_y, max_y = min(all_qubit_y_coords) - 1, max(all_qubit_y_coords) + 1
    else:
        diff_from_max_coord_to_margin_no_ancilla = (
            2 if not unrotated_code or not (code.linear_tr == np.eye(2)).all() else 1
        )
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
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    stabilisers = tuple(chain.from_iterable(code._stabilisers))
    stabilisers = code._sort_stabilisers(stabilisers)

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
        if 0 in data_qubit_x_coords and 2 * code.width - 1 in data_qubit_x_coords:
            if data_qubit_x_coords.count(0) > data_qubit_x_coords.count(
                2 * code.width - 1
            ):
                data_qubit_x_coords = [
                    -1 if x == 2 * code.width - 1 else x for x in data_qubit_x_coords
                ]
            else:
                data_qubit_x_coords = [
                    2 * code.width if x == 0 else x for x in data_qubit_x_coords
                ]

        # Wrap boundary stabilisers on the Y axis
        if 0 in data_qubit_y_coords and 2 * code.height - 1 in data_qubit_y_coords:
            if data_qubit_y_coords.count(0) > data_qubit_y_coords.count(
                2 * code.height - 1
            ):
                data_qubit_y_coords = [
                    -1 if y == 2 * code.height - 1 else y for y in data_qubit_y_coords
                ]
            else:
                data_qubit_y_coords = [
                    2 * code.height if y == 0 else y for y in data_qubit_y_coords
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
    for qubit in code._data_qubits:
        cc = plt.Circle(
            qubit.unique_identifier,
            0.2,
            color=DrawingColours.DATA_QUBIT_COLOUR.value,
            alpha=1,
        )
        ax.set_aspect(1)
        ax.add_artist(cc)

    if code._use_ancilla_qubits:
        # Draw X stabiliser ancilla qubits
        for qubit in code._x_ancilla_qubits:
            cc = plt.Circle(
                qubit.unique_identifier,
                0.2,
                color=DrawingColours.ANCILLA_QUBIT_COLOUR.value,
                alpha=1,
            )
            ax.set_aspect(1)
            ax.add_artist(cc)

        # Draw Z stabiliser ancilla qubits
        for qubit in code._z_ancilla_qubits:
            cc = plt.Circle(
                qubit.unique_identifier,
                0.2,
                color=DrawingColours.ANCILLA_QUBIT_COLOUR.value,
                alpha=1,
            )
            ax.set_aspect(1)
            ax.add_artist(cc)

    # Create the legend;
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
        output_directory = Path(filename).parent
        if not output_directory.exists():
            output_directory.parent.mkdir(parents=True)
        if backend == "matplotlib":
            fig.savefig(filename, bbox_extra_artists=(legend,), bbox_inches="tight")
        elif backend == "pgf":
            fig.savefig(
                filename.parent / f"{filename.name}.pgf", bbox_extra_artists=(legend,), bbox_inches="tight"
            )
        elif backend == "svg":
            fig.savefig(
                filename.parent / f"{filename.name}.svg", bbox_extra_artists=(legend,), bbox_inches="tight"
            )
    return fig, ax

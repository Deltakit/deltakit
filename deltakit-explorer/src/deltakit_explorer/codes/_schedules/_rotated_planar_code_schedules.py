# (c) Copyright Riverlane 2020-2025.
"""
Collection of dictionaries for different types of schedules for Rotated Planar
codes.
"""

from dataclasses import dataclass
from typing import ClassVar

from deltakit_circuit._basic_types import Coord2DDelta


@dataclass
class RotatedPlanarCodeSchedules:
    """
    Dataclass to capture default syndrome extraction schedules for the rotated
    planar code.
    """

    # In default_N_Z_dict,
    #     "N" means the N-shaped schedule, i.e. for that plaquette we use the
    #     following schedule:
    #             ---------------
    #             |             |
    #             |   ||    |   |
    #             |   | |   ˅   |
    #             |   ˅  |  |   |
    #             |   |   | |   |
    #             |             |
    #             ---------------
    #     and "Z" means the Z-shaped schedule, i.e. for that plaquette we use the
    #     following schedule:
    #             --------------
    #             |            |
    #             |    --<--   |
    #             |    |       |
    #             |     |      |
    #             |      |     |
    #             |  --<--     |
    #             |            |
    #             --------------
    default_N_Z_dict: ClassVar[
        dict[str, tuple[Coord2DDelta, Coord2DDelta, Coord2DDelta, Coord2DDelta]]
    ] = {
        "N": (
            Coord2DDelta(1, 1),
            Coord2DDelta(1, -1),
            Coord2DDelta(-1, 1),
            Coord2DDelta(-1, -1),
        ),
        "Z": (
            Coord2DDelta(1, 1),
            Coord2DDelta(-1, 1),
            Coord2DDelta(1, -1),
            Coord2DDelta(-1, -1),
        ),
    }

    # Horizontally reflected version of default_N_Z_dict, i.e. with
    #         ---------------
    #         |             |
    #         |   |    ||   |
    #         |   |   | ˅   |
    #         |   ˅  |  |   |
    #         |   | |   |   |
    #         |             |
    #         ---------------
    # and
    #         --------------
    #         |            |
    #         |    -->--   |
    #         |       |    |
    #         |     |      |
    #         |   |        |
    #         |  -->--     |
    #         |            |
    #         --------------
    horizontally_reflected_N_Z_dict: ClassVar[
        dict[str, tuple[Coord2DDelta, Coord2DDelta, Coord2DDelta, Coord2DDelta]]
    ] = {
        "N": (
            Coord2DDelta(-1, 1),
            Coord2DDelta(-1, -1),
            Coord2DDelta(1, 1),
            Coord2DDelta(1, -1),
        ),
        "Z": (
            Coord2DDelta(-1, 1),
            Coord2DDelta(1, 1),
            Coord2DDelta(-1, -1),
            Coord2DDelta(1, -1),
        ),
    }

    # Vertically reflected version of default_N_Z_dict, i.e. with
    #         ---------------
    #         |             |
    #         |   |   | |   |
    #         |   |  |  |   |
    #         |   ^ |   ^   |
    #         |   ||    |   |
    #         |             |
    #         ---------------
    # and
    #         --------------
    #         |            |
    #         |    --<--   |
    #         |       |    |
    #         |     |      |
    #         |   |        |
    #         |  --<--     |
    #         |            |
    #         --------------
    vertically_reflected_N_Z_dict: ClassVar[
        dict[str, tuple[Coord2DDelta, Coord2DDelta, Coord2DDelta, Coord2DDelta]]
    ] = {
        n_or_z: schedule[::-1]
        for n_or_z, schedule in horizontally_reflected_N_Z_dict.items()
    }

    # Doubly reflected version of default_N_Z_dict, i.e. with
    #         ---------------
    #         |             |
    #         |   ||    |   |
    #         |   | |   ^   |
    #         |   ^  |  |   |
    #         |   |   | |   |
    #         |             |
    #         ---------------
    # and
    #         --------------
    #         |            |
    #         |    -->--   |
    #         |    |       |
    #         |     |      |
    #         |      |     |
    #         |  -->--     |
    #         |            |
    #         --------------
    doubly_reflected_N_Z_dict: ClassVar[
        dict[str, tuple[Coord2DDelta, Coord2DDelta, Coord2DDelta, Coord2DDelta]]
    ] = {n_or_z: schedule[::-1] for n_or_z, schedule in default_N_Z_dict.items()}

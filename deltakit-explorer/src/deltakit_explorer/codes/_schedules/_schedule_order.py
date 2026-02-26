# (c) Copyright Riverlane 2020-2025.
"""
Enum to encapsulate the different types of schedules available for
planar codes, and a function to return the X and Z schedules for a
planar code.
"""

from enum import Enum

from deltakit_circuit._basic_types import Coord2DDelta

from deltakit_explorer.codes._schedules._rotated_planar_code_schedules import (
    RotatedPlanarCodeSchedules,
)
from deltakit_explorer.codes._schedules._unrotated_planar_code_schedules import (
    UnrotatedPlanarCodeSchedules,
)

ScheduleClass = type[RotatedPlanarCodeSchedules] | type[UnrotatedPlanarCodeSchedules]


class ScheduleOrder(Enum):
    """
    Enum for specifying the schedule type for measuring stabilisers for a planar
    code.
    There are four options available:
        - ScheduleOrder.STANDARD, using, in general, the default_N_Z_dict
          schedules
        - ScheduleOrder.HORIZONTALLY_REFLECTED using, in general, the
          horizontally_reflected_N_Z_dict schedules
        - ScheduleOrder.VERTICALLY_REFLECTED using, in general, the
          vertically_reflected_N_Z_dict schedules
        - ScheduleOrder.DOUBLY_REFLECTED using, in general, the
          doubly_reflected_N_Z_dict schedules
    """

    STANDARD = "standard"
    HORIZONTALLY_REFLECTED = "horizontally_reflected"
    VERTICALLY_REFLECTED = "vertically_reflected"
    DOUBLY_REFLECTED = "doubly_reflected"


def get_x_and_z_schedules(
    schedule_class: ScheduleClass,
    schedule_order: ScheduleOrder,
    x_type_has_N_shape: bool = True,
) -> tuple[
    tuple[Coord2DDelta, Coord2DDelta, Coord2DDelta, Coord2DDelta],
    tuple[Coord2DDelta, Coord2DDelta, Coord2DDelta, Coord2DDelta],
]:
    """
    Function to retrieve the relevant schedule dictionary and assign the correct
    schedule to the X or Z plaquette. Returns a tuple containing first the X and
    then the Z schedules.

    Params
    ------
    schedule_class : ScheduleClass
        The class containing the schedules for the relevant code.
    schedule_order : ScheduleOrder
        The schedule order for the code.
    x_type_has_N_shape : bool, optional
        Bool specifying whether the X-type plaquette follows the N-type schedule,
        or if it should follow the Z-type schedule. The Z-type plaquette takes the
        other. By default, True.
    """
    if schedule_order == ScheduleOrder.STANDARD:
        dict_to_use = schedule_class.default_N_Z_dict
    elif schedule_order == ScheduleOrder.HORIZONTALLY_REFLECTED:
        dict_to_use = schedule_class.horizontally_reflected_N_Z_dict
    elif schedule_order == ScheduleOrder.VERTICALLY_REFLECTED:
        dict_to_use = schedule_class.vertically_reflected_N_Z_dict
    elif schedule_order == ScheduleOrder.DOUBLY_REFLECTED:
        dict_to_use = schedule_class.doubly_reflected_N_Z_dict
    else:
        msg = f"Did not recognise ScheduleOrder {schedule_order}"
        raise ValueError(msg)

    if x_type_has_N_shape:
        return dict_to_use["N"], dict_to_use["Z"]
    return dict_to_use["Z"], dict_to_use["N"]

# (c) Copyright Riverlane 2020-2025.
"""Module explores how the error suppression factor (lambda)
builds a connection between QuOps and code distance.

References
----------
- https://doi.org/10.48550/arXiv.2408.13687
"""

from typing import Callable


def _equal_or_less_descending_bisection(
    func: Callable,
    target: float,
    minimum: int,
    maximum: int,
) -> int | None:
    """Binary search in the given boundaries for a descending function.
    func(...) is expected to be a monotonic descending function.
    The function searches for smallest D from [min, max],
    with func(D) <= target.

    Parameters:
        func (Callable): a descending function.
        target (float): a value to search for.
        minimum (int): minimum tested value of D.
        maximum (int): maximum tested value of D.

    Returns:
        int | None:
            a value, which safisfies the search, or None.
    """
    while minimum < maximum:
        mid = (minimum + maximum) // 2
        if func(mid) <= target:
            maximum = mid
        else:
            minimum = mid + 1
    if func(maximum) <= target:
        return maximum
    else:
        return None

def _calculate_lep(p0: float, lambda_: float, distance: int, num_rounds: int) -> float:
    """Returns the probability of observing a logical error on a code of fixed
    distance after a number of rounds.

    It uses the formula in Section VI.B of Supplementary Information in
    https://doi.org/10.48550/arXiv.2408.13687 which is the sum of the probabilities
    of all ways of there being an odd number of errors in fixed number of rounds.
    """
    lep_per_round = p0 * lambda_ ** (-(distance + 1) / 2)
    # At `lep_per_round` << 1 this is be approximated as `lep_per_round * num_rounds`
    return 0.5 * (1 - (1 - 2 * lep_per_round) ** num_rounds)

def predict_quops_at_distance(p0: float, lambda_: float, distance: int) -> float:
    """Returns the number of QuOps, given distance. This
    uses the definition that the number of QuOps achievable is 1 / pL, where pL is
    the probability of a logical error occurring in a dxdxd block.

    Parameters
    ----------
    p0 (float):
        SPAM error.
    lambda_ (float):
        Error suppression factor.
    distance (int):
        The distance at which to calculate the number of QuOps.
    """
    return 1. / _calculate_lep(p0, lambda_, distance, distance)

def predict_distance_for_quops(p0: float, lambda_: float, num_quops: float) -> int:
    """Returns the nearest distance that achieves the desired number of QuOps to one
    decimal place. Uses the definition that the number of QuOps achievable at a
    particular distance is 1 / pL, where pL is the probability of a logical error
    occurring during a dxdxd memory experiment.

    Parameters
    ----------
    p0 (float):
        SPAM error.
    lambda_ (float):
        Error suppression factor.
    num_quops (int):
        Number of desired QuOps, must be a positive integer greater than 2.

    Raises
    ------
    ValueError
        - if solution is not found
    """

    if num_quops < 2:
        raise ValueError("Number of QuOps should be at least 2")

    required_lep = 1. / num_quops
    distance = _equal_or_less_descending_bisection(
        lambda x: _calculate_lep(p0, lambda_, x, x),
        required_lep,
        minimum=2,
        maximum=999,
    )
    if distance is not None:
        return distance
    else:
        raise ValueError("Could not find a solution for LEP(distance) < 1 / QoOps")

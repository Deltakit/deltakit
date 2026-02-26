# (c) Copyright Riverlane 2020-2025.

import math
from collections import Counter, defaultdict
from collections.abc import Collection, Generator, Iterable, Iterator, Sequence
from itertools import chain, combinations, islice, product
from warnings import warn

import numpy as np
import numpy.typing as npt
import pathos
from deltakit_core.decoding_graphs import NXDecodingGraph

PijData = dict[frozenset[int], float]


def _compute_combinations_of_detectors(
        samples: Collection,
        max_degree: int = 2,
) -> Counter:
    """Calculate combinations of detectors for a
    given set of samples of detectors.
    Parameters
    ----------
    samples : Collection[Collection[int]]
        Sample data from which to calculate combinations
    max_degree : int
        Maximum degree of (hyper)edges/combinations to consider.
    Returns
    -------
    Counter
        Counter object containing the combinations of detectors
        observed in the samples.
    """
    return Counter(
        chain.from_iterable(
            combinations(s, weight)
            for s in samples for weight in range(1, max_degree + 1)
        )
    )


def _generate_expectation_data_multiprocess(
    samples: Collection[Iterable[int]],
    only_even: bool = False,
    only_odd: bool = False,
    max_degree: int = 2,
    num_processes: int = -1,
) -> PijData:
    if num_processes <= 0:
        if num_processes == -1:
            num_processes = pathos.helpers.mp.cpu_count()
        else:
            warn("num_processes 0 or < -1, falling back to using single process.",
                 UserWarning,
                 stacklevel=3)
            num_processes = 1

    if only_even and not only_odd:
        samples = list(islice(samples, 0, len(samples), 2))
    if only_odd and not only_even:
        samples = list(islice(samples, 1, len(samples), 2))
    elif only_even and only_odd:
        warn("Both only_odd and only_even are True. Selecting whole batch.",
             stacklevel=3)

    # if num_processes > samples, floor to 1
    step = max(len(samples)//num_processes, 1)+1
    split_samples = [list(islice(samples, n, n+step))
                     for n in range(0, len(samples), step)]
    with pathos.helpers.mp.Pool(num_processes) as p:
        results = p.starmap(_compute_combinations_of_detectors, [
            (samples, max_degree) for samples in split_samples])
    data: Counter = Counter()
    for c in results:
        data.update(c)

    # divide count values of each Xi, Xj etc by number of
    # samples to get expectation value
    expectation_values: PijData = {frozenset(key):
                                   data[key] / len(samples)
                                   for key in data}

    return expectation_values


def _generate_expectation_data_singleprocess(
    samples: Generator | Iterator,
    only_even: bool = False,
    only_odd: bool = False,
    max_degree: int = 2,
    num_processes: int = 1,
):
    if num_processes != 1:
        msg = (
            "Generators may only be passed in single-threaded mode. Please set "
            "num_processes=1"
        )
        raise NotImplementedError(msg)

    if only_even and not only_odd:
        samples = islice(samples, 0, None, 2)
    if only_odd and not only_even:
        samples = islice(samples, 1, None, 2)
    elif only_even and only_odd:
        warn("Both only_odd and only_even are True. Selecting whole batch.",
             stacklevel=3)

    counter: Counter[Iterable[int]] = Counter()
    num_samples = 0
    for s in samples:
        counter.update(chain.from_iterable(combinations(s, weight)
                                           for weight in range(1, max_degree + 1)))
        num_samples += 1

    # divide count values of each Xi, Xj etc by number of
    # samples to get expectation value
    expectation_values: PijData = {frozenset(key):
                                   counter[key] / num_samples
                                   for key in counter}
    return expectation_values


def generate_expectation_data(
    samples: Iterable[Iterable[int]],
    only_even: bool = False,
    only_odd: bool = False,
    max_degree: int = 2,
    num_processes: int = 1,
) -> PijData:
    """Generates the <Xi>, <Xj>, <Xij> etc expectation values
    for deriving Pij probabilities.

    You may provide a collection or generator as input for samples.
    In the case of a generator, you must set num_processes=1 as
    multiprocessing is not supported for samples of generator type.

    Parameters
    ----------
    samples : Iterable[Iterable[int]]
        Iterable of iterable of integers signifying which
        detectors lit up in that shot.
    only_even : bool
        Boolean to specify whether only the even-indexed samples
        should be used to generate expectation data.
    only_odd : bool
        Boolean to specify whether only the odd-indexed samples
        should be used to generate expectation data.
    max_degree : int
        Maximum degree hyperedges to be considered. For instance,
        if a sample contains 4 detector events, and max_degree is
        set to 4, the program will consider a degree 4 hyperedge
        consisting of the 4 present detectors.
        Default value is 2.
    num_processes: int
        Number of processes across which to distribute the data
        to parallelise the task.
        Default value is 1. Value of -1 will use all available
        cores.

    Returns
    -------
    PijData
        The keys are the pairs of detectors that were observed
        firing together. The frequency of a single detector
        firing is stored as a tuple of the single detector
        index, e.g, detector 1 is (1,).
        The float value represents the expectation value of
        the given pair of detectors firing.
    """
    if isinstance(samples, Generator):
        return _generate_expectation_data_singleprocess(samples,
                                                        only_even,
                                                        only_odd,
                                                        max_degree,
                                                        num_processes)
    if isinstance(samples, Collection):
        return _generate_expectation_data_multiprocess(samples,
                                                       only_even,
                                                       only_odd,
                                                       max_degree,
                                                       num_processes)
    msg = (
        f"Unrecognised argument type: {type(samples)}, argument must be a Generator or "
        "Collection."
    )
    raise NotImplementedError(msg)


def _calculate_g_value(p: float, q: float) -> float:
    """Given a pair of probabilities, calculate g as in equation
    (S15) in https://arxiv.org/pdf/2102.06132.pdf"""
    return p + q - (2 * p * q)


def _calculate_p_i_sigma(nodes: list[float]) -> list[float]:
    """Recursively collapse list of Pij values into PiSigma value,
    equation (S14) in https://arxiv.org/pdf/2102.06132.pdf

    Parameters
    ----------
    nodes : List[float]
        This begins as the initial list of probabilities for the connected edges
        of the node we wish to adjust the probability for. Is progressively
        reduced down to a list of length 1 as the function recursively
        calls itself.

    Returns
    -------
    List[float]
        Will return a List of length 1 containing the final p_i_sigma value.
    """
    if len(nodes) == 1:
        return nodes
    if len(nodes) == 2:
        return [_calculate_g_value(nodes[0], nodes[1])]
    return _calculate_p_i_sigma([nodes[0], *_calculate_p_i_sigma(nodes[1:])])


def _calculate_edge_prob_with_higher_degrees(
    edge: frozenset[int],
    pij_data: PijData,
    min_prob: float,
) -> float:
    """Having computed the 'local' Pij.. we must then update
    this probability, taking into account higher degree edges
    that may cause the given edge to light up. This is only
    needed for edges that are of degree less than max_degree.
    Equations (S14), (S15) and (S16) from page 20
    in Google RepCode Paper:
    https://arxiv.org/pdf/2102.06132.pdf
    By "local", we mean the Pij formula when considering only
    nodes "within" the edge, vs pij formula being adjusted for
    higher edges that can trigger the current edge.

    Parameters
    ----------
    edge : FrozenSet[int]
        Edge for which to recalculate the Pij.. but taking into account
        higher degree edges.
    pij_data : PijData
        Pij probabilities calculated from expectation data.

    Returns
    -------
    float
        Probability for the given edge having taken into account
        any higher degree edges that could light up the detectors
        in the given edge.
    """
    # collect all edges our current edge is a subset of -
    # thereby all edges that can turn on our edge
    connected_edges = [pij_data[k]
                       for k in pij_data if edge < k]
    if len(connected_edges) == 0:
        return pij_data.get(edge, 0.0)
    pi_sigma = _calculate_p_i_sigma(connected_edges)[0]
    if math.isclose(pi_sigma, 0.5, rel_tol=0.05):
        warn("pi_sigma is converging to 0.5. Please consider trimming"
             " the number of edges used in calculations.", stacklevel=3)
        return max(min_prob, 0.0)
    return (pij_data.get(edge, 0.0) - pi_sigma) / (1 - (2 * pi_sigma))


def _n1_formula(
    exp_values: PijData,
    keys: Sequence[frozenset[int]]
) -> float:
    """Formula for parameter n1 in hyperedge calculations.

    Parameters
    ----------
    exp_values : PijData
        Expectation values for edges.

    keys : Sequence[FrozenSet[int]]
        Sequence of single nodes to use in calculation.

    Returns
    -------
    float
        The value of the calculation.
    """
    return math.prod((1 - (2 * exp_values.get(n, 0.0))) for n in keys)


def _d1_formula(
    exp_values: PijData,
    keys: Sequence[frozenset[int]]
) -> float:
    """Formula for parameter d1 in hyperedge calculations.

    Parameters
    ----------
    exp_values : PijData
        Expectation values for edges.

    keys : Sequence[FrozenSet[int]]
        Sequence of single nodes to use in calculation.

    Returns
    -------
    float
        The value of the calculation.
    """
    tuple_keys = [tuple(x) for x in keys]
    return math.prod(1 - (2 * exp_values.get(frozenset(n[:1]), 0.0))
                     - (2 * exp_values.get(frozenset(n[1:]), 0.0))
                     + (4 * exp_values.get(frozenset(n), 0.0))
                     for n in tuple_keys)


def _n2_formula(exp_values: PijData,
                key: frozenset[int],
                keys: Sequence[frozenset[int]]
                ) -> float:
    """Formula for parameter n2 in hyperedge calculations.

    Parameters
    ----------
    exp_values : PijData
        Expectation values for edges.

    key : FrozenSet[int]
        Single weight 3 edge to be used in calculation.

    keys : Sequence[FrozenSet[int]]
        Sequence of pairs of nodes to use in calculation, a combination
        of the nodes contained within key. E.g, collections.combinations(key, 2)

    Returns
    -------
    float
        The value of the calculation.
    """
    return 1 - (2 * math.fsum(exp_values.get(frozenset((n,)), 0.0) for n in key)) \
        + (4 * math.fsum(exp_values.get(x, 0.0) for x in keys)) \
        - (8 * exp_values.get(key, 0.0))


def create_correlation_matrix(
    pij_data: PijData,
    graph: NXDecodingGraph,
    plot_boundary_edges: bool = False,
) -> tuple[npt.NDArray[np.float64], dict[tuple[float, ...], list[int]]]:
    """Generate a correlation matrix for a given Pij matrix.
    Will plot qubit labels as major ticks, minor ticks
    within the major ticks are rounds (time). Matrix will be
    symmetric.

    Parameters
    ----------
    pij_data : PijData
        Pij values calculated from experimental data
    graph : NXDecodingGraph
        Accompanying graph for Pij data.
    plot_boundary_edges : bool
        Boolean to specify whether the boundary edges/nodes
        should be plotted on the heatmap. These will be the
        entries along the direct diagonal from bottom-left
        to top-right.
        Default value is False.

    Returns
    -------
    Tuple[List[List[float]], Dict[Tuple[int, ...], List[int]]]
        Returns the correlation matrix and coordinate mapping.
        E.g, if all nodes are [0,1,2,3] then 2 & 3 correspond to
        the parity checks 0 & 1 in the future, respectively. E.g,
        {0: [0, 2], 1: [1, 3]}
    """

    # create a mapping - each major tick is an ancilla qubit,
    # and each minor tick inside a major corresponds to a round
    major_minor_mapping: defaultdict[tuple[float, ...],
                                     list[int]] = defaultdict(list)
    for k, v in graph.detector_records.items():
        if k not in graph.boundaries:
            major_minor_mapping[v.spatial_coord].append(k)
    for _, detectors in major_minor_mapping.items():
        detectors.sort()
    if len(major_minor_mapping) == 0:
        return np.array([]), {}

    # if there are not a consistent number of rounds for each qubit,
    # throw an error, since we can only plot NxN matrices
    if not all(len(x) == len(next(iter(major_minor_mapping.values())))
                for x in major_minor_mapping.values()):
        msg = "Inconsistent qubit time mapping"
        raise ValueError(msg)

    # helper function for converting coordinates into their Pij value
    # as per the Pij matrix
    def coord_to_pij(x):
        xf = frozenset((x[0],)) if x[0] == x[1] else frozenset(x)
        if xf in pij_data:
            return pij_data[xf]
        return 0.0

    # going through each minor tick, convert that tick's coordinates
    # into the corresponding Pij value, and place in corresponding row
    num_major_squares = len(major_minor_mapping.keys())
    num_minor_squares = len(next(iter(major_minor_mapping.values())))
    correlation_matrix = np.array([np.array([0.0
                                             for _ in range(num_major_squares
                                                            * num_minor_squares)])
                                   for _ in range(num_major_squares
                                                  * num_minor_squares)])
    for i, coord in enumerate(product(sorted(major_minor_mapping.keys()), repeat=2)):
        for j, qubit in enumerate(product(major_minor_mapping[coord[0]],
                                          major_minor_mapping[coord[1]])):
            row_coord = ((i // num_major_squares) * num_minor_squares) + \
                (j // num_minor_squares)
            col_coord = ((i % num_major_squares) * num_minor_squares) + \
                (j % num_minor_squares)
            correlation_matrix[row_coord][col_coord] = coord_to_pij(qubit)

    if not plot_boundary_edges:
        for i in range(num_major_squares * num_minor_squares):
            correlation_matrix[i][i] = 0.0

    return correlation_matrix, major_minor_mapping

# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
from functools import partial
from itertools import chain, combinations, product, repeat, starmap
from math import comb, floor, prod
from typing import (Any, Callable, Dict, FrozenSet, Iterable, Iterator, List,
                    Optional, Sequence, Tuple)

import numpy as np
from deltakit_core.decoding_graphs import (EdgeT, HyperMultiGraph,
                                           OrderedDecodingEdges)
from deltakit_decode.noise_sources._generic_noise_sources import (
    CombinedSequences, MonteCarloNoise, SequentialNoise, offset_seed)
from typing_extensions import TypeAlias


def _empty_generator() -> Iterator[OrderedDecodingEdges]:
    for _ in []:
        yield OrderedDecodingEdges()


EdgeFilterT: TypeAlias = Callable[[HyperMultiGraph], Sequence[EdgeT]]


class NoNoiseMatchingSequence(SequentialNoise[HyperMultiGraph,
                                              OrderedDecodingEdges]):
    """A noise model that outputs a single empty list of decoding edges.
    """

    def error_generator(
        self, code_data: HyperMultiGraph, seed: Optional[int] = None
    ) -> Iterator[OrderedDecodingEdges]:
        yield OrderedDecodingEdges()

    def split_error_generator(
        self,
        code_data: HyperMultiGraph,
        num_splits: int,
        seed: Optional[int] = None
    ) -> Tuple[Tuple[Iterator[OrderedDecodingEdges], int], ...]:
        return ((self.error_generator(code_data, seed), 1),) + \
            tuple((_empty_generator(), 0) for _ in range(num_splits-1))

    def sequence_size(self, code_data: HyperMultiGraph) -> int:
        return 1

    def __repr__(self) -> str:
        return "NO_NOISE_SEQ"


class IndependentMatchingNoise(MonteCarloNoise[HyperMultiGraph,
                                               OrderedDecodingEdges]):
    """Class for all independent noise sources defined over matching graphs.
    """

    def __radd__(self, other: IndependentMatchingNoise) -> IndependentMatchingNoise:
        """Called when add implementations return `NotImplemented`, which is typically
        when the type of other is not self.

        For the purpose of addition of noise sources, when noise model types do not match
        it's typically the `AdditiveMatchingNoise` class that deals with the addition.
        The exception being if self or other is ZERO.
        """
        if self == _ZERO:
            return other

        if other == _ZERO:
            return self

        if isinstance(self, AdditiveMatchingNoise):
            return self + other

        return AdditiveMatchingNoise((self, other))

    @staticmethod
    def empty_filter(graph: HyperMultiGraph):
        """Filter that does not change the input."""
        return graph.edges


class NoMatchingNoise(IndependentMatchingNoise):
    """A noise model that outputs an infinite stream of empty decoding edge lists.

    Used to define ZERO for the purpose of monte carlo model addition.
    """

    def error_generator(
        self, code_data: HyperMultiGraph, seed: Optional[int] = None
    ) -> Iterator[OrderedDecodingEdges]:
        while True:
            yield OrderedDecodingEdges()

    def as_exhaustive_sequential_model(self) -> SequentialNoise[HyperMultiGraph,
                                                                OrderedDecodingEdges]:
        return NoNoiseMatchingSequence()

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, UniformMatchingNoise):
            return __o.basic_error_prob == 0
        if isinstance(__o, FixedWeightMatchingNoise):
            return __o.weight == 0
        return isinstance(__o, NoMatchingNoise)

    def __hash__(self):
        return hash(NoMatchingNoise)

    def __repr__(self) -> str:
        return "NO_NOISE"

    def __add__(self, other):
        if self == other:
            return self
        return other


_ZERO = NoMatchingNoise()


class FixedWeightMatchingNoise(IndependentMatchingNoise):
    """A noise model that outputs errors over basic events with fixed weight of activated
    events.
    """

    def __init__(
            self, weight: int,
            edge_filter: Optional[EdgeFilterT] = None):
        self.weight = weight
        self.edge_filter = edge_filter or self.empty_filter

    def error_generator(
        self, code_data: HyperMultiGraph, seed: Optional[int] = None
    ) -> Iterator[OrderedDecodingEdges]:
        filtered = list(self.edge_filter(code_data))
        num_edges = len(filtered)
        # Need to copy into 1D array to avoid inner object conversion to array
        filtered_edges = np.empty(num_edges, dtype=object)
        filtered_edges[:] = filtered
        if num_edges < self.weight:
            raise ValueError(f"Fixed weight {self.weight} generator cannot be created "
                             f"as after filtering there are {num_edges} edges.")
        rng = self.get_rng(seed)

        while True:
            selected_edges = rng.integers(0, num_edges, self.weight)
            while (curr_len := len(set(selected_edges))) < self.weight:
                new_edges = rng.integers(0, num_edges, self.weight - curr_len)
                selected_edges = np.concatenate((selected_edges, new_edges))

            yield OrderedDecodingEdges(filtered_edges[selected_edges],
                                       mod_2_filter=False)

    def as_exhaustive_sequential_model(self) -> SequentialNoise[HyperMultiGraph,
                                                                OrderedDecodingEdges]:
        return ExhaustiveMatchingNoise(self.weight, self.edge_filter)

    def __eq__(self, __o: object) -> bool:
        if self.weight == 0 and __o == _ZERO:
            return True
        return (isinstance(__o, FixedWeightMatchingNoise) and
                self.weight == __o.weight and self.edge_filter == __o.edge_filter)

    def __hash__(self):
        if self.weight == 0:
            return hash(_ZERO)
        return hash((self.weight, self.edge_filter))

    def __repr__(self) -> str:
        return f"FIXED_WEIGHT_{self.weight}"

    def __add__(self, other: IndependentMatchingNoise) -> IndependentMatchingNoise:
        if not isinstance(other, FixedWeightMatchingNoise):
            return NotImplemented

        if self.weight == 0:
            return other

        if other.weight == 0:
            return self

        return AdditiveMatchingNoise((self, other))

    def field_values(self) -> Dict[str, Any]:
        base_dict = super().field_values()
        base_dict["weight"] = self.weight
        return base_dict


class UniformMatchingNoise(IndependentMatchingNoise):
    """A noise model that defines noise over a single decoding graph, where
    all decoding edges are assigned a uniform weight and sampled randomly with
    given probability.
    """

    def __init__(self,
                 basic_error_prob: float,
                 edge_filter: Optional[EdgeFilterT] = None):
        self.basic_error_prob = basic_error_prob
        self.edge_filter = edge_filter or self.empty_filter

    def error_generator(
        self, code_data: HyperMultiGraph, seed: Optional[int] = None
    ) -> Iterator[OrderedDecodingEdges]:
        filtered = list(self.edge_filter(code_data))
        num_edges = len(filtered)
        # Need to copy into 1D array to avoid inner object conversion to array
        filtered_edges = np.empty(num_edges, dtype=object)
        filtered_edges[:] = filtered
        rng = self.get_rng(seed)
        while True:
            selected_edges = np.flatnonzero(
                rng.random(num_edges) < self.basic_error_prob)
            yield OrderedDecodingEdges(filtered_edges[selected_edges],
                                       mod_2_filter=False)

    def importance_sampling_decomposition(
        self,
        code_data: HyperMultiGraph,
        coefficient_limit: float = 1e-20
    ) -> List[Tuple[MonteCarloNoise, float]]:
        coefficients = error_weight_probabilities(len(self.edge_filter(code_data)),
                                                  self.basic_error_prob,
                                                  coefficient_limit)
        return [(FixedWeightMatchingNoise(error_weight, self.edge_filter), coefficient)
                for error_weight, coefficient in enumerate(coefficients)]

    def __eq__(self, __o: object) -> bool:
        if self.basic_error_prob == 0 and __o == _ZERO:
            return True
        return (isinstance(__o, UniformMatchingNoise) and
                self.basic_error_prob == __o.basic_error_prob and
                self.edge_filter == __o.edge_filter)

    def __hash__(self):
        if self.basic_error_prob == 0:
            return hash(_ZERO)
        return hash((self.basic_error_prob, self.edge_filter))

    def __repr__(self) -> str:
        return f"UNIFORM_{self.basic_error_prob}"

    def __add__(self, other: IndependentMatchingNoise) -> IndependentMatchingNoise:
        if not isinstance(other, UniformMatchingNoise):
            return NotImplemented

        if self.basic_error_prob == 0:
            return other

        if other.basic_error_prob == 0:
            return self

        if self.edge_filter == other.edge_filter:
            return UniformMatchingNoise(min(1, self.basic_error_prob +
                                            other.basic_error_prob), self.edge_filter)

        return AdditiveMatchingNoise((self, other))

    def field_values(self) -> Dict[str, Any]:
        base_dict = super().field_values()
        base_dict["p"] = self.basic_error_prob
        return base_dict


class EdgeProbabilityMatchingNoise(IndependentMatchingNoise):
    """A noise model that defines noise over a decoding graph based on the
    edge probabilities. A random variable is generated for each edge and an
    error occurs on the edge if the edge's p_err is greater than this random
    variable. Therefore edges with higher p_err have more chance of being
    selected.
    """

    def __init__(self, edge_filter: Optional[EdgeFilterT] = None):
        self.edge_filter = edge_filter or self.empty_filter

    def error_generator(
        self,
        code_data: HyperMultiGraph,
        seed: Optional[int] = None
    ) -> Iterator[OrderedDecodingEdges]:
        rng = self.get_rng(seed)
        filtered = list(self.edge_filter(code_data))
        num_edges = len(filtered)
        # Need to copy into 1D array to avoid inner object conversion to array
        filtered_edges = np.empty(num_edges, dtype=object)
        filtered_edges[:] = filtered
        edge_probabilities = np.array([code_data.edge_records[edge].p_err
                                       for edge in filtered_edges])
        while True:
            selected_edges = np.flatnonzero(
                rng.random(num_edges) < edge_probabilities)
            yield OrderedDecodingEdges(filtered_edges[selected_edges],
                                       mod_2_filter=False)

    def importance_sampling_decomposition(
        self,
        code_data: HyperMultiGraph,
        coefficient_limit: float = 1e-20
    ) -> List[Tuple[MonteCarloNoise, float]]:
        """Decompose this noise model to be used in importance sampling.
        Decomposition is done by creating a uniform matching noise model for
        each unique p_err in the graph. Then each of these matching noises is
        decomposed and every combination of models is evaluated as the sum of
        the individual models with product of the individual probabilities.

        In this way, if an edge in a graph has 2 different probabilities the
        decomposition of noise is the sum of each of the individual noise
        models defined over the edge and the probability of that noise
        happening is the product of the probabilities.

        Parameters
        ----------
        code_data : HyperMultiGraph
            The graph to decompose noise models over.
        coefficient_limit : float, optional
            If the probabilities drop below this level then the combined
            model is discarded, by default 1e-20

        Returns
        -------
        List[Tuple[MonteCarloNoise, float]]
        """
        p_err_edges = defaultdict(list)
        for edge, record in code_data.edge_records.items():
            p_err_edges[record.p_err].append(edge)

        inner_model_modes = [
            float(binomial_pmf(len(edges),
                               floor((len(edges) + 1) * p_err),
                               Decimal(p_err)))
            for p_err, edges in p_err_edges.items()]

        mode_product = prod(inner_model_modes)

        inner_model_coefficient_limits = [coefficient_limit / (mode_product/mode)
                                          for mode in inner_model_modes]

        # The lambda needs to be annotated with a default argument so that the
        # argument is captured within the for loop.
        edge_p_err_decompositions = (
            UniformMatchingNoise(p_err, lambda _, e=edges: e)
            .importance_sampling_decomposition(code_data, inner_coefficient_limit)
            for (p_err, edges), inner_coefficient_limit in zip(
                p_err_edges.items(), inner_model_coefficient_limits)
        )

        combined_edge_decomposition = []
        for model_product in product(*edge_p_err_decompositions):
            # Doing sum and product with individual iterators is faster than
            # writing a for loop since they are written in C.
            model_sum = sum((model for model, _ in model_product),
                            start=NoMatchingNoise())
            combined_probability = prod(
                (Decimal(prob) for _, prob in model_product),
                start=Decimal(1))
            # Only add the decomposition if the probability is above the
            # cutoff threshold.
            if combined_probability >= coefficient_limit:
                combined_edge_decomposition.append(
                    (model_sum, float(combined_probability)))

        return combined_edge_decomposition

    def __add__(self, other):
        raise NotImplementedError("Cannot add edge probability noise to "
                                  "any other noise.")

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, self.__class__):
            return self.edge_filter == __o.edge_filter
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.__class__, self.edge_filter))

    def __repr__(self) -> str:
        return "EdgeProbability"


class AdditiveMatchingNoise(IndependentMatchingNoise):
    """Given some independent matching noise sources, act as a noise model with errors
    generated from all constituent sources at once.

    Use add operator to add noise sources, not the constructor for this class.
    """
    _filter_cache: Dict[FrozenSet[Callable], Callable] = {}

    def __init__(self, internal_sources: Iterable[IndependentMatchingNoise]):
        self._internal_sources_multiset = Counter(internal_sources)
        self.internal_sources = list(self._internal_sources_multiset.elements())

        if len(self.internal_sources) < 2:
            raise ValueError(
                "Additive noise is for two or more noise sources that do not reduce "
                "to less sources when summed.")

    def error_generator(
        self, code_data: HyperMultiGraph, seed: Optional[int] = None
    ) -> Iterator[OrderedDecodingEdges]:
        internal_generators = [
            model.error_generator(code_data, seed_)
            for seed_, model in zip(offset_seed(seed), self.internal_sources)]
        for error in zip(*internal_generators):
            yield OrderedDecodingEdges(chain.from_iterable(error))

    def as_exhaustive_sequential_model(
            self) -> SequentialNoise[HyperMultiGraph, OrderedDecodingEdges]:
        return AdditiveSequentialMatchingNoise(
            model.as_exhaustive_sequential_model() for model in self.internal_sources)

    def __repr__(self) -> str:
        return ' + '.join(map(str, self.internal_sources))

    def __hash__(self) -> int:
        return hash(frozenset(self._internal_sources_multiset.items()))

    def __eq__(self, __o: object) -> bool:
        return (isinstance(__o, AdditiveMatchingNoise) and
                self._internal_sources_multiset == __o._internal_sources_multiset)

    def __add__(self, other: IndependentMatchingNoise) -> IndependentMatchingNoise:
        if other == _ZERO:
            return self

        if isinstance(other, AdditiveMatchingNoise):
            return sum(chain(self.internal_sources, other.internal_sources), start=_ZERO)

        for model in self.internal_sources:
            if (isinstance(other, model.__class__) and
                    not isinstance(model_sum := model + other, AdditiveMatchingNoise)):
                inner_counter = self._internal_sources_multiset.copy()
                inner_counter.subtract([model])
                inner_counter.update([model_sum])
                return AdditiveMatchingNoise(inner_counter.elements())
        return AdditiveMatchingNoise(self.internal_sources + [other])

    @classmethod
    def _get_combined_filter(cls, filter_list):
        frozen_filter_set = frozenset(filter_list)
        if frozen_filter_set not in cls._filter_cache:
            cls._filter_cache[frozen_filter_set] = \
                lambda edges: list(
                    chain.from_iterable(edge_filter(edges)
                                        for edge_filter in frozen_filter_set))
        return cls._filter_cache[frozen_filter_set]

    def importance_sampling_decomposition(self,
                                          code_data: HyperMultiGraph[EdgeT],
                                          coefficient_limit: float = 1e-20
                                          ) -> List[Tuple[MonteCarloNoise, float]]:
        filtered_edges: List[EdgeT] = []
        internal_filters = []
        internal_probs = set()
        for model in self.internal_sources:
            if isinstance(model, UniformMatchingNoise):
                internal_filters.append(model.edge_filter)
                internal_probs.add(model.basic_error_prob)
                filtered_edges.extend(model.edge_filter(code_data))
            else:
                return super().importance_sampling_decomposition(code_data,
                                                                 coefficient_limit)

        if (len(filtered_edges) == len(set(filtered_edges)) and
                len(internal_probs) == 1):
            affective_model = UniformMatchingNoise(
                internal_probs.pop(), self._get_combined_filter(internal_filters))
            return affective_model.importance_sampling_decomposition(code_data,
                                                                     coefficient_limit)

        return super().importance_sampling_decomposition(code_data,
                                                         coefficient_limit)


class AdditiveSequentialMatchingNoise(
    SequentialNoise[HyperMultiGraph, OrderedDecodingEdges]
):
    """Class which represents multiple independent noise sources where the
    errors are the product of all possible errors on the internal sources
    returning each product as a single error.
    """

    def __init__(self, internal_sources: Iterable[SequentialNoise]):
        self.internal_sources = tuple(internal_sources)

    def error_generator(
        self,
        code_data: HyperMultiGraph,
        seed: int | None = None
    ) -> Iterator[OrderedDecodingEdges]:
        error_generators = (
            partial(model.error_generator, code_data, seed_)
            for model, seed_ in zip(self.internal_sources, offset_seed(seed))
        )
        for error in CombinedSequences._lazy_product(*error_generators):
            yield OrderedDecodingEdges(chain.from_iterable(error))

    def sequence_size(self, code_data: Any) -> int:
        return prod(model.sequence_size(code_data)
                    for model in self.internal_sources)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, AdditiveSequentialMatchingNoise):
            return self.internal_sources == __o.internal_sources
        return NotImplemented

    def __hash__(self) -> int:
        raise NotImplementedError(
            "Implementation voluntarily not provided. If you think you need it, please "
            "open an issue at https://github.com/Deltakit/deltakit/issues/new/choose"
        )

class ExhaustiveMatchingNoise(SequentialNoise[HyperMultiGraph,
                                              OrderedDecodingEdges]):
    """A noise model that outputs all errors with a given weight.

    If weight is set to None, this will result in all weights being evaluated in
    ascending order.
    """

    def __init__(
            self,
            weight: Optional[int],
            edge_filter: Optional[EdgeFilterT] = None):
        self.weight = weight
        self.edge_filter = edge_filter or IndependentMatchingNoise.empty_filter

    def error_generator(
        self, code_data: HyperMultiGraph, seed: Optional[int] = None
    ) -> Iterator[OrderedDecodingEdges]:
        filtered = list(self.edge_filter(code_data))
        # Need to copy into 1D array to avoid inner object conversion to array
        filtered_edges = np.empty(len(filtered), dtype=object)
        filtered_edges[:] = filtered
        if self.weight is None:
            weights = list(range(0, len(filtered_edges)+1))
        else:
            weights = [self.weight]

        for weight in weights:
            for errors in combinations(filtered_edges, weight):
                yield OrderedDecodingEdges(errors, mod_2_filter=False)

    def sequence_size(self, code_data: HyperMultiGraph) -> int:
        len_filtered_edges = len(self.edge_filter(code_data))
        if self.weight is None:
            weights = list(range(0, len_filtered_edges+1))
        else:
            weights = [self.weight]

        return sum(comb(len_filtered_edges, weight)
                   for weight in weights)

    def __eq__(self, __o: object) -> bool:
        return (isinstance(__o, ExhaustiveMatchingNoise) and
                self.weight == __o.weight and self.edge_filter == __o.edge_filter)

    def __hash__(self) -> int:
        raise NotImplementedError(
            "Implementation voluntarily not provided. If you think you need it, please "
            "open an issue at https://github.com/Deltakit/deltakit/issues/new/choose"
        )

    def __repr__(self) -> str:
        if self.weight is None:
            return "ASCENDING_EXHAUSTIVE"
        return f"EXHAUSTIVE_{self.weight}"

    def split_error_generator(
        self,
        code_data: HyperMultiGraph,
        num_splits: int,
        seed: Optional[int] = None
    ) -> Tuple[Tuple[Iterator[OrderedDecodingEdges], int], ...]:
        if self.weight is None:
            raise NotImplementedError("Ascending exhaustive split generator is not "
                                      "implemented.")
        if self.weight == 0:
            return NoNoiseMatchingSequence().split_error_generator(code_data, num_splits,
                                                                   seed)
        edges = list(self.edge_filter(code_data))
        num_edges = len(edges)
        sizes = [0] * num_splits
        iterators: List[Iterable[OrderedDecodingEdges]] = \
            [_empty_generator() for _ in range(num_splits)]
        # split based on first element, bin greedily
        for edge_i in range(num_edges-self.weight+1):
            iterator: Iterable[OrderedDecodingEdges] = starmap(
                lambda x, y: OrderedDecodingEdges(x+y), zip(repeat(
                    (edges[edge_i], )),
                    combinations(edges[edge_i+1:], self.weight-1)))
            size = comb(num_edges - edge_i - 1, self.weight - 1)
            min_bucket_i = sizes.index(min(sizes))
            sizes[min_bucket_i] += size
            iterators[min_bucket_i] = chain(iterators[min_bucket_i], iterator)
        return tuple(zip(iterators, sizes))

    def field_values(self) -> Dict[str, Any]:
        base_dict = super().field_values()
        base_dict["weight"] = self.weight
        return base_dict


class ExhaustiveWeightedMatchingNoise(SequentialNoise[HyperMultiGraph,
                                                      OrderedDecodingEdges]):
    """A noise model that outputs all errors where their weighted
    error locations sum up to less than a given exhaustion ceiling
    """

    def __init__(
            self,
            exhaustion_ceiling: float,
            edge_filter: Optional[EdgeFilterT] = None):
        self.exhaustion_ceiling = exhaustion_ceiling
        self.edge_filter = edge_filter or IndependentMatchingNoise.empty_filter

    def prune_edges(self, code_data: HyperMultiGraph) -> Tuple[Any, ...]:
        """From a set of weighted edges, get the subset of edges and their combinations
        that have weights that can be summed without exceeding a given target
        weight (exhaustion ceiling).

        Parameters
        ----------
        code_data : OrderedDecodingEdges
            A weighted graph of decoding edges

        Returns
        -------
        Tuple[List[OrderedDecodingEdges], int]
            Tuple of edges with weights that are within a given exhaustion target
            and the maximum number of those edges' weights that can be summed while
            still remaining within the exhaustion target
        """
        sorted_edges = sorted(self.edge_filter(code_data),
                              key=lambda edge: code_data.edge_records[edge].weight)

        pruned_edges = [edge for edge in sorted_edges
                        if code_data.edge_records[edge].weight < self.exhaustion_ceiling]

        max_edges_within_ceiling = 0
        total_weight = 0.0
        for i, edge in enumerate(pruned_edges):
            total_weight += code_data.edge_records[edge].weight
            if total_weight >= self.exhaustion_ceiling:
                max_edges_within_ceiling = i
                break

        return pruned_edges, max_edges_within_ceiling

    def error_generator(
        self, code_data: HyperMultiGraph, seed: Optional[int] = None
    ) -> Iterator[OrderedDecodingEdges]:
        pruned_edges, max_edges_within_ceiling = self.prune_edges(code_data)

        for distance in range(max_edges_within_ceiling):
            for errors in combinations(pruned_edges, distance+1):
                if sum(code_data.edge_records[error].weight for error in errors) \
                        < self.exhaustion_ceiling:
                    yield OrderedDecodingEdges(errors)

    def sequence_size(self, code_data: HyperMultiGraph) -> int:
        pruned_edges, max_edges_within_ceiling = self.prune_edges(code_data)

        size = 0
        for distance in range(max_edges_within_ceiling):
            for errors in combinations(pruned_edges, distance+1):
                if sum(code_data.edge_records[error].weight for error in errors) \
                        < self.exhaustion_ceiling:
                    size += 1

        return size

    def __repr__(self) -> str:
        return f"EXHAUSTIVE_WEIGHTED_{self.exhaustion_ceiling}"

    def __eq__(self, __o: object) -> bool:
        return (isinstance(__o, ExhaustiveWeightedMatchingNoise) and
                self.exhaustion_ceiling == __o.exhaustion_ceiling
                and self.edge_filter == __o.edge_filter)

    def __hash__(self) -> int:
        raise NotImplementedError(
            "Implementation voluntarily not provided. If you think you need it, please "
            "open an issue at https://github.com/Deltakit/deltakit/issues/new/choose"
        )

    def field_values(self) -> Dict[str, Any]:
        base_dict = super().field_values()
        base_dict["exhaustion_ceiling"] = self.exhaustion_ceiling
        return base_dict


class UniformErasureNoise(MonteCarloNoise[HyperMultiGraph, Tuple[OrderedDecodingEdges,
                                                                 OrderedDecodingEdges]]):
    """Noise model that simulates a simple erasure channel. Edges are selected for
    erasure independently at random with given probability. Independently at random,
    each erased edge has a 50% chance of also causing a pauli error.
    This noise source returns tuples of all pauli error edges and erased edges.
    """

    def __init__(self,
                 erasure_probability: float,
                 pauli_noise_model: IndependentMatchingNoise | None = None,
                 edge_filter: Optional[EdgeFilterT] = None):
        self.erasure_probability = erasure_probability
        self.edge_filter = edge_filter or IndependentMatchingNoise.empty_filter
        self.pauli_noise_model = (
            pauli_noise_model if pauli_noise_model is not None else NoMatchingNoise()
        )

    def error_generator(
        self, code_data: HyperMultiGraph, seed: Optional[int] = None
    ) -> Iterator[Tuple[OrderedDecodingEdges,
                        OrderedDecodingEdges]]:
        filtered = list(self.edge_filter(code_data))
        num_edges = len(filtered)
        filtered_edges = np.empty(num_edges, dtype=object)
        filtered_edges[:] = filtered
        rng = self.get_rng(seed)

        inner_seed = seed if seed is None else seed+abs(hash(UniformErasureNoise))

        pauli_generator = self.pauli_noise_model.error_generator(code_data,
                                                                 inner_seed)
        while True:
            selected_edges = np.flatnonzero(rng.random(num_edges)
                                            < self.erasure_probability)
            erased_edges = filtered_edges[selected_edges]
            pauli_errors = next(pauli_generator)

            selected_erased_errors = np.flatnonzero(rng.random(len(erased_edges)) < 0.5)
            erased_errors = erased_edges[selected_erased_errors]

            yield (OrderedDecodingEdges(chain(pauli_errors, erased_errors)),
                   OrderedDecodingEdges(erased_edges, mod_2_filter=False))

    def __eq__(self, __o: object) -> bool:
        return (isinstance(__o, UniformErasureNoise) and
                self.erasure_probability == __o.erasure_probability and
                self.edge_filter == __o.edge_filter and
                self.pauli_noise_model == __o.pauli_noise_model)

    def __hash__(self):
        return hash((self.erasure_probability, self.edge_filter, self.pauli_noise_model))

    def __repr__(self) -> str:
        return f"UNIFORM_ERASURE_{self.erasure_probability}"

    def field_values(self) -> Dict[str, Any]:
        base_dict = super().field_values()
        base_dict["p"] = self.erasure_probability
        return base_dict

    def __add__(self, other):
        raise NotImplementedError("Cannot add uniform erasure noise to any other noise.")


def error_weight_probabilities(
    num_error_mechanisms: int,
    probability: float,
    coefficient_limit: float = 1e-20
) -> List[float]:
    """The binomial distribution of error weights when each possible error mechanism
    experiences an error independently with the given probability. The following code
    can be achieved more simply by using the `scipy.stats.binom.pmf()` method but not
    using it here to break the dependency on SciPy.

    Parameters
    ----------
    num_error_mechanisms : int
        The number of possible error sites, i.e. edges in the decoding graph.
    probability : float
        The probability of an error occurring on an edge.
    coefficient_limit : float
        Weights with coefficients less than this value will not be included in the
        output.

    Returns
    -------
    List[float]
        Entry i is the probability that exactly i errors occur.
    """
    if probability == 0:  # 0 ** 0 undefined for Decimal
        return [1.0] if coefficient_limit > 0 else [1.0] + [0.0] * num_error_mechanisms

    # OverflowError: int too large to convert to float
    _probability = Decimal(probability)
    max_coeff = 0.0
    error_weight_ps = []
    for k in range(num_error_mechanisms + 1):
        coeff = float(binomial_pmf(num_error_mechanisms, k, _probability))
        max_coeff = max(coeff, max_coeff)

        if coeff <= coefficient_limit:
            if coeff < max_coeff:
                break
            continue

        error_weight_ps.append(coeff)

    return error_weight_ps


def binomial_pmf(max_k: int, k: int, probability: Decimal) -> Decimal:
    """The binomial probability mass function

    Parameters
    ----------
    max_k : int
        Number of trials
    k : int
        Number of successful trials within max_k trials.
    probability : Decimal
        Probability of success on a single trial

    Returns
    -------
    float
    """
    return comb(max_k, k) * probability ** k \
        * (1 - probability) ** (max_k - k)

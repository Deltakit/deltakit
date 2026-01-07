# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from functools import reduce
from itertools import chain, count, islice, product, repeat
from math import prod
from operator import itemgetter, mul
from typing import Any, Generic, TypeAlias, TypeVar

import numpy.random as npr

ErrorT = TypeVar('ErrorT')
CodeT = TypeVar('CodeT')
BatchErrorT = TypeVar('BatchErrorT')


class BatchErrorGenerator(Generic[BatchErrorT]):
    """Wrapper class of generating batches of errors.

    Parameters
    ----------
    batch_generator : Callable[[int], BatchErrorT]
        Callable that generates. a batch of errors given a batch size.
    """

    def __init__(self, batch_generator: Callable[[int], BatchErrorT]):
        self._batch_generator = batch_generator

    def __call__(self, batch_size: int):
        """Generate a batch of size `batch_size`."
        """
        return self._batch_generator(batch_size)


class _NoiseModel(ABC, Generic[CodeT, ErrorT]):
    """Base abstract class for noise sources.

    Generic over `CodeT`, the type for data from the code needed to produce
    errors over, and `ErrorT`, the type of error produced

    Many noise sources have the commonality of needing a random number generator, and
    that number generator being created via some seed.
    """
    @staticmethod
    def get_rng(seed: int | None) -> npr.Generator:
        """Return a numpy random number generator, using the member data seed.
        """
        return npr.default_rng(seed)

    def split_error_generator(
        self,
        code_data: CodeT,
        num_splits: int,
        seed: int | None = None
    ) -> tuple[tuple[Iterator[ErrorT], int], ...]:
        """Given some representation of a code, return `num_splits` number of generators
        of errors for that code and the respective sizes for those generators.
        """
        msg = f"Noise model {self} is currently not splittable."
        raise NotImplementedError(msg)

    @abstractmethod
    def error_generator(
        self,
        code_data: CodeT,
        seed: int | None = None
    ) -> Iterator[ErrorT]:
        """Given some representation of a code, return a generator of errors for that
        code.
        """

    def build_batch_error_generator(
        self,
        code_data: CodeT,
        seed: int | None = None
    ) -> BatchErrorGenerator:
        """Given some representation of a code, return a generator of batches of errors
        for that code.
        """
        error_gen = self.error_generator(code_data, seed)
        return BatchErrorGenerator(
            lambda batch_size: list(islice(error_gen, int(batch_size))))

    def build_split_batch_error_generators(self,
                                           code_data: CodeT,
                                           num_splits: int,
                                           seed: int | None = None
                                           ) -> tuple[
                                               tuple[BatchErrorGenerator, int], ...]:
        """Given some representation of a code, return `num_splits` number of batch
        generators of errors for that code and the respective sizes for those generators.
        """
        msg = f"Noise model {self} is currently not splittable."
        raise NotImplementedError(msg)

    def field_values(self) -> dict[str, Any]:
        """Return the values of data that characterises this noise model.
        """
        return {"noise_name": str(self)}


class SequentialNoise(_NoiseModel[CodeT, ErrorT]):
    """Abstract class to manage the production of deterministically sized terminating
    noise generators.
    """

    def error_list(self, code_data: CodeT, seed: int | None = None
                   ) -> list[ErrorT]:
        """Given some representation of a code, return a list of errors for that code.
        These will match the errors returned by `error_generator`.
        """
        return list(self.error_generator(code_data, seed))

    @abstractmethod
    def sequence_size(self, code_data: CodeT,
                      ) -> int:
        """Return the number of elements in the sequence that would be generated for the
        given `code_data`.

        This should be a O(1) computation that does not compute the entire error
        sequence.

        Property should hold that: `len(error_list(x)) == sequence_size(x)`.
        """


class MonteCarloNoise(_NoiseModel[CodeT, ErrorT]):
    """Abstract class to manage the production of non-terminating independent noise
    generators.
    """

    # MonteCarloNoise generators are continuous and therefore given size infinite
    infinite_generator_size = -1

    def importance_sampling_decomposition(
        self,
        code_data: CodeT,
        coefficient_limit: float = 1e-20
    ) -> list[tuple[MonteCarloNoise, float]]:
        """Expresses the independent error distribution as a statistical mixture of other
        error distributions.  This decomposition can be used for importance sampling.

        Parameters
        ----------
        code_data : CodeT
            The code data needed to define the component noise sources.
        coefficient_limit : float, optional
            Only noise models with a probability of occurring above this limit
            will be kept in the decomposition.

        Returns
        -------
        List[Tuple[MonteCarloNoise, float]]
            List of component noise sources, with normalised weightings.
        """

        msg = f"Noise model {self} is currently not decomposable."
        raise NotImplementedError(msg)

    def as_exhaustive_sequential_model(self) -> SequentialNoise[CodeT, ErrorT]:
        """Return the equivalent exhaustive model for this noise model. The exhaustive
        model will be a sequential model that contains all elements that could possibly
        be generated by this model.

        Returns
        -------
        SequentialNoise[CodeT, ErrorT]
            Corresponding sequential exhaustive version of this noise model.
        """
        msg = f"Noise model {self} currently has no exhaustion."
        raise NotImplementedError(msg)

    def split_error_generator(
        self,
        code_data: CodeT,
        num_splits: int,
        seed: int | None = None
    ) -> tuple[tuple[Iterator[ErrorT], int], ...]:
        return tuple((self.error_generator(code_data, seed),
                      self.infinite_generator_size)
                     for seed in islice(offset_seed(seed), num_splits))

    def build_split_batch_error_generators(
        self,
        code_data: CodeT,
        num_splits: int,
        seed: int | None = None
    ) -> tuple[tuple[BatchErrorGenerator, int], ...]:
        return tuple((self.build_batch_error_generator(code_data, seed),
                      self.infinite_generator_size)
                     for seed in islice(offset_seed(seed), num_splits))

    @abstractmethod
    def __add__(self, other):
        """Addition defined over independent noise sources.
        Equivalent to taking noise from both independent sources and then adding the
        generated errors together.
        """


NoiseModel: TypeAlias = SequentialNoise[CodeT, ErrorT] | MonteCarloNoise[CodeT, ErrorT]


class CombinedIndependent(MonteCarloNoise[tuple[CodeT, ...], tuple[ErrorT, ...]],
                          Generic[CodeT, ErrorT]):
    """Class to combine several independent noise sources into one combined model, where
    each error returned from one of the internal sources becomes an element in a tuple.
    """

    def __init__(self, internal_sources: tuple[MonteCarloNoise[CodeT, ErrorT], ...]):
        self.internal_sources = internal_sources

    def error_generator(
        self,
        code_data: tuple[CodeT, ...],
        seed: int | None = None
    ) -> Iterator[tuple[ErrorT, ...]]:
        internal_generators = (
            model.error_generator(inner_code_data, seed)
            for seed, model, inner_code_data in zip(offset_seed(seed),
                                                    self.internal_sources, code_data))
        yield from zip(*internal_generators)

    def as_exhaustive_sequential_model(self) -> SequentialNoise[tuple[CodeT, ...],
                                                                tuple[ErrorT, ...]]:
        return CombinedSequences(tuple(model.as_exhaustive_sequential_model()
                                       for model in self.internal_sources))

    def importance_sampling_decomposition(
        self,
        code_data: tuple[CodeT, ...],
        coefficient_limit: float = 1e-20
    ) -> list[tuple[MonteCarloNoise, float]]:
        internal_decompositions = (
            model.importance_sampling_decomposition(inner_code_data, coefficient_limit)
            for model, inner_code_data in zip(self.internal_sources, code_data))
        combined_decomposition: list[tuple[MonteCarloNoise, float]] = []
        for decomposition_product in product(*internal_decompositions):
            sources, coefficients = zip(*decomposition_product)
            if (prod_coeff := prod(coefficients)) > coefficient_limit:
                combined_decomposition.append((CombinedIndependent(sources),
                                               prod_coeff))

        return combined_decomposition

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, CombinedIndependent) and
                self.internal_sources == other.internal_sources)

    def __hash__(self):
        return hash(self.internal_sources)

    def __add__(self: CombinedIndependent[tuple[CodeT, ...], tuple[ErrorT, ...]],
                other: CombinedIndependent[tuple[CodeT, ...], tuple[ErrorT, ...]]
                ) -> CombinedIndependent[tuple[CodeT, ...], tuple[ErrorT, ...]]:
        return CombinedIndependent(tuple(
            model_1 + model_2
            for model_1, model_2 in zip(self.internal_sources, other.internal_sources)))

    def __repr__(self) -> str:
        return f"({', '.join(map(str, self.internal_sources))})"

    def field_values(self) -> dict[str, Any]:
        base_dict = super().field_values()
        for model_i, inner_model in enumerate(self.internal_sources):
            inner_dict = {f"{key}_{model_i}": value
                          for key, value in inner_model.field_values().items()
                          if key != "noise_name"}
            base_dict.update(inner_dict)
        return base_dict


class CombinedSequences(SequentialNoise[tuple[CodeT, ...], tuple[ErrorT, ...]],
                        Generic[CodeT, ErrorT]):
    """Class to combine several sequential noise sources into one combined model, where
    each error returned from one of the internal sources becomes an element in a tuple.
    This is the product of all the internal sources.
    """

    def __init__(self, internal_sources: tuple[SequentialNoise[CodeT, ErrorT], ...]):
        self.internal_sources = internal_sources

    def split_error_generator(
        self,
        code_data: tuple[CodeT, ...],
        num_splits: int,
        seed: int | None = None
    ) -> tuple[tuple[Iterator[tuple[ErrorT, ...]], int], ...]:
        inner_data = list(zip(offset_seed(seed), self.internal_sources, code_data))

        def _get_gen(
                inner_id: int, split_id: int) -> tuple[Iterator[ErrorT], int]:
            inner_seed, inner_model, inner_code_data = inner_data[inner_id]
            return inner_model.split_error_generator(
                inner_code_data, num_splits, inner_seed)[split_id]

        combined_splits = []
        for split_ids in product(range(num_splits), repeat=len(inner_data)):
            gen_sizes = []
            for inner_id, split_id in enumerate(split_ids):
                _, size = _get_gen(inner_id, split_id)
                gen_sizes.append(size)
            combined_splits.append(
                (self._lazy_product(_get_gen, split_ids),
                 reduce(mul, gen_sizes))
            )

        # bin error generators greedily
        gens = [(iter(()))] * num_splits
        sizes = [0] * num_splits
        for gen, size in sorted(combined_splits, key=itemgetter(1), reverse=True):
            index_smallest_bucket = sizes.index(min(sizes))
            sizes[index_smallest_bucket] += size
            gens[index_smallest_bucket] = chain(gens[index_smallest_bucket], gen)

        return tuple(zip(gens, sizes))

    @staticmethod
    def _lazy_product(
            _get_gen: Callable[[int, int], tuple[Iterator[ErrorT], int]],
            split_ids: tuple[int, ...]):
        """Itertools product forcefully evaluates entire inputs given. This version
        is lazy, but requires re-creation of inner generators.

        Parameters
        ----------
        _get_gen : Callable[[int, int], Tuple[Iterator[ErrorT], int]]
            Function to re-create inner generators
        split_ids : Tuple[int, ...]
            The indices of the generators to take the products of.
        """
        gens = []
        for inner_id, split_id in enumerate(split_ids):
            gen, _ = _get_gen(inner_id, split_id)
            gens.append(iter(gen))

        try:
            next_values = [next(i) for i in gens]
        except StopIteration:
            return

        while True:

            yield tuple(next_values)
            for index in reversed(range(len(gens))):
                try:
                    next_values[index] = next(gens[index])
                    break
                except StopIteration:
                    gen, _ = _get_gen(index, split_ids[index])
                    gens[index] = iter(gen)
                    next_values[index] = next(gens[index])
            else:
                return

    def error_generator(
        self,
        code_data: tuple[CodeT, ...],
        seed: int | None = None
    ) -> Iterator[tuple[ErrorT, ...]]:
        yield from product(
            *(model.error_generator(inner_code_data, seed)
              for seed, model, inner_code_data in zip(
                offset_seed(seed),
                self.internal_sources, code_data)))

    def sequence_size(self, code_data: tuple[CodeT, ...]) -> int:
        return prod(model.sequence_size(inner_code_data)
                    for model, inner_code_data in zip(self.internal_sources,
                                                      code_data))

    def __repr__(self) -> str:
        return f"({', '.join(map(str, self.internal_sources))})"

    def field_values(self) -> dict[str, Any]:
        base_dict = super().field_values()
        for model_i, inner_model in enumerate(self.internal_sources):
            inner_dict = {f"{key}_{model_i}": value
                          for key, value in inner_model.field_values().items()
                          if key != "noise_name"}
            base_dict.update(inner_dict)
        return base_dict


def offset_seed(seed: int | None) -> Iterable[int | None]:
    """Given a starting seed, produce an iterator of distinct offset seeds.
    """
    if seed is None:
        return repeat(seed)
    return count(seed)

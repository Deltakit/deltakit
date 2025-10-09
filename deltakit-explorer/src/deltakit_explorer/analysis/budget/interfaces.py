"""Defines a generic noise interface required to perform error-budgeting."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, Generic, Mapping, Self, TypeVar

import numpy
import numpy.typing as npt

Computation = TypeVar("Computation")


class NoiseInterface(ABC, Generic[Computation]):
    """A minimal interface around a noise model to perform error-budgeting.

    Args:
        noise_parameters (Sequence[float] | npt.NDArray[numpy.float64]):
            the floating-point values representing noise parameters for the underlying
            noise model.
        parameter_names (Sequence[str]): a name representing the noise parameter for
            each entry in ``noise_parameters``.
        name (str | None): name of the noise model.

    """

    num_noise_parameters: ClassVar[int]
    parameter_names: ClassVar[tuple[str, ...]]

    def __init__(
        self,
        noise_parameters: Sequence[float] | npt.NDArray[numpy.float64],
        name: str | None = None,
    ) -> None:
        self._noise_parameters = numpy.asarray(noise_parameters, dtype=numpy.float64)
        self._name = (
            name if name is not None else "_".join(self.parameter_names)
        )

    @abstractmethod
    def apply(self, computation: Computation) -> Computation:
        """Apply the noise model represented by ``self`` to the provided computation."""

    @classmethod
    def is_valid(cls, parameters: npt.NDArray[numpy.float64]) -> str | None:
        """Check if the provided ``parameters`` are valid for the noise model
        represented by ``cls``."""
        if parameters.size != cls.num_noise_parameters:
            return (
                f"Invalid number of parameters (got {parameters.size}, expected "
                f"{cls.num_noise_parameters})."
            )
        return None

    @property
    def noise_parameters(self) -> npt.NDArray[numpy.float64]:
        return self._noise_parameters

    def variate_by(
        self,
        noise_parameters_deltas: (
            Sequence[float] | npt.NDArray[numpy.float64] | Mapping[str, float]
        ),
    ) -> Self:
        """Returns a new noise model with its noise parameters changed by the provided
        ``noise_parameters_deltas``.

        Args:
            noise_parameters_deltas \
            (Sequence[float] | npt.NDArray[numpy.float64] | Mapping[str, float]):
                the amount by which each noise parameter should be offset in order to
                create a new noise model.

        Returns:
            a new noise model obtained from ``self`` and the provided
            ``noise_parameters_deltas``.
        """
        if isinstance(noise_parameters_deltas, Mapping):
            noise_parameters_deltas = [
                noise_parameters_deltas[name] for name in self.parameter_names
            ]
        noise_parameters_deltas = numpy.asarray(
            noise_parameters_deltas, dtype=numpy.float64
        )
        return type(self)(self.noise_parameters + noise_parameters_deltas, self._name)

    def is_variation_valid(
        self, parameters_variation: npt.NDArray[numpy.float64]
    ) -> str | None:
        return self.is_valid(self._noise_parameters + parameters_variation)

    def is_variation_on_parameter_valid(
        self, noise_parameter_index: int, variation: float
    ) -> str | None:
        e = numpy.zeros((self.num_noise_parameters,), dtype=numpy.float64)
        e[noise_parameter_index] = variation
        return self.is_variation_valid(e)

    def variate_noise_parameter_by(
        self, noise_parameter_index: int, variation: float
    ) -> Self:
        e = numpy.zeros((self.num_noise_parameters,), dtype=numpy.float64)
        e[noise_parameter_index] = variation
        return self.variate_by(e)

    def get_heuristic_steps(self, factor: float = 0.02) -> npt.NDArray[numpy.float64]:
        return factor * self._noise_parameters

    def to_dict(self) -> dict[str, float]:
        return {
            name: value
            for name, value in zip(self.parameter_names, self._noise_parameters)
        }

    def _get_index(self, parameter_name: str) -> int:
        return (
            self.parameter_names.index(parameter_name)
            if parameter_name in self.parameter_names
            else -1
        )

    def _get_value(self, parameter_name: str) -> float:
        if (index := self._get_index(parameter_name)) != -1:
            return self._noise_parameters[index]
        raise IndexError(f"Parameter {parameter_name} not found.")

    @property
    def name(self) -> str:
        return self._name

    @property
    def approximate_measurement_error_rate(self) -> float | None:
        return None

"""Defines a generic noise interface required to perform error-budgeting."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, Generic, TypeVar

import numpy
import numpy.typing as npt

Computation = TypeVar("Computation")


class NoiseInterface(ABC, Generic[Computation]):
    """A minimal interface around a noise model to perform error-budgeting.

    Args:
        noise_parameters (Sequence[float] | npt.NDArray[numpy.floating]):
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
        noise_parameters: Sequence[float] | npt.NDArray[numpy.floating],
        name: str | None = None,
    ) -> None:
        self._noise_parameters = numpy.asarray(noise_parameters, dtype=numpy.floating)
        self._name = name if name is not None else "_".join(self.parameter_names)

    @abstractmethod
    def apply(self, computation: Computation) -> Computation:
        """Apply the noise model represented by ``self`` to the provided computation."""

    @classmethod
    def is_valid(cls, parameters: npt.NDArray[numpy.floating]) -> str | None:
        """Check if the provided ``parameters`` are valid for the noise model
        represented by ``cls``."""
        if parameters.size != cls.num_noise_parameters:
            return (
                f"Invalid number of parameters (got {parameters.size}, expected "
                f"{cls.num_noise_parameters})."
            )
        return None

    @property
    def noise_parameters(self) -> npt.NDArray[numpy.floating]:
        return self._noise_parameters

    def _get_index(self, parameter_name: str) -> int:
        return (
            self.parameter_names.index(parameter_name)
            if parameter_name in self.parameter_names
            else -1
        )

    def _get_value(self, parameter_name: str) -> float:
        if (index := self._get_index(parameter_name)) != -1:
            return self._noise_parameters[index]
        msg = f"Parameter {parameter_name} not found."
        raise IndexError(msg)

    @property
    def name(self) -> str:
        return self._name

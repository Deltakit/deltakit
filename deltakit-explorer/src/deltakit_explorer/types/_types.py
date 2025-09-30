# (c) Copyright Riverlane 2020-2025.
"""This file contains types to be used as input fields
for the generate_circuit mutation
"""

from __future__ import annotations

import dataclasses
import enum
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from io import BytesIO, StringIO, TextIOWrapper
from math import ceil
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import stim
from deltakit_explorer.data._converter import (read_01, read_b8, read_csv,
                                               write_01, write_b8, write_csv)
from deltakit_explorer.enums._api_enums import APIEndpoints, DataFormat, DecoderType
from deltakit_explorer.types._data_string import DataString


class JSONable:
    """Provide a utility static method ``to_json`` to inheriting classes."""

    @staticmethod
    def _to_json(obj) -> Any:
        """Convert dataclass or object to dict for JSON serialization."""
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return JSONable._to_json(dataclasses.asdict(obj))
        if isinstance(obj, dict):
            return {key: JSONable._to_json(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [JSONable._to_json(value) for value in obj]
        if isinstance(obj, enum.Enum):
            return JSONable._to_json(obj.value)
        if isinstance(obj, (str, int, float)):
            return obj
        if obj is None:
            return None
        raise ValueError(
            f"Object of type {type(obj)} is not a dataclass or serializable object."
        )

    def to_json(self) -> dict:
        return JSONable._to_json(self)


# pylint: disable=duplicate-code, too-few-public-methods, invalid-name
@dataclass
class MatrixSpecifications(JSONable):
    """Parameters to be passed to BivariateBicycle code type, IBM Bivariate
    Bicycle qLDPC codes, as specified in arXiv:2308.07915.
    The parameter `n` below as in the [[n,k,d]] specification of an
    error-correcting code refers to a number of physical qubits.

    Args:
        param_l (int):
            The parameter `l`, used to construct a code of length `n = 2lm`.
        param_m (int):
            The parameter `m`, used to construct a code of length `n = 2lm`.
        m_A_powers (List[int]):
            The powers of the polynomial terms used to construct
            the matrix A. Each polynomial is of the form
            `x^a + y^b + y^c` so the sequence should specify
            `[a, b, c]`.
        m_B_powers (List[int]):
            The powers of the polynomial terms used to construct
            the matrix B. Each polynomial is of the form
            `y^a + x^b + x^c` so the sequence should specify
            `[a, b, c]`.
    """

    param_l: int
    param_m: int
    m_A_powers: list[int]
    m_B_powers: list[int]

    def to_gql(self):
        """Get a dictionary for the current object"""

        return {
            "paramL": self.param_l,
            "paramM": self.param_m,
            "mAPowers": self.m_A_powers,
            "mBPowers": self.m_B_powers,
        }


# pylint: disable=duplicate-code, too-few-public-methods
@dataclass
class Sizes(JSONable):
    """The size parameter to be passed to generate_circuit() method.

    Args:
        sizes (Sequence[int]):
            - QUANTUM_MEMORY: `(x_distance, z_distance)` for RotatedPlanarCode,
              `(distance,)` for repetition, `(width, height)` for UnrotatedPlanarCode.
            - STABILITY: patch size `(width, height)`.

    """

    sizes: Sequence[int]

    def to_gql(self):
        """Get a dictionary for the current object"""

        return {"sizes": self.sizes}


# pylint: disable=duplicate-code, too-few-public-methods
@dataclass
class CircuitParameters(JSONable):
    """Parameter to be passed as a type for additional parameters
    for generate_circuit"""

    matrix_specifications: MatrixSpecifications | None = None
    sizes: Sizes | None = None

    @staticmethod
    def from_sizes(sizes: Iterable[int]) -> CircuitParameters:
        """Use with Repetition, Planar and Toric codes. Defines parameters
        for a circuit, which is initialised by a grid. Repetition code accepts
        a single number, other codes - two.

        Args:
            sizes (Iterable[int]): List or tuple of integer sizes.

        Return:
            CircuitParameters: parameters object for circuit generation.

        Examples:
            Creation and usage of parameters::

                circuit = client.generate_circuit(
                    types.QECExperimentDefinition(
                        experiment_type=types.QECExperimentType.QUANTUM_MEMORY,
                        code_type=types.QECECodeType.ROTATED_PLANAR,
                        observable_basis=PauliBasis.Z,
                        num_rounds=3,
                        basis_gates=["CZ", "H", "MZ", "RZ"],
                        parameters=CircuitParameters.from_sizes((3, 3))
                    )
                )

        """
        return CircuitParameters(sizes=Sizes(list(sizes)))

    @staticmethod
    def from_matrix_specification(
        param_l: int, param_m: int, m_A_powers: list[int], m_B_powers: list[int]
    ) -> CircuitParameters:
        """Use with Bivariate Bicycle codes.

        Args:
            param_l (int):
                The parameter `l`, used to construct a code of length `n = 2lm`.
            param_m (int):
                The parameter `m`, used to construct a code of length `n = 2lm`.
            m_A_powers (List[int]):
                The powers of the polynomial terms used to construct
                the matrix A. Each polynomial is of the form
                `x^a + y^b + y^c` so the sequence should specify
                `[a, b, c]`.
            m_B_powers (List[int]):
                The powers of the polynomial terms used to construct
                the matrix B. Each polynomial is of the form
                `y^a + x^b + x^c` so the sequence should specify
                `[a, b, c]`.

        Returns:
            CircuitParameters: object to pass to a circuit generation function.

        Examples:
            Creation and usage of parameters::

                circuit = client.generate_circuit(
                    types.QECExperimentDefinition(
                        experiment_type=types.QECExperimentType.QUANTUM_MEMORY,
                        code_type=types.QECECodeType.BIVARIATE_BICYCLE,
                        observable_basis=PauliBasis.Z,
                        num_rounds=6,
                        basis_gates=["CZ", "H", "MZ", "RZ"],
                        parameters=CircuitParameters.from_matrix_specification(
                            param_l=6,
                            param_m=6,
                            m_A_powers=[3, 1, 2],
                            m_B_powers=[3, 1, 2],
                        )
                    )
                )

        """
        return CircuitParameters(
            matrix_specifications=MatrixSpecifications(
                param_l=param_l,
                param_m=param_m,
                m_A_powers=m_A_powers,
                m_B_powers=m_B_powers
            )
        )

    def to_gql(self) -> dict[str, Any]:
        """Get a GraphQL-compatible dictionary
        representation of the object."""

        return {
            "sizes": self.sizes.to_gql() if self.sizes else None,
            "matrixSpecifications": (
                self.matrix_specifications.to_gql()
                if self.matrix_specifications
                else None
            ),
        }


@dataclass
class TypedData(ABC):  # pragma: nocover
    """Basic class for all typed data manipulation activities."""

    data_format: DataFormat
    """Format in which the data is stored. b8, 01, or csv."""

    content: Any
    """File name, on in-RAM array object."""

    data_width: int = -1
    """If data is serialisable, data_width supports b8 format parsing."""

    @abstractmethod
    def as_01_string(self) -> str:
        """Generate or return a 01 string for the stored data."""
        msg = "This is an abstract class"
        raise NotImplementedError(msg)

    @abstractmethod
    def as_b8_bytes(self) -> bytes:
        """Generate or return a b8 string for the stored data."""
        msg = "This is an abstract class"
        raise NotImplementedError(msg)

    @abstractmethod
    def as_numpy(self) -> npt.NDArray[np.uint8]:
        """Generate or return an numpy array for the stored data."""
        msg = "This is an abstract class"
        raise NotImplementedError(msg)

    def get_width(self) -> int:
        """Derives data width, infers from the data in 01 or CSV format."""
        if self.data_width != -1:
            return self.data_width
        if self.data_format != DataFormat.B8:
            self.data_width = self.as_numpy().shape[1]
            return self.data_width
        return -1


@dataclass
class RAMData(TypedData):
    """An wrapper around a collection stored in RAM."""

    content: list[list[int]] | npt.NDArray

    def get_width(self) -> int:
        """Infer the width of the string from the data."""
        if self.data_width != -1:
            return self.data_width
        if len(self.content) > 0:
            return len(self.content[0])
        return -1

    def as_01_string(self) -> str:
        string_builder = StringIO()
        write_01(self.content, string_builder)
        return string_builder.getvalue()

    def as_b8_bytes(self) -> bytes:
        bytes_builder = BytesIO()
        write_b8(self.content, bytes_builder)
        return bytes_builder.getvalue()

    def as_numpy(self) -> npt.NDArray[np.uint8]:
        return np.array(self.content, dtype=np.uint8)


@dataclass
class TypedDataFile(TypedData):
    """Data file with typed content."""

    content: Path

    def as_01_string(self) -> str:
        if self.data_format == DataFormat.F01:
            with Path.open(self.content, encoding="utf-8") as file:
                return file.read()
        elif self.data_format == DataFormat.CSV:
            with Path.open(self.content, encoding="utf-8") as file:
                string_builder = StringIO()
                write_01(read_csv(file), string_builder)
                return string_builder.getvalue()
        elif self.data_format == DataFormat.B8:
            with open(self.content, "rb") as file:
                string_builder = StringIO()
                write_01(
                    read_b8(file, self.data_width), string_builder)
                return string_builder.getvalue()
        else:
            msg = "Type conversion is not implemented"
            raise NotImplementedError(msg)

    def as_b8_bytes(self) -> bytes:
        if self.data_format == DataFormat.B8:
            with open(self.content, "rb") as file:
                return file.read()
        elif self.data_format == DataFormat.CSV:
            with Path.open(self.content, encoding="utf-8") as file:
                bytes_builder = BytesIO()
                write_b8(read_csv(file), bytes_builder)
                return bytes_builder.getvalue()
        elif self.data_format == DataFormat.F01:
            with Path.open(self.content, encoding="utf-8") as file:
                bytes_builder = BytesIO()
                write_b8(read_01(file), bytes_builder)
                return bytes_builder.getvalue()
        msg = f"Type conversion is not implemented for {self.data_format}"
        raise NotImplementedError(
            msg)

    def as_numpy(self) -> npt.NDArray[np.uint8]:
        if self.data_format == DataFormat.B8:
            with open(self.content, "rb") as file:
                return np.array(
                    list(read_b8(file, self.data_width)), dtype=np.uint8)
        elif self.data_format == DataFormat.CSV:
            with Path.open(self.content, encoding="utf-8") as file:
                return np.array(list(read_csv(file)), dtype=np.uint8)
        elif self.data_format == DataFormat.F01:
            with Path.open(self.content, encoding="utf-8") as file:
                return np.array(list(read_01(file)), dtype=np.uint8)
        msg = f"Type conversion is not implemented for {self.data_format}"
        raise NotImplementedError(
            msg)


@dataclass
class TypedDataString(TypedData):
    """Typed data string."""

    content: DataString

    def as_01_string(self) -> str:
        if self.data_format == DataFormat.F01:
            return self.content.to_string()
        if self.data_format == DataFormat.B8:
            string_builder = StringIO()
            write_01(
                read_b8(BytesIO(self.content.data), self.data_width),
                string_builder,
            )
            return string_builder.getvalue()
        if self.data_format == DataFormat.CSV:
            string_builder = StringIO()
            write_01(
                read_csv(
                    TextIOWrapper(BytesIO(self.content.data), encoding="utf-8")
                ),
                string_builder,
            )
            return string_builder.getvalue()
        msg = "Type conversion is not implemented"
        raise NotImplementedError(msg)

    def as_b8_bytes(self) -> bytes:
        if self.data_format == DataFormat.B8:
            return self.content.data
        if self.data_format == DataFormat.CSV:
            bytes_builder = BytesIO()
            write_b8(
                read_csv(
                    TextIOWrapper(BytesIO(self.content.data), encoding="utf-8")
                ),
                bytes_builder
            )
            return bytes_builder.getvalue()
        if self.data_format == DataFormat.F01:
            bytes_builder = BytesIO()
            write_b8(
                read_01(
                    TextIOWrapper(BytesIO(self.content.data), encoding="utf-8")
                ),
                bytes_builder
            )
            return bytes_builder.getvalue()
        msg = "Type conversion is not implemented"
        raise NotImplementedError(msg)

    def as_numpy(self) -> npt.NDArray[np.uint8]:
        wrapper = TextIOWrapper(BytesIO(self.content.data), encoding="utf-8")

        if self.data_format == DataFormat.B8:
            return np.array(
                list(
                    read_b8(
                        stream=BytesIO(self.content.data),
                        width=self.data_width
                    )
                ),
                dtype=np.uint8,
            )
        if self.data_format == DataFormat.CSV:
            return np.array(
                list(read_csv(stream=wrapper)),
                dtype=np.uint8,
            )
        if self.data_format == DataFormat.F01:
            return np.array(
                list(read_01(stream=wrapper)),
                dtype=np.uint8,
            )
        msg = "Type conversion is not implemented"
        raise NotImplementedError(msg)

    @staticmethod
    def from_data(data: TypedData) -> TypedDataString:
        """
        Prepare a DataString of arbitrary table-like data,

        Args:
            data (QECDataType): File or table with the data to send.

        Returns:
            TypedDataString: DataString with associated data type.
        """
        if isinstance(data, TypedDataFile):
            return TypedDataString(
                data_format=data.data_format,
                content=DataString.from_file(data.content),
            )
        if isinstance(data, TypedDataString):
            return data
        if isinstance(data, RAMData):
            if data.data_format == DataFormat.F01:
                return TypedDataString(
                    data_format=data.data_format,
                    data_width=data.get_width(),
                    content=DataString(data.as_01_string()),
                )
            if data.data_format == DataFormat.CSV:
                string_builder = StringIO()
                write_csv(data.as_numpy(), string_builder)
                return TypedDataString(
                    data_format=data.data_format,
                    data_width=data.get_width(),
                    content=DataString(string_builder.getvalue()),
                )
            if data.data_format == DataFormat.B8:
                bytes_builder = BytesIO()
                write_b8(data.as_numpy(), bytes_builder)
                return TypedDataString(
                    data_format=data.data_format,
                    data_width=data.get_width(),
                    content=DataString(bytes_builder.getvalue()),
                )
        msg = f"Cannot extract binary data from {type(data)}, {data.data_format}."
        raise ValueError(
            msg)

    def as_data_string(self) -> str:
        """Gets a datastring for the object."""
        return str(self.content)


@dataclass
class BinaryDataType:
    """Any binary table, stored in a file, datastring, or RAM."""

    data: TypedData

    def __init__(
        self,
        data: Path | DataString | list[list[int]] | npt.NDArray,
        data_format: DataFormat = DataFormat.B8,
        data_width: int = -1,
    ):
        if isinstance(data, Path):
            self.data = TypedDataFile(
                data_format=data_format,
                content=data,
                data_width=data_width
            )
        elif isinstance(data, DataString):
            self.data = TypedDataString(
                data_format=data_format,
                content=data,
                data_width=data_width,
            )
        elif isinstance(data, (list, np.ndarray)):
            self.data = RAMData(
                data_format=data_format,
                content=np.array(data, dtype=np.uint8),
                data_width=data_width,
            )
        else:
            msg = f"Not supported argument of type {type(data)}"
            raise ValueError(msg)

    def as_data_string(self, data_format: DataFormat = DataFormat.B8) -> str:
        """Converts the object to a compact datastring."""
        if data_format == DataFormat.B8:
            return str(DataString(TypedDataString.from_data(self.data).as_b8_bytes()))
        if data_format == DataFormat.F01:
            return str(DataString(TypedDataString.from_data(self.data).as_01_string()))
        if data_format == DataFormat.CSV:
            string = StringIO()
            write_csv(
                TypedDataString.from_data(self.data).as_numpy(), string)
            return str(DataString(string.getvalue()))
        msg = f"Saving to data string is not implemented for {data_format}"
        raise NotImplementedError(
            msg)

    def as_01_string(self) -> str:
        """Represent content of the object as 01 string."""
        return self.data.as_01_string()

    def as_numpy(self) -> npt.NDArray[np.uint8]:
        """
        Represent the data as
        unsigned 1 byte/number (uint8) numpy table.
        """
        return self.data.as_numpy()

    def as_b8_bytes(self) -> bytes:
        """Represent underlying data as b8 bit-packed bytes."""
        return self.data.as_b8_bytes()

    def split(self, shards: int) -> list[BinaryDataType]:
        """Split a data file into multiple blocks based on the
        number of output files. Return objects of the same class.

        Args:
            shards (int): Number of approximately equal chunks.

        Returns:
            List[BinaryDataType]:
                List of chuck objects, of the same
                type as the calling object.

        Examples:
            Splitting the data file::

                object = Measurements(data_string, DataFormat.F01)
                parts = object.split(shards=shards)

        """
        # heavy lifting
        assert shards > 0, \
            f"Number of batches should be bigger than 0, but was {shards}."
        numpy_data = self.as_numpy()
        lines = numpy_data.shape[0]
        shard_size = ceil(lines / shards)
        if shards == 1:
            return [self]
        results = []
        cls = type(self)
        for i in range(shards):
            start, end = i * shard_size, (i + 1) * shard_size
            # this may produce empty objects, as in 6 / 4 -> [2, 2, 2, 0]
            results.append(
                cls(
                    data=numpy_data[start: end],
                    data_format=self.data.data_format,
                    data_width=self.data.get_width(),
                )
            )
        return results

    def to_batches(self, batch_size: int = 100_000) -> Iterable[BinaryDataType]:
        """Split a data file into batches of fixed size.
        Returns objects of the same class.

        Args:
            batch_size (int):
                Number of approximately equal chunks. Default is 100k.

        Returns:
            Iterable[BinaryDataType]:
                Iterable of batch objects, of the same
                type as the calling object.

        Examples:
            Splitting the data file::

                object = Measurements(data_string, DataFormat.F01)
                for batch in object.to_batches(1000):
                    ...

        """
        assert batch_size > 0, \
            f"Batch size should be bigger than 0, but was {batch_size}."
        numpy_data = self.as_numpy()
        lines = numpy_data.shape[0]
        cls = type(self)
        ptr = 0
        while ptr < lines:
            yield cls(
                data=numpy_data[ptr: ptr + batch_size],
                data_format=self.data.data_format,
                data_width=self.data.get_width(),
            )
            ptr += batch_size

    @classmethod
    def combine(cls, list_of_data: list[BinaryDataType]) -> BinaryDataType:
        """Stack multiple binary tables into one using numpy representation.

        Args:
            list_of_data (List[BinaryDataType]):
                list of objects, e.g. decoding predictions or measurements.

        Returns:
            BinaryDataType:
                an object of the class, which invoked the method.

        """
        data = np.vstack([d.as_numpy() for d in list_of_data])
        return cls(
            data=data,
            data_format=DataFormat.B8,
            data_width=data.shape[1],
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return np.allclose(self.as_numpy(), other.as_numpy())


class Measurements(BinaryDataType):
    """Binary table representing measurements."""

    @staticmethod
    def _deinstrument_leakage_circuit(circuit: str) -> str:
        """This method removes leakage heralding, noise, and
        detectors to make sure"""
        bits: list[int | None] = []
        measurement_bits_count = 0
        lines = circuit.splitlines()
        result_lines = []
        for line in lines:
            if line.strip() == "":
                continue
            if "HERALD_LEAKAGE_EVENT" in line:
                herald_gap = len(line.strip().split()) - 1
                bits.extend([None] * herald_gap)
                continue
            # measurement happened, adding new bits
            if line.strip()[0] == "M":
                measurements = len(line.strip().split()) - 1
                bits.extend(
                    range(measurement_bits_count, measurement_bits_count + measurements)
                )
                measurement_bits_count += measurements
            # detector/observable, update indices or skip if refers to leakage
            if ("DETECTOR" in line) or ("OBSERVABLE_INCLUDE" in line):
                skip_me = False
                # absolute values may become smaller, we should not re-remap!
                bits_from_latest_to_earliest = list(enumerate(bits))[::-1]
                for i, new_i in bits_from_latest_to_earliest:
                    # old relative indexing
                    neg_i = i - len(bits)
                    # this was a heralded leakage
                    if new_i is None:
                        if f"rec[{neg_i}]" in line:
                            # this line addresses leakage heralding
                            # we remove it
                            skip_me = True
                            break
                        # just do not update, if referred to leakage
                        continue
                    # new relative indexing
                    neg_new_i = new_i - measurement_bits_count
                    line = line.replace(f"rec[{neg_i}]", f"rec[{neg_new_i}]")
                if skip_me:
                    continue
            if ("RELAX" in line) or ("LEAKAGE" in line):
                continue
            result_lines.append(line)

        return "\n".join(result_lines)

    def to_detectors_and_observables(
        self,
        stim_circuit: str | stim.Circuit,
        sweep_bits: BinaryDataType | None = None,
    ) -> tuple[DetectionEvents, ObservableFlips]:
        """Converts measurements object into detectors and observables tuple.

        Args:
            stim_circuit (str | stim.Circuit):
                STIM circuit content.
            sweep_bits: (Optional[BinaryDataType]):
                If data was generated with initial sweep bits,
                they are needed to obtain correct detectors.

        Returns:
            Tuple[DetectionEvents, ObservableFlips]:
                Detector and observable data.
        """
        if isinstance(stim_circuit, stim.Circuit):
            circuit = stim_circuit
        else:
            try:
                circuit = stim.Circuit(stim_circuit)
            except ValueError:
                stim_circuit = self._deinstrument_leakage_circuit(stim_circuit)
                try:
                    circuit = stim.Circuit(stim_circuit)
                except ValueError as stim_ex:
                    msg = "Provided circuit is not a valid stim circuit."
                    raise ValueError(
                        msg
                    ) from stim_ex

        converter = circuit.compile_m2d_converter()
        sweeps = None
        if sweep_bits is not None:
            sweeps = sweep_bits.as_numpy().astype(np.bool_)
        dets, obs = converter.convert(
            measurements=self.as_numpy().astype(np.bool_),
            sweep_bits=sweeps,
            append_observables=False,
            separate_observables=True,
        )
        return (
            DetectionEvents(dets, self.data.data_format, circuit.num_detectors),
            ObservableFlips(obs, self.data.data_format, circuit.num_measurements),
        )


class DetectionEvents(BinaryDataType):
    """Binary table, representing detectors."""


class ObservableFlips(BinaryDataType):
    """Binary table, representing observables."""


class LeakageFlags(BinaryDataType):
    """Binary table, representing heralded leakage flags."""


@dataclass
class NoiseModel(ABC):
    """Generic noise model class."""

    ENDPOINT: ClassVar[APIEndpoints]
    """
    Each noise model has a corresponding service endpoint,
    which is responsible for applying this model to a circuit.
    """
    ENDPOINT_RESULT_FIELDNAME: ClassVar[str]
    """Endpoints name noise addition results differently."""


@dataclass
# pylint: disable=too-many-instance-attributes
class PhysicalNoiseModel(NoiseModel):
    """
    Physically-inspired noise model. This noise model
    is built on physical characteristics of qubits and
    gates.
    """

    ENDPOINT: ClassVar[APIEndpoints] = APIEndpoints.ADD_NOISE
    ENDPOINT_RESULT_FIELDNAME: ClassVar[str] = "addNoiseToStimCircuit"

    t_1: float
    """T1 time (relaxation from |1> to |0>), seconds."""

    t_2: float
    """T2 time (dephasing), seconds."""

    time_1_qubit_gate: float
    """Time to execute a 1-qubit gate."""

    time_2_qubit_gate: float
    """Time to execute a 2-qubit gate."""

    time_measurement: float
    """Time to measure a qubit."""

    time_reset: float
    """Time to reset a qubit."""

    p_1_qubit_gate_error: float
    """Probability of a flip while doing a 1-qubit gate."""

    p_2_qubit_gate_error: float
    """Probability of a flip while doing a 2-qubit gate."""

    p_reset_error: float
    """Probability of a flip while doing a reset."""

    p_meas_qubit_error: float
    """Probability of incorrect measurement."""

    p_readout_flip: float
    """Probability of a flip while measuring a qubit."""

    @staticmethod
    def get_floor_superconducting_noise():
        """Return minimum supercondicting noise."""
        return PhysicalNoiseModel(
            t_1=20e-6,
            t_2=30e-6,
            time_1_qubit_gate=25e-9,
            time_2_qubit_gate=34e-9,
            time_measurement=500e-9,
            time_reset=160e-9,
            p_1_qubit_gate_error=0.312e-3,
            p_2_qubit_gate_error=0.425e-3,
            p_reset_error=1.99e-3,
            p_meas_qubit_error=6.17e-3,
            p_readout_flip=0.0
        )

    @staticmethod
    def get_superconducting_noise():
        """Generic superconducting noise model, derived from Google's
        https://www.nature.com/articles/s41586-022-05434-1 experiment."""
        return PhysicalNoiseModel(
            t_1=20e-6,
            t_2=30e-6,
            time_1_qubit_gate=25e-9,
            time_2_qubit_gate=34e-9,
            time_measurement=500e-9,
            time_reset=160e-9,
            p_1_qubit_gate_error=0.001,
            p_2_qubit_gate_error=0.01,
            p_reset_error=0.01,
            p_meas_qubit_error=0.01,
            p_readout_flip=0.01
        )

    @staticmethod
    def get_ion_trap_noise():
        """Generic ion trap noise model, derived from
        IonQ Aria public device specification
        https://ionq.com/quantum-systems/aria."""

        return PhysicalNoiseModel(
            t_1=100.0,
            t_2=1.0,
            time_1_qubit_gate=5e-6,
            time_2_qubit_gate=2.5e-5,
            time_measurement=1.2e-4,
            time_reset=1e-5,
            p_1_qubit_gate_error=6e-4,
            p_2_qubit_gate_error=6e-3,
            p_reset_error=1e-5,
            p_meas_qubit_error=39e-4,
            p_readout_flip=3e-3
        )


@dataclass
class SI1000NoiseModel(NoiseModel):
    """
    The SI 1000
    (Superconducting Inspired with 1000 nanosecond cycle)
    noise model, with leakage.
    See https://arxiv.org/pdf/2312.04522v1 (Appendix A) for details.
    """

    ENDPOINT: ClassVar[APIEndpoints] = APIEndpoints.ADD_SI1000_NOISE
    ENDPOINT_RESULT_FIELDNAME: ClassVar[str] = "addSi1000NoiseToStimCircuit"

    p: float
    """Probability of a Pauli error."""

    p_l: float
    """Probability of a leakage error."""


class DecodingResult:
    """Results of decoding"""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        fails: int,
        shots: int,
        times: list[float] | None = None,
        counts: list[int] | None = None,
        predictions_format: DataFormat = DataFormat.F01,
        predictionsFile: dict[str, str | None] | None = None,
    ):
        """
        Args:
            fails (int): number of times decoder failed
            shots (int): number of shots performed
            times (List[float]): CPU time for decoding workers
            counts (List[int]): number of shots each worker processed
            predictions_format (DataFormat): format of predictions
            predictionsFile (Optional[Dict[str, Optional[str]]]):
                definition of filename or datastring with decoding predictions.
        """
        self.fails = fails
        self.shots = shots
        if times is None:
            times = []
        if counts is None:
            counts = [shots]
        self.times = times
        self.counts = counts
        self.predictions_format = predictions_format
        self.predictions_file = predictionsFile
        if self.predictions_file is not None:
            string = self.predictions_file.get("uid")
            if string is None:
                self.predictions = None
            else:
                self.predictions = BinaryDataType(
                    data=DataString.from_data_string(string),
                    data_format=self.predictions_format,
                )
        else:
            self.predictions = None

    def get_logical_error_probability(self) -> float:
        """Logical error probability, fails / shots."""
        if self.shots <= 0:
            return float("inf")
        return self.fails / self.shots

    def get_standard_deviation(self) -> float:
        """
        Return standard deviation of result
        for a given number of shots and decoder fails."""
        if self.shots <= 0:
            return float("inf")
        hits = self.shots - self.fails
        return (self.fails * hits / self.shots ** 3) ** .5

    @classmethod
    def combine(cls, results: Iterable[DecodingResult]) -> DecodingResult:
        """Concatenate decoding results from multiple experiments.
        Predictions are also merged.

        Args:
            results (Iterable[DecodingResult]):
                collection of decoding results. E.g. from batch decoding.

        Returns:
            DecodingResult: combined results, including predictions.
        """
        fails = 0
        shots = 0
        times: list[float] = []
        counts: list[int] = []
        predictions = []
        for result in results:
            fails += result.fails
            shots += result.shots
            times.extend(result.times)
            counts.extend(result.counts or [result.shots])
            # pylint: disable=protected-access
            if result.predictions is not None:
                predictions.append(result.predictions)
        total_result = cls(
            fails=fails,
            shots=shots,
            times=times,
            counts=counts,
        )
        if len(predictions) > 0:
            total_result.predictions = BinaryDataType.combine(predictions)
        return total_result

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, DecodingResult)
            and self.counts == other.counts
            and self.fails == other.fails
            and self.shots == other.shots
            and self.predictions == other.predictions
        )

    def __str__(self) -> str:
        return (
            f"DecodingResult(shots={self.shots}, "
            f"fails={self.fails}, "
            f"LEP={self.get_logical_error_probability():.5f}"
            f" ± {self.get_standard_deviation():.5f})"
        )

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Decoder:
    """Common settings for decoding."""

    decoder_type: DecoderType
    """Decoding algorithm."""

    use_experimental_graph: bool = False
    """If set to True, data will be split into halves.
    One half of the data will be used for training experimental
    decoder error graph, and the other half will be decoded with
    this graph. And vice versa."""

    parallel_jobs: int = 1
    """Server will perform data-parallel decoding, if more cores
    are requested."""

    parameters: dict[str, Any] = field(default_factory=dict)
    """Decoders may have individual settings. They may be passed as
    dictionary items."""


@dataclass
class QubitCoordinateToDetectorMapping:
    """Mapping of detectors to qubit coordinates."""

    detector_map: dict[tuple[float, ...], list[int]]
    """Keys are detector coordinates, values are a list of
    all detectors IDs, associated with these coordinates."""

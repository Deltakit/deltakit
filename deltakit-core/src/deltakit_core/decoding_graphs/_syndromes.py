# (c) Copyright Riverlane 2020-2025.
"""Datastructures for decoding syndromes."""

from __future__ import annotations

import sys
from collections import Counter, UserDict, defaultdict
from functools import cached_property
from itertools import chain, repeat
from typing import (
    AbstractSet,
    Any,
    cast,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    SupportsIndex,
    Tuple,
    Union,
    overload,
)

import numpy as np

Bit = Literal[0, 1]


class DetectorRecord(UserDict):
    """Dictionary for recording information about a detector.
    String attributes for arbitrary values.
    Coordinate and time given as special data that is always
    defined.

    Parameters
    ----------
    spatial_coord : Tuple[int, ...], optional
        Variadic spatial coordinate of this detector, by default ().
    time : int, optional
        Time coordinate of this detector, by default 0.
    """

    def __init__(
        self, spatial_coord: Tuple[float | int, ...] = (), time: int = 0, **kwargs
    ) -> None:
        super().__init__(spatial_coord=spatial_coord, time=time, **kwargs)

    @property
    def spatial_coord(self) -> Tuple[float, ...]:
        """Spatial coordinate of this detector."""
        return self.data["spatial_coord"]

    @spatial_coord.setter
    def spatial_coord(self, value: Tuple[float, ...]):
        self.data["spatial_coord"] = value

    @property
    def time(self) -> int:
        """Time coordinate of this detector."""
        return self.data["time"]

    @time.setter
    def time(self, value: int):
        self.data["time"] = value

    @property
    def full_coord(self) -> Tuple[float, ...]:
        """Return the full coordinate for this detector, which is given in the form
        spatial coordinates then time coordinate.
        """
        return self.spatial_coord + (self.time,)

    @classmethod
    def from_dict(
        cls,
        property_dict: Dict[str, Any],
    ) -> "DetectorRecord":
        """Create a DetectorRecord from a given property dict of optional values.
        This is included for compatibility for network-x vertex representations.

        Parameters
        ----------
        property_dict : Dict[str, Any]
            Any optional properties such as spatial_coord or time.

        Returns
        -------
        DetectorRecord
        """
        spatial_coord = property_dict.get("spatial_coord", tuple())
        # ensure a tuple
        spatial_coord = tuple(spatial_coord)
        time = property_dict.get("time", 0)
        return DetectorRecord(time=time, spatial_coord=spatial_coord)

    @classmethod
    def from_sequence(cls, sequence: Sequence[float | int]) -> DetectorRecord:
        """Create a DetectorRecord given a sequence if floats or ints. The
        first n-1 items in the sequence are used for the spatial coordinate
        and the nth item is used as the time coordinate.

        Parameters
        ----------
        sequence : Sequence[float  |  int]
            A sequence of floats or ints specifying the coordinates.

        Returns
        -------
        DetectorRecord
        """
        if len(sequence) == 0:
            return DetectorRecord()
        *spatial_coord, time = sequence
        return DetectorRecord(spatial_coord=tuple(spatial_coord), time=int(time))


class OrderedSyndrome(Sequence[int], AbstractSet[int]):
    """Immutable ordered mod 2 set of detectors. Where detectors
    are represented by integers.

    Parameters
    ----------
    detectors : Iterable[int], optional
        Detectors to put into this ordered mod 2 collection, by default ().
    enforce_mod_2 : bool, optional
        Whether to enforce that there's only an odd number of each detector in
        the syndrome, by default True.
    """

    def __init__(self, detectors: Iterable[int] = (), enforce_mod_2: bool = True):
        if enforce_mod_2:
            detector_counts = Counter(detectors)
            self._detectors = dict.fromkeys(
                [edge for edge, count in detector_counts.items() if count % 2 == 1]
            )
        else:
            self._detectors = dict.fromkeys(detectors)

    @cached_property
    def _as_tuple(self) -> Tuple[int, ...]:
        """Defined to create immutable object to hash, and to make `__getitem__` a O(1)
        method. Exists as a property to avoid duplication of data in core member
        attributes, and to have this be created only when needed.
        """
        return tuple(self._detectors)

    def __len__(self) -> int:
        return len(self._detectors)

    def __getitem__(self, index):
        return self._as_tuple.__getitem__(index)

    def __contains__(self, x: object) -> bool:
        return self._detectors.__contains__(x)

    def __iter__(self) -> Iterator[int]:
        return self._detectors.__iter__()

    def __repr__(self) -> str:
        return f"[{', '.join(map(str, self))}]"

    def __hash__(self) -> int:
        return hash(self._as_tuple)

    def __eq__(self, __o: object) -> bool:
        # Could allow other sequences to be comparable?
        return isinstance(__o, OrderedSyndrome) and self._as_tuple == __o._as_tuple

    @classmethod
    def from_bitstring(cls, bitstring: Sequence[Bit]) -> OrderedSyndrome:
        """Create an ordered syndrome from a bistring.

        Parameters
        ----------
        bitstring : Sequence[Bit]
            Bitstring to convert from.

        Returns
        -------
        OrderedSyndrome
            Indices of the non zero elements in the bitstring as a syndrome.
        """
        return OrderedSyndrome(np.flatnonzero(bitstring).tolist())

    def as_bitstring(self, num_bits: int) -> List[Bit]:
        """Convert OrderedSyndrome to a bitstring, with `num_bits` bits."""
        syndrome_bits = np.zeros(num_bits, dtype=np.uint8)
        syndrome_bits[list(self._detectors)] = 1
        result = syndrome_bits.tolist()
        return cast(list[Bit], result)

    def split_at_symptom(
        self, split_detection_event: int
    ) -> Tuple[List[int], List[int]]:
        """Split OrderedSyndrome in to two lists of detection events in
        order, ending the first at the given detector event.

        Parameters
        ----------
        split_detection_event : int
            The largest detection event symptom.
            Events after this are interpreted to be leakage heralding events.

        Returns
        -------
        Tuple[List[int], List[int]]
            Lists of post and pre symptom detection events.
        """
        pre_dets, post_dets = [], []
        for event_id in self:
            if event_id > split_detection_event:
                post_dets.append(event_id - split_detection_event - 1)
            else:
                pre_dets.append(event_id)
        return pre_dets, post_dets

    def split_by_time_coord(
        self, detector_records: Mapping[int, DetectorRecord], layers: int
    ) -> List[OrderedSyndrome]:
        """Split the OrderedSyndrome into a time-ordered list of OrderedSyndromes, with
        `layers` number of items in the list.
        """
        syndromes_by_time: Dict[int, List[int]] = defaultdict(list)
        for detector in self:
            detector_time = detector_records[detector].time
            syndromes_by_time[detector_time].append(detector)

        result = [OrderedSyndrome() for _ in range(layers)]
        for time, syndromes in syndromes_by_time.items():
            result[time] = OrderedSyndrome(syndromes)

        return result

    def as_layers(
        self,
        syndromes_per_layer: Union[int, List[int]],
        total_layers: Optional[int] = None,
    ) -> List[List[int]]:
        """Create a sequence of layers from a syndrome, where each layer is a
        collection of integers representing the detectors triggered on that layer.
        Each layer should be contiguously indexed from 0.

        Parameters
        ----------
        syndromes_per_layer : Union[int, List[int]]
            Number of possible detectors on each layer. If int, assume all layers
            are the same size.
        total_layers : Optional[int]
            Number of layers returned. If None, use len(syndromes_per_layer).
            If syndromes_per_layer is int, use however many layers is necessary to fit
            all the syndromes. By default, None.
        """
        if total_layers is None:
            if isinstance(syndromes_per_layer, int):
                total_layers = (
                    max(self._detectors, default=-1) // syndromes_per_layer
                ) + 1
            else:
                total_layers = len(syndromes_per_layer)
        if isinstance(syndromes_per_layer, int):
            syndromes_per_layer = [syndromes_per_layer for _ in range(total_layers)]
        layers: List[List[int]] = [[] for _ in range(total_layers)]

        if len(syndromes_per_layer) == 0:
            return layers
        current_layer = 0
        syndromes_before_current_layer = 0
        syndromes_including_current_layer = syndromes_per_layer[0]
        for symptom in sorted(self._detectors):
            # check if crossed to new layer
            while syndromes_including_current_layer <= symptom:
                syndromes_before_current_layer += syndromes_per_layer[current_layer]
                current_layer += 1
                syndromes_including_current_layer += syndromes_per_layer[current_layer]
            layers[current_layer].append(symptom - syndromes_before_current_layer)

        return layers

    @classmethod
    def from_layers(
        cls, layers: Sequence[Iterable[int]], syndromes_per_layer: Union[int, List[int]]
    ) -> OrderedSyndrome:
        """Create a syndrome from a sequence of layers, where each layer is a
        collection of integers representing the detectors triggered on that layer.
        Each layer should be contiguously indexed from 0.

        Parameters
        ----------
        layers : Sequence[Iterable[int]]
            Sequence of layers of positive detectors.
        syndromes_per_layer : Union[int, List[int]]
            Number of possible detectors on each layer. If int, assume all layers same
            size.

        Returns
        -------
        OrderedSyndrome
        """
        syndrome: List[int] = []
        layer_synd_add = 0
        for i, layer in enumerate(layers):
            syndrome.extend([synd + layer_synd_add for synd in layer])
            if isinstance(syndromes_per_layer, int):
                layer_synd_add += syndromes_per_layer
            else:
                layer_synd_add += syndromes_per_layer[i]
        return cls(syndrome)


class Bitstring:
    """Class which efficiently represents a bitstring."""

    __slots__ = ("_bits",)

    def __init__(self, value: int = 0):
        if value < 0:
            raise ValueError("Bitstring cannot be a negative value.")
        self._bits = value

    @classmethod
    def from_bits(cls, bits: Iterable[Bit]):
        """Create a bitstring from a list of bits. The bits are
        assumed to be in little endian form, i.e. the lsb is the first item
        in the iterable.

        >>> Bitstring.from_bits([1, 0, 0, 1, 1])
        0b11001
        """
        value = 0
        for bit_num, bit in enumerate(bits):
            value |= bit << bit_num
        return cls(value)

    @classmethod
    def from_bytes(cls, bytes_: Collection[SupportsIndex]):
        """Create a bitstring from a collection of bytes. For example, a numpy array of
        uint8s. Assumed to be in little endian form.

        >>> Bitstring.from_bytes(np.array([2, 128], dtype=np.uint8))
        0b1000000000000010
        """
        return cls(int.from_bytes(bytes_, byteorder="little"))

    @classmethod
    def from_indices(cls, indices: Iterable[int]):
        """Create a bitstring from a list of non-zero indices.

        >>> Bitstring.from_indices([0, 2, 5])
        0b100101
        """
        value = 0
        for index in indices:
            value |= 1 << index
        return cls(value)

    def to_indices(self) -> List[int]:
        """Create a list of indices of the non-zero elements in this bitstring

        Returns
        -------
        List[int]
            The indices of non-zero elements in the bitstring.
        """
        return [index for index, value in enumerate(self) if value == 1]

    def to_words(self, num_bits_per_word: int) -> Iterator[FixedWidthBitstring]:
        """Convert the bitstring into words of a given length."""
        for shift in range(0, len(self), num_bits_per_word):
            yield FixedWidthBitstring(num_bits_per_word, self._bits >> shift)

    def bit_count(self) -> int:
        """Get the number of 1s in this bitstring."""
        if sys.version_info >= (3, 10):
            return self._bits.bit_count()
        return bin(self._bits).count("1")

    def __or__(self, __value: object) -> Bitstring:
        if isinstance(__value, Bitstring):
            return Bitstring(self._bits | __value._bits)
        return NotImplemented

    def __ior__(self, __value: object) -> Bitstring:
        if isinstance(__value, Bitstring):
            self._bits |= __value._bits
            return self
        return NotImplemented

    def __and__(self, __value: object) -> Bitstring:
        if isinstance(__value, Bitstring):
            return Bitstring(self._bits & __value._bits)
        return NotImplemented

    def __iand__(self, __value: object) -> Bitstring:
        if isinstance(__value, Bitstring):
            self._bits &= __value._bits
            return self
        return NotImplemented

    def __xor__(self, __value: object) -> Bitstring:
        if isinstance(__value, Bitstring):
            return Bitstring(self._bits ^ __value._bits)
        return NotImplemented

    def __ixor__(self, __value: object) -> Bitstring:
        if isinstance(__value, Bitstring):
            self._bits ^= __value._bits
            return self
        return NotImplemented

    def __lshift__(self, __value: object) -> Bitstring:
        if isinstance(__value, int):
            return Bitstring(self._bits.__lshift__(__value))
        return NotImplemented

    def __rshift__(self, __value: object) -> Bitstring:
        if isinstance(__value, int):
            return Bitstring(self._bits.__rshift__(__value))
        return NotImplemented

    def __iter__(self) -> Iterator[Bit]:
        # max is required so that list(Bitstring(0)) == [0] and not []
        for i in range(max(1, self._bits.bit_length())):
            yield (self._bits >> i) & 1  # type: ignore[misc]

    def __reversed__(self) -> Iterator[Bit]:
        # max is required so that list(Bitstring(0)) == [0] and not []
        for i in reversed(range(max(1, self._bits.bit_length()))):
            yield (self._bits >> i) & 1  # type: ignore[misc]

    @overload
    def __getitem__(self, index: int) -> Bit: ...

    @overload
    def __getitem__(self, index: slice) -> Bitstring: ...

    def __getitem__(self, index):
        if isinstance(index, int):
            return (self._bits >> index) & 1
        if isinstance(index, slice):
            start, stop, _ = index.indices(len(self))
            mask = (1 << (stop - start)) - 1
            return Bitstring((self._bits >> start) & mask)
        raise TypeError(
            f"Bitstring indices must be integers or slices, not {type(index)}"
        )

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Bitstring):
            return self._bits == __value._bits
        return NotImplemented

    def __len__(self) -> int:
        return max(1, self._bits.bit_length())

    def __index__(self):
        return self._bits

    def __repr__(self) -> str:
        return bin(self._bits)

    def __hash__(self) -> int:
        return hash(self._bits)


class FixedWidthBitstring(Bitstring):
    """Class which represents a fixed precision bitstring."""

    __slots__ = ("_width",)

    def __init__(self, width: int, value: int = 0):
        if width < 1:
            raise ValueError("Width of bitstring must be greater than zero.")
        super().__init__(value & ((1 << width) - 1))
        self._width = width

    @classmethod
    def from_indices(cls, indices: Iterable[int]):
        value = 0
        for index in indices:
            value |= 1 << index
        return cls(value.bit_length(), value)

    @classmethod
    def from_bits(cls, bits: Iterable[Bit]):
        value = 0
        # Use variable shadowing in case the enumerate block is never entered
        bit_num = -1
        for bit_num, bit in enumerate(bits):
            value |= bit << bit_num
        return cls(bit_num + 1, value)

    @classmethod
    def from_bytes(cls, bytes_: Collection[SupportsIndex]):
        return cls(len(bytes_) * 8, int.from_bytes(bytes_, byteorder="little"))

    def change_width(self, new_width: int):
        """Change the width of this bitstring internally. If the new width is
        less than the current width then any excess bits will be removed.
        """
        if new_width < self._width:
            self._bits &= (1 << new_width) - 1
        self._width = new_width

    def __or__(self, __value: object) -> FixedWidthBitstring:
        if isinstance(__value, Bitstring):
            return FixedWidthBitstring(self._width, self._bits | __value._bits)
        return NotImplemented

    def __ior__(self, __value: object) -> FixedWidthBitstring:
        if isinstance(__value, Bitstring):
            # Need the mask here to remove the upper bits if __value has more
            # bits than self.
            self._bits |= __value._bits & ((1 << self._width) - 1)
            return self
        return NotImplemented

    def __xor__(self, __value: object) -> FixedWidthBitstring:
        if isinstance(__value, Bitstring):
            return FixedWidthBitstring(self._width, self._bits ^ __value._bits)
        return NotImplemented

    def __ixor__(self, __value: object) -> FixedWidthBitstring:
        if isinstance(__value, Bitstring):
            # Need the mask here to remove the upper bits if __value has more
            # bits than self.
            self._bits ^= __value._bits & ((1 << self._width) - 1)
            return self
        return NotImplemented

    def __and__(self, __value: object) -> FixedWidthBitstring:
        if isinstance(__value, Bitstring):
            return FixedWidthBitstring(self._width, self._bits & __value._bits)
        return NotImplemented

    def __iand__(self, __value: object) -> FixedWidthBitstring:
        if isinstance(__value, Bitstring):
            # Need the mask here to remove the upper bits if __value has more
            # bits than self.
            self._bits &= __value._bits & ((1 << self._width) - 1)
            return self
        return NotImplemented

    def __invert__(self) -> FixedWidthBitstring:
        return FixedWidthBitstring(self._width, ~self._bits)

    def __iter__(self) -> Iterator[Bit]:
        yield from chain(
            super().__iter__(), repeat(0, self._width - max(1, self._bits.bit_length()))
        )

    def __reversed__(self) -> Iterator[Bit]:
        yield from chain(
            repeat(0, self._width - max(1, self._bits.bit_length())),
            super().__reversed__(),
        )

    def __len__(self) -> int:
        return self._width

    def __getitem__(self, index):
        if isinstance(index, int):
            return (self._bits >> index) & 1
        if isinstance(index, slice):
            start, stop, _ = index.indices(len(self))
            mask = (1 << (stop - start)) - 1
            return FixedWidthBitstring(
                width=stop - start, value=(self._bits >> start) & mask
            )
        raise TypeError(
            f"Bitstring indices must be integers or slices, not {type(index)}"
        )

    def __add__(self, __value: object) -> FixedWidthBitstring:
        if isinstance(__value, FixedWidthBitstring):
            return FixedWidthBitstring(
                self._width + __value._width,
                (self._bits << __value._width) | __value._bits,
            )
        return NotImplemented


def get_round_words(
    detector_indices: Sequence[int], num_detectors_per_round: int, num_rounds: int
) -> Iterator[FixedWidthBitstring]:
    """For given sparse-form sequence, return the corresponding bit-strings
    that correspond to individual decoding rounds.

    The sparse sequence is expected to contain defects, boundary adjacency, logical
    adjacency or other bits that are specified per-detector.

    Parameters
    ----------
    detector_indices : Sequence[int]
        Sparse form of detector aligned data.
    num_detectors_per_round : int
        Number of detectors in each decoding round.
    num_rounds : int
        Number of decoding rounds.

    Yields
    ------
    Iterator[FixedWidthBitstring]
        Bitstrings for each decoding round, containing dense-form data.
    """

    total_width = num_detectors_per_round * num_rounds
    if len(detector_indices) == 0:
        bitstring = FixedWidthBitstring(total_width)
    else:
        bitstring = FixedWidthBitstring.from_indices(detector_indices)
        bitstring.change_width(total_width)

    return bitstring.to_words(num_detectors_per_round)

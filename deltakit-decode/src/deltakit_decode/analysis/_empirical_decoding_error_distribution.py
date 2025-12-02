# (c) Copyright Riverlane 2020-2025.

from __future__ import annotations

from itertools import product
from typing import overload

import numpy as np
import numpy.typing as npt


class EmpiricalDecodingErrorDistribution:
    """Data structure for storing data related to decoding events. Stores the
    distribution of failures over logicals. Failures can be represented as integers
    or boolean tuples assumed to be in little endian form, i.e. the lsb is
    the first item in the tuple.

    Parameters
    ----------
    number_of_logicals : int
        The number of logicals over which all error combinations
        are derived from.
    """

    def __init__(self, number_of_logicals: int):
        self._number_of_logicals = number_of_logicals
        self._distribution_size = 1 << number_of_logicals
        self._shots = 0
        self._fails = 0
        self._fails_per_logical = np.zeros(number_of_logicals, dtype=np.int32)
        self._error_distribution = np.zeros(self._distribution_size, dtype=np.uint32)

    def add_event(self, event: tuple[bool, ...], frequency: int = 1):
        """Adds given tuple to error disitrubtion.

        Parameters
        ----------
        event : Tuple[bool, ...]
            Tuple of bools representing the event of errors
            which occurred on each logical.
        frequency : Tuple[bool, ...]
            Frequency of the event, by default 1.
        """
        if frequency < 0:
            msg = f"Event frequency = {frequency} must be non-negative."
            raise ValueError(msg)
        event_index = sum((1 << i) * parity
                          for i, parity in enumerate(event))
        self._error_distribution[event_index] += frequency
        self._shots += frequency
        if event_index != 0:
            self._fails += frequency

        for logical, error in enumerate(event):
            if error:
                self._fails_per_logical[logical] += frequency

    @property
    def number_of_logicals(self) -> int:
        """The number of logicals over which all error combinations
        are derived from.
        """
        return self._number_of_logicals

    def record_error(self, correction: tuple[bool, ...], target: tuple[bool, ...]):
        """Computes and adds error event based on predicted and target value of
        the logicals.

        Parameters
        ----------
        correction : Tuple[bool, ..]
            List of logical flips as booleans. True if the homology class is 1 (flipped),
            False if the homology class is 0 (not flipped).
        target : Tuple[bool, ..]
            The target logical flips. True if the homology
            class is 1 (flipped), False if the homology class is 0 (not flipped).
        """
        logical_parity = sum((1 << i) * int(l_error ^ l_correction)
                             for i, (l_error, l_correction) in
                             enumerate(zip(correction, target)))
        self._error_distribution[logical_parity] += 1
        self._shots += 1
        self._fails += int(correction != target)

    def reset(self):
        """Resets all data to empty distribution.
        """
        self._shots = 0
        self._fails = 0
        self._fails_per_logical = np.zeros(self._number_of_logicals, dtype=np.int32)
        self._error_distribution = np.zeros(self._distribution_size, dtype=np.uint32)

    def batch_record_errors(self, corrections: npt.NDArray[np.uint8],
                            target: npt.NDArray[np.uint8]):
        """Computes and adds a batch of error events based on batches
        of predicted and target values of the logicals.

        Parameters
        ----------
        corrections :  npt.NDArray[np.uint8]
            2D Array indicating predicted corrections of shape
            (number of shots, number of edges). Each element is a 1
            or 0. prediction of each logical from the decoder.

        target :  npt.NDArray[np.uint8]
            2D Array indicating target homolgies of shape
            (number of shots, number of edges). Each element is a 1
            or 0.
        """
        batch_size, num_observables = corrections.shape
        logicals_xor = corrections ^ target
        logical_parities = np.zeros(batch_size, dtype=np.int32)
        for col in range(num_observables):
            logical_parities += (1 << col) * logicals_xor[:, col]
        np.add.at(self._error_distribution, logical_parities, 1)

        fails = np.sum(
            np.any(corrections != target, axis=1)
        ).astype(np.int_)

        self._shots += batch_size
        self._fails += fails

        batch_fails_per_observable = np.sum(
            corrections != target, axis=0
        ).astype(np.int_)

        self._fails_per_logical += batch_fails_per_observable

    def __add__(self,
                other: EmpiricalDecodingErrorDistribution
                ) -> EmpiricalDecodingErrorDistribution:
        if (isinstance(other, EmpiricalDecodingErrorDistribution)
                and self._number_of_logicals == other.number_of_logicals):
            sum_distr = EmpiricalDecodingErrorDistribution(self._number_of_logicals)
            for event in product((False, True), repeat=self._number_of_logicals):
                sum_distr.add_event(event, self[event] + other[event])
            return sum_distr
        return NotImplemented

    @overload
    def __getitem__(self, index: int) -> int:
        ...

    @overload
    def __getitem__(self, index: tuple[bool, ...]) -> int:
        ...

    def __getitem__(self, index) -> int:
        if isinstance(index, int):
            return self._error_distribution[index]

        if isinstance(index, tuple) and all(isinstance(item, bool) for item in index):
            if len(index) != self.number_of_logicals:
                msg = (
                    f"EmpiricalDecodingErrorDistribution index tuples must be of "
                    f"length {self.number_of_logicals}, not {len(index)}"
                )
                raise TypeError(msg)
            event = sum((1 << i) * parity
                        for i, parity in enumerate(index))
            return self[event]
        msg = (
            "EmpiricalDecodingErrorDistribution indices must be integers or "
            f"Tuple[bool], not {type(index)}"
        )
        raise TypeError(msg)

    def __len__(self):
        return self._distribution_size

    def get_num_errors_on_logical(self, logical: int) -> int:
        """Helper function that can compute the number of errors that happened
        on a given logical.

        Parameters
        ----------
        logical : int
            The zero indexed logical over which to compute the number of
            errors that happened on that logical.

        Returns
        -------
        int
            The number of errors that happened on a given logical.
        """
        return self._fails_per_logical[logical]

    def to_dict(self) -> dict[tuple[bool, ...], int]:
        """Returns the error distribution in dictionary representation of
        boolean keys.

        Examples
        --------
            {(False, False, False): 4, (False, False, True): 1, ...,
            (True, True, True): 2}

            indicates that there were 4 cases where the
            correction and error did not differ at all, 1 case where the third
            logical alone differed and 2 cases where all the logicals differed.

        Returns
        -------
        Dict[Tuple[bool, ...], int]
            A dictionary that describes the distribution of failures across
            all combinations of logicals.
        """
        return {parity: self[parity] for parity in
                product((False, True), repeat=self._number_of_logicals)}

    @classmethod
    def from_dict(cls, distribution_dict: dict[tuple[bool, ...], int],
                  ) -> EmpiricalDecodingErrorDistribution:
        """Create a EmpiricalDecodingErrorDistribution from a given distribution
        dict of boolean tuples.

        Parameters
        ----------
        distribution_dict : Dict[Tuple[bool], int]
            A dictionary that describes the distribution of failures across
            all combinations of logicals.

        Returns
        -------
        EmpiricalDecodingErrorDistribution
            Returns an instance of the class based on the specified distribution dict
        """
        if len(distribution_dict) == 0:
            return EmpiricalDecodingErrorDistribution(number_of_logicals=0)
        num_logicals = len(next(iter(distribution_dict)))
        distr = EmpiricalDecodingErrorDistribution(num_logicals)
        for event, occurrences in distribution_dict.items():
            distr.add_event(event, occurrences)
        return distr

    @property
    def shots(self) -> int:
        """ Number of recorded shots.
        """
        return self._shots

    @property
    def fails(self) -> int:
        """ Number of recorded failures.
        """
        return self._fails

    @property
    def fails_per_logical(self) -> npt.NDArray[np.int32]:
        """Numpy array of failure occurrences on each logical.
        """
        return self._fails_per_logical

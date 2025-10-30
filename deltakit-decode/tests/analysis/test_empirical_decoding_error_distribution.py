# (c) Copyright Riverlane 2020-2025.
import random
from itertools import product
from typing import Dict, Tuple

import numpy as np
import pytest
from deltakit_decode.analysis._empirical_decoding_error_distribution import \
    EmpiricalDecodingErrorDistribution


class TestEmpiricalDecodingErrorDistribution:

    @pytest.fixture(params=[
        EmpiricalDecodingErrorDistribution(5),
        EmpiricalDecodingErrorDistribution(1),
        EmpiricalDecodingErrorDistribution(12)
    ], scope='function')
    def empirical_decoding_error_distribution(
            self, request) -> EmpiricalDecodingErrorDistribution:
        return request.param

    def _generate_random_bool_tuple(self, tuple_size: int):
        return tuple(random.choice([True, False]) for _ in range(tuple_size))

    def _distribution_data_is_equal(self,
                                    distr1: EmpiricalDecodingErrorDistribution,
                                    distr2: EmpiricalDecodingErrorDistribution):
        "Check that the data of two distributions are equal"
        assert len(distr1) == len(distr2)

        assert distr1.shots == distr2.shots
        assert distr1.fails == distr2.fails

        assert np.array_equal(distr1.fails_per_logical, distr2.fails_per_logical)
        for event in range(len(distr1)):
            assert distr1[event] == distr2[event]


    @pytest.mark.parametrize("num_logicals", [5, 1, 13])
    def test_from_dict_then_to_dict_is_identity(self, num_logicals: int):
        """Test that converting converting distribution to dict and then back
        results in exactly the same distribution."""
        empirical_decoding_error_distribution = EmpiricalDecodingErrorDistribution(
            num_logicals)
        event_length = empirical_decoding_error_distribution.number_of_logicals
        for _ in range(10):
            empirical_decoding_error_distribution.add_event(self._generate_random_bool_tuple(event_length),
                                                            random.randint(1, 99))
        assert (EmpiricalDecodingErrorDistribution.from_dict(empirical_decoding_error_distribution.to_dict()).to_dict() ==
                empirical_decoding_error_distribution.to_dict())

    def test_to_dict_returns_expected_dict(self):
        empirical_decoding_error_distribution = EmpiricalDecodingErrorDistribution(3)

        empirical_decoding_error_distribution.add_event((True, True, False), 3)
        empirical_decoding_error_distribution.add_event((False, True, False), 1)
        empirical_decoding_error_distribution.add_event((True, True, False), 2)

        distr_dict = empirical_decoding_error_distribution.to_dict()

        expected_dict = {parity: 0 for parity in product((False, True), repeat=3)}
        expected_dict[(True, True, False)] = 5
        expected_dict[(False, True, False)] = 1
        assert distr_dict == expected_dict

    @pytest.mark.parametrize("negative_int", [-5, -1, -13])
    def test_distribution_cannot_have_negative_frequency_event(self, negative_int):
        empirical_decoding_error_distribution = EmpiricalDecodingErrorDistribution(3)
        with pytest.raises(
                ValueError,
                match=(f"Event frequency = {negative_int} must be non-negative.")):
            empirical_decoding_error_distribution.add_event(
                (True, False, False), negative_int)

    @pytest.mark.parametrize("bad_tuple", [
        (True,),
        (True, False, False, True)
    ])
    def test_error_is_raised_if_index_tuple_is_incorrect_length(self, bad_tuple):
        distr = EmpiricalDecodingErrorDistribution(3)
        with pytest.raises(TypeError,
                           match="EmpiricalDecodingErrorDistribution index tuples must be of length "
                           f"{distr.number_of_logicals}, not {len(bad_tuple)}"):
            distr[bad_tuple]

    @pytest.mark.parametrize("bad_index", [
        "bad",
        0.0,
    ])
    def test_error_is_raised_if_index_is_not_int_or_bool_tuple(self, bad_index):
        distr = EmpiricalDecodingErrorDistribution(3)
        with pytest.raises(TypeError,
                           match="EmpiricalDecodingErrorDistribution indices "
                           r"must be integers or Tuple\[bool\], not .*"):
            distr[bad_index]

    @pytest.mark.parametrize("error_distribution_dict", [
        (
            {(True, False, False): 5,
             (False, True, False): 6,
             (False, False, True): 9}),
        (
            {(True, False, False): 5,
             (True, True, True): 6,
             (False, False, True): 9}),
        (
            {(False,): 0})
    ])
    def test_from_dict_gives_expected_frequency_from_bool_tuple(self,
                                                                error_distribution_dict: Dict[Tuple[bool, ...], int]):
        distr = EmpiricalDecodingErrorDistribution.from_dict(error_distribution_dict)
        for key, val in error_distribution_dict.items():
            assert distr[key] == val

    def test_get_by_index_gives_expected_frequency(self):
        distr_dict = {(True, False, False): 5,
                      (False, True, False): 6,
                      (False, False, False): 9}

        distr = EmpiricalDecodingErrorDistribution.from_dict(distr_dict)

        assert distr[0] == 9
        assert distr[2] == 6
        assert distr[1] == 5

        nonzero_entries = {0, 1, 2}
        for i in range(len(distr)):
            if i not in nonzero_entries:
                assert distr[i] == 0

    def test_record_error_adds_correct_error(self):
        distr = EmpiricalDecodingErrorDistribution(3)

        distr.record_error((True, True, False), (False, True, False))
        distr.record_error((True, True, False), (False, True, False))
        distr.record_error((True, False, False), (False, True, False))
        distr.record_error((False, False, True), (False, True, False))

        assert distr[(True, False, False)] == 2
        assert distr[(True, True, False)] == 1
        assert distr[6] == 1

        nonzero_entries = {1, 3, 6}
        for i in range(len(distr)):
            if i not in nonzero_entries:
                assert distr[i] == 0

    def test_add_two_distributions(self):
        distr1 = EmpiricalDecodingErrorDistribution(3)
        distr1.add_event((False,True, False),5)

        distr1.add_event((False,True, True),1)

        distr2 = EmpiricalDecodingErrorDistribution(3)
        distr2.add_event((True,True,False),4)
        distr2.add_event((False,True, True),2)

        result = distr1 + distr2

        assert result[(True,True,False)] == 4
        assert result[(False,True, False)] == 5
        assert result[(False,True, True)] == 3

        nonzero_entries = {3, 2, 6}
        for i in range(len(distr2)):
            if i not in nonzero_entries:
                assert result[i] == 0

    @pytest.mark.parametrize("num_logicals", [5, 1, 13])
    def test_addition_is_commutative(self, num_logicals: int):
        distr1 = EmpiricalDecodingErrorDistribution(num_logicals)
        distr2 = EmpiricalDecodingErrorDistribution(num_logicals)

        for _ in range(10):
            distr1.add_event(self._generate_random_bool_tuple(num_logicals),random.randint(1, 99))
            distr2.add_event(self._generate_random_bool_tuple(num_logicals),random.randint(1, 99))

        self._distribution_data_is_equal(distr1 + distr2,distr2 + distr1)

    @pytest.mark.parametrize("num_logicals", [5, 1, 13])
    def test_addition_is_associative(self, num_logicals: int):
        distr1 = EmpiricalDecodingErrorDistribution(num_logicals)
        distr2 = EmpiricalDecodingErrorDistribution(num_logicals)
        distr3 = EmpiricalDecodingErrorDistribution(num_logicals)

        for _ in range(10):
            distr1.add_event(self._generate_random_bool_tuple(num_logicals),random.randint(1, 99))
            distr2.add_event(self._generate_random_bool_tuple(num_logicals),random.randint(1, 99))
            distr3.add_event(self._generate_random_bool_tuple(num_logicals),random.randint(1, 99))

        self._distribution_data_is_equal(distr1 + (distr2 + distr3),(distr1 + distr2)+ distr3)


    @pytest.mark.parametrize("num_logicals", [5, 1, 13])
    def test_add_empty_distribution_is_identity(self, num_logicals: int):
        distr1 = EmpiricalDecodingErrorDistribution(num_logicals)
        for _ in range(10):
            distr1.add_event(self._generate_random_bool_tuple(num_logicals),random.randint(1, 99))

        self._distribution_data_is_equal(distr1 + EmpiricalDecodingErrorDistribution(num_logicals), distr1)

    def test_batch_record_adds_correct_errors(self):
        target = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [0, 1, 1]])

        corrections = np.array([[1, 1, 0],
                              [0, 1, 1],
                              [1, 0, 1]])

        distr = EmpiricalDecodingErrorDistribution(3)

        distr.batch_record_errors(corrections, target)
        print(distr.to_dict())

        assert distr[4] == 1
        assert distr[(True, True, False)] == 2
        nonzero_entries = {3, 4}
        for i in range(len(distr)):
            if i not in nonzero_entries:
                assert distr[i] == 0

    @pytest.mark.parametrize("event_length", [5, 1])
    def test_batch_record_equivalent_to_singly_record_error(self, event_length: int):
        distr1 = EmpiricalDecodingErrorDistribution(event_length)
        distr2 = EmpiricalDecodingErrorDistribution(event_length)

        # Generated random error data and singly record
        targets = []
        corrections = []
        for _ in range(1000):
            target = self._generate_random_bool_tuple(event_length)
            correction = self._generate_random_bool_tuple(event_length)
            targets.append(target)
            corrections.append(correction)
            distr1.record_error(correction, target)

        # Convert to 2d numpy ndarray for batch operations
        distr2.batch_record_errors(np.array(corrections).astype(np.uint8),
                                   np.array(targets).astype(np.uint8))

        assert len(distr1) == len(distr2)
        for error in range(len(distr1)):
            assert distr1[error] == distr2[error]

    @pytest.mark.parametrize("expected_errors_per_logical, error_distribution_dict", [
        ([1, 3, 2],
         {(True, False, False): 1,
         (False, True, False): 3,
         (False, False, True): 2}),
        ([12, 7, 21],
         {(True, False, True): 5,
         (True, True, True): 7,
         (False, False, True): 9}),
        ([0],
         {(False,): 0})
    ])
    def test_errors_per_logical_matches_expected(
            self, expected_errors_per_logical, error_distribution_dict):
        distr = EmpiricalDecodingErrorDistribution.from_dict(error_distribution_dict)
        assert np.array_equal(distr.fails_per_logical, expected_errors_per_logical)

    @pytest.mark.parametrize("expected_error_counts, error_distribution_dict", [
        ([5, 6, 9],
         {(True, False, False): 5,
         (False, True, False): 6,
         (False, False, True): 9}),
        ([11, 6, 15],
         {(True, False, False): 5,
         (True, True, True): 6,
         (False, False, True): 9}),
        ([0],
         {(False,): 0})
    ])
    def test_number_of_failures_on_a_given_logical_can_be_computed_from_dict(
            self, expected_error_counts, error_distribution_dict):
        distr = EmpiricalDecodingErrorDistribution.from_dict(error_distribution_dict)
        for i, expected_error_count in enumerate(expected_error_counts):
            assert expected_error_count == distr.get_num_errors_on_logical(
                logical=i)

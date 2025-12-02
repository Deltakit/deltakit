# (c) Copyright Riverlane 2020-2025.
from unittest.mock import Mock

import pytest
from deltakit_decode.analysis._matching_decoder_managers import \
    GraphDecoderManager


class TestGraphDecoderManager:

    @pytest.fixture()
    def logicals(self):
        return [set(range(10)), {1}]

    @pytest.fixture()
    def mock_graph_decoder(self, mocker, logicals) -> Mock:
        decode_to_logical_flip = mocker.Mock(return_value=(True,))
        logicals = mocker.PropertyMock(return_value=logicals)
        decoder: Mock = mocker.Mock()
        decoder.attach_mock(decode_to_logical_flip, "_decode_to_logical_flip")
        type(decoder).logicals = logicals
        return decoder

    @pytest.fixture()
    def mock_noise_model(self, mocker) -> Mock:
        noise_model: Mock = mocker.Mock()
        return noise_model

    @pytest.fixture()
    def example_graph_decoder_manager(self, mock_noise_model, mock_graph_decoder):
        return GraphDecoderManager(mock_noise_model, mock_graph_decoder)

    @pytest.mark.parametrize("error, expected_logical_flip", [
        (set(), (False, False)),
        ({1}, (True, True)),
        ({1, 2}, (False, True)),
        ({3}, (True, False)),
    ])
    def test_analyse_correction_success(self,
                                        example_graph_decoder_manager: GraphDecoderManager,
                                        error,
                                        expected_logical_flip):
        assert not example_graph_decoder_manager._analyse_correction(
            error, expected_logical_flip)

    @pytest.mark.parametrize("error, expected_logical_flip", [
        (set(), (False, True)),
        ({1}, (True, False)),
        ({1, 2}, (True, True)),
        ({3}, (False, False)),
    ])
    def test_analyse_correction_fail(self,
                                     example_graph_decoder_manager: GraphDecoderManager,
                                     error,
                                     expected_logical_flip):
        assert example_graph_decoder_manager._analyse_correction(
            error, expected_logical_flip)

    @pytest.mark.parametrize("logicals, error, expected_logical_flip", [
        ([{0}], {0}, (True,)),
        ([{0}], {1}, (False,)),
        ([{0}, {1}], {1}, (False, True))
    ])
    def test_decoder_manager_uses_different_logicals_if_given(
        self,
        mock_noise_model: Mock,
        mock_graph_decoder: Mock,
        logicals: list[set[int]],
        error: set[int],
        expected_logical_flip: tuple[bool, ...]
    ):
        decoder_manager: GraphDecoderManager = GraphDecoderManager(
            mock_noise_model, mock_graph_decoder, logicals=logicals)
        assert not decoder_manager._analyse_correction(error, expected_logical_flip)

import os

import deltakit_circuit
import numpy as np
import pytest
import stim
from deltakit_explorer import Client, types, _api
from deltakit_explorer._cloud_decoders import (ACDecoder,
                                               BeliefMatchingDecoder,
                                               BPOSDecoder, CCDecoder,
                                               LCDecoder, MWPMDecoder,
                                               _CloudDecoder)


class TestCloudDecoder:

    def setup_method(self):
        # make sure we have a token specified. If there is not
        # token in the environment, client instance cannot be constructed.
        # This is by design.
        os.environ[_api._auth.TOKEN_VARIABLE] = "fake token"

    def teardown_method(self):
        os.environ.pop(_api._auth.TOKEN_VARIABLE)

    @pytest.mark.parametrize(
        "decoder_class",
        [
            _CloudDecoder, MWPMDecoder, ACDecoder,
            CCDecoder, LCDecoder, BPOSDecoder, BeliefMatchingDecoder
        ]
    )
    def test_raises_with_no_client(self, decoder_class):
        with pytest.raises(NotImplementedError, match="a `client` must be provided"):
            client = None
            decoder_class("", client=client)

    @pytest.mark.parametrize(
        "circuit",
        [
            "M 0 1",
            stim.Circuit("M 0 1"),
            deltakit_circuit.Circuit.from_stim_circuit(
                stim.Circuit("M 0 1")
            ),
        ]
    )
    @pytest.mark.parametrize(
        "decoder_class",
        [
            _CloudDecoder, MWPMDecoder, ACDecoder,
            CCDecoder, LCDecoder, BPOSDecoder, BeliefMatchingDecoder
        ]
    )
    def test_accept_formats_and_converts_to_string(self, decoder_class, circuit):
        client = Client("http://localhost")
        decoder = decoder_class(circuit, client=client)
        # every format should be converted to string eventually
        assert decoder.text_circuit.strip() == "M 0 1"

    @pytest.mark.parametrize(
        "circuit",
        [
            "M 0 1",
            stim.Circuit("M 0 1"),
            deltakit_circuit.Circuit.from_stim_circuit(
                stim.Circuit("M 0 1")
            ),
        ]
    )
    def test_cloud_decoder_raises_with_no_observables(self, circuit):
        client = Client("http://localhost")
        with pytest.raises(ValueError, match="Circuit must have at least one observable"):
            decoder = _CloudDecoder(circuit, client=client)
            decoder.decode_batch_to_logical_flip(
                np.array([[0, 1], [1, 0]], dtype=np.uint8)
            )

    @pytest.mark.parametrize(
        "decoder_class",
        [
            MWPMDecoder, ACDecoder, CCDecoder,
            LCDecoder, BPOSDecoder, BeliefMatchingDecoder,
        ]
    )
    def test_decoder_batch_to_logical_flip(self, decoder_class, mocker):
        circuit = stim.Circuit("M 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]")
        client = Client("http://localhost")
        result = types.DecodingResult(
            # some 01-compatible string
            predictionsFile={"uid": "duck://310a300a"},
            fails=1,
            shots=2,
            times=[],
            counts=[2],
        )
        mocker.patch.object(client, "decode", return_value=result)
        decoder = decoder_class(circuit, client=client)
        result = decoder.decode_batch_to_logical_flip(
            np.array([[0, 1], [1, 0]], dtype=np.uint8)
        )
        # two shots, one bit each
        assert result.shape == (2, 1)
        assert np.allclose(result, np.array([[1], [0]]))
        client.decode.assert_called_once()

    def test_lc_decoder_batch_to_logical_flip(self, mocker):
        circuit = "SOME NON_STIM TEXT"
        client = Client("http://localhost")
        result = types.DecodingResult(
            # some 01-compatible string
            predictionsFile={"uid": "duck://310a300a"},
            fails=1,
            shots=1000,
            times=[],
            counts=[1000],
        )
        mocker.patch.object(client, "decode", return_value=result)
        decoder = LCDecoder(circuit, num_observables=1, client=client)
        result = decoder.decode_batch_to_logical_flip(
            np.array([[0, 1], [1, 0]], dtype=np.uint8)
        )
        # two shots, one bit each
        assert result.shape == (2, 1)
        assert np.allclose(result, np.array([[1], [0]]))
        client.decode.assert_called_once()

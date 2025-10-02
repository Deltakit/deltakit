from typing import TYPE_CHECKING, Any, Optional

import deltakit_circuit
import numpy as np
import numpy.typing as npt
import stim

if TYPE_CHECKING:
    from deltakit_core.decoding_graphs import OrderedDecodingEdges, OrderedSyndrome

from deltakit_explorer import Client, enums, types


class _CloudDecoder:
    """Decoder (Cloud-based).

    Parameters
    ----------
    circuit : deltakit_circuit.Circuit | stim.Circuit | str
        Circuit to use to construct the decoder.
    parameters : Optional[dict[str, Any]]
        Some decoders may require additional parameters. Please refer
        to decoder documentation.
    use_experimental_graph_method : bool
        If enabled, the decoder tries to extract noise model from the
        experimental data.
    client : Optional[Client]
        Client with which to perform operations.

    Raises
    ------
    NotImplementedError
        When a method or attribute other than `decode_batch_to_logical_flip` is
        accessed.

    Notes
    -----
    Currently, only `decode_batch_to_logical_flip` is implemented; this class can be
    used with `deltakit.decode.analysis.run_decoding_on_circuit`, but not other
    decoder workflows.

    """

    _decoder_type: enums.DecoderType = enums.DecoderType.MWPM

    def __init__(
        self,
        circuit: deltakit_circuit.Circuit | stim.Circuit | str,
        parameters: Optional[dict[str, Any]] = None,
        use_experimental_graph_method: bool = False,
        client: Optional[Client] = None,
    ):
        # some leakage-aware decoders will need to hint this value,
        # if they are using non-standard STIM extension
        self.num_observables = 0
        self.decoder_parameters = {} if parameters is None else parameters
        self.use_experimental_graph = use_experimental_graph_method
        self.stim_circuit: Optional[stim.Circuit] = None
        # to communicate to the server and to support leakage,
        # circuit must be a string
        if isinstance(circuit, deltakit_circuit.Circuit):
            circuit = circuit.as_stim_circuit()
        if isinstance(circuit, stim.Circuit):
            self.stim_circuit = circuit
            self.num_observables = self.stim_circuit.num_observables
            circuit = str(circuit)
        self.text_circuit = circuit
        if client is None:
            raise NotImplementedError(
                "Currently, a `client` must be provided to instantiate this class."
            )
        self.client = client

    def decode_batch_to_logical_flip(
            self,
            syndrome_batch: npt.NDArray[np.uint8],
            leakage_batch: Optional[npt.NDArray[np.uint8]] = None,
        ):
        """The method decodes the batch of syndromes to boolean values."""
        num_shots = syndrome_batch.shape[0]
        detectors = types.DetectionEvents(syndrome_batch)
        if self.num_observables < 1:
            raise ValueError(
                "Circuit must have at least one observable. "
                "Please make sure your circuit has observables or provide "
                f"`num_observables` when instantiating an `{self.__class__.__name__}`."
            )
        observables = types.ObservableFlips(
            np.zeros((num_shots, self.num_observables), dtype=syndrome_batch.dtype)
        )
        leakage = None
        if leakage_batch is not None:
            leakage = types.LeakageFlags(leakage_batch)
        decoder = types.Decoder(
            decoder_type=self._decoder_type,
            use_experimental_graph=self.use_experimental_graph,
            parameters=self.decoder_parameters
        )
        decoding_result = self.client.decode(
            detectors=detectors,
            observables=observables,
            decoder=decoder,
            noisy_stim_circuit=self.text_circuit,
            leakage_flags=leakage,
        )
        return decoding_result.predictions.as_numpy()

    # use inherited `decode_to_boolean` and `decode_batch_to_full_correction`;
    # they will raise `NotImplementedError`, too.

    def decode_to_full_correction(self, syndrome: 'OrderedSyndrome') -> 'OrderedDecodingEdges':
        raise NotImplementedError()

    @property
    def logicals_edge_list(self) -> list[list[int]]:
        raise NotImplementedError()


class MWPMDecoder(_CloudDecoder):
    """Minimum Weight Perfect Matching Decoder (Cloud-based).

    Parameters
    ----------
    circuit : deltakit_circuit.Circuit | stim.Circuit | str
        Circuit to use to construct the decoder.
    parameters : Optional[dict[str, Any]]
        Not used in MWPM.
    use_experimental_graph_method : bool
        If enabled, the decoder tries to extract noise model from the
        experimental data.
    client : Optional[Client]
        Client with which to perform operations.

    Raises
    ------
    NotImplementedError
        When a method or attribute other than `decode_batch_to_logical_flip` is
        accessed.

    Notes
    -----
    Currently, only `decode_batch_to_logical_flip` is implemented; this class can be
    used with `deltakit.decode.analysis.run_decoding_on_circuit`, but not other
    decoder workflows.

    This cloud-based decoder implements Minimum Weight Perfect Matching
    (https://arxiv.org/abs/2303.15933).
    """
    _decoder_type = enums.DecoderType.MWPM


class CCDecoder(_CloudDecoder):
    """Collision Clustering Decoder (Cloud-based).

    Parameters
    ----------
    circuit : deltakit_circuit.Circuit | stim.Circuit | str
        Circuit to use to construct the decoder.
    parameters : Optional[dict[str, Any]]
        Not used in CC.
    use_experimental_graph_method : bool
        If enabled, the decoder tries to extract noise model from the
        experimental data.
    client : Optional[Client]
        Client with which to perform operations.

    Raises
    ------
    NotImplementedError
        When a method or attribute other than `decode_batch_to_logical_flip` is
        accessed.

    Notes
    -----
    Currently, only `decode_batch_to_logical_flip` is implemented; this class can be
    used with `deltakit.decode.analysis.run_decoding_on_circuit`, but not other
    decoder workflows.

    This cloud-based decoder implements Collision Clustering
    (https://arxiv.org/abs/2309.05558).
    """
    _decoder_type = enums.DecoderType.CC


class BeliefMatchingDecoder(_CloudDecoder):
    """Belief Matching Decoder (Cloud-based).

    Parameters
    ----------
    circuit : deltakit_circuit.Circuit | stim.Circuit | str
        Circuit to use to construct the decoder.
    parameters : Optional[dict[str, Any]]
        Not used in Belief Matching.
    use_experimental_graph_method : bool
        If enabled, the decoder tries to extract noise model from the
        experimental data.
    client : Optional[Client]
        Client with which to perform operations.

    Raises
    ------
    NotImplementedError
        When a method or attribute other than `decode_batch_to_logical_flip` is
        accessed.

    Notes
    -----
    Currently, only `decode_batch_to_logical_flip` is implemented; this class can be
    used with `deltakit.decode.analysis.run_decoding_on_circuit`, but not other
    decoder workflows.

    This cloud-based decoder implements Belief Matching
    (https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.031007).
    """
    _decoder_type = enums.DecoderType.BELIEF_MATCHING


class BPOSDecoder(_CloudDecoder):
    """Belief Propagation - Ordered Statistics Decoder (Cloud-based).

    Parameters
    ----------
    circuit : deltakit_circuit.Circuit | stim.Circuit | str
        Circuit to use to construct the decoder.
    parameters : Optional[dict[str, Any]]
        - `max_bp_rounds` is an integer, and specifies the maximum number
          of iterations of message passing that should be performed during
          the execution of belief propagation.
          It may terminate earlier. By default, this is 20.
        - `combination_sweep_order` is the depth of the OSD search.
    use_experimental_graph_method : bool
        If enabled, the decoder tries to extract noise model from the
        experimental data.
    client : Optional[Client]
        Client with which to perform operations.

    Raises
    ------
    NotImplementedError
        When a method or attribute other than `decode_batch_to_logical_flip` is
        accessed.

    Notes
    -----
    Currently, only `decode_batch_to_logical_flip` is implemented; this class can be
    used with `deltakit.decode.analysis.run_decoding_on_circuit`, but not other
    decoder workflows.

    This cloud-based decoder implements Belief Propagation - Ordered Statistics Decoding
    (https://quantum-journal.org/papers/q-2021-11-22-585/).
    """
    _decoder_type = enums.DecoderType.BP_OSD


class ACDecoder(_CloudDecoder):
    """Ambiguity Clustering Decoder (Cloud-based).

    Parameters
    ----------
    circuit : deltakit_circuit.Circuit | stim.Circuit | str
        Circuit to use to construct the decoder.
    parameters : Optional[dict[str, Any]]
        - `bp_rounds` is an integer, and specifies how many iterations
          of message passing should be performed during the execution
          of belief propagation. Note that bp_rounds in AC is different
          `from max_bp_rounds` in BP_OSD as early termination is not
          allowed. Typically, setting this equal to the distance of
          the code is sufficient. By default, this is 20.
        - `ac_kappa_proportion` is a float, between 0.0 and 1.0, and
          reflects the number of error mechanisms, in addition to those
          used to find a first solution, that should be used to grow
          clusters to search for additional solutions, expressed as a
          proportion of the total number of error mechanisms. Setting
          this number higher results in better accuracy at the cost of slower
          performance. Start with 0.0 and increase by 0.01 until the desired
          accuracy is reached. Reasonable values lie between 0 and 0.1, as
          larger values will typically lead to a significant slow down.
          By default, this is 0.01.
    use_experimental_graph_method : bool
        If enabled, the decoder tries to extract noise model from the
        experimental data.
    client : Optional[Client]
        Client with which to perform operations.

    Raises
    ------
    NotImplementedError
        When a method or attribute other than `decode_batch_to_logical_flip` is
        accessed.

    Notes
    -----
    Currently, only `decode_batch_to_logical_flip` is implemented; this class can be
    used with `deltakit.decode.analysis.run_decoding_on_circuit`, but not other
    decoder workflows.

    This cloud-based decoder implements Ambiguity Clustering
    (https://arxiv.org/abs/2406.14527).
    """
    _decoder_type = enums.DecoderType.AC


class LCDecoder(_CloudDecoder):
    """Local Clustering Decoder (Cloud-based).

    Parameters
    ----------
    circuit : deltakit_circuit.Circuit | stim.Circuit | str
        Circuit to use to construct the decoder.
    parameters : Optional[dict[str, Any]]
        Some decoders may require additional parameters. Please refer
        to decoder documentation.
    use_experimental_graph_method : bool
        If enabled, the decoder tries to extract noise model from the
        experimental data.
    client : Optional[Client]
        Client with which to perform operations.
    num_observables : Optional[int]
        If provided, sets the number of observables in the decoder.
        If not provided, the number of observables is inferred from
        the circuit (if possible). This parameter is useful when
        the circuit uses non-standard STIM extensions to represent
        observables.

    Raises
    ------
    NotImplementedError
        When a method or attribute other than `decode_batch_to_logical_flip` is
        accessed.

    Notes
    -----
    Currently, only `decode_batch_to_logical_flip` is implemented; this class can be
    used with `deltakit.decode.analysis.run_decoding_on_circuit`, but not other
    decoder workflows.

    This cloud-based decoder implements the Local Clustering Decoder algorithm
    (https://arxiv.org/abs/2411.10343).
    """
    _decoder_type = enums.DecoderType.LCD


    def __init__(
        self,
        circuit: deltakit_circuit.Circuit | stim.Circuit | str,
        parameters: Optional[dict[str, Any]] = None,
        use_experimental_graph_method: bool = False,
        client: Optional[Client] = None,
        num_observables: Optional[int] = None,
    ):
        """Local Clustering Decoder (Cloud-based).

        Parameters
        ----------
        circuit : deltakit_circuit.Circuit | stim.Circuit | str
            Circuit to use to construct the decoder.
        parameters : Optional[dict[str, Any]]
            Some decoders may require additional parameters. Please refer
            to decoder documentation.
        use_experimental_graph_method : bool
            If enabled, the decoder tries to extract noise model from the
            experimental data.
        client : Optional[Client]
            Client with which to perform operations.
        num_observables : Optional[int]
            If provided, sets the number of observables in the decoder.
            If not provided, the number of observables is inferred from
            the circuit (if possible). This parameter is useful when
            the circuit uses non-standard STIM extensions to represent
            observables.

        """
        super().__init__(
            circuit,
            parameters,
            use_experimental_graph_method,
            client,
        )
        if num_observables is not None:
            self.num_observables = num_observables

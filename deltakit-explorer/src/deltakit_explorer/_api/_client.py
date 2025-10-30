# (c) Copyright Riverlane 2020-2025.
"""This file contains implementation of an API for accessing
the Deltakit service.
"""

from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt
import stim
from deltakit_explorer import simulation
from deltakit_explorer._api._api_client import APIClient
from deltakit_explorer._api._api_v2_client import APIv2Client
from deltakit_explorer._api._auth import TOKEN_VARIABLE
from deltakit_explorer._api._gql_client import GQLClient
from deltakit_explorer._utils import _utils as utils
from deltakit_explorer._utils._decorators import (
    validate_and_split_decoding, validate_and_split_simulation,
    validate_generation)
from deltakit_explorer._utils._logging import Logging
from deltakit_explorer.data._data_analysis import has_leakage
from deltakit_explorer.types._experiment_types import (QECExperiment,
                                                       QECExperimentDefinition)
from deltakit_explorer.types._types import (BinaryDataType, Decoder,
                                            DecodingResult, DetectionEvents,
                                            LeakageFlags, Measurements,
                                            ObservableFlips,
                                            PhysicalNoiseModel,
                                            QubitCoordinateToDetectorMapping,
                                            SI1000NoiseModel)
from deltakit_explorer.qpu._noise import SI1000Noise


# pylint: disable=too-many-locals,unsubscriptable-object
class Client:
    """The `Client` class provides convenient methods to access circuit generation,
    noise addition, decoding, simulation, and analysis calls. To create a connection
    to a particular server, use the following syntax::

        client = Client(
            base_url="https://deltakit.riverlane.com/proxy"
        )

    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 60000,
        api_version: int = 2,
    ):
        """
        Args:
          base_url : str
              Url of the Dektakit service endpoint.
          timeout: int
              Request timeout, seconds.
          api_version: int
              1 - Long living GraphQL requests.
              2 - Task-based requests.
        """
        if api_version == 1:
            self._api: APIClient = GQLClient(base_url, timeout)
        elif api_version == 2:
            self._api = APIv2Client(base_url, timeout)
        else:
            raise NotImplementedError(f"API version {api_version} is not implemented.")
        self._api_version = api_version

    @staticmethod
    def get_instance(api_version: int=2) -> Client:
        """Return a pre-configured instance of a cloud-based Deltakit client.
        If you need to connect to a custom instance, please use `Client(base_url)`
        syntax.

        Args:
            api_version (int):
                version of API, GraphQL-based (1),
                or queue-based (2)

        Returns:
            Client: configured instance of a Deltakit API.
        """
        utils.load_environment_variables_from_drive()
        server_name = os.environ.get(
            utils.DELTAKIT_SERVER_URL_ENV,
            default=utils.DELTAKIT_SERVER_DEFAULT_URL_ENV,
        )
        return Client(base_url=server_name, api_version=api_version)

    @classmethod
    def set_token(cls, token: str, validate: bool = True):
        """Set and validate a token."""
        # NB: .get_instance() call fails if there is no token provided
        # as an env variable, or on a drive. For cold start, we supply
        # the variable first:
        os.environ[TOKEN_VARIABLE] = token
        return cls.get_instance()._api.set_token(token, validate)

    def add_noise(
        self,
        stim_circuit: str | stim.Circuit,
        noise_model: PhysicalNoiseModel | SI1000NoiseModel | SI1000Noise,
    ) -> str:
        """Given a stim circuit, changes all noise in the circuit to noise
        defined by user-given parameters. If the circuit is noiseless,
        noise is added into the circuit.

        Args:
            stim_circuit: (str | stim.Circuit):
                Noiseless circuit.
            noise_model (PhysicalNoiseModel | SI1000NoiseModel | SI1000Noise):
                Noise model to apply to a circuit.

        Returns:
            str:
                STIM circuit with error mechanisms derived from the
                noise model. Note, that adding SI1000NoiseModel may
                add leakage terms, which are not compatible with vanilla
                STIM, and can only be simulated using Deltakit client.

        Examples:
            Populate the circuit with the noise, and then simulate::

                leakage_noise_model = types.SI1000NoiseModel(p=0.001, p_l=0.001)
                noisy_circuit = client.add_noise(
                    stim_circuit=compiled_circuit,
                    noise_model=leakage_noise_model,
                )
                measurements, leakage = client.simulate_stim_circuit(
                    stim_circuit=noisy_circuit,
                    shots=num_shots,
                )

        """
        query_id = Logging.info_and_generate_uid(locals())
        try:
            if isinstance(noise_model, SI1000Noise):
                noise_model = SI1000NoiseModel(p=noise_model.p, p_l=noise_model.pL)
            return self._api.add_noise(stim_circuit, noise_model, query_id)
        except Exception as ex:
            Logging.error(ex, query_id)
            raise

    @validate_and_split_decoding
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def decode(
        self,
        detectors: DetectionEvents,
        observables: ObservableFlips,
        decoder: Decoder,
        noisy_stim_circuit: str | stim.Circuit,
        leakage_flags: LeakageFlags | None = None,
    ) -> DecodingResult:
        """Accepts detectors, observables and a stim circuit with integrated noise
        and returns decoding results for a requested decoder.

        Args:
            detectors (DetectionEvents):
                Syndrome data object.
            observables (ObservableFlips):
                Observables data.
            decoder (Decoder):
                Definition of decoder and its features.

                - `MWPM` for Minimum-Weight Perfect Matching
                  (https://arxiv.org/abs/2303.15933),
                - `CC` for Collision Clustering (https://arxiv.org/abs/2309.05558), or
                - `BELIEF_MATCHING` for Belief Matching
                  (https://arxiv.org/abs/2203.04948).
                - `BP_OSD` for Belief Propagation - Ordered Statistics Decoding
                  (BP-OSD) (https://quantum-journal.org/papers/q-2021-11-22-585/),
                - `AC` for Ambiguity Clustering (https://arxiv.org/abs/2406.14527).
                - `LCD` for Local Clustering Decoder (https://arxiv.org/abs/2411.10343).

            noisy_stim_circuit (str | stim.Circuit):
                STIM circuit with defined noise terms. Used to derive detector
                error model.

            leakage_flags (Optional[LeakageFlags]):
                Leakage information may be used by Local Clustering Decoder (LCD) to
                improve decoding quality.

        Returns:
            DecodingResult: Server decoding response.

        Examples:
            Decoding simulated data::

                measurements, leakage = client.simulate_stim_circuit(
                    stim_circuit=noisy_circuit,
                    shots=10000,
                )
                detectors, observables = measurements.to_detectors_and_observables(
                    stim_circuit=noisy_circuit,
                )
                decoding_result = client.decode(
                    detectors=detectors,
                    observables=observables,
                    decoder=types.Decoder(types.DecoderType.AC),
                    noisy_stim_circuit=noisy_circuit,
                )

        """
        query_id = Logging.info_and_generate_uid(locals())
        try:
            return self._api.decode(
                detectors,
                observables,
                decoder,
                noisy_stim_circuit,
                leakage_flags,
                query_id,
            )
        except Exception as ex:
            Logging.error(ex, query_id)
            raise

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def decode_measurements(
        self,
        measurements: Measurements,
        decoder: Decoder,
        ideal_stim_circuit: str | stim.Circuit,
        noise_model: PhysicalNoiseModel,
        leakage_flags: LeakageFlags | None = None,
        sweep_bits: BinaryDataType | None = None,
    ) -> DecodingResult:
        """Accept raw measurement file and a clean stim circuit and
        return decoding results with a requested decoder.

        Args:
            measurements (Measurements):
                Measurements object.
            decoder (Decoder):
                Predefined decoder object.

                - `MWPM` for Minimum-Weight Perfect Matching
                  (https://arxiv.org/abs/2303.15933),
                - `CC` for Collision Clustering (https://arxiv.org/abs/2309.05558), or
                - `BELIEF_MATCHING` for Belief Matching
                  (https://arxiv.org/abs/2203.04948).
                - `BP_OSD` for Belief Propagation - Ordered Statistics Decoding
                  (BP-OSD) (https://quantum-journal.org/papers/q-2021-11-22-585/),
                - `AC` for Ambiguity Clustering (https://arxiv.org/abs/2406.14527).
                - `LCD` for Local Clustering Decoder (https://arxiv.org/abs/2411.10343).

            ideal_stim_circuit: (str | stim.Circuit): clean stim circuit.
            noise_model (PhysicalNoiseModel):
                Noise model used to generate a circuit, or a noise model close
                to hardware, which has generated the measurements data.
            leakage_flags (Optional[LeakageFlags]):
                Heralded leakage events, if available.
            sweep_bits (Optional[BinaryDataType]):
                Initialisation bits, if were used in experiments.

        Returns:
            DecodingResult: server response.

        Examples:
            Decoding using experimental graph method::

                decoder = Decoder(
                    decoder_type=DecoderType.MWPM,
                    use_experimental_graph=True
                )
                client.decode_measurements(
                    measurements=measurements,
                    decoder=decoder,
                    ideal_stim_circuit=circuit,
                    noise_model=PhysicalNoiseModel.get_floor_superconducting_noise(),
                    sweep_bits=sweep_bits,
                )

            Decoding with AC with additional parameters::

                decoder = types.Decoder(
                    decoder_type=types.DecoderType.AC,
                    parallel_jobs=8,
                    parameters={
                        "decompose_errors": False,
                        "bp_rounds": 99,
                        "ac_kappa_proportion": 0.02
                    }
                )
                client.decode_measurements(
                    measurements=measurements,
                    decoder=decoder,
                    ideal_stim_circuit=compiled_circuit,
                    noise_model=noise_model,
                )

        """
        noisy_stim_circuit = self.add_noise(
            stim_circuit=ideal_stim_circuit,
            noise_model=noise_model,
        )
        noisy_stim_file_content = noisy_stim_circuit
        dets, obs = measurements.to_detectors_and_observables(
            stim_circuit=noisy_stim_file_content,
            sweep_bits=sweep_bits,
        )
        return self.decode(
            detectors=dets,
            observables=obs,
            decoder=decoder,
            noisy_stim_circuit=noisy_stim_circuit,
            leakage_flags=leakage_flags,
        )

    def defect_rates(
        self,
        detectors: DetectionEvents,
        stim_circuit: str | stim.Circuit,
    ) -> dict[tuple[float, ...], list[float]]:
        """Obtain defect rates for the detectors
        defined in the stim circuit.

        Args:
            detectors (DetectionEvents):
                Detectors data.
            stim_circuit (str | stim.Circuit):
                STIM circuit content.

        Returns:
            Dict[Tuple[float, ...], List[float]]:
                Keys are detectors coordinates, values are
                their defect rates.

        Examples:
            Obtaining a defect rate list for a single coordinate::

                coordinate = (2.0, 5.0)
                defect_rates = client.defect_rates(detectors, stim_circuit)
                print(f"{coordinate}: {defect_rates[coordinate]}")

        """
        query_id = Logging.info_and_generate_uid(locals())
        try:
            return self._api.defect_rates(
                detectors, stim_circuit, query_id,
            )
        except Exception as ex:
            Logging.error(ex, query_id)
            raise

    def get_correlation_matrix_for_trimmed_data(
        self,
        detectors: DetectionEvents,
        noise_floor_circuit: str | stim.Circuit,
        use_default_noise_model_edges: bool = False,
    ) -> tuple[npt.NDArray[np.float64], QubitCoordinateToDetectorMapping]:
        """Obtain a correlation matrix for a given set of detection events
        and corresponding noisy circuit. Also returns a qubit-detector mapping
        that can be used to plot detectors in groups according to the qubit
        they correspond to. This method may be used separately, but we recommend
        considering the simpler method Client.get_correlation_matrix, which calls it.

        Args:
            detectors (DetectionEvents):
                Detectors data.
            noise_floor_circuit (str | stim.Circuit):
                Cricuit with defined minimal noise. For example, this noise may
                be obtained with
                PhysicalNoiseModel.get_floor_superconducting_noise().
            use_default_noise_model_edges (bool):
                If set to True, uses noise edges defined in the circuit, otherwise
                derives noise from detectors data.

        Returns:
            Tuple[npt.NDArray, QubitCoordinateToDetectorMapping]:
                A matrix with correlation values,
                and a mapping of coordinates to detectors.

        """
        query_id = Logging.info_and_generate_uid(locals())
        try:
            return self._api.get_correlation_matrix_for_trimmed_data(
                detectors, noise_floor_circuit, use_default_noise_model_edges, query_id,
            )
        except Exception as ex:
            Logging.error(ex, query_id)
            raise

    def get_correlation_matrix(
        self,
        detectors: DetectionEvents,
        stim_circuit: str | stim.Circuit,
        use_default_noise_model_edges: bool,
    ) -> tuple[npt.NDArray[np.float64], QubitCoordinateToDetectorMapping]:
        """Obtain a correlation matrix for a given set of detection events
        and corresponding circuit. Also returns a qubit-detector mapping
        that can be used to plot detectors in groups according to the qubit
        they correspond to. First trims the circuit and detector data.

        Args:
            detectors (DetectionEvents):
                Detectors data.
            stim_circuit (str | stim.Circuit):
                Clean STIM circuit.
            use_default_noise_model_edges (bool):
                If set to True, uses noise edges defined in the circuit, otherwise
                derives noise from detectors data.

        Returns:
            Tuple[npt.NDArray, QubitCoordinateToDetectorMapping]:
                - The correlation matrix for the given data;
                - Mapping of a qubit (in coordinates) to the detectors it
                  corresponds to.

        Examples:
            Getting the matrix and plotting it::

                matrix, mapping = client.get_correlation_matrix(
                    detectors, stim_circuit,
                    use_default_noise_model_edges=True,
                )
                plt = visualisation.correlation_matrix(matrix, mapping)

        """
        query_id = Logging.info_and_generate_uid(locals())
        try:
            noisy_circuit = self.add_noise(
                stim_circuit=stim_circuit,
                noise_model=PhysicalNoiseModel.get_floor_superconducting_noise(),
            )
            (
                trimmed_circuit,
                trimmed_dets
            ) = self.trim_circuit_and_detectors(noisy_circuit, detectors)
            return self.get_correlation_matrix_for_trimmed_data(
                trimmed_dets, trimmed_circuit, use_default_noise_model_edges)
        except Exception as ex:
            Logging.error(ex, query_id)
            raise

    def trim_circuit_and_detectors(
        self,
        stim_circuit: str | stim.Circuit,
        detectors: DetectionEvents,
    ) -> tuple[str, DetectionEvents]:
        """Trims a circuit to remove redundant detectors (detectors that do not
        affect the observable). It then compares the trimmed circuit to the
        original to make a list of the removable detectors that are then used
        to also trim the detector data.

        Trimming a circuit:
            removing detectors that are not associated
            with any obervables.
        Trimming detector data:
            having removed detectors from the circuit
            as per the above definition, remove those same detectors
            from the given detector data.

        Args:
            stim_circuit (str | stim.Circuit):
                Noisy STIM circuit.
            detectors (DetectionEvents):
                Corresponding detectors.


        Returns:
            Tuple[str, DetectionEvents]
                Tuple containing trimmed circuit and
                detection events.

        Examples:
            Trim a disconnected circuit and related detection events,
            keeping only the part of it, connected to observables::

                trimmed_circuit, trimmed_dets = client.trim_circuit_and_detectors(
                    stim_circuit=noisy_circuit,
                    detectors=detectors,
                )


        """
        query_id = Logging.info_and_generate_uid(locals())
        try:
            return self._api.trim_circuit_and_detectors(
                stim_circuit, detectors, query_id,
            )
        except Exception as ex:
            Logging.error(ex, query_id)
            raise

    def get_experiment_detectors_and_defect_rates(
        self,
        experiment: QECExperiment,
    ) -> tuple[dict[int, tuple[float, ...]], dict[tuple[float, ...], list[float]]]:
        """Extract detector to coordinate mapping,
        and defect rates for these detectors.

        Args:
            experiment (QECExperiment):
                Experiment with measurements or detectors defined.

        Returns:
            Tuple[Dict[int, Tuple[float, ...]], Dict[Tuple[float, ...], List[float]]]:
                Two dictionaries: the first is a mapping from
                detector index to coordinates, the second is mapping
                from coordinates to defect rates per round.

        Examples:
            Defining experiment and extracting defect rate information::

                experiment = QECExperiment.from_circuit_and_measurements(
                    folder / "circuit_noisy.stim",
                    folder / "measurements.b8",
                    DataFormat.B8,
                )
                mapping, rates = client.get_experiment_detectors_and_defect_rates(
                    experiment
                )
        """
        if experiment.measurements is None and experiment.detectors is None:
            msg = (
                "Experiment object should have at least measurements or detectors. "
                "Provided object has neither."
            )
            raise ValueError(
                msg
            )
        experiment.compute_detectors_and_observables()
        # trim the circuit to remove redundant detectors
        trimmed_circuit, trimmed_dets = self.trim_circuit_and_detectors(
            stim_circuit=experiment.noisy_circuit,
            # detectors are computed above
            detectors=experiment.detectors,
        )
        # get defect rates for trimmed circuit
        all_qubit_defect_rates = self.defect_rates(
            detectors=trimmed_dets,
            stim_circuit=trimmed_circuit,
        )
        circuit = stim.Circuit(trimmed_circuit)
        all_detector_coordinates = {
            key: tuple(value) for key, value in
            circuit.get_detector_coordinates().items()
        }
        return all_detector_coordinates, all_qubit_defect_rates

    @validate_generation
    def generate_circuit(
        self,
        experiment_definition: QECExperimentDefinition,
    ) -> str:
        """Generate a STIM circuit for a quantum error correction experiment.
        `experiment_definition` holds all information essential for building
        an experiment.

        Args:
            experiment_definition (QECExperimentDefinition):
                Definition of an experiment.

                - experiment_type: Type of experiment, e.g. Quantum Memory or Stability.
                - code_type: QEC code, e.g. Rotated Planar Code.
                - observable_basis: Pauli Basis, in which observable is measured.
                  E.g. X or Z.
                - num_rounds: Number of experiment rounds.
                - basis_gates: If circuit is generated with a specific basis gate set.
                - parameters: Optional parameters of circuit generation.

        Returns:
            str: STIM circuit.

        Examples:

            Using shortcut experiment definition::

                compiled_circuit = client.generate_circuit(
                    QECExperimentDefinition.get_rotated_planar_z_quantum_memory(
                        distance, distance, ["CZ", "H", "MZ", "RZ"]
                    )
                )

            Using explicit definition::

                circuit = client.generate_circuit(
                    types.QECExperimentDefinition(
                        experiment_type=types.QECExperimentType.QUANTUM_MEMORY,
                        code_type=types.QECECodeType.BIVARIATE_BICYCLE,
                        observable_basis=PauliBasis.Z,
                        num_rounds=6,
                        basis_gates=["CZ", "H", "MZ", "RZ"],
                        parameters=types.CircuitParameters.from_matrix_specification(
                            param_l=6,
                            param_m=6,
                            m_A_powers=[3, 1, 2],
                            m_B_powers=[3, 1, 2],
                        )
                    )
                )

        """
        query_id = Logging.info_and_generate_uid(locals())
        try:
            return self._api.generate_circuit(experiment_definition, query_id)
        except Exception as ex:
            Logging.error(ex, query_id)
            raise

    @validate_and_split_simulation
    def simulate_stim_circuit(
        self,
        stim_circuit: str | stim.Circuit,
        shots: int,
    ) -> tuple[Measurements, LeakageFlags | None]:
        """
        Simulate STIM circuit with Deltakit client.

        Args:
            stim_circuit (str | stim.Circuit):
                Any STIM circuit. May include leakage instructions.
            shots (int):
                Number of shots.

        Returns:
            Tuple[Measurements, Optional[LeakageFlags]]:
                (Measurements, Leakage). Leakage may be None.

        Examples:
            Generate and simulate a circuit with leakage::

                compiled_circuit = client.generate_circuit(
                    types.QECExperimentDefinition.get_rotated_planar_z_quantum_memory(
                        distance=distance,
                        num_rounds=num_rounds,
                        basis_gates=["CZ", "H", "MZ", "RZ"],
                    )
                )
                # Add leakage noise
                leakage_noise_model = types.SI1000NoiseModel(p=0.001, p_l=0.001)
                noisy_circuit = client.add_noise(
                    stim_circuit=compiled_circuit,
                    noise_model=leakage_noise_model,
                )
                measurements, leakage = client.simulate_stim_circuit(
                    stim_circuit=noisy_circuit,
                    shots=num_shots,
                )

        """
        query_id = Logging.info_and_generate_uid(locals())
        try:
            # always simulate pure stim with no leakage locally
            if not has_leakage(str(stim_circuit)):
                return simulation.simulate_with_stim(stim_circuit, shots)
            return self._api.simulate_circuit(stim_circuit, shots, query_id)
        except Exception as ex:
            Logging.error(ex, query_id)
            raise

    def kill(self, request_id: str) -> int:
        """Kill a decoding task submitted to a client.
        Decoding methods report a request ID. Calling this
        method with an ID tries to kill all CPU
        workers associated with the task. Method returns the number
        of CPU workers it could terminate. If the task is already finished,
        the method should return 0.

        Args:
            request_id (str):
                ID of the requests which started the decoding task.

        Returns:
            int: number of workers killed
        """
        return self._api.kill(request_id)

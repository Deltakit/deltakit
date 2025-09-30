---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Decoding

In this guide, you will learn how to run a decoder on measurement results from a QEC experiment. To do this, you'll need to follow three steps:

1. Convert the raw measurement results into detector events and
   observable flips;
2. Provide the decoder with the circuit and noise model; and
3. Run the decoder on the detector measurements,
   and compare the results to the experimental outcomes.

Additionally, on this page, you'll learn how to tune your decoder parameters to improve its performance.

## Getting detector and observable results

If you run a QEC experiment in a simulator or on a real QPU,
your results are measurement bits.
Thus, you need to convert the measurement results into detectors and logical
observables.
Detectors are used to inform a decoder where errors have occurred in the quantum circuit,
and observables describe the quantum state.
The aim of a decoder is to use the detector measurements to determine if
errors have caused the observable measurement result to flip.

Detectors and observables can be computed for a given stim circuit and set of
measurement results using the {class}`deltakit.explorer.types.Measurements` class:

```{code-cell} ipython3
import stim
from deltakit.explorer import types
from deltakit.explorer.qpu import QPU, ToyNoise
from deltakit.explorer.codes import RepetitionCode, css_code_memory_circuit
from deltakit.circuit.gates import PauliBasis

repcode = RepetitionCode(distance=5)
noiseless_deltakit_circuit = css_code_memory_circuit(repcode, num_rounds=5, logical_basis=PauliBasis.Z)

qpu = QPU(qubits=noiseless_deltakit_circuit.qubits, noise_model=ToyNoise(p=0.03))
deltakit_circuit = qpu.compile_and_add_noise_to_circuit(noiseless_deltakit_circuit)
stim_circuit = deltakit_circuit.as_stim_circuit()

measurements = stim_circuit.compile_sampler().sample(10_000)

# convert them using deltakit
deltakit_measurements = types.Measurements(measurements)
detectors, observables = deltakit_measurements.to_detectors_and_observables(stim_circuit)

print("Measurements:", measurements.shape)
print("Detectors   :", detectors.as_numpy().shape)
print("Observables :", observables.as_numpy().shape)
```

The sweep file is an optional additional file containing sweep bits.
These are typically used to inform the code as to what states the qubits were
in at the start of each run of the circuit.
If not specified, it is assumed that all qubits were initialised
in the $\vert 0\rangle$ state.

```{code-cell} ipython3
import numpy as np

circuit_with_sweeps = """
CX sweep[0] 0 sweep[1] 2 sweep[2] 4
CZ 0 1 2 3
CZ 1 2 3 4
M 1 3
DETECTOR(1, 0) rec[-2]
DETECTOR(3, 0) rec[-1]
M 0 4
OBSERVABLE_INCLUDE(0) rec[-1]
OBSERVABLE_INCLUDE(1) rec[-2]
"""

sweeps_stim_circuit = stim.Circuit(circuit_with_sweeps)
measurements = np.zeros((1000, 4), dtype=np.uint8)

# assume these were initial states of your data qubits
sweeps = np.random.choice([0, 1], size=(1000, 3)).astype(np.uint8)

deltakit_measurements = types.Measurements(measurements)

# provide a circuit, and sweep bits
deltakit_measurements.to_detectors_and_observables(
    sweeps_stim_circuit,
    types.BinaryDataType(sweeps),
)
```

## Decoding from detectors

The most common way to decode your simulation results is to provide detector events to a decoder.
Please be attentive to the data you pass to decoders.
Check if the circuit has noise, and that the noise is relevant to the data.

### Decoding on your machine

Deltakit provides easy access to the MWPM decoder, which will run on your local machine.
Please refer to the documentation on {class}`PyMatchingDecoder <deltakit.decode.PyMatchingDecoder>`.

```{code-cell} ipython3
from deltakit.circuit import Circuit
from deltakit.decode import PyMatchingDecoder
from deltakit.explorer.types import DecodingResult

decoder, circuit = PyMatchingDecoder.construct_decoder_and_stim_circuit(deltakit_circuit)
predictions = decoder.decode_batch_to_logical_flip(
    syndrome_batch=detectors.as_numpy(),
)
mismatch = (predictions != observables.as_numpy())
fails = int(sum(np.any(mismatch, axis=1)))

DecodingResult(shots=predictions.shape[0], fails=fails)
```

### Decoding via the cloud

A broader selection of decoders is accessible via the cloud:

* [Minimum-Weight Perfect Matching](https://arxiv.org/abs/2303.15933), using {class}`MWPMDecoder <deltakit.decode.MWPMDecoder>`;
* [Collision Clustering](https://arxiv.org/abs/2309.05558), using {class}`CCDecoder <deltakit.decode.CCDecoder>`;
* [Belief Matching](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.031007), using {class}`BeliefMatchingDecoder <deltakit.decode.BeliefMatchingDecoder>`;
* [Belief Propagation - Ordered Statistics Decoding (BP-OSD)](https://quantum-journal.org/papers/q-2021-11-22-585/), using {class}`BPOSDecoder <deltakit.decode.BPOSDecoder>`;
* [Ambiguity Clustering](https://arxiv.org/abs/2406.14527), using {class}`ACDecoder <deltakit.decode.ACDecoder>`;
* [Local Clustering Decoder](https://arxiv.org/abs/2411.10343), using {class}`LCDecoder <deltakit.decode.LCDecoder>`;

```{code-cell} ipython3
from deltakit.decode import MWPMDecoder, CCDecoder, BeliefMatchingDecoder, BPOSDecoder, ACDecoder, LCDecoder
from deltakit.explorer import Client

cloud = Client.get_instance()

decoders = [
    MWPMDecoder(circuit=deltakit_circuit, use_experimental_graph_method=True, client=cloud),
    CCDecoder(circuit=deltakit_circuit, client=cloud),
    BeliefMatchingDecoder(circuit=deltakit_circuit, client=cloud),
    BPOSDecoder(circuit=deltakit_circuit, parameters={}, client=cloud),
    ACDecoder(circuit=deltakit_circuit, parameters={"bp_rounds": 200}, client=cloud),
    LCDecoder(circuit=deltakit_circuit, parameters={"weighted": True}, client=cloud),
]
for decoder in decoders:
    print(f"{decoder.__class__.__name__:22}: ", end="")
    predictions = decoder.decode_batch_to_logical_flip(detectors.as_numpy())
    mismatch = (predictions != observables.as_numpy())
    fails = int(sum(np.any(mismatch, axis=1)))
    print(f"{DecodingResult(fails, predictions.shape[0])}")
```

For cloud-based decoders, if the `use_experimental_graph_method` argument is set to `True`,
then rather than using the
noise model directly to decode, we will instead derive a noise model based on the
measurement results.

In order for this to be possible, a minimal noise model must be provided.
This will be used as a lower bound when deriving the noise model.
We provide a minimal noise model in {class}`PhysicalNoiseModel <deltakit.explorer.types.PhysicalNoiseModel>`.

```{code-cell} ipython3
from deltakit.explorer.types import PhysicalNoiseModel

very_low_noise = PhysicalNoiseModel.get_floor_superconducting_noise()
low_noise_circuit = cloud.add_noise(stim_circuit, very_low_noise)
low_noise_stim = stim.Circuit(low_noise_circuit).flattened()

decoder = MWPMDecoder(circuit=low_noise_stim, use_experimental_graph_method=True, client=cloud)
```

The six decoders use different approaches to correcting errors.

**Minimum-Weight Perfect Matching** (MWPM) tries to identify errors by grouping
flipped detector measurements in pairs based on how close they are.
Collision Clustering works similarly, by identifying small clusters of flipped
detector measurements. Belief Matching works by initially using a
technique called Belief Propagation to identify likely errors, and then uses
this knowledge to inform an MWPM decoder. These three decoders are designed for certain
types of decoding problems, such as those that arise when considering the surface
code. For other codes, such as general qLDPC codes, a hypergraph decoder, such as
BP-OSD or Ambiguity Clustering (AC), is required.

**Ambiguity Clustering** (AC) is Riverlane's proprietary decoder for decoding
general qLDPC codes. It works on any code with a parity check matrix (or a Stim detector error model),
and in a typical code allows orders of magnitude faster decoding whilst achieving the same
logical fidelity as BP-OSD, the industry standard qLDPC decoder. It works by combining the
ideas of clustering and Gaussian elimination to find local vector spaces of solutions to
isolated bits of syndrome.

AC also presents several advantages over decoders such as MWPM:

* AC is a hypergraph decoder, and can work with codes beyond surface codes (e.g. colour codes, 
  qLDPC codes). MWPM only works in the world of surface codes.
* MWPM only has the notion of X and Z noise. If these noises correlate, such that the system has Y-noise,
  AC may be more accurate than MWPM.

**Local Clustering Decoder** (LCD) is the first decoder that retains the performance advantage offered by hardware decoders,
while achieving levels of accuracy and flexibility that are competitive with their software counterparts.
It balances speed and accuracy, both of which are required to reach fault-tolerant quantum computing.
Higher decoder accuracy means more of the error-correction burden is placed on the decoder,
and less on the qubits, so you can do more with fewer qubits.
The LCD contains two main components to achieve this balance between speed and accuracy:

1. A decoding engine that allows the decoder to scale;
2. An adaptivity engine that helps deal with leakage.

Leakage is a source of noise where qubits no longer occupy the $\vert 0\rangle$ and $\vert 1\rangle$ computational basis states, and instead 
drift into higher energy 'leaked' states, specified as $\vert 2\rangle$, $\vert 3\rangle$, $\vert 4\rangle$, etc. Leakage noise is long-living and may 
spread to other qubits through multi-qubit gates.

`parameters` is an optional dictionary, which may contain flags and values
specific to each decoder. For more details on which parameters are available for each decoder, see the subsection below.

+++

#### Tuning decoder parameters

The performance of the decoders may depend on parameters. We allow decoders
to accept arbitrary named values using the `parameters` argument.
All decoders have access to the following parameters:

* ``decompose_errors`` (bool) - if set to `True`, Stim tries to
  decompose composite error mechanisms into simpler errors with at
  most two detectors affected. This option is important for decoders
  which do not support hypergraph detector error models (e.g. `MWPMDecoder`, `CCDecoder`).
* `approximate_disjoint_errors` (bool) -- approximates the noise as independent errors.

These two parameters work together and allow decoders to deal with composite
error mechanisms defined in the Stim circuit. By default, both are False.
More about these two parameters can be found in the
[official Stim documentation](https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md#stim.Circuit.detector_error_model).

Particular decoders may also expose parameters for tuning. These are the cases for the supported decoders.

* The `BPOSDecoder` supports two optional parameters:

  * `max_bp_rounds` is an integer, and specifies the maximum number of iterations 
    of message passing that should be performed during the execution of belief propagation.
    It may terminate earlier. By default, this is 20.

  * `combination_sweep_order` is the depth of the OSD search.

* The `ACDecoder` supports two optional parameters:

  * `bp_rounds` is an integer, and specifies how many iterations of message passing 
    should be performed during the execution of belief propagation. Note that `bp_rounds`
    in AC is different from `max_bp_rounds` in `BP_OSD` as early termination is not
    allowed. Typically, setting this equal to the distance of the code is sufficient. By
    default, this is 20.

  * `ac_kappa_proportion` is a float, between 0.0 and 1.0, and reflects the number 
    of error mechanisms, in addition to those used to find a first solution, that should be 
    used to grow clusters to search for additional solutions, expressed as a proportion 
    of the total number of error mechanisms. Setting this number higher results in better
    accuracy at the cost of slower performance. Start with 0.0 and increase by 0.01 until
    the desired accuracy is reached. Reasonable values lie between 0 and 0.1, as larger values will
    typically lead to a significant slowdown. By default, this is 0.01.

* The `LCDecoder` supports one optional parameter:

  * `weighted` is a boolean, and specifies whether to add weights to the decoding graph. Weighted
    leakage-aware decoding achieves higher decoding accuracy by more accurately modelling the stochastic
    process of qubits becoming leaked. It accounts for the fact that the probability of a leakage event
    occurring in some time interval after reset is proportional to the length of time. The unweighted
    implementation of LCD assumes constant probability for leakage in time. By default, `weighted` is False.

+++

## Leakage-aware decoding

Deltakit supports simulation and decoding for circuits with leakage information.
Such circuits produce additional information about the leakage state of qubits together with their measurements.
To generate such circuits, please refer to the chapter
[Adding Noise](./adding_noise.md).

The `LCDecoder` supports leakage as an additional input, and uses it for logical error rate reduction.

```{code-cell} ipython3
from deltakit.explorer.types import SI1000NoiseModel
from deltakit.explorer.simulation import simulate_with_stim

leaky = cloud.add_noise(stim_circuit, SI1000NoiseModel(p=0.01, p_l=0.02))
measurements, leakage = simulate_with_stim(leaky, shots=10_000, client=cloud)
leaky_detectors, leaky_observables = measurements.to_detectors_and_observables(leaky)

lc_decoder = LCDecoder(leaky, client=cloud, num_observables=stim_circuit.num_observables)
predictions_without_leakage = lc_decoder.decode_batch_to_logical_flip(leaky_detectors.as_numpy())
predictions_with_leakage = lc_decoder.decode_batch_to_logical_flip(leaky_detectors.as_numpy(), leakage.as_numpy())

for attempt, data in [("without", predictions_without_leakage), ("with", predictions_with_leakage)]:
    mismatch = (data != leaky_observables.as_numpy())
    fails = int(sum(np.any(mismatch, axis=1)))
    print(f"{attempt:7} leakage:", DecodingResult(fails, data.shape[0]))
```

An example of leakage-aware decoding is provided in the
[Leakage-Aware Decoding](../examples/notebooks/simulation/leakage-aware_decoding.ipynb) notebook.

## Per-shot analysis

While exploring decoding capabilities, you may be interested in the cases when decoding
predictions fail.

For example, for a quantum memory experiment which starts from the $\vert 0\rangle$ state,
a decoder failure is a mismatch between the prediction and the observable.
By analysing detector combinations (also known as syndromes) which led to failures,
you may derive important information about decoder behaviour and device properties.

To see how predictions can be used for analysis, please refer to the
[Analysis of Per-shot Decoding](../examples/notebooks/simulation/shot_analysis.ipynb)
example notebook.

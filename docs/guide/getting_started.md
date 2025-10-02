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

# Getting Started

Performing a QEC experiment with Deltakit typically involves four steps.

1. In the *circuit generation* step, you generate a quantum circuit to implement the experiment
   of interest, such as quantum memory or stability. In this step, you compile a noisy physical
   circuit based on your QPU's native gates and their characteristics.
2. In the *simulation* step, you simulate the circuit using [Stim](https://github.com/quantumlib/Stim).
3. In the *decoding* step, you choose a decoder to decode your measurement results. You can use
   both open-source decoders, like minimum weight perfect matching (MWPM), and propriety
   decoders, like Ambiguity Clustering (AC).
4. In the *analysis* step, you interpret the results of your QEC experiment, and calculate and visualize
   the logical error probability and error suppression factor $\Lambda$.

## Circuit Generation

### Idealized Circuit
To generate the underlying circuit for quantum memory experiments, there are several common quantum error correction codes you can use, including:

- Rotated planar codes ({class}`RotatedPlanarCode <deltakit.explorer.codes.RotatedPlanarCode>`),
- Unrotated planar codes ({class}`UnrotatedPlanarCode <deltakit.explorer.codes.UnrotatedPlanarCode>`),
- Unrotated toric codes ({class}`UnrotatedToricCode <deltakit.explorer.codes.UnrotatedToricCode>`),
- Repetition codes ({class}`RepetitionCode <deltakit.explorer.codes.RepetitionCode>`),
- Bivariate bicycle codes ({class}`BivariateBicycleCode <deltakit.explorer.codes.BivariateBicycleCode>`), and
- (More generally) any CSS code ({class}`CSSCode <deltakit.explorer.codes.CSSCode>`).

To get the logical error rate with a quantum memory experiment, for example, you can use the rotated planar code to encode a logical qubit:

```{code-cell} ipython3
from deltakit.explorer import codes
# configure the rotated planar code
code = codes.RotatedPlanarCode(width=2, height=2)
code.draw_patch()
```

You can then pass this `code` to the `css_code_memory_circuit` function to generate a noiseless circuit for a quantum memory experiment, for example, using a single round and measuring in the logical Pauli Z basis:

```{code-cell} ipython3
from deltakit.explorer.codes import css_code_memory_circuit
from deltakit.circuit.gates import PauliBasis

circuit = css_code_memory_circuit(code, num_rounds=1, logical_basis=PauliBasis.Z)
```

This returns a `Circuit` object, which contains the quantum circuit that encodes a memory experiment using the error correcting code you chose in the first cell.

```{code-cell} ipython3
circuit
```

To convert the `circuit` to [`stim`](https://github.com/quantumlib/Stim), you can run:

```{code-cell} ipython3
stim_circuit = circuit.as_stim_circuit()
```

which is then easy to visualize:

```{code-cell} ipython3
stim_circuit.diagram(type="timeline-svg")
```

### Realistic Circuit


You can now add a noise model and compile the circuit into the native gate set of a realistic QPU.

#### Noise

To simulate a realistic noise environment on a QPU, you can use several common noise models with
{ref}`api-deltakit-explorer-qpu`, including:


- Phenomenological noise ({class}`PhenomenologicalNoise <deltakit.explorer.qpu.PhenomenologicalNoise>`),
  which uses a fixed probability of depolarisation after each gate;
- Standard depolarising noise ({class}`SD6Noise <deltakit.explorer.qpu.SD6Noise>`), which
  has a single probability of incorrect measurement and depolarisation after each gate, idle cycle,
  and reset operation;
- "Toy" noise ({class}`ToyNoise <deltakit.explorer.qpu.ToyNoise>`), which parametrizes
  the probabilities of incorrect measurement and depolarisation after gate and idle cycle with two
  parameters; and
- SI1000 noise ({class}`SI1000Noise <deltakit.explorer.qpu.SI1000Noise>`), a noise model
  inspired by superconducting qubits.

For example:

```{code-cell} ipython3
from deltakit.explorer.qpu import SI1000Noise
noise_model = SI1000Noise(p=1e-3)  # 0.001 physical error rate
```

You can also configure more general noise models using {class}`NoiseParameters <deltakit.explorer.qpu.NoiseParameters>`.

#### Native Gates

With Deltakit, you can define a QPU's native gate set by specifying its one qubit,
two qubit, reset, and measurement gates.

```{code-cell} ipython3
from deltakit.explorer.qpu import NativeGateSet
from deltakit.circuit.gates import SQRT_X, SQRT_X_DAG, S_DAG, S, X, Z, CZ,RZ, MZ
native_gates = NativeGateSet(
    one_qubit_gates={SQRT_X, SQRT_X_DAG, S_DAG, S, X, Z},
    two_qubit_gates={CZ},
    reset_gates={RZ},
    measurement_gates={MZ},
)
```

With the noise model and native gate set defined, you can now generate an object to represent the `QPU` object, and compile your circuit to simulate the target QPU.

```{code-cell} ipython3
from deltakit.explorer.qpu import QPU
qpu = QPU(qubits=code.qubits,
          native_gates_and_times=native_gates,
          noise_model=noise_model)
qpu_circuit = qpu.compile_and_add_noise_to_circuit(circuit)
qpu_circuit.as_stim_circuit().diagram(type="timeline-svg")
```

With the noisy, qpu-native circuit generated, you're now ready to tackle the next step of your quantum error correction experiment: simulation.

+++

## Simulation
To sample raw output from the circuit, you can use `stim` features directly:

1. Convert to `stim` circuit
2. Compile a `stim` "sampler" from the circuit
3. Sample using the sampler

```{code-cell} ipython3
stim_circuit = qpu_circuit.as_stim_circuit()  # 1
sampler = stim_circuit.compile_sampler()  # 2
sampler.sample(shots=3)  # 3
```

The output is a boolean NumPy array representing the seven measurement outcomes for three separate shots.

Typically, the focus is not on the individual measurement outcomes but on logical errors. Assessing those is the job of the decoder.

+++

## Decoding

Deltakit currently exposes one decoder for use locally, [PyMatching](https://github.com/oscarhiggott/PyMatching). (However, there are many other decoder options when using the remote workflow, and there are many local decoder analysis and manipulation tools in {ref}`api-deltakit-decode`.)

({meth}`PyMatchingDecoder.construct_decoder_and_stim_circuit <deltakit.decode.PyMatchingDecoder.construct_decoder_and_stim_circuit>`) is a helper factor that accepts a circuit and returns two things:

- an object representing a Minimum Weight Perfect Matching decoder and
- a modified version of the original circuit configured to operate with the decoder.

```{code-cell} ipython3
from deltakit.decode import PyMatchingDecoder
decoder, decoder_circuit = PyMatchingDecoder.construct_decoder_and_stim_circuit(qpu_circuit)
```

({meth}`run_decoding_on_circuit <deltakit.decode.analysis.run_decoding_on_circuit>`) accepts these objects and the maximum number of shots to sample, returning a dictionary that contains the actual number of shots and the number of failures (shots for which the decoder incorrectly assessed whether the observable had flipped or not).

```{code-cell} ipython3
from deltakit.decode.analysis import run_decoding_on_circuit
results = run_decoding_on_circuit(decoder_circuit, 1000, decoder)
print(f"There were {results['fails']} failures out of {results['shots']} shots.")
```

To scale beyond simple experiments, it is useful to define the concept of a "Decoder Manager", which represents the combined experiment circuit and decoder system. The {class}`StimDecoderManager <deltakit.decode.analysis.StimDecoderManager>` accepts the circuit and decoder objects and returns a decoder manager. Similar to the `run_decoding_on_circuit` function, its {class}`run_batch_shots <deltakit.decode.analysis.StimDecoderManager.run_batch_shots>` method returning the total number of shots and the number of failures.

```{code-cell} ipython3
from deltakit.decode.analysis import StimDecoderManager
decoder_manager = StimDecoderManager(decoder_circuit, decoder)
n_shots, n_fails = decoder_manager.run_batch_shots(1000)
print(f"There were {n_fails} failures out of {n_shots} shots.")
```

## Analysis
Deltakit also provides tools for summarizing the results of experiments. To calculate the LEP (Logical Error Probability) and its standard error, use the
{meth}`calculate_lep_and_lep_stddev <deltakit.explorer.analysis.calculate_lep_and_lep_stddev>` function.

```{code-cell} ipython3
from deltakit.explorer.analysis import calculate_lep_and_lep_stddev
lep, lep_stddev = calculate_lep_and_lep_stddev(
    fails=[n_fails],
    shots=[n_shots],
)
print(f"LEP: {lep[0]}, LEP std: {lep_stddev[0]}")

# confirm the calculation manually
import numpy as np
from scipy import stats
lep0 = n_fails / n_shots
lep_stddev0 = stats.sem([1]*n_fails + [0]*(n_shots - n_fails), ddof=0)
np.testing.assert_allclose(lep, lep0)
np.testing.assert_allclose(lep_stddev, lep_stddev0)
```

Now you can start varying different parameters to see how that changes the logical error probability, such as the code distance. You can pass a list of decoder managers and number of shots to {class}`RunAllAnalysisEngine <deltakit.decode.analysis.RunAllAnalysisEngine>`, and use the `run` method to run your simulations. This method then returns a dataframe that summarizes the results.

```{code-cell} ipython3
from deltakit.decode.analysis import RunAllAnalysisEngine

distances = [5, 7, 9]
all_shots, all_fails = [], []
decoder_managers = []
for d in distances:
    code = codes.RotatedPlanarCode(width=d, height=d)
    circuit = css_code_memory_circuit(code, num_rounds=d, logical_basis=PauliBasis.Z)
    qpu = QPU(qubits=code.qubits, native_gates_and_times=native_gates, noise_model=noise_model)
    qpu_circuit = qpu.compile_and_add_noise_to_circuit(circuit)
    decoder, decoder_circuit = PyMatchingDecoder.construct_decoder_and_stim_circuit(qpu_circuit)
    decoder_managers.append(StimDecoderManager(decoder_circuit, decoder))

engine = RunAllAnalysisEngine("my experiment", decoder_managers=decoder_managers, max_shots=100000)
df = engine.run()
df['distance'] = distances
df
```

`calculate_lep_and_lep_stddev` can calculate all the logical error probabilities in batch.

```{code-cell} ipython3
from deltakit.explorer.analysis import calculate_lep_and_lep_stddev
leps, leps_std = calculate_lep_and_lep_stddev(df['fails'], df['shots'])
df['LEP'] = leps
df['LEP std'] = leps_std
df
```

To estimate the error scaling parameter $\Lambda$, use the
{meth}`calculate_lambda_and_lambda_stddev <deltakit.explorer.analysis.calculate_lambda_and_lambda_stddev>`
function:

```{code-cell} ipython3
from deltakit.explorer.analysis import calculate_lambda_and_lambda_stddev
l, _ = calculate_lambda_and_lambda_stddev(distances, leps, leps_std)
l
```

Now that you have an overview of the four steps of QEC experimentation with Deltakit,
please continue with the following guides.

```{toctree}
:maxdepth: 1

authentication
circuit_generation
adding_noise
simulation
decoding
analysis

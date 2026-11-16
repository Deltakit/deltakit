---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  display_name: deltakit-explorer-py3.10
  language: python
  name: python3
---

# Decoding with Ambiguity Clustering

*Want to follow along? {nb-download}`Download this notebook.<ac_decoding.ipynb>`*

In this notebook, we demonstrate decoding with Riverlane's proprietary decoder: the [Ambiguity Clustering (AC) decoder](https://www.nature.com/articles/s41586-024-07107-7). In particular, we

* generate a circuit for a surface code quantum memory experiment, define a noise model, and generate measurement and detector data from the circuit with the noise model applied;
* decode the detector data with both AC and Pymatching (MWPM), and show that AC can achieve a better logical error probability;
* generate a stim circuit implementation of IBM's 72-qubit bivariate bicycle code to generate measurement and detector data;
* decode this data with both AC and BP-OSD to show that AC can decode much faster whilst maintaining a similar logical error probability.

+++

## Introduction

Ambiguity Clustering (AC) is a prototype decoding algorithm for general qLDPC codes being developed by Riverlane. The purpose of this notebook is to demonstrate the decoder's capability decoding a surface code, benchmarking its performance against the [minimum weight perfect matching](https://arxiv.org/pdf/2105.13082) (MWPM) decoder. Then, we look at its performance on a qLDPC code ([IBM's 72-qubit bivariate bicycle code](https://arxiv.org/pdf/2308.07915)), and compare its performance with the [BP-OSD](https://github.com/quantumgizmos/bp_osd) decoder.

Unlike 'matching' decoders such as MWPM (e.g. Pymatching), which operate on decoding *graphs* (that is, each error mechanism must trigger at most two checks), Ambiguity Clustering operates on a more general structure, the decoding *hypergraph* - or in other words, any Stim detector error model (DEM). This makes it suitable for decoding any QEC code for which a detector error model can be generated, including qLDPC codes.

Even for the surface code, whose DEM can be well-approximated as a decoding graph so that we can run MWPM on it, AC can still potentially provide a logical fidelity gain over matching decoders, as by considering the full hypergraph it is able to 'see' correlations between Pauli X and Pauli Z errors occurring on qubits, to which MWPM is blind (unless you use a correlated matching technique). We demonstrate this accuracy gain here for a distance-7 surface code memory experiment.

We then generate a stim circuit of IBM's 72-qubit bivariate bicycle code using the Deltakit API, and then decode with AC, demonstrating AC's superior speed with comparable accuracy to BP-OSD.

### How it works

AC is a two phase algorithm. First, like BP-OSD, it runs the belief propagation graphical inference algorithm on the decoding Tanner graph to get an approximate probability of occurrence for every error mechanism. Secondly, it applies a series of linear algebra operations to the parity check matrix to identify connected clusters of syndrome bits in the Tanner graph. It can then search over solutions to these clusters individually to find the most likely solutions to the syndrome.

+++

## Preparation
### Perform all necessary imports

```{code-cell} ipython3
# Import prerequisites
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import stim
from deltakit.explorer import Client
from deltakit.explorer.enums import DecoderType
from deltakit.explorer.types import Decoder, PhysicalNoiseModel, QECExperimentDefinition
```

```{code-cell} ipython3
client = Client.get_instance()
example_circuits_folder = Path("../..") / "data/examples/qldpc"
```

## Generate Surface Code Data
We generate data for rotated planar codes of distances 3, 5, 7, with the number of rounds of syndrome extraction matching the distance. We use a circuit-level noise model with p=0.5% for all the noise parameters. Gate times, T1 and T2 times allow to compute idleness durations and apply realistic idleness noise. For more details see the {ref}`client-add-noise-to-the-circuit` example.

```{code-cell} ipython3
# How many shots to run
num_shots = 10000
distances = [3, 5, 7]

# Define p = 0.5% noise model
noise_model = PhysicalNoiseModel(
    t_1=50e-6,
    t_2=50e-6,
    time_1_qubit_gate=50e-9,
    time_2_qubit_gate=100e-9,
    time_measurement=500e-9,
    time_reset=500e-9,
    p_1_qubit_gate_error=0.002,
    p_2_qubit_gate_error=0.002,
    p_reset_error=0.002,
    p_meas_qubit_error=0.002,
    p_readout_flip=0.002,
)

# Generate noiseless circuits
compiled_circuits = []
measurements_files = []
for distance in distances:
    compiled_circuit = client.generate_circuit(
        QECExperimentDefinition.get_rotated_planar_z_quantum_memory(
            distance, distance, ["CZ", "H", "MZ", "RZ"]
        )
    )
    compiled_circuits.append(compiled_circuit)
    
    # Add noise to circuit
    noisy_circuit = client.add_noise(
        stim_circuit=compiled_circuit,
        noise_model=noise_model
    )
    
    # Simulate noisy circuit to generate measurement data
    measurements, _ = client.simulate_stim_circuit(
        stim_circuit=noisy_circuit,
        shots=num_shots,
    )
    measurements_files.append(measurements)
```

## Decoding the surface code data with MWPM

```{code-cell} ipython3
mwpm_errors = []
mwpm_error_bars = []
# Loop through the circuits and perform MWPM decoding on each
for _distance, compiled_circuit, measurements in zip(distances, compiled_circuits, measurements_files):
    mwpm_result = client.decode_measurements(
        measurements=measurements,
        decoder=Decoder(DecoderType.MWPM, parallel_jobs=8),
        ideal_stim_circuit=compiled_circuit,
        noise_model=noise_model,
    )
    lep = mwpm_result.get_logical_error_probability()
    std_deviation = mwpm_result.get_standard_deviation()
    mwpm_errors.append(mwpm_result.get_logical_error_probability())
    mwpm_error_bars.append(mwpm_result.get_standard_deviation())
```

## Decoding the surface code data with AC
We see that although AC takes longer to decode per shot than MWPM, it can achieve a lower logical error probability (note there is significant variance per run). This is because it operates on the whole, undecomposed, detector error model, and so can 'see' correlations between X and Z errors.

```{code-cell} ipython3
ac_errors, ac_error_bars = [], []
kappa = 0.02
# Loop through the circuits and perform AC decoding on each
for distance, compiled_circuit, measurements in zip(
    distances, compiled_circuits, measurements_files):
    ac_result = client.decode_measurements(
        measurements=measurements,
        noise_model=noise_model,
        decoder=Decoder(
            DecoderType.AC,
            parallel_jobs=8,
            parameters={
                "decompose_errors": False,
                "bp_rounds": distance,
                "ac_kappa_proportion": kappa,
            }
        ),
        ideal_stim_circuit=compiled_circuit,
    )
    error_probability = ac_result.get_logical_error_probability()
    std_deviation = ac_result.get_standard_deviation()
    
    ac_errors.append(error_probability)
    ac_error_bars.append(std_deviation)
    
```

```{code-cell} ipython3
plt.errorbar(distances, mwpm_errors, yerr = mwpm_error_bars, color="#00968f", capsize = 3, label = "MWPM", fmt = 's')
plt.errorbar(distances, ac_errors, yerr = ac_error_bars, color="#ff7500", capsize = 3, label = "Ambiguity Clustering", fmt = 'd')
plt.grid(color="gray", linestyle=":", linewidth=1)
plt.xticks(distances)
plt.xlabel("Code distance")
plt.ylabel("Logical error probability")
plt.yscale('log')
plt.legend()
```

## Decoding qLDPC codes
Quantum low-density parity check (qLDPC) codes are the current leading family of error correcting codes. These codes allow for the possibility of fault-tolerant quantum computation with lower overheads (fewer physical qubits needed per encoded logical qubit) compared to conventional codes, such as the surface code.

We will deal with one such qLDPC code, IBM's 72-qubit bivariate bicycle code, which makes use of 72 physical qubits to encode 12 logical qubits, and has a distance of 6. This gives it code parameters [[72, 12, 6]]. 

+++

### Read-in circuit
Here we use a Stim circuit implementing the 72-qubit code at a 0.5% physical error rate (generated using a script from [Gong et al.](https://arxiv.org/pdf/2403.18901); repo [here](https://github.com/gongaa/SlidingWindowDecoder)). The noise model is a realistic noise model for superconducting qubits. Parameters are listed below.

```{code-cell} ipython3
# Code parameters, for naming files
code_name = "[[72,12,6]]"

# How many shots to run
num_shots = 1000
noise_model = PhysicalNoiseModel.get_superconducting_noise()
# Read-in the pre-generated circuit
noisy_circuit = str(stim.Circuit.from_file(example_circuits_folder / f"{code_name}p=0.005.stim"))
```

### Simulate measurements, generate detectors and observables

```{code-cell} ipython3
# Simulate noisy circuit to generate measurement data
measurements, _ = client.simulate_stim_circuit(
    stim_circuit=noisy_circuit,
    shots=num_shots,
)

# Convert the measurement data to detectors and observables
detectors, observables = measurements.to_detectors_and_observables(
    stim_circuit=noisy_circuit,
)
```

### Decode with AC and BP-OSD

Here we demonstrate decoding of IBM's 72 qubit bivariate bicycle code with AC, and, for comparison, with BP-OSD.
These BP parameters for the two algorithms are empirically found to work well. Since BP-OSD has a much heavier post-processing step than AC, it makes sense that the algorithm benefits from more rounds of BP.

In our example, we demonstrate AC's superior decoding time compared to BP_OSD, whilst maintaining a comparable logical error probability.

```{code-cell} ipython3
decoding_results = {}
server_cpu_workers = 8
decoders = [
    Decoder(
        decoder_type=DecoderType.AC,
        parameters={"decompose_errors": False, "bp_rounds": 99, "ac_kappa_proportion": 0.02},
        parallel_jobs=server_cpu_workers,
        use_experimental_graph=False,
    ),
    Decoder(
        decoder_type=DecoderType.BP_OSD,
        parameters={"decompose_errors": False, "max_bp_rounds": 10_000, "combination_sweep_order": 0},
        parallel_jobs=server_cpu_workers,
        use_experimental_graph=False,
    )
]

for decoder in decoders:
    print(f"Decoding {code_name} code with {decoder.decoder_type.value}...")
    qldpc_results = client.decode(
        detectors=detectors,
        observables=observables,
        decoder=decoder,
        noisy_stim_circuit=noisy_circuit,
    )
    decoding_results[decoder.decoder_type] = qldpc_results

print("\n Decoder | Logical error probability | Total CPU time (all cores), s")
print("-" * 69)
for decoder, result in decoding_results.items():
    decoding_time = sum(result.times)
    print(f" {decoder.value:7s} | {result.get_logical_error_probability():25.2} |\t\t{decoding_time:.4f}")
```

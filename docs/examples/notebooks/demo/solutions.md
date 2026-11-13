---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
---

## Exercise #1

*Want to follow along? {nb-download}`Download this notebook.<solutions.ipynb>`*

We decoded one particular experiment, and every time you run it, you should have similar numbers.
What we are often interested in is HOW logical error probability responds to different parameter tweaking.

Your first task is to complete the following code. For a fixed noise model and a fixed decoder,
plot how repetition code scales with size.

```{code-cell} ipython3
from __future__ import annotations
from deltakit.circuit.gates import PauliBasis
import matplotlib.pyplot as plt

distances = [3, 5, 7, 9, 11]
le_probabilities = []
for distance in distances:
    # update experiment parameters
    experiment.parameters = types.CircuitParameters.from_sizes([distance])
    experiment.num_rounds = distance
    noise = types.SI1000NoiseModel(0.01, 0.0)

    circuit = client.generate_circuit(experiment)
    noisy_circuit = client.add_noise(circuit, noise)

    measurements, _ = client.simulate_stim_circuit(noisy_circuit, 100_000)
    detectors, observables = measurements.to_detectors_and_observables(noisy_circuit)

    decoder = types.Decoder(enums.DecoderType.MWPM)
    decoding_result = client.decode(detectors, observables, decoder, noisy_circuit)
    le_probabilities.append(decoding_result.get_logical_error_probability())

plt.plot(distances, le_probabilities, label="logical error probability,\nrepcode+MWPM")
plt.xlabel("distance")
plt.ylabel("logical error probability")
plt.legend()
plt.grid()
```

## Exercise #3

Plot a 3D diagram for a rotated planar code, Z-memory experiment which has error distance 3 (sizes=[3, 3]), and 3 rounds of syndrome measurements. Set basis gates to `None`.

```{code-cell} ipython3
experiment = types.QECExperimentDefinition(
    experiment_type=enums.QECExperimentType.QUANTUM_MEMORY,
    code_type=enums.QECECodeType.ROTATED_PLANAR,
    observable_basis=PauliBasis.Z,
    num_rounds=3,
    parameters=types.CircuitParameters.from_sizes([3, 3]),
)
experiment = types.QECExperimentDefinition.get_rotated_planar_z_quantum_memory(3, 3)
circuit = client.generate_circuit(experiment)
display(stim.Circuit(circuit).diagram(type="timeline-3d"))
```

---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

# Modifying Stim Circuits

deltakit-compile compiles QEC circuits by running many compiler passes over the circuit, each pass transforming the circuit in a specific, well-defined way. Some of these passes are, for example, optimisations on a Deltakit stim circuit where the specifics of the optimisation will depend on the QPU.

You can run these passes individually to see the effect of each. In this notebook, we show how to run two of these passes: `add-noise`, and `remap-qubits`.

```{code-cell} ipython3
# This file contains information which is proprietary to Riverlane Ltd
# ("Riverlane") and is Riverlane Confidential Information.
# (c) Copyright Riverlane 2025-2026. All rights reserved.
import deltakit_stim
import stim

from deltakit_compile.frontend.deltakit_stim import (
    deltakit_stim_circuit_to_dialect,
    deltakit_stim_context,
    deltakit_stim_dialect_to_circuit,
)
from deltakit_compile.noise_models.sd6_noise import SD6NoiseConfig
from deltakit_compile.noise_models.si1000_noise import SI1000NoiseConfig
from deltakit_compile.passes.add_noise import AddNoise
from deltakit_compile.passes.remap_qubits import RemapQubits
```

The following circuit is the beginning of a repetition code quantum memory experiment, and will be our example.

```{code-cell} ipython3
circuit = deltakit_stim.Circuit("""
QUBIT_COORDS(0, 4) 0
QUBIT_COORDS(0, 0) 1
QUBIT_COORDS(0, 1) 2
QUBIT_COORDS(0, 2) 3
QUBIT_COORDS(0, 3) 4
R 0 3 1 2 4
TICK
SQRT_X 2 4
TICK
CZ 2 1 4 3
TICK
CZ 2 3 4 0
TICK
SQRT_X 2 4
TICK
M 2 4
TICK
DETECTOR(1, 0, 0) rec[-2]
DETECTOR(3, 0, 0) rec[-1]
""")
```

While the circuit above is defined using Deltakit stim, we can obtain a Stim circuit object from it by parsing it back in as a Deltakit stim circuit.

```{code-cell} ipython3
stim_circuit = stim.Circuit("R 0 1")
deltakit_stim_circuit = deltakit_stim.Circuit(str(stim_circuit))
print(deltakit_stim_circuit)
```

deltakit-compile passes require a `Context` object, which is loaded with all the dialects (e.g. `stim` or `qcore`) used in the Intermediate Representation (IR) the pass is operating on.

For all the passes in this notebook we will be using a standard context that sets us up for using Deltakit stim and Physical Circuit IR dialects. Note that if we were using any more dialects, such as `stab` for stabiliser flows, we would need to additionally load those dialects into the context.

```{code-cell} ipython3
context = deltakit_stim_context()
```

## Adding noise to a circuit

deltakit-compile provides user-configurable noise models, which can be used to add noise to Deltakit stim circuits. In the code below we add SI1000 noise to the circuit with `p = 0.01`.

```{code-cell} ipython3
# Convert the circuit to an Intermediate Representation (IR)
module_op = deltakit_stim_circuit_to_dialect(circuit)

# Apply the AddNoise pass using the SI1000 noise model, with a Context object loaded with all the
# required dialects
AddNoise(SI1000NoiseConfig(p=0.01)).apply(context, module_op)

# Convert the IR back to a Deltakit stim circuit
circuit_out = deltakit_stim_dialect_to_circuit(module_op)
print(circuit_out)
```

As an alternate example, here is the same thing using the SD6 noise model.

```{code-cell} ipython3
module_op = deltakit_stim_circuit_to_dialect(circuit)
AddNoise(SD6NoiseConfig(p=0.01)).apply(context, module_op)
circuit_out = deltakit_stim_dialect_to_circuit(module_op)
print(circuit_out)
```

## Qubit remapping

Another common circuit modification is remapping qubit IDs such that they match those on the QPU the circuit will eventually be run on. To do this, the compiler needs a mapping between qubit ID and the coordinates of the qubit on the QPU. Given this information, it can relabel qubits in those locations with the correct IDs.

```{code-cell} ipython3
# We are imagining our QPU as having 10 qubits in a row
qubit_mapping = {
    0: [0, 0],
    1: [0, 1],
    2: [0, 2],
    3: [0, 3],
    4: [0, 4],
    5: [0, 5],
    6: [0, 6],
    7: [0, 7],
    8: [0, 8],
    9: [0, 9],
}

module_op = deltakit_stim_circuit_to_dialect(circuit)
RemapQubits(qubit_coord_offset=None, qubit_mapping=qubit_mapping).apply(context, module_op)
circuit_out = deltakit_stim_dialect_to_circuit(module_op)
print(circuit_out)
```

The circuit can also be translated around the QPU and the IDs will be automatically adjusted for the qubit's new locations.

```{code-cell} ipython3
# Configure the pass to shift the experiment by 5 and use the previous qubit mapping
module_op = deltakit_stim_circuit_to_dialect(circuit)
RemapQubits(qubit_coord_offset=[0, 5], qubit_mapping=qubit_mapping).apply(context, module_op)
circuit_out = deltakit_stim_dialect_to_circuit(module_op)
print(circuit_out)
```

## Combining passes

Once you know what passes you want for your QPU, you can combine them together.

```{code-cell} ipython3
module_op = deltakit_stim_circuit_to_dialect(circuit)
RemapQubits(qubit_coord_offset=[0, 5], qubit_mapping=qubit_mapping).apply(context, module_op)
AddNoise(SI1000NoiseConfig(p=0.01)).apply(context, module_op)
circuit_out = deltakit_stim_dialect_to_circuit(module_op)
print(circuit_out)
```

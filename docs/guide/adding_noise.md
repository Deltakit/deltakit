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

# Adding Noise

## Motivation

The goal of Quantum Error Correction is to correct errors, which are caused by different noise mechanisms.
A noise channel is a way to describe a probabilistic change of a quantum state.
In order for a decoder to more accurately correct errors,
it's important to pass a model of the noise, described using noise channels.

Whether you are exploring error-correcting codes or designing a new decoder,
it's important to have realistic data to validate them.
You can obtain those data from a real QPU, however, this might be expensive or time consuming, due to requiring calibration. This is where realistic QEC simulation comes in.

The physics of noise in quantum systems is complicated and
differs between qubit architectures.
For example, superconducting systems don't typically lose their qubits,
while neutral atom ones can.
Cross-talk, on the other hand, is specific to
superconducting QPUs.
Shuttling noise is something you would observe in ion trap devices,
but cannot even be defined in superconducting systems,
and so on.

Each noise source has its own theory, and you may try to model it quite accurately.
The problem comes with the experiment size.
As you explore probabilistic processes, you want your evidence to be statistically reliable.
But due to the exponential improvement in logical error-probability when scaling an error-correction code,
typical experiment repetition numbers (also known as number of shots) may exceed $10^8$.
Doing accurate simulation with this number of repetitions is practically impossible,
so you may have to sacrifice model accuracy for simulation speed.

## Practical solution

The [Stim](https://github.com/quantumlib/Stim) library offers tools for both quick simulation (thanks to its limitation to Clifford gates)
and a reasonably rich palette of error mechanisms,
which may be used to approximate different types of noise.
Deltakit benefits from both Stim's simulation and error implementation approaches.
You may read about these building blocks in detail in the
[Noise Channels section](https://github.com/quantumlib/Stim/blob/main/doc/gates.md#noise-channels)
of the Stim documentation.

With these tools in your hands, you may implement different **noise models**.
A noise model is a way to define realistic errors across all qubits and gates
using a small number of controllable parameters.

## `NoiseParameters` class
The {class}`NoiseParameters <deltakit.explorer.qpu.NoiseParameters>` base class
allows you to define arbitrary functions which insert
noise events into the circuit when the necessary conditions are met.
You may define your realistic noise model by specifying what should happen
each time a gate or measurement is executed, or when a qubit remains idle.
Please refer to the documentation for this class.

All other noise models in Deltakit inherit from this class.
It defines a uniform way of adding Stim noise channels to the circuit.

## `ToyNoise`

The {class}`ToyNoise <deltakit.explorer.qpu.ToyNoise>` is a simple toy model for noise that
has only two parameters.
One parameter, `p`, gives the two-qubit operation error rate,
which is assumed to be the dominant error.
Another optional parameter, `p_measurement_flip` gives the probability of
obtaining an incorrect measurement result.

All other noise channels have an error rate `p/10`.
Idle noise is therefore not time dependent.
All these noise channels are depolarising channels --
the noise channels occurring after a two-qubit gate are two-qubit
depolarising channels, and all others are one-qubit depolarising channels.

In the example below, you can see how `p/10`-scaled noise follows gates,
and `p_measurement_flip` becomes a parameter of the measurement gate.

```{code-cell} ipython3
import stim
from deltakit.circuit import Circuit

simple_stim = """
RZ 0 1
TICK
H 0
X 1
TICK
CX 0 1
TICK
H 0
TICK
I 1
TICK
MZ 0 1
"""
stim_circuit = stim.Circuit(simple_stim)
deltakit_circuit = Circuit.from_stim_circuit(stim_circuit)
```

```{code-cell} ipython3
from deltakit.explorer.qpu import QPU, ToyNoise

toy_noise = ToyNoise(p=0.01, p_measurement_flip=0.005)

toy_qpu = QPU(
    qubits=deltakit_circuit.qubits,
    noise_model=toy_noise,
)

deltakit_toy_noisy = toy_qpu.compile_and_add_noise_to_circuit(deltakit_circuit, remove_paulis=False)
deltakit_toy_noisy.as_stim_circuit().diagram("timeline-html")
```

## `PhysicalNoise` class

On the other end of the modeling spectrum is a model with seven parameters.
All these parameters are characteristics of the QPU and can be derived during device calibration.
You can often find these numbers in the datasheets of hardware companies.

The {class}`PhysicalNoise <deltakit.explorer.qpu.PhysicalNoise>` class is parameterized with the following values:
- `t_1`: $T_1$ time (relaxation from $\vert 1\rangle$ to $\vert 0\rangle$), in seconds.
- `t_2`: $T_2$ time (dephasing), in seconds.
- `p_1_qubit_gate_error`: Probability of a flip during a 1-qubit gate.
- `p_2_qubit_gate_error`: Probability of a flip during a 2-qubit gate.
- `p_reset_error`: Probability of a flip during reset.
- `p_meas_qubit_error`: Probability of incorrect measurement.
- `p_readout_flip`: Probability of a flip during qubit measurement.

This class applies noise terms to the gates, and also computes idle noise,
based on gate durations and the provided $T_1$ and $T_2$ qubit times.

```{code-cell} ipython3
from deltakit.explorer.qpu import PhysicalNoise, NativeGateSetAndTimes
from deltakit.circuit import gates

physical_noise = PhysicalNoise(
    t1=50e-6,
    t2=50e-6,
    p_1_qubit_gate_error=0.013,
    p_2_qubit_gate_error=0.014,
    p_reset_error=0.011,
    p_meas_qubit_error=0.012,
    p_readout_flip=0.001,
)

physical_qpu = QPU(
    qubits=deltakit_circuit.qubits,
    native_gates_and_times=NativeGateSetAndTimes(
        one_qubit_gates={gates.H: 20e-9, gates.X: 20e-9},
        two_qubit_gates={gates.CZ: 30e-9},
        reset_gates={gates.RZ: 500e-9},
        measurement_gates={gates.MZ: 500e-9},
    ),
    noise_model=physical_noise,
)
deltakit_physical_noisy = physical_qpu.compile_and_add_noise_to_circuit(deltakit_circuit, remove_paulis=False)
deltakit_physical_noisy.as_stim_circuit().diagram("timeline-html")
```

### `SD6Noise` and `SI1000Noise` classes

The literature provides good guesses for simple yet accurate noise models.

The {class}`SD6Noise <deltakit.explorer.qpu.SD6Noise>` class implements the standard
depolarising noise model as seen in QEC literature
(see Table 2 in [A Fault-Tolerant Honeycomb Memory, Gidney et al.](https://arxiv.org/abs/2108.10457)).
This model sets all one- and two-qubit depolarising error probabilities to `p`,
as well as setting all measurement flip probabilities to `p`.
This model can be used for theory and simulation but is not
representative of other models seen in real hardware devices.

The {class}`SI1000Noise <deltakit.explorer.qpu.SI1000Noise>` class implements another
superconducting-inspired noise model from [A Fault-Tolerant Honeycomb Memory, Gidney et al.](https://arxiv.org/abs/2108.10457).
This noise model assumes that after every measurement there is a reset.
This model also includes an optional parameter `pL` for leakage, implemented as described
in [Local Clustering Decoder: a fast and adaptive hardware decoder for the surface code, Ziad et al.](https://arxiv.org/abs/2411.10343).

Note that to apply `SI1000Noise` **with leakage**, Deltakit makes a call to the cloud API.

```{code-cell} ipython3
from deltakit.explorer.qpu import SI1000Noise, SD6Noise
from deltakit.circuit import gates

sd6_noise = SD6Noise(p=0.01)

sd6_qpu = QPU(
    qubits=deltakit_circuit.qubits,
    noise_model=sd6_noise,
)

deltakit_sd6_noisy = sd6_qpu.compile_and_add_noise_to_circuit(deltakit_circuit, remove_paulis=False)
deltakit_sd6_noisy.as_stim_circuit().diagram("timeline-html")
```

No leakage example:

```{code-cell} ipython3
si1000_noise = SI1000Noise(p=0.01, pL=0.0)
si1000_qpu = QPU(
    qubits=deltakit_circuit.qubits,
    noise_model=si1000_noise,
)
deltakit_si1000_noisy = si1000_qpu.compile_and_add_noise_to_circuit(deltakit_circuit, remove_paulis=False)
deltakit_si1000_noisy.as_stim_circuit().diagram("timeline-html")
```

Example with leakage:

```{code-cell} ipython3
from deltakit.explorer import Client

cloud = Client.get_instance()

# NB we use stim circuit here
si1000_leakage_noise = SI1000Noise(p=0.01, pL=0.005)
stim_text = cloud.add_noise(stim_circuit, si1000_leakage_noise)
print(stim_text)
```

In this notebook, you have explored different ways of adding noise. To summarise:
- There is a simple phenomenological model, {class}`ToyNoise <deltakit.explorer.qpu.ToyNoise>`, which is a good starting point.
- The model {class}`PhysicalNoise <deltakit.explorer.qpu.PhysicalNoise>` is inspired by the physical properties of the QPU. It has many parameters and depends on gate times.
- There are models {class}`SD6Noise <deltakit.explorer.qpu.SD6Noise>` and {class}`SI1000Noise <deltakit.explorer.qpu.SI1000Noise>` which are inspired by the literature.
  `SI1000Noise` can also be used to simulate leakage noise using the cloud features.
- If you want to implement your noise model, you may use {class}`NoiseParameters <deltakit.explorer.qpu.NoiseParameters>` class.

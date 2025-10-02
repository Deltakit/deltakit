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

# Simulation

## 1. Simulation of Measurements

Decoding a quantum error correction experiment requires measurement data.
This can either be data simulated on a classical computer or data obtained from a QPU. 
If you have already run a quantum error correction experiment,
you may have measurement results in one of the formats.
Otherwise, you can use Deltakit and Stim to generate simulated measurement data, as described on this page.

You should prepare a quantum circuit that you want to simulate.
You can generate one by following the instructions of this documentation.
Or you may write circuits following the Stim circuit format.

### 1.1. Stim Circuit Simulation

In this chapter we consider the case,
when noise always preserves qubits in the computational space of $\vert 0\rangle$ and $\vert 1\rangle$.
Such Stim circuits can be simulated using the ({meth}`simulation.simulate_with_stim <deltakit.explorer.simulation.simulate_with_stim>`) method:

```{code-cell} ipython3
from deltakit.explorer import simulation
import numpy as np
import stim

simple_circuit = stim.Circuit("""
    RZ 0 1
    H 0
    CX 0 1
    MZ 0 1
    """
)

measurements, leakage = simulation.simulate_with_stim(simple_circuit, 1_000_000)
corr = np.corrcoef(measurements.as_numpy().T)[0, 1]
print(f"Bell state correlation = {corr:.4f}")
```

And now let's see how noise changes the correlation:

```{code-cell} ipython3
from deltakit.circuit import Circuit, gates
from deltakit.explorer.qpu import QPU, NativeGateSetAndTimes, ToyNoise

def add_noise(circuit: Circuit, amount: float = 0.01) -> Circuit:
    qpu = QPU(
        circuit.qubits,
        native_gates_and_times=NativeGateSetAndTimes(
            # times are given approximately for a superconducting device
            one_qubit_gates={gates.H: 20e-9, gates.SQRT_X: 20e-9},
            two_qubit_gates={gates.CZ: 30e-9},
            reset_gates={gates.RZ: 160e-9},
            measurement_gates={gates.MZ: 500e-9, gates.MRZ: 500e-9},
        ),
        noise_model=ToyNoise(p=amount)
    )
    return qpu.compile_and_add_noise_to_circuit(circuit, remove_paulis=False)
```

```{code-cell} ipython3
noisy_simple_circuit = add_noise(
    Circuit.from_stim_circuit(simple_circuit),
    0.02,
)
measurements, leakage = simulation.simulate_with_stim(
    noisy_simple_circuit.as_stim_circuit(), 1_000_000
)
corr = np.corrcoef(measurements.as_numpy().T)[0, 1]
print(f"Bell state correlation = {corr:.4f}")
```

You may use Deltakit to generate a full-scale QEC experiment circuit and simulate it.

```{code-cell} ipython3
from deltakit_explorer.codes import css_code_memory_circuit
from deltakit_explorer.codes import RotatedPlanarCode
from deltakit_circuit.gates import PauliBasis

rplanar = RotatedPlanarCode(width=3, height=3)
circuit = css_code_memory_circuit(rplanar, num_rounds=3, logical_basis=PauliBasis.Z)
noisy_rplanar_qmem = add_noise(circuit, 0.005)
measurements, leakage = simulation.simulate_with_stim(noisy_rplanar_qmem.as_stim_circuit(), 100_000)
with open("measurements.01", "w") as file:
    file.write(measurements.as_01_string())
```

### 1.2. Leakage Simulation

When a device, like a superconducting QPU, may experience leakage noise,
it is good to be able to model this.
Leakage is when a qubit leaves a computational space (e.g. becomes $\vert 2\rangle$).
Relaxation is when a qubit returns into a $\vert 0\rangle$ or $\vert 1\rangle$ state.

Good news is than superconducting devices at the stage of measurement and
state classification may also report if the qubit is in a leaked state.
This is called leakage heralding, and it produces an additional bit of information.

Leakage simulation and heralding are not supported by the Stim language.
Deltakit provides an extension to this language
which allows you to define leakage and relaxation events, as well as leakage heralding.

```
LEAKAGE(event_probability) [list of targets]
RELAX(event_probability) [list of targets]
HERALD_LEAKAGE_EVENT(error_probability) [list of targets]  
```

`HERALD_LEAKAGE_EVENT` commands typically go together with measurements.

To be able to run these commands, you will need to provide a cloud client object to the simulation call.
Note that you will have to use plain-text circuits, as Stim does not support these commands.

```{code-cell} ipython3
from deltakit.explorer import Client

cloud = Client.get_instance()

# in this circuit the qubit 0 will probably leak, but will relax
# after measurement
leaky_circuit= """
    RZ 0 1
    LEAKAGE(0.8) 0
    MZ 0 1
    HERALD_LEAKAGE_EVENT(0.01) 0 1
    RELAX(1.0) 0
    MZ 0 1
    HERALD_LEAKAGE_EVENT(0.01) 0 1
    """

# note, that leakage object is not None any more
measurements, leakage = simulation.simulate_with_stim(leaky_circuit, 30, client=cloud)
print("measurements (shots are columns):")
print(measurements.as_numpy().T)
print("leakage flags (shots are columns):")
print(leakage.as_numpy().T)
```

You may add leakage noise automatically to your circuit. For this, you may use Deltakit's modified `SI1000` noise model.

```{code-cell} ipython3
from deltakit.explorer import types

noise = types.SI1000NoiseModel(p=0.01, p_l=0.01)
# note: this circuit is not compatible with Stim any more
leaky_circuit = cloud.add_noise(circuit.as_stim_circuit(), noise_model=noise)
measurements, leakage = simulation.simulate_with_stim(leaky_circuit, 100_000, client=cloud)
```

## 2. Simulation of Detectors and Observables

### 2.1. Measurements to Syndromes

If you perform your experiments on hardware, or in other simulator (like qiskit), you will end up with a measurements file.
For example, a CSV or a 01 file with qubit measurements, one experiment per line.
Some of them will preserve the same order of measurements, some (like qiskit) may reverse or reorder them.
Before moving this data to Deltakit, please make sure the data is given **in the order of measurement**.

Decoders, however, need syndromes to predict.
In literature the word "syndrome" is used to name a collection of detector values.
So, after you've done a simulation or a QPU experiment,
you will need to convert measurements to syndromes.
You will also need a circuit to define the connection between them.

```{code-cell} ipython3
from deltakit.explorer import types, enums
from pathlib import Path

loaded_measurements = types.Measurements(Path("measurements.01"), enums.DataFormat.F01)

detectors, observables = loaded_measurements.to_detectors_and_observables(
    stim_circuit=noisy_rplanar_qmem.as_stim_circuit(),
)
print(detectors.as_numpy().shape)
```

### 2.2. Sampling Detectors

If you perform local simulation, you may rely on the built-in Stim sampler functionality
to generate syndromes and observables, avoiding the measurements construction stage. This is
typically faster. You cannot avoid measurements when you simulate with leakage.

You may also sample from the decoding graph (detector error model). This is useful
when the error model was defined without involving a circuit (e.g., restored from experimental data).

```{code-cell} ipython3
# sampling from a circuit
sampler = noisy_rplanar_qmem.as_stim_circuit().compile_detector_sampler(seed=1337)
detectors, observables = sampler.sample(1000, separate_observables=True)
# these objects can be passed to a cloud decoder or used to represent data in different formats
dets, obs = types.DetectionEvents(detectors), types.ObservableFlips(observables)
with open("dets_from_circuit_sampler.b8", "wb") as file:
    file.write(dets.as_b8_bytes())

# sampling from a detector error model
dem = noisy_rplanar_qmem.as_stim_circuit().detector_error_model()
dem_sampler = dem.compile_sampler(seed=1337)
detectors, observables, _ = dem_sampler.sample(shots=1000)
# these objects can be passed to a cloud decoder or used to represent data in different formats
dets, obs = types.DetectionEvents(detectors), types.ObservableFlips(observables)
with open("dets_from_dem_sampler.01", "w") as file:
    file.write(dets.as_01_string())
```

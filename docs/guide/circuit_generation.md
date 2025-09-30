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

# QEC Experiment Circuit Generation

Compilation of error-corrected quantum programs is a crucial task.
With advances in codes, decoders, and control systems, we know more and more about
how quantum algorithms designed for perfect qubits should be translated into
physical gates and programs for control systems.

We already know that future compilation approaches will consist of such building blocks as
logical gates, patches, and magic state factories.
Contemporary research combines them into
experiments.
QEC experiments show that these blocks are compatible and behave as theory
predicts.

They are also a powerful benchmark for hardware providers.
By running multiple such experiments you may observe both design flaws and optimal calibration parameters.
By running quality simulations with realistic noise models you may predict how the hardware will scale.

Deltakit provides you with a set of classes and functions, which enable various experiment generation.
You may use our existing functionality or build your own on top of existing code.

## 1. Generating a quantum memory experiments with planar codes

Quantum Memory Experiment (QMEM) is a core benchmark for both QPU hardware and decoders.
Logical Error Probability (LEP) value, derived from the experiment, is influenced by both
the quality of qubits and the capability of the decoder.

With Deltakit, you can generate experimental circuits, compatible with QPUs with different architectures and qubit-qubit connectivity.

### 1.1. Repetition code: minimal viable experiment

The code below generates the circuit for a 5-qubit repetition code, representing a single logical qubit.
5-qubit code is defined as distance-3 code: this is the number of error-corrected data qubits in this code.
We will generate 3 rounds of error correction, which means that detectors (auxiliary or ancillary qubits)
will be reset, entangled with data qubits, and measured. This will repeat 3 times.
In the end of the experiment, all data qubits will be measured as well. One of them will be used as
a representative of the logical qubit state.
In this experiment we assume, that data qubits are initialised as $\vert 000\rangle$ and should remain in the same state.
If one of them (chosen as an observable) due to noise in the system flips its value to $\vert 1\rangle$, then
a good decoder should be able to predict this event.

```{code-cell} ipython3
from deltakit_circuit import Coord2D
from deltakit_explorer.codes import css_code_memory_circuit, RepetitionCode, RotatedPlanarCode
from deltakit_circuit.gates import PauliBasis

rep_code = RepetitionCode(distance=3, stabiliser_type=PauliBasis.Z)
circuit = css_code_memory_circuit(
    css_code=rep_code,
    num_rounds=3,
    logical_basis=PauliBasis.Z,  # Z-basis stabilisers protect from bit flips
)
stim_circuit = circuit.as_stim_circuit()
text_stim_circuit = str(stim_circuit)
text_qasm_circuit = stim_circuit.to_qasm(open_qasm_version=3, skip_dets_and_obs=True)

stim_circuit.diagram("timeline-svg-html")
```

The repetition code above is not yet runnable on the real QPU hardware.
It has a mixture of reset and measurement gates in different bases,
its 2-qubit gate is chosen arbitrarily, and its qubits do not order in a chain.

To address these issues, use the ({class}`QPU class <deltakit.explorer.qpu.QPU>`).
This class encapsulates qubit grid, native gates, and gate noise model.
In this example we will create a noiseless QPU with exactly the needed number of qubits.

```{code-cell} ipython3
from deltakit_circuit import gates
from deltakit_decode.noise_sources import StimNoise
from deltakit_explorer.qpu import QPU, NativeGateSetAndTimes, ToyNoise, NoiseParameters

qpu_instance = QPU(
    qubits=circuit.qubits,  # only qubits we need are defined
    native_gates_and_times=NativeGateSetAndTimes(
        # times are given approximately for a superconducting device
        one_qubit_gates={gates.H: 20e-9, gates.SQRT_X: 20e-09},
        two_qubit_gates={gates.CZ: 30e-9},
        reset_gates={gates.RZ: 160e-9},
        measurement_gates={gates.MZ: 500e-9, gates.MRZ: 500e-9},
    ),
    # let's keep it noiseless for now, you may omit this line
    noise_model=NoiseParameters(),
)
deltakit_compiled_circuit = qpu_instance.compile_circuit(circuit)

# map qubit numbers with respect to their coordinates
mapping = {
    q: q.unique_identifier[0]
    for q in circuit.qubits
}
stim_circuit = deltakit_compiled_circuit.as_stim_circuit(mapping)
stim_circuit.diagram("timeline-svg-html")
```

In a more realistic example, ion trap devices are often using [Mølmer–Sørensen gate](https://en.wikipedia.org/wiki/M%C3%B8lmer%E2%80%93S%C3%B8rensen_gate), which in Stim is represented as `SQRT_XX`. You can use a ({class}`QPU class <deltakit.explorer.qpu.QPU>`) to compile down your circuit to use that gate.

```{code-cell} ipython3
ions_qpu_instance = QPU(
    qubits=circuit.qubits,
    native_gates_and_times=NativeGateSetAndTimes(
        # times here are 0, as we are not using them for noise definition
        one_qubit_gates={gates.SQRT_X: 0.0, gates.X: 0.0, gates.Z: 0.0, gates.SQRT_X_DAG: 0.0, gates.S_DAG: 0.0, gates.S: 0.0},
        two_qubit_gates={gates.SQRT_XX: 0.0},
        reset_gates={gates.RZ: 0.0},
        measurement_gates={gates.MZ: 0.0},
    ),
    # let's keep it noiseless for now, you may omit this line
    noise_model=NoiseParameters(),
)
deltakit_ions_compiled_circuit = ions_qpu_instance.compile_circuit(circuit)
deltakit_ions_compiled_circuit.as_stim_circuit().diagram("timeline-svg-html")
```

## 1.2. Planar codes

You can use Deltakit to generate the following planar codes:
- Rotated planar ({class}`RotatedPlanarCode <deltakit.explorer.codes.RotatedPlanarCode>`),
- Unrotated planar ({class}`UnrotatedPlanarCode <deltakit.explorer.codes.UnrotatedPlanarCode>`),
- Unrotated toric ({class}`UnrotatedToricCode <deltakit.explorer.codes.UnrotatedToricCode>`),
- Repetition codes ({class}`RepetitionCode <deltakit.explorer.codes.RepetitionCode>`).

A lot of contemporary experiments are built using the rotated planar code.
You can explore them in different native gate sets, schedules, and different patch locations.
A lot of parameters are optional and have reasonable default values. In the example below you see them all explicitly.

```{code-cell} ipython3
from deltakit_explorer.codes import ScheduleOrder, ScheduleType
from deltakit_circuit import Coord2DDelta

rotated_code = RotatedPlanarCode(
    width=3,
    height=3,
    schedule_type=ScheduleType.SIMULTANEOUS,
    schedule_order=ScheduleOrder.HORIZONTALLY_REFLECTED,
    use_ancilla_qubits=True,  # otherwise MPP gates will be used
    shift=Coord2DDelta(3, 1),  # you may move the patch
    horizontal_bump_with_top_left=True,  # bumps location and their type
    top_bumps_are_z=True,
)
rotated_code.draw_patch()
```

Now you can generate a quantum memory experiment using this code. Its native gates will be a mixture or `CX` and `CZ`.

```{code-cell} ipython3
circuit = css_code_memory_circuit(
    css_code=rotated_code,
    num_rounds=3,
    logical_basis=PauliBasis.Z,  # Z-basis stabilisers protect from bit flips
)
stim_circuit = circuit.as_stim_circuit()
stim_circuit.diagram("timeline-svg-html")
```

Some experiments explore `ISWAP` gates as the entangling gate.
`ISWAP` circuits are generated differently, so you have to express your desire to use these gates directly at code construction (and not when compiling).

Native `ISWAP` circuit generation is accessible using Deltakit Cloud API.

```{code-cell} ipython3
import stim
from deltakit.explorer import Client, types, enums
from deltakit.circuit import Circuit

cloud = Client.get_instance()
rotate_planar_2x2 = RotatedPlanarCode(width=2, height=2)
rotate_planar_2x2.draw_patch()
iswap_qmem_circuit = css_code_memory_circuit(
    css_code=rotate_planar_2x2,
    num_rounds=2,
    logical_basis=PauliBasis.Z,
    client=cloud,
    use_iswap_gates=True,
)
iswap_qmem_circuit.as_stim_circuit().diagram("timeline-svg-html")
```

And now you may compile these circuits within `*SWAP` gates domain.

```{code-cell} ipython3
czwap_qpu_instance = QPU(
    qubits=iswap_qmem_circuit.qubits,  # only qubits we need are defined
    native_gates_and_times=NativeGateSetAndTimes(
        # times are given approximately for a superconducting device
        one_qubit_gates={gates.H: 20e-9, gates.SQRT_X: 20e-09},
        two_qubit_gates={gates.CZSWAP: 30e-9},
        reset_gates={gates.RX: 160e-9},
        measurement_gates={gates.MX: 500e-9},
    ),
    # let's keep it noiseless for now, you may omit this line
    noise_model=NoiseParameters(),
)

deltakit_czswap_qmen = czwap_qpu_instance.compile_circuit(iswap_qmem_circuit)
deltakit_czswap_qmen.as_stim_circuit().diagram("timeline-svg-html")
```

## 2. Stability experiment

Memory experiments check how well logical observables are preserved through time.
[Stability experiments](https://arxiv.org/abs/2204.13834) check how well a quantum error
correction system can move logical observables through space.

Below you will generate a small stability experiment.
You will need an API client object to generate a stability experiments.

```{code-cell} ipython3
from deltakit_explorer.codes import css_code_stability_circuit

deltakit_stability = css_code_stability_circuit(
    css_code=RotatedPlanarCode(width=3, height=5),  # rectangular patch
    num_rounds=7,  # time dimension
    logical_basis=PauliBasis.Z,
    client=cloud,
)
stim_stability = deltakit_stability.as_stim_circuit()
```

## 3. qLDPC codes

You may use Deltakit to generate QEC experiments, which use qLDPC codes.
qLDPC codes are interesting to explore, as for the price of connectivity you may achieve higher density of logical qubits.
The code below will generate a memory experiment using the bivariate bicycle code.
You will compile it for the ion trap native gate set.
In this experiment you remove Pauli gates (`X`, `Y`, `Z`, `I`).
Pauli gate effect on the result of simulations can be tracked classically, so in the
context of a Stim simulation (or a simple experiment on a real QPU) they may be removed,
thus making execution faster (without loss of correctness).
You prefer to keep Pauli gates, if you plan to run the experiment in other simulators or using real QPUs.

```{code-cell} ipython3
from deltakit_explorer.codes import BivariateBicycleCode

code = BivariateBicycleCode(
    param_m=6,
    param_l=6,
    m_A_powers=[3, 1, 2],
    m_B_powers=[3, 1, 2],
)
textbook_circuit = css_code_memory_circuit(code, num_rounds=3, logical_basis=PauliBasis.Z)

ions_qpu_instance = QPU(
    qubits=textbook_circuit.qubits,
    native_gates_and_times=NativeGateSetAndTimes(
        # times here are 0, as we are not using them for noise definition
        one_qubit_gates={
            # support X and Z rotations:
            gates.X: 0.0, gates.SQRT_X: 0.0, gates.SQRT_X_DAG: 0.0,
            gates.Z: 0.0, gates.S: 0.0, gates.S_DAG: 0.0,
        },
        two_qubit_gates={gates.SQRT_XX: 0.0},
        reset_gates={gates.RZ: 0.0},
        measurement_gates={gates.MZ: 0.0},
    ),
)

ion_trap_qmem_qldpc = ions_qpu_instance.compile_circuit(
    textbook_circuit,
    remove_paulis=True,  # this is to remove Pauli gates from the circuits
)
ion_trap_qmem_qldpc.as_stim_circuit().diagram("timeline-3d")
```

## Creating your own codes

In this document you have generated Quantum Memory and Stability experiments with Repetition, Rotated Planar, and qLDPC codes in different native gate sets.
If you want to go beyond, you may implement your own QEC codes and experiments by inheriting from the base classes defined in Deltakit.

({class}`CSSCode <deltakit.explorer.codes.CSSCode>`) and its parent ({class}`StabiliserCode <deltakit.explorer.codes.StabiliserCode>`)
lay the ground to generation of stabiliser codes. All codes considered in this examples are build on top of these classes.

You may also want to build your own experiment.
Please refer to the implementation of ({func}`css_code_memory_circuit <deltakit.explorer.codes.css_code_memory_circuit>`) function
to see how ({class}`CSSStage <deltakit.explorer.codes.CSSStage>`) class can be used for experiment construction.

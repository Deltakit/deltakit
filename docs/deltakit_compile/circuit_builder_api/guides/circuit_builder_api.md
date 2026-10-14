---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  display_name: deltakit (3.12.x)
  language: python
  name: python3
---

# Getting Started with Circuit Builder API

The Circuit Builder API is a Python-Based Domain Specific Language (DSL) for defining the physical circuits used in QEC experiments. 

The Circuit Builder is designed for QEC experts, offering features such as automatic detector generation, composable circuit chunks, precise control of parallelism, and control flow, all while being completely code-agnostic. 

While it shares infrastructure with the higher-level Logical Assembly API (to learn more about LogASM, see [`logical_assembly_api.ipynb`](../../logical_assembly_api/guides/logical_assembly_api.ipynb)), Circuit Builder offers considerably more freedom in physical circuit design with the caveat that it provides fewer guarantees on the validity of the output.

```{code-cell} ipython3
# This file contains information which is proprietary to Riverlane Ltd
# ("Riverlane") and is Riverlane Confidential Information.
# (c) Copyright Riverlane 2025-2026. All rights reserved.

from deltakit_compile.frontend.circuit import (
    Circuit,
    CircuitBuilder,
    MeasurementReg,
    Observable,
    Pauli,
    QubitReg,
    ParallelAlignment,
    Result,
)
from deltakit_compile.frontend.logasm import LogAsmBuilder, RotatedPlanarPatch
from deltakit_compile.frontend.logical_assembler import (
    LogicalAssembler,
    LogicalAssemblerConfig,
)
from deltakit_compile.passes.stim.stim_export.pipeline import StimExportPipelineConfig
```

Let's start by seeing how to define a Bell Pair circuit using the Circuit Builder API.

```{code-cell} ipython3
# A circuit that forms a bell pair between two qubits (it is recommended to do this in a
# function to keep things compartmentalised and make type hinting clear)
def build_bell_pair_circ() -> Circuit[[QubitReg], MeasurementReg]:
    circ_builder = CircuitBuilder()
    qubits = circ_builder.add_arg(QubitReg(2))
    circ_builder.gate("R", qubits)
    circ_builder.gate("H", qubits[0])
    circ_builder.gate("CX", qubits)
    readouts = circ_builder.measure("Z", qubits)
    circ_builder.add_return(readouts)
    return circ_builder.build("bell_pair")


bell_pair = build_bell_pair_circ()
print(bell_pair)  # Prints the xDSL IR for the circuit
```

You add registers and qubits to a circuit using `CircuitBuilder` and then build a `Circuit` object by running `CircuitBuilder.build("circuit_name")`. The `Circuit` object is then used as an argument to `LogAsmBuilder.call_circuit` in a LogASM program, where the arguments to the circuit are objects returned by previous calls to the `LogAsmBuilder`. Printing the `LogAsmSubroutine` returned by `LogAsmBuilder.build("program_name")` shows the full program as xDSL IR, including the circuit.

```{code-cell} ipython3
# A LogASM program that calls the bell pair circuit on its qubits
builder = LogAsmBuilder()
qubits = builder.declare_patch(QubitReg(4))
readouts = builder.call_circuit(bell_pair(qubits[0:2]))
builder.add_return(readouts)

program = builder.build_program()
print(program)
```

The fundamental structure of a circuit is similar to that of a LogASM subroutine but with a special builder that has access to much lower level operations. Writing a circuit is analogous to writing a custom assembly function as part of a C program - you get significantly more control at the expense of safety and simplicity.

Compiling these circuits is just the same as compiling LogASM programs using the LogicalAssembler to deltakit's xDSL Physical Circuit IR. But you can also compile and export many circuits for Deltakit stim, as long as they are compatible with Deltakit stim's limitations.

```{code-cell} ipython3
result = LogicalAssembler(
    config=LogicalAssemblerConfig(stabiliser_flow_config=None),
).compile(program)
print("Bell Pair as Physical Circuit IR:")
print(result.program)

print()

deltakit_stim_assembler = LogicalAssembler(
    config=LogicalAssemblerConfig(
        export_config=StimExportPipelineConfig(), stabiliser_flow_config=None
    ),
)
deltakit_stim_result = deltakit_stim_assembler.compile(program)
print("Bell Pair exported for Deltakit stim:")
print(deltakit_stim_result.program)
```

As you can see from the exported program, a detector was automatically added to the circuit by the stabiliser flow pipeline.

+++



## Defining a Physical Circuit

While the patches in a LogASM program may have more specific definitions (rotated planar code, etc.), in a circuit everything is treated as a QubitReg with indexing/slicing used to retrieve specific Qubits from a register.

```{code-cell} ipython3
circ_builder = CircuitBuilder()
q = circ_builder.add_arg(QubitReg(25))

# Apply CX(0, 1) and CX(7, 4)
circ_builder.gate("CX", [*q[0:2], q[7], q[4]])
```

Here, qubits are not declared inside circuits, rather, they must be declared in the main LogASM program and passed in as arguments. A good option for improving circuit readability is separating large sets of qubits into registers with descriptive names.

When you are defining these registers, you can omit the number of qubits to create an unsized register. This allows the circuit to accept a `QubitReg` of any size. 

Moreover, qubits can also be measured and their measurements can either be used inside the circuit (detectors, observables) or returned to be used in the context the circuit is called in.

```{code-cell} ipython3
data_qubits = circ_builder.add_arg(QubitReg())
ancilla_qubits = circ_builder.add_arg(QubitReg())
circ_builder.gate("CZ", [ancilla_qubits[7], data_qubits[2]])

ro = circ_builder.measure("Z", q[2])
_ = circ_builder.add_return(ro)
```

Measurement records (append-only lists of measurements) can be declared for later use, such as in repeats (more on this in the control flow section) and for measurement grouping (e.g. per qubit).

```{code-cell} ipython3
q2_rec = circ_builder.declare_record()
ro = circ_builder.measure("Z", q[2])
q2_rec.append(ro)
assert ro == q2_rec[-1]
```

When grouping measurements into rounds, the logical assembler, by default, will assume that these measurements are performed in parallel and they are part of the same round. 

If this is not the case (e.g. when using spaced scheduling), measurements can be explicitly assigned to rounds.

```{code-cell} ipython3
ro0 = circ_builder.measure("Z", q[2])
circ_builder.gate("H", q[1])
ro1 = circ_builder.measure("Z", q[3])
circ_builder.measurement_round(ro0, ro1)
```

You can call circuits from other circuit objects. This is made to allow an intelligent composition of reusable circuit fragments.

When defining multiple circuits, it is strongly encouraged that each circuit is defined in a separate function to make them more readable and reusable. A particularly strong benefit to this approach is clear and explicit return typing of the arguments and returns of the circuit when called.

```{code-cell} ipython3
def build_reset_circ(qubit_reg_size: int) -> Circuit[[QubitReg], None]:
    """Build a circuit that resets a qubit reg."""
    circ_builder = CircuitBuilder()
    qubits = circ_builder.add_arg(QubitReg(qubit_reg_size))
    circ_builder.gate("R", qubits)
    return circ_builder.build("reset")


def build_measure_circ(qubit_reg_size: int) -> Circuit[[QubitReg], MeasurementReg]:
    """Circuit that measures out a qubit reg."""
    circ_builder = CircuitBuilder()
    qubits = circ_builder.add_arg(QubitReg(qubit_reg_size))
    ro = circ_builder.measure("Z", qubits)
    circ_builder.add_return(ro)
    return circ_builder.build("measure")


def build_compound_circ(
    qubit_reg_size: int,
) -> Circuit[[QubitReg, QubitReg], tuple[MeasurementReg, MeasurementReg]]:
    """Circuit that applies the previous circuits to multiple patches."""
    circ_builder = CircuitBuilder()
    reset = build_reset_circ(qubit_reg_size)
    measure = build_measure_circ(qubit_reg_size)

    p0 = circ_builder.add_arg(QubitReg(qubit_reg_size))
    p1 = circ_builder.add_arg(QubitReg(qubit_reg_size))
    circ_builder.call_circuit(reset(p0))
    p0_ro = circ_builder.call_circuit(measure(p0))
    circ_builder.call_circuit(reset(p1))
    p1_ro = circ_builder.call_circuit(measure(p1))
    circ_builder.add_return(p0_ro)
    circ_builder.add_return(p1_ro)
    return circ_builder.build("compound_circ")


qubit_reg_size = 4
builder = LogAsmBuilder()
compound_circ = build_compound_circ(qubit_reg_size)

qreg0 = builder.declare_patch(QubitReg(qubit_reg_size))
qreg1 = builder.declare_patch(QubitReg(qubit_reg_size))
qreg0_ro, qreg1_ro = builder.call_circuit(compound_circ(qreg0, qreg1))
builder.add_return(qreg0_ro)
builder.add_return(qreg1_ro)

print(builder.build_program())
```

## Detectors

There are three ways to add detectors to your circuit object:
- Define stabiliser flows for your circuit (the logical assembler will verify them and compute the corresponding detectors).
- Define detectors by hand (the logical assembler will attempt to verify them but otherwise take them as-is).
- Don’t define anything (the logical assembler will attempt to compute stabiliser flows from the circuit’s physical gates, followed by computing detectors from the stabiliser flows).

The recommended method is defining your own stabiliser flows, as that gives the logical assembler more to work with and it can verify that the flows you’ve defined are valid.

```{code-cell} ipython3
circ_builder = CircuitBuilder()
data = circ_builder.add_arg(QubitReg(2))
ancilla = circ_builder.add_arg(QubitReg(1))
circ_builder.gate("R", ancilla)
circ_builder.gate("CX", [data[0], ancilla[0]])
circ_builder.gate("CX", [data[1], ancilla[0]])

ro = circ_builder.measure("Z", ancilla[0])
circ_builder.add_return(ro)
circ_builder.add_flow(
    input_paulis={data[0]: Pauli.Z, data[1]: Pauli.Z},
    output_paulis={data[0]: Pauli.Z},
    measurements=[ro],
    parity=True,
)

print(circ_builder.build("flow_circ"))
```

If you choose to define the detectors by hand, the logical assembler is going to “take your word for it” so you will get fewer guarantees on whether the compiled output works.

```{code-cell} ipython3
qubits = circ_builder.add_arg(QubitReg(4))
ro = circ_builder.measure("Z", qubits)
det0 = circ_builder.detector([ro[0], ro[2]], coordinates=(0, 0))
det1 = circ_builder.detector([ro[1], ro[3]], coordinates=(0, 1))
circ_builder.detector_round(det0, det1)
```

Notice the difference in how detectors are defined here compared to the traditional way (in e.g. Stim) is that we make detector rounds explicit rather than by convention using a 3rd coordinate dimension to represent time. 

This is made to reduce ambiguity further down the compilation flow. Defining coordinates is still possible, they are just not considered meaningful by the logical assembler. Similarly, in scenarios where detector time coordinates would not be useful detector rounds are not required.

## Observables

The recommended way to define observables is by specifying the qubits the logical sheet is on and how it moves.

```{code-cell} ipython3
circ_builder = CircuitBuilder()
qubits = circ_builder.add_arg(QubitReg(17))
obs = circ_builder.declare_observable(qubits[0:3])  # Observable on 3 qubits

# Gates happen...

# At this point the observable moves to 4 different qubits through a
# series of measurements
ro = circ_builder.measure("Z", qubits[3:7])
obs.move(qubits[7:11], [ro[0], ro[2]])
```

An observable may also be defined at a lower level using Stim-like observable includes.

```{code-cell} ipython3
ro = circ_builder.measure("Z", qubits)
obs = circ_builder.declare_observable()
obs.include([ro[0], ro[3], ro[4]])
```

You can ask an observable for its uncorrected value, logical correction, or corrected value.

```{code-cell} ipython3
uncorrected = obs.get_uncorrected()

# These will block circuit execution until decoding has finished
correction = obs.get_correction()
corrected = obs.get_corrected()
```

Waiting for the correction or corrected observable value can be done in a non-blocking fashion by using control flow in the main LogASM program (more on control flow later).

```{code-cell} ipython3
def build_circ() -> Circuit[[QubitReg], Observable]:
    circ_builder = CircuitBuilder()
    qubits = circ_builder.add_arg(QubitReg(17))
    obs = circ_builder.declare_observable([qubits[0], qubits[2], qubits[10]])
    # Do gates...
    circ_builder.add_return(obs)
    return circ_builder.build("circ")


builder = LogAsmBuilder()
circ = build_circ()

p0 = builder.declare_patch(RotatedPlanarPatch(3, 3))
p1 = builder.declare_patch(RotatedPlanarPatch(3, 3))
obs: Observable = builder.call_circuit(circ(p0))

# Do memory on p1 until p0 correction has been decoded
# TODO: Support context managers
# with builder.while_(not obs.correction_ready()):
#     p1.measure_stabilisers(1)

p0_val = obs.get_corrected()
```

There might be cases where the observable is expected to be defined before the circuit, and the circuit needs to take that observable as an argument and potentially change it before returning it. 

To support this, the circuit can declare the qubits at which it is expecting the observable to be supported on, and the qubits on which it would be returning that observable.

```{code-cell} ipython3
# TODO: Implement observable interfaces
# circ_builder = CircuitBuilder()
# qubits = circ_builder.add_arg(QubitReg(10))
# observable = circ_builder.add_interface_observable(
#     inp=PauliString(
#         {qubits[0]: Pauli.X, qubits[2]: Pauli.X, qubits[5]: Pauli.X}
#     ),
#     out=PauliString(
#         {qubits[0]: Pauli.X, qubits[2]: Pauli.X, qubits[5]: Pauli.X}
#     ),
# )
# circ = circ_builder.build("passing_through_obs")
# Can be called like a normal circuit, the observable interface is just
# here to make that circuit well-behaved in a log_asm context.
#     builder.call_circuit(circ(qreg))
```

## Control Flow

The only control flow supported inside a circuit is a repeat with a fixed number of repetitions.

```{code-cell} ipython3
circ_builder = CircuitBuilder()
qubits = circ_builder.add_arg(QubitReg(10))

q0_rec = circ_builder.declare_record()
q0_rec.append(circ_builder.measure("Z", qubits[0]))
# TODO: Support context managers
# with circ_builder.repeat(10):
#     circ_builder.gate("X", qubits)
#     ro = circ_builder.measure("Z", qubits[0])
#     det = circ_builder.detector([ro, q0_rec[-1]])
#     circ_builder.detector_round(det)
#     q0_rec.append(ro)
```

If you wish to mix other types of control flow with circuits, you will need to do it at the LogASM program level. This may require splitting what you would typically consider one circuit into multiple circuit chunks.

```{code-cell} ipython3
def build_logical_meas() -> Circuit[[QubitReg], Result]:
    circ_builder = CircuitBuilder()
    qubits = circ_builder.add_arg(QubitReg(17))
    obs = circ_builder.declare_observable(qubits[0:3])
    circ_builder.measure("Z", qubits[0:3])
    corrected_obs = obs.get_corrected()
    circ_builder.add_return(corrected_obs)
    return circ_builder.build("logical_meas")


def build_flip() -> Circuit[[QubitReg], None]:
    circ_builder = CircuitBuilder()
    qubits = circ_builder.add_arg(QubitReg(17))
    circ_builder.gate("X", qubits)
    return circ_builder.build("flip")


builder = LogAsmBuilder()
logical_meas = build_logical_meas()
flip = build_flip()

qubits = builder.declare_patch(QubitReg(17))
obs_val = builder.call_circuit(logical_meas(qubits))
# TODO: Support context managers
# with builder.if_(obs_val == 1):
#     builder.call_circuit(flip(qubits))

obs_val = builder.call_circuit(logical_meas(qubits))
builder.add_return(obs_val)

print(builder.build_program())
```

## Parallelism

In contrast to the implicit parallelism in normal LogASM programs, parallelism inside circuits is entirely explicit.

```{code-cell} ipython3
circ_builder = CircuitBuilder()
qubits = circ_builder.add_arg(QubitReg(2))

with circ_builder.parallel() as p:  # The Hadamard gate and measure start at the same time
    with p():
        circ_builder.gate("H", qubits[0])
        circ_builder.gate("X", qubits[0])
    with p():
        ro = circ_builder.measure(Pauli.X, qubits[1])
```

As with the context managers defined for control flow, this only puts circuit builder related statements in parallel - other Python statements will be executed in series at Python runtime like normal.

Parallels have multiple options for how their contents are aligned:
- TOP (default): Parallel operations/circuits start at the same time
- BOTTOM: Parallel operations/circuits end at the same time
- LOCKSTEP: The contents of parallel circuits are aligned operation-by-operation

LOCKSTEP alignment operates similarly to a Python zip, specifically `itertools.zip_longest` as lockstepping circuits of uneven length doesn’t truncate the longer circuit, the shorter circuit stops early like in TOP alignment.

```{code-cell} ipython3
circ_builder = CircuitBuilder()
p1 = circ_builder.add_arg(QubitReg())
p2 = circ_builder.add_arg(QubitReg())
p3 = circ_builder.add_arg(QubitReg())
with circ_builder.parallel(align=ParallelAlignment.LOCKSTEP) as p:
    with p():
        circ_builder.gate("X", p1)
        circ_builder.gate("H", p1)
    with p():
        circ_builder.gate("Z", p2)
        circ_builder.gate("CX", p2)
        circ_builder.gate("CX", p2)
    with p():
        circ_builder.gate("H", p3)

# Time steps of execution:
# 0: X(p1)   Z(p2)   H(p3)
# 1: H(p1)   CX(p2)
# 2:         CX(p2)
```

The types of circuit operations that will lockstep are: gates, measurements, sub-circuits, parallels, and repeats. This means that things like detector and observable definitions (which are really forms of annotation) will not be put into time steps, but the entire contents of a repeat or circuit call would be put into a single time step.

```{code-cell} ipython3
circ_builder = CircuitBuilder()
p1 = circ_builder.add_arg(QubitReg())
p2 = circ_builder.add_arg(QubitReg())
# TODO: Support lockstep
# with circ_builder.parallel(align=ParallelAlignment.LOCKSTEP) as p:
#     with p():
#         with circ_builder.repeat(3):
#             circ_builder.gate("X", p0)
#     with p():
#         circ_builder.gate("Z", p1)
#         circ_builder.gate("CX", p1)

# Time steps of execution:
# 0: X(p1)   Z(p2)
#    X(p1)
#    X(p1)
# 1:         CX(p2)
```

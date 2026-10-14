# Quantum Memory with Circuit Builder

```{code-cell} ipython3
# This file contains information which is proprietary to Riverlane Ltd
# ("Riverlane") and is Riverlane Confidential Information.
# (c) Copyright Riverlane 2025-2026. All rights reserved.
```

Example of using the Circuit Builder to generate a Stim file containing a quantum memory
experiment.

```{code-cell} ipython3
import argparse
import sys
from enum import IntEnum
from pathlib import Path
from typing import Literal

from deltakit_compile.frontend.circuit import (
    Circuit,
    CircuitBuilder,
    ParallelAlignment,
    Qubit,
    QubitReg,
    Result,
)
from deltakit_compile.frontend.logasm import LogAsmBuilder, LogAsmProgram
from deltakit_compile.frontend.logical_assembler import LogicalAssembler, LogicalAssemblerConfig
from deltakit_compile.passes.stim.stim_export.pipeline import StimExportPipelineConfig
```

```{code-cell} ipython3
# The 4 schedule slots for a single stabiliser round, in order; None marks an idle slot.
StabiliserSchedule = tuple[Qubit | None, Qubit | None, Qubit | None, Qubit | None]
```

```{code-cell} ipython3
def build_prepare(basis: Literal["X", "Y", "Z"]) -> Circuit[[QubitReg], None]:
    """Build a circuit that prepares a patch of any size in the provided basis."""
    builder = CircuitBuilder()
    qubits = builder.add_arg(QubitReg())
    builder.gate("R" + basis, qubits)
    return builder.build("prepare")
```

```{code-cell} ipython3
def inline_stabiliser(
    ancilla: Qubit,
    data: StabiliserSchedule,
    basis: Literal["X", "Z"],
    builder: CircuitBuilder,
) -> None:
    """Add the ops for a single round of a stabiliser to the provided builder. The schedule is
    implicit in `data`'s order, with None entries producing an idle gate on the ancilla instead
    of a two-qubit gate."""
    for trgt in data:
        if trgt is None:
            builder.gate("I", ancilla)
        else:
            builder.gate(f"C{basis}", [ancilla, trgt])
```

```{code-cell} ipython3
def inline_stabiliser_2q_gates(
    x_stabiliser_qubits: dict[Qubit, StabiliserSchedule],
    z_stabiliser_qubits: dict[Qubit, StabiliserSchedule],
    builder: CircuitBuilder,
) -> None:
    """Add the two qubit ops for a single round of stabiliser measurement for the full patch to the
    provided builder."""

    with builder.parallel(ParallelAlignment.LOCKSTEP) as p:
        for anc, data in x_stabiliser_qubits.items():
            with p():
                inline_stabiliser(anc, data, "X", builder)
        for anc, data in z_stabiliser_qubits.items():
            with p():
                inline_stabiliser(anc, data, "Z", builder)
```

```{code-cell} ipython3
class StabRound(IntEnum):
    FIRST = 0
    MIDDLE = 1
    LAST = 2
```

```{code-cell} ipython3
def build_measure_stabilisers(stab_round: StabRound) -> Circuit[[QubitReg], None]:
    """Build a circuit that measures stabilisers on a d3 patch for a single round."""
    builder = CircuitBuilder()
    qubits = builder.add_arg(QubitReg(17))

    # Define the ordered schedule for each stabiliser
    x_stabiliser_qubits: dict[Qubit, StabiliserSchedule] = {
        qubits[9]: (qubits[1], qubits[4], qubits[0], qubits[3]),
        qubits[12]: (qubits[5], qubits[8], qubits[4], qubits[7]),
        qubits[13]: (None, None, qubits[2], qubits[5]),
        qubits[15]: (qubits[3], qubits[6], None, None),
    }
    z_stabiliser_qubits: dict[Qubit, StabiliserSchedule] = {
        qubits[10]: (qubits[2], qubits[1], qubits[5], qubits[4]),
        qubits[11]: (qubits[4], qubits[3], qubits[7], qubits[6]),
        qubits[14]: (qubits[8], qubits[7], None, None),
        qubits[16]: (None, None, qubits[1], qubits[0]),
    }
    all_stabiliser_qubits = x_stabiliser_qubits | z_stabiliser_qubits

    if stab_round != StabRound.FIRST:
        builder.gate("RX", list(all_stabiliser_qubits.keys()))

    inline_stabiliser_2q_gates(x_stabiliser_qubits, z_stabiliser_qubits, builder)

    if stab_round != StabRound.LAST:
        builder.measure("X", list(all_stabiliser_qubits.keys()))

    return builder.build(f"{stab_round.name.lower()}_measure_stabilisers")
```

```{code-cell} ipython3
def build_measure(basis: Literal["X", "Y", "Z"]) -> Circuit[[QubitReg], Result]:
    """Build a circuit that measures out the logical observable of a d3 patch."""
    builder = CircuitBuilder()
    qubits = builder.add_arg(QubitReg(17))
    readouts = builder.measure(basis, qubits)

    # Include the readouts from the first column in the observable
    obs = builder.declare_observable()
    obs.include(readouts[0:3])
    log = obs.get_corrected()

    builder.add_return(log)
    return builder.build("measure")
```

```{code-cell} ipython3
def build_quantum_memory(stab_rounds: int) -> LogAsmProgram:
    r"""
    Build a LogASM program that implements a d3 quantum memory experiment.

    The qubit IDs of the patch are laid out as follows::

    3   |           13
        |          /   \
        |        2-------5-------8
        |        |       |       | \
    2   |        |  10   |  12   |  14
        |        |       |       | /
        |        1-------4-------7
        |      / |       |       |
    1   |   16   |   9   |  11   |
        |      \ |       |       |
        |        0-------3-------6
        |                  \   /
    0   |                   15
        +-----------------------------=
            0       1       2       3

    Args:
        stab_rounds: The number of rounds of stabiliser measurement.

    Returns:
        Instantiated subroutine object to be provided to the LogicalAssembler.
    """
    builder = LogAsmBuilder()
    prepare = build_prepare("X")
    first_measure_stabilisers = build_measure_stabilisers(StabRound.FIRST)
    middle_measure_stabilisers = build_measure_stabilisers(StabRound.MIDDLE)
    last_measure_stabilisers = build_measure_stabilisers(StabRound.LAST)
    measure = build_measure("X")

    qubit_locations = [
        (0.5, 0.5),
        (0.5, 1.5),
        (0.5, 2.5),
        (1.5, 0.5),
        (1.5, 1.5),
        (1.5, 2.5),
        (2.5, 0.5),
        (2.5, 1.5),
        (2.5, 2.5),
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (1, 3),
        (3, 2),
        (2, 0),
        (0, 1),
    ]
    patch = builder.declare_patch(QubitReg(17, qubit_locations=qubit_locations))
    builder.call_circuit(prepare(patch))
    for rnd in range(stab_rounds):
        if rnd == 0:
            builder.call_circuit(first_measure_stabilisers(patch))
        elif rnd == stab_rounds - 1:
            builder.call_circuit(last_measure_stabilisers(patch))
        else:
            builder.call_circuit(middle_measure_stabilisers(patch))
    log = builder.call_circuit(measure(patch))
    builder.add_return(log)

    return builder.build_program()
```

```{code-cell} ipython3
qmem_circuit = build_quantum_memory(5)
```

```{code-cell} ipython3
assembler = LogicalAssembler(LogicalAssemblerConfig(export_config=StimExportPipelineConfig()))
compiled_circuit = assembler.compile(qmem_circuit)
print(compiled_circuit.program)
```

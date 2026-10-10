# CNOT Gate

+++

An example of using the Logical Assembler to generate a circuit containing a logical CNOT.

```{code-cell} ipython3
# This file contains information which is proprietary to Riverlane Ltd
# ("Riverlane") and is Riverlane Confidential Information.
# (c) Copyright Riverlane 2025-2026. All rights reserved.
from typing import Literal
from deltakit_compile.frontend.logasm import LogAsmBuilder, LogAsmProgram, RotatedPlanarPatch
```

```{code-cell} ipython3
def build_cnot(
    width: int,
    height: int,
    prep_bases: tuple[Literal["X", "Y", "Z"], Literal["X", "Y", "Z"]] = ("Z", "Z"),
    meas_bases: tuple[Literal["X", "Y", "Z"], Literal["X", "Y", "Z"]] = ("Z", "Z"),
) -> LogAsmProgram:
    """
    Build a LogASM program that implements a logical CNOT gate.

    Args:
        width: The width of the patches involved.
        height: The height of the patches involved.
        prep_bases: Preparation bases of the control and target patches respectively.
        meas_bases: Measurement bases of the control and target patches respectively.

    Returns:
        Instantiated subroutine object to be provided to the LogicalAssembler.
    """
    builder = LogAsmBuilder()
    lq0 = builder.declare_patch(
        RotatedPlanarPatch(width=width, height=height, location=(0, 0), vertical_z=False)
    )
    anc = builder.declare_patch(
        RotatedPlanarPatch(width=width, height=height, location=(width + 1, 0), vertical_z=False)
    )
    lq1 = builder.declare_patch(
        RotatedPlanarPatch(
            width=width, height=height, location=(width + 1, height + 1), vertical_z=False
        )
    )
    bridge0anc = builder.declare_patch(
        RotatedPlanarPatch(width=1, height=height, location=(width, 0), vertical_z=False)
    )
    bridge1anc = builder.declare_patch(
        RotatedPlanarPatch(width=width, height=1, location=(width + 1, height), vertical_z=False)
    )
    stab_rounds = max(width, height)
    # Patch prepares - these are arbitrary for lq0 and lq1, necessarily X for ancilla
    lq0.prepare(prep_bases[0])
    lq1.prepare(prep_bases[1])
    anc.prepare("X")

    # Measure stabilisers
    lq0.measure_stabilisers(stab_rounds)
    lq1.measure_stabilisers(stab_rounds)
    anc.measure_stabilisers(stab_rounds)

    # Measure lq1 an extra 5 rounds
    lq1.measure_stabilisers(stab_rounds)

    # Multi Pauli measurement
    builder.multi_pauli_measure(
        operand_patches=[lq0, anc],
        bridges=[bridge0anc],
        rounds=stab_rounds,
        pauli_bases=["Z", "Z"],
    )

    # Measure lq0 an extra 5 rounds
    lq0.measure_stabilisers(stab_rounds)

    # Multi Pauli measurement
    builder.multi_pauli_measure(
        operand_patches=[lq1, anc],
        bridges=[bridge1anc],
        rounds=stab_rounds,
        pauli_bases=["X", "X"],
    )

    # Measure ancilla qubit (fixed basis - ancilla)
    anc.measure("Z")

    # Measure lq0/lq1 for an extra d rounds
    lq1.measure_stabilisers(stab_rounds)
    lq0.measure_stabilisers(stab_rounds)

    # Measure lq0 and lq1 (arbitrary basis)

    lq1.measure(meas_bases[0])
    lq0.measure(meas_bases[1])

    return builder.build_program()
```

```{code-cell} ipython3
cnot_gate = build_cnot(3, 3, ("Z", "Z"), ("Z", "Z"))
print(cnot_gate)
```

# GHZ state

+++

Example of using the Logical Assembler to generate a circuit containing GHZ state prep.

```{code-cell} ipython3
# This file contains information which is proprietary to Riverlane Ltd
# ("Riverlane") and is Riverlane Confidential Information.
# (c) Copyright Riverlane 2025-2026. All rights reserved.
from deltakit_compile.frontend.logasm import LogAsmBuilder, LogAsmProgram, RotatedPlanarPatch
```

```{code-cell} ipython3
def build_ghz_prep(width: int, height: int) -> LogAsmProgram:
    """
    Build a LogASM program that implements a GHZ state prep circuit.

    Args:
        width: The width of the patches involved.
        height: The height of the patches involved.

    Returns:
        Instantiated subroutine object to be provided to the LogicalAssembler.

    """
    builder = LogAsmBuilder()

    # Patch declarations, lq0, bridge, lq1, bridge, lq2, bridge
    lq0 = builder.declare_patch(
        RotatedPlanarPatch(width=width, height=height, location=(0, 0), vertical_z=False)
    )
    bridge01 = builder.declare_patch(
        RotatedPlanarPatch(width=1, height=height, location=(width, 0), vertical_z=False)
    )
    lq1 = builder.declare_patch(
        RotatedPlanarPatch(width=width, height=height, location=(width + 1, 0), vertical_z=False)
    )
    bridge12 = builder.declare_patch(
        RotatedPlanarPatch(width=1, height=height, location=(2 * width + 1, 0), vertical_z=False)
    )
    lq2 = builder.declare_patch(
        RotatedPlanarPatch(
            width=width, height=height, location=(2 * width + 2, 0), vertical_z=False
        )
    )
    stab_rounds = max(width, height)

    # Prepare all 3 qubits in |+> state (+X)
    lq0.prepare("X")
    lq1.prepare("X")
    lq2.prepare("X")

    # Measure all stabilisers for d rounds
    lq0.measure_stabilisers(stab_rounds)
    lq1.measure_stabilisers(stab_rounds)
    lq2.measure_stabilisers(stab_rounds)

    # Multi-pauli Z_L Z_L measurements between qubits 1 and 2 and 0 and 1
    # Measure stabilisers for lq0 while we do MPP on 1 and 2
    lq0.measure_stabilisers(stab_rounds)
    builder.multi_pauli_measure(
        operand_patches=[lq1, lq2],
        bridges=[bridge12],
        rounds=stab_rounds,
        pauli_bases=["Z", "Z"],
    )

    # Measure stabilisers for lq2 while we do MPP on 0 and 1
    lq2.measure_stabilisers(stab_rounds)
    builder.multi_pauli_measure(
        operand_patches=[lq0, lq1],
        bridges=[bridge01],
        rounds=stab_rounds,
        pauli_bases=["Z", "Z"],
    )

    # Further d rounds of stabiliser measurement on all qubits
    lq0.measure_stabilisers(stab_rounds)
    lq1.measure_stabilisers(stab_rounds)
    lq2.measure_stabilisers(stab_rounds)

    return builder.build_program()
```

```{code-cell} ipython3

ghz_prep = build_ghz_prep(3, 3)
print(ghz_prep)
```

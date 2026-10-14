# Lattice Surgery

+++

Example of using the Logical Assembler to generate a Stim circuit containing a ZZ-Lattice surgery
experiment.

```{code-cell} ipython3
# This file contains information which is proprietary to Riverlane Ltd
# ("Riverlane") and is Riverlane Confidential Information.
# (c) Copyright Riverlane 2025-2026. All rights reserved.
from deltakit_compile.frontend.logasm import LogAsmBuilder, LogAsmProgram, RotatedPlanarPatch
```

```{code-cell} ipython3
def build_lattice_surgery(width: int, height: int, bridge_columns: int = 1) -> LogAsmProgram:
    """
    Build a LogASM program that implements ZZ-lattice surgery.

    Args:
        width: The width of the patches involved.
        height: The height of the patches involved.
        bridge_columns: The number of data qubit columns in the bridge between the patches.

    Returns:
        Instantiated subroutine object to be provided to the LogicalAssembler.
    """
    builder = LogAsmBuilder()
    # Patch declarations, lq0 and lq1 with a bridge in between them.
    lq0 = builder.declare_patch(
        RotatedPlanarPatch(width=width, height=height, location=(0, 0), vertical_z=False)
    )
    lq1 = builder.declare_patch(
        RotatedPlanarPatch(
            width=width, height=height, location=(width + bridge_columns, 0), vertical_z=False
        )
    )
    bridge = builder.declare_patch(
        RotatedPlanarPatch(
            width=bridge_columns, height=height, location=(width, 0), vertical_z=False
        )
    )
    stab_rounds = max(width, height)

    # Prepare the logical patches in the Z basis
    lq0.prepare("Z")
    lq1.prepare("Z")

    lq0.measure_stabilisers(stab_rounds)
    lq1.measure_stabilisers(stab_rounds)

    # Do a multi pauli ZZ measurements across the X boundaries of lq0 and lq1 via the bridge
    builder.multi_pauli_measure(
        operand_patches=[lq0, lq1],
        bridges=[bridge],
        rounds=stab_rounds,
        pauli_bases=["Z", "Z"],
    )

    lq0.measure_stabilisers(stab_rounds)
    lq1.measure_stabilisers(stab_rounds)

    # Measure the patches in the Z basis
    lq0.measure("Z")
    lq1.measure("Z")

    return builder.build_program()
```

```{code-cell} ipython3
lattice_surgery_circuit = build_lattice_surgery(3, 3, 1)
print(lattice_surgery_circuit)
```

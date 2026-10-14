# Hadamard Gate

+++

Example of using the Logical Assembler to generate a circuit containing a logical
Hadamard.

```{code-cell} ipython3
# This file contains information which is proprietary to Riverlane Ltd
# ("Riverlane") and is Riverlane Confidential Information.
# (c) Copyright Riverlane 2025-2026. All rights reserved.
from deltakit_compile.frontend.logasm import LogAsmBuilder, LogAsmProgram, RotatedPlanarPatch
```

```{code-cell} ipython3
def build_hadamard(width: int, height: int) -> LogAsmProgram:
    """
    Build a LogASM program that implements a logical Hadamard gate.

    Args:
        width: The width of the patch.
        height: The height of the patch.

    Returns:
        Instantiated subroutine object to be provided to the LogicalAssembler.
    """
    builder = LogAsmBuilder()
    lq0 = builder.declare_patch(
        RotatedPlanarPatch(width=width, height=height, location=(0, 0), vertical_z=False)
    )
    stab_rounds = max(width, height)

    lq0.prepare("Z")
    lq0.measure_stabilisers(stab_rounds)

    # Apply the transversal hadamard gate, which implicitly rotates the patch
    lq0.transversal("H")

    # Rotate the patch back to its original orientation, changing its location in the process
    lq0.rotate(offset=(width, 0))
    lq0.measure("X")

    return builder.build_program()
```

```{code-cell} ipython3
hadamard_circuit = build_hadamard(3, 3)
print(hadamard_circuit)
```

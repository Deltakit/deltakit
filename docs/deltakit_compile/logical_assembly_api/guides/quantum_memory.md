# Quantum Memory with Logical Assembly

+++

Example of using the Logical Assembler to generate a circuit containing a quantum memory
experiment.

```{code-cell} ipython3
# This file contains information which is proprietary to Riverlane Ltd
# ("Riverlane") and is Riverlane Confidential Information.
# (c) Copyright Riverlane 2025-2026. All rights reserved.
from deltakit_compile.frontend.logasm import LogAsmBuilder, LogAsmProgram, RotatedPlanarPatch
```

```{code-cell} ipython3
def build_quantum_memory(width: int, height: int, stab_rounds: int) -> LogAsmProgram:
    """
    Build a LogASM program that implements a quantum memory experiment.

    Args:
        width: The width of the patch.
        height: The height of the patch.
        stab_rounds: The number of rounds of stabiliser measurement.

    Returns:
        Instantiated subroutine object to be provided to the LogicalAssembler.
    """
    builder = LogAsmBuilder()
    patch = builder.declare_patch(RotatedPlanarPatch(width=width, height=height, location=(0, 0)))

    patch.prepare("Z")
    patch.measure_stabilisers(stab_rounds)
    ro = patch.measure("Z")
    builder.add_return(ro)

    return builder.build_program()
```

```{code-cell} ipython3
memory_circuit = build_quantum_memory(width=3, height=3, stab_rounds=10)
print(memory_circuit)
```

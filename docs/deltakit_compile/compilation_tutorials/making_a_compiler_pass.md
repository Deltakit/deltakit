---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

# Making a Simple Compiler Pass with xDSL

This tutorial will describe how to make a simple xDSL pass that replaces measurements such that they use a new measurement basis while maintaining a logically equivalent program.
This pass will operate on our `stim` dialect so first we'll familiarise ourselves with the `stim` dialect operations and how they map to and from Deltakit stim circuits:

```{code-cell} ipython3
# This file contains information which is proprietary to Riverlane Ltd
# ("Riverlane") and is Riverlane Confidential Information.
# (c) Copyright Riverlane 2025-2026. All rights reserved.
from deltakit_stim import Circuit
from xdsl.builder import Builder
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue
from xdsl.parser import IntegerType
from xdsl.rewriter import InsertPoint

from deltakit_compile.dialects import stim
from deltakit_compile.frontend.deltakit_stim import deltakit_stim_circuit_to_dialect, deltakit_stim_dialect_to_circuit

circuit = Circuit("""
H 0
MX 0
""")
circuit_ir = deltakit_stim_circuit_to_dialect(circuit)

print(circuit_ir)
circuit_ir.verify()
```

We can see from the code above that our tiny Deltakit stim circuit with just a H gate and X-basis measurement is transformed into the `stim` dialect as a `ModuleOp` containing:
- qubit allocation (`stim.qubit_alloc`)
- the H gate (`stim.clifford H`)
- the measurement (`stim.measure X`)

In Deltakit stim, qubit indices are implicit and so can just be used without initialisation. Whereas in the `stim` dialect, we use SSA values that represent each qubit (`%0` in this case) and these are passed to each gate or other operation that uses those qubits.

Secondly, in Deltakit stim the measurement record is automatic, so every measurement records the result implicitly. When we translate this into the `stim` dialect, we make the results of measurements explicit (`%1` in this case) and these can then be passed into detectors or other operations.

We can generate the exact same IR using each operation's constructors to see how this works:

```{code-cell} ipython3
program = ModuleOp([])
builder = Builder(InsertPoint.at_end(program.body.block))

# Operation that will allocate qubit with ID 0
qubit_alloc_op = stim.QubitAllocOp(0)
builder.insert(qubit_alloc_op)

# `res` is the result of the QubitAllocOp. It is a reference to the allocated qubit
qubit_0 = qubit_alloc_op.res

# Operation that represents `H 0`
h_gate_op = stim.CliffordGateOp(stim.SingleQubitUnitaryEnum.H, [qubit_0])
builder.insert(h_gate_op)

# Operation that measures our qubit_0 in the X basis
measurement_op = stim.MeasurementGateOp([qubit_0], stim.PauliOperatorEnum.X)
builder.insert(measurement_op)

# The SSA results of measurement_op.
readouts: tuple[SSAValue[IntegerType], ...] = measurement_op.readouts

print(program)
program.verify()

equal = circuit_ir == program
structurally_equiv = circuit_ir.is_structurally_equivalent(program)
print(f"circuit_ir == program                         : {equal}")
print(f"circuit_ir.is_structurally_equivalent(program): {structurally_equiv}")
```

Note that `circuit_ir` is not equal to `program`, instead they are "structurally equivalent". Equality is defined using `is` and so refers to specific python objects in memory, whereas `is_structurally_equivalent` checks that all the operations have all the same operands, properties, regions, blocks, etc.

If our IR only contains `stim` dialect operations then we can convert back to a `Circuit` object:

```{code-cell} ipython3
print("Our IR as a deltakit-stim program:")
circuit_out = deltakit_stim_dialect_to_circuit(program)
print(circuit_out)

print(f"circuit == circuit_out: {circuit == circuit_out}")
```

At this point, try to add some more Deltakit stim instructions to `circuit` and look at the resulting IR that is produced. Then use the operation constructors from the `stim` dialect to produce an equivalent IR. As the Deltakit stim circuit and IR gets more complicated you may find that `circuit_out` no longer precisely matches `circuit`, but the meaning should remain the same.

The circuit doesn't need to do anything interesting so don't make it too long. Try adding Gates, Detectors, and Coordinates. As an extension you can add Repeats too.

|                         | Deltakit stim              | `stim` dialect                | Constructor           |
|-------------------------|---------------------|-------------------------------|-----------------------|
| Qubit allocation        |                     | `stim.qubit_alloc`            | `QubitAllocOp()`      |
| Gates                   | `X`, `H`, `CY`, etc | `stim.clifford`               | `CliffordGateOp()`    |
| Measurements            | `M`, `MX`, etc      | `stim.measure`                | `MeasurementGateOp()` |
| Detectors               | `DETECTOR`          | `stim.detector`               | `DetectorOp()`        |
| Qubit Coords            | `QUBIT_COORDS`      | `stim.assign_qubit_coord`     | `QubitCoordsOp()`     |
| Ticks                   | `TICK`              | `stim.tick`                   | `TickAnnotationOp()`  |
| Repeats                 | `REPEAT`            | `stim.repeat`                 | `RepeatOp()`          |
| implicit loop arguments |                     | `stim.yield`                  | `YieldOp()`           |


+++

## Writing a ModulePass

`ModulePass`es are the main way we manipulate and transform our IR. The compiler runs the `add-noise` pass to add noise ops, for example. Passes fall into 4 main categories:
 - Analysis passes either don't actually change the IR, or the changes they do make are just to add attributes to existing operations to provide information to future passes.
 - Lowering passes take a higher-level dialect and map it down to a lower-level dialect. Eg. `log_asm` dialect to `stim`. This process is often 'lossy' in that contextual / domain specific knowledge can be lost in translation. Eg. Converting an X Gate operation into the control systems specific pulse lengths - once you're that low level it's difficult to go back.
 - Transformation passes generally modify the IR in a similar level of abstraction. These are often optimisation passes, used for injecting extra code, or translating to use a different dialect that then make other passes easier. Eg. `parallelise-circuit`. 
 - Backend passes are used to get our IR out of the compiler, they often use a printer to produce code, or export specific parts of the IR. They include things like printer passes that don't make any changes to the program but instead produce e.g. OpenQASM 3 program files and write them to disk. 
 
The different passes offered by `ModulePass` can be found in `deltakit_compile.passes`. These categories are rough and there aren't really any rules, but when done well they allow for more combinations of small passes which is more flexible and maintainable than large monolithic passes that essentially try to compile everything on one big step.

```{code-cell} ipython3
from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class TransformMeasurementBasis(ModulePass):
    """Pass to Transform stim dialect measurements to use a specified basis"""

    # ModulePasses are designed to be run in a generic way, from the command line, so they must
    # define name
    name = "stim-transform-measurement-basis"

    # They can be parametrised but the parameters can only have simple types:
    # int | float | bool | string  or  list | dict[str, ] | tuple of those types.
    # To make it easier to call passes from python we use type unions and then validate the
    # arguments in the main apply method.
    new_basis: str | stim.PauliOperatorEnum = field(
        default_factory=lambda: stim.PauliOperatorEnum.X
    )

    # Apply is the main method of any ModulePass - it's what actually does work on the IR
    # ctx stores information about what dialects are currently loaded, but we don't need it.
    # op is the program to operate on. All operations are done in-place, and we can do pretty much
    # anything we like until we return from apply(). The program should verify before our pass, and
    # we just guarantee that it verifies again by the end of our pass.
    def apply(self, ctx: Context, op: ModuleOp):

        # This will throw an error if an invalid str was given and gives us back type safety
        new_basis = stim.PauliOperatorEnum(self.new_basis)

        # This is where an implementation would go. We'll see what it looks like later
        print(
            f"This pass should transform measurements into the {new_basis} basis, but doesn't yet"
        )


# The Context object stores all the dialects in use.
context = Context()
context.load_dialect(stim.Stim)
context.load_dialect(Builtin)

program.verify()  # The program must verify before we apply a pass
module_pass = TransformMeasurementBasis(stim.PauliOperatorEnum.Z)
module_pass.apply(context, program)
program.verify()  # The program must verify after the pass is applied
```

## RewritePattern

To make writing passes easier, there are a variety of tools and helpers, the most important of which is the RewritePattern class. This class can be inherited such that the child class implements the transformation of a single matched operation. Many of these RewritePatterns can then be easily coordinated from the ModulePass's apply method.

Every `RewritePattern` must implement a `match_and_rewrite` method that is called for every operation in the program. The `PatternRewriter` is used to modify the program by inserting, removing, or modifying operations. It is important that `rewriter` is used rather than directly modifying the operation / program because it tracks if modifications have been made. Only when no changes are made for any of the operations in the program will the `PatternRewriteWalker` that manages the `RewritePattern`s stop trying to apply them.

The `@op_type_rewrite_pattern` annotation adds a filter to the `match_and_rewrite` method that uses the type hint of `op` and effectively inserts `if not isinstance(op, type_hint): return` to the beginning of the method. This is a common practise to neaten how we write the `match_and_rewrite` method and is not required. 

The implementation itself then:
- Checks if the existing basis is already the desired new basis, and simply returns if it is
- Calculates the correct gate to preserve logical equivalence when measuring in the desired new basis
- Creates a new `CliffordGateOp` (`stim.clifford`) for the required gate
- Uses `rewriter.insert_op` to add this instruction before `op` in the program
- Creates a new `MeasurementGateOp` that is the same as `op` except that it uses the new basis
- Uses `rewriter.replace_matched_op` to remove `op` from the program and replace it with `new_measurement_op`, and update any operations that used the `op.readouts` SSA values with the new `new_measurement_op.readouts` SSA values. 

```{code-cell} ipython3
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class RewriteMeasurementBasis(RewritePattern):
    def __init__(self, new_basis: stim.PauliOperatorEnum):
        self.new_basis = new_basis

    @op_type_rewrite_pattern
    # This annotation enables the MeasurementGateOp type hint to act as a pattern matcher
    def match_and_rewrite(self, op: stim.MeasurementGateOp, rewriter: PatternRewriter) -> None:
        # match_and_rewrite will be called for each MeasurementGateOp in the program.
        # rewriter is used to modify the program

        # The IR starts out as:
        #   ...
        #   %1 = stim.measure {original_basis} (%0)  //  << matched op
        #   ...

        original_basis = op.pauli_modifier.data

        if original_basis == self.new_basis:
            return  # Measurement already in correct basis

        # Work out what gate we need to apply
        bases_set = {original_basis, self.new_basis}
        if bases_set == {stim.PauliOperatorEnum.X, stim.PauliOperatorEnum.Y}:
            required_gate = stim.SingleQubitUnitaryEnum.HXY
        elif bases_set == {stim.PauliOperatorEnum.X, stim.PauliOperatorEnum.Z}:
            required_gate = stim.SingleQubitUnitaryEnum.H
        else:
            assert bases_set == {stim.PauliOperatorEnum.Y, stim.PauliOperatorEnum.Z}
            required_gate = stim.SingleQubitUnitaryEnum.HYZ

        # Use the rewriter to insert an op for the required gate, using the arguments of the
        # MeasurementGateOp
        rewriter.insert_op(stim.CliffordGateOp(required_gate, op.targets), InsertPoint.before(op))
        #   ...
        #   stim.clifford {HXY or HXZ or HYZ} (%0)    // newly inserted op
        #   %1 = stim.measure {original_basis} (%0)   // matched op
        #   ...

        # Make the new measurement op - generally we prefer to make new instances of operations
        # rather than modifying properties since the original may have extra attributes attached
        # whose meaning would be invalidated by our modifications.
        # The exception is that we can insert new operations into the bodies of ops (such as
        # loops) rather than making whole copies of the parent ops, since that is usually
        # excessive.
        new_measurement_op = stim.MeasurementGateOp(
            op.targets, pauli_modifier=self.new_basis, noise=op.noise
        )

        # If we did want to modify the operation in-place we must tell the rewriter with:
        # rewriter.notify_op_modified(op)

        # Finally, use the rewriter to replace the matched_op, ensuring the results of the matched
        # op are all replaced with the new results of our new_measurement_op
        rewriter.replace_op(op, new_measurement_op, new_results=new_measurement_op.readouts)
        #   ...
        #   stim.clifford {HXY or HXZ or HYZ} (%0)   // inserted op
        #   %1 = stim.measure {self.new_basis} (%0)  // replaced op
        #   ...
```

We can now use a `RewriteMeasurementBasis` object to make our ModulePass. This is done with a `PatternRewriteWalker` that we can give any `RewritePattern` to and when we call `.rewrite_module(op)` it will apply the `RewritePattern` (call its `match_and_rewrite` method) for every operation repeatedly until calling it no longer modifies the program. The `GreedyRewritePatternApplier` is also a `RewritePattern` but all it does is try to apply each of its given `RewritePattern`s in turn, stopping as soon as one makes a modification to the program. This is helpful for applying many different `RewritePattern`s all in one `PatternRewriteWalker`.

```{code-cell} ipython3
@dataclass(frozen=True)
class TransformMeasurementBasis(ModulePass):
    """Pass to Transform stim dialect measurements to use a specified basis"""

    name = "stim-transform-measurement-basis"

    new_basis: str | stim.PauliOperatorEnum = field(
        default_factory=lambda: stim.PauliOperatorEnum.X
    )

    def apply(self, ctx: Context, op: ModuleOp):

        # This will throw an error if an invalid str was given and gives us back type safety
        new_basis = stim.PauliOperatorEnum(self.new_basis)

        # A PatternRewriteWalker handles walking (iterating recursively) through the ModuleOp and
        # calling the patterns when and where they are needed
        PatternRewriteWalker(
            # The GreedyRewritePatternApplier allows us to apply many RewritePatterns all together
            GreedyRewritePatternApplier(
                # Our list of rewritePatterns only has one Rewrite but we could easily have more if
                # we had more transformations to do / patterns to match
                [
                    RewriteMeasurementBasis(new_basis),
                ]
            )
        ).rewrite_module(op)
```

We can now use the `TransformMeasurementBasis(PauliOperatorEnum.Z)` pass on our example program.

Our `RewriteMeasurementBasis` rewrite will be called once for each `stim.measure` operation, and this will produce the new IR.
But since this new IR also has a `stim.measure` operation, our rewrite will be called again, but this time will do nothing (it returns before using the `rewriter`). This signals to the `PatternRewriteWalker` (via the `GreedyRewritePatternApplier`) to stop applying rewrites and finish the pass.

```{code-cell} ipython3
print("Our program before the pass:")
print(program)
print()

program_copy = program.clone()  # cloning the program to save the original

program_copy.verify()  # The program must verify before we apply a pass
module_pass = TransformMeasurementBasis(stim.PauliOperatorEnum.Z)
module_pass.apply(context, program_copy)
program_copy.verify()  # The program must verify after the pass is applied


print("Our program after the pass:")
print(program_copy)
print()

print("Our new IR as a deltakit-stim program:")
print(deltakit_stim_dialect_to_circuit(program_copy))
```

Notice that when we added the new H gate we may have just inserted it immediately after another H gate. While there may be uses for this, if we assume we want to try to minimise the number of gates then we could choose to remove both H gates as they should logically cancel out.

Try to write your own `RewritePattern` below that matches on a `CliffordGateOp` doing one of the Hadamard gates, checks that the next operation is also a `CliffordGateOp` that operates on the same qubit and that they should cancel out. And when it finds these operations, erase them both from the program.

Some useful hints:
 - `op.next_op` returns the next operation in the `Block` (or `None` if `op` is the last operation).
 - `op.prev_op` returns the previous operation in the `Block` (or `None` if `op` is the first operation).
 - `rewriter.erase_op()` will remove the given op from the program.

```{code-cell} ipython3
class RewriteIdentityHGates(RewritePattern):
    # Add your implementation here:
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        pass


@dataclass(frozen=True)
class TransformMeasurementBasis(ModulePass):
    """Pass to Transform stim dialect measurements to use a specified basis and also remove gates
    that obviously cancel out"""

    name = "stim-transform-measurement-basis"

    new_basis: str | stim.PauliOperatorEnum = field(
        default_factory=lambda: stim.PauliOperatorEnum.X
    )

    def apply(self, ctx: Context, op: ModuleOp):
        new_basis = stim.PauliOperatorEnum(self.new_basis)

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [RewriteMeasurementBasis(new_basis), RewriteIdentityHGates()]
            )
        ).rewrite_module(op)
```

```{code-cell} ipython3
print("Our program before the pass:")
print(program)
print()

program_copy = program.clone()  # cloning the program to save the original

program_copy.verify()
module_pass = TransformMeasurementBasis(stim.PauliOperatorEnum.Z)
module_pass.apply(context, program_copy)
program_copy.verify()


print("Our program after the pass:")
print(program_copy)
print()

print("Our new IR as a deltakit-stim program:")
print(deltakit_stim_dialect_to_circuit(program_copy))
```

Now we're going to add this pass to deltakit-compile:

 - Put the code for `TransformMeasurementBasis` and the required `RewriterPass`es in `src/deltakit_compile/passes/transform_measurement_basis.py`.
 - Add the pass to `deltakit_compile.pass_runner.PASS_MAP`, so it can be called from the command line and tested with the 'filecheck' and 'lit' test tooling.

There are two main ways of testing our code, pytest and filecheck tests. Pytests are best for testing exceptions and corner cases, whereas filecheck tests are useful for testing that the input and output IR are as we expect them - ie. that the pass has done what we wanted it to. We won't cover pytest as it is similar to most python unit testing, but you can see examples in `tests/unit/passes/`. We will, however, demonstrate filecheck tests as they are less well known.

To setup filecheck testing:
 - Make the file: `tests/filecheck/passes/transform_measurement_basis/test_Z.mlir` and put the following contents in:

```{raw-cell}
---
vscode:
  languageId: raw
---
// RUN: deltakit-compile compile-passes %s -p stim-transform-measurement-basis \
// RUN: --pass-args '{"new_basis":"Z"}' -O %t && filecheck %s --input-file %t

builtin.module {
// CHECK:       builtin.module {

    %0 = stim.qubit_alloc 0 -> !stim.qubit
    %1 = stim.qubit_alloc 1 -> !stim.qubit
    stim.assign_qubit_coord <4.0, 0.0>(%0 : !stim.qubit)
    stim.assign_qubit_coord <0.0, 0.0>(%1 : !stim.qubit)
// CHECK-NEXT:    %[[QUBIT_0:[0-9]+]] = stim.qubit_alloc 0 -> !stim.qubit
// CHECK-NEXT:    %[[QUBIT_1:[0-9]+]] = stim.qubit_alloc 1 -> !stim.qubit
// CHECK-NEXT:    stim.assign_qubit_coord <4.0, 0.0>(%[[QUBIT_0]] : !stim.qubit)
// CHECK-NEXT:    stim.assign_qubit_coord <0.0, 0.0>(%[[QUBIT_1]] : !stim.qubit)

    stim.clifford H (%0)
    %2 = stim.measure Z (%0)
// CHECK-NEXT:    stim.clifford H (%[[QUBIT_0]])
// CHECK-NEXT:    %2 = stim.measure Z (%[[QUBIT_0]])

    stim.clifford H (%1)
    %3 = stim.measure X (%1)
// CHECK-NOT:     stim.clifford
// CHECK-NEXT:    %3 = stim.measure Z (%[[QUBIT_1]])

}
// CHECK-NEXT:  }
```

We won't go into detail about `lit` and `filecheck` but the existing filecheck files in `tests/filecheck/` serve as useful examples and there are some good resources at:
 - [llvm.org/docs/CommandGuide/lit](https://llvm.org/docs/CommandGuide/lit.html)
 - [llvm.org/docs/CommandGuide/FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html)

Once the filecheck test file has been made and the `TransformMeasurementBasis` is in the compiler, you can use the following command to run that particular test.

```{raw-cell}
---
vscode:
  languageId: raw
---
lit -vv tests/filecheck/transforms/transform_measurement_basis
```

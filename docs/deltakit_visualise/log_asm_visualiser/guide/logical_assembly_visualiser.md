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

# Logical Assembly Visualiser

The Logical Assembly Visualiser renders a Logical Assembly (LogASM) program so that you can
check what it does before compiling it into a physical circuit. It offers two views: a 3D
spacetime diagram of the logical operations in a program, and a 2D view of the plaquettes making
up a single logical patch.

A program is handed to the visualiser either as Python objects built with the Logical Assembly
API (to learn more about LogASM, see
[`logical_assembly_api.ipynb`](../../../deltakit_compile/compilation_tutorials/making_a_compiler_pass.ipynb))
for more or as Logical Assembly IR read from a `.mlir` file.

```{code-cell} ipython3
# This file contains information which is proprietary to Riverlane Ltd
# ("Riverlane") and is Riverlane Confidential Information.
# (c) Copyright Riverlane 2025-2026. All rights reserved.

from deltakit_compile.passes.log_asm_api.pipeline import LogAsmApiToLogAsmPipeline
from deltakit_compile.frontend.logasm import LogAsmBuilder, RotatedPlanarPatch
from deltakit_compile.frontend.circuit import QubitReg

from deltakit_visualise.logical_assembly_visualiser import LogicalAssemblyVisualiser
from deltakit_visualise.pipelines.spacetime import SpacetimePipeline
from deltakit_visualise.utils.patch import visualise_logical_patch
from deltakit_visualise.visualiser import get_visualisation_data, show
```

## A program to visualise

The program below is the lattice surgery example from the Logical Assembly API guide: two
distance-`d` rotated planar patches are prepared, their stabilisers are measured for `d` rounds,
a lattice surgery `ZZ` measurement is performed between them, and finally both patches are
measured out.

```{code-cell} ipython3
d = 5
builder = LogAsmBuilder()
p0 = builder.declare_patch(RotatedPlanarPatch(d, d, location=(0, 0)))
p1 = builder.declare_patch(RotatedPlanarPatch(d, d, location=(d + 1, 0)))

# Prepare two logical patches and measure their stabilisers for d rounds
p0.prepare("Z")
p1.prepare("Z")
p0.measure_stabilisers(d)
p1.measure_stabilisers(d)

# Do a lattice surgery merge and split between the logical patches, connected by an implicit bridge
builder.multi_pauli_measure([p0, p1], pauli_bases=["Z", "Z"])

# Measure stabilisers for d rounds and measure out the patches
p0.measure_stabilisers(d)
p1.measure_stabilisers(d)
b0 = p0.measure("Z")
b1 = p1.measure("Z")

# Return the patches' corrected logical bits to the user
builder.add_return(b0)
builder.add_return(b1)

program = builder.build_program()
```

`print(program)` shows the built `LogAsmProgram` as xDSL IR. These are the operations the
visualiser draws.

```{code-cell} ipython3
print(program)
```

## The spacetime view

`from_log_asm_ir` takes a built `LogAsmProgram` (or a `ModuleOp` parsed from IR), and
`visualise_space_time` runs the spacetime pipeline over it and opens the interactive 3D view,
returning the path to the generated HTML.

Every logical operation has a representation in the diagram. Declaring a patch renders a grey
surface, and preparing it in a basis colours that surface by the basis it was prepared in: `X`
renders red and `Z` renders blue. Measuring stabilisers is drawn as the sides carrying the patch
through its rounds, and measuring a patch out is a surface sitting on top of them. The vertical
axis of the diagram is time measured in rounds.

`visualise_space_time()` defaults to `static=True`: inside a notebook the view is embedded in the
cell output with an *Open link* button next to it, and outside a notebook it is opened in the
system browser. On a remote or headless machine, use `static=False` instead, see
*Serving the view over HTTP* below.

```{code-cell} ipython3
LogicalAssemblyVisualiser.from_log_asm_ir(program).visualise_space_time()
```

```{code-cell} ipython3
module = program.module.clone()  # a LogAsmProgram's .module is already a builtin.module
ctx = LogicalAssemblyVisualiser.make_context()
LogAsmApiToLogAsmPipeline().apply(ctx, module)
SpacetimePipeline().apply(ctx, module)

data = get_visualisation_data(module)
print(data["type"], "-", len(data["ops"]), "items")
print(data["ops"][0])
show(data)
```

If you are a QEC expert you can go a step further: `deltakit-compile` lets you add a compiler pass of
your own, which is covered in
[`making_a_compiler_pass.ipynb`](../../../deltakit_compile/compilation_tutorials/making_a_compiler_pass.ipynb).
The same approach applies here, so a pass of yours can shape the visualisation in a different
way. To keep using the default frontend library,
[`deltakit-visualise`](https://www.npmjs.com/package/deltakit-visualise), follow the same data
API; `pipelines/spacetime.py` and `pipelines/surfacecodes.py` in `deltakit-visualise` show how the
existing pipelines manage it.

+++

## Visualising IR from a file

If a program is already saved as Logical Assembly IR, load it with `from_log_asm_file` rather
than building it in Python. The example below renders a logical CNOT: two lattice surgery
measurements, through an ancilla patch, between a control and a target patch.

```{code-cell} ipython3
LogicalAssemblyVisualiser.from_log_asm_file("../examples/cnot.mlir").visualise_space_time()
```

```{image} images/spacetime_cnot.png
:alt: Spacetime view of a logical CNOT
:width: 300px
```

## Serving the view over HTTP

On a remote or headless machine, pass `static=False` to serve the view over HTTP on port `8899`
instead of opening a local browser tab. Note that this **blocks** the kernel while the server
runs, so interrupt the cell to stop it.

```LogicalAssemblyVisualiser.from_log_asm_ir(program).visualise_space_time(static=False)```

```{code-cell} ipython3
LogicalAssemblyVisualiser.from_log_asm_ir(program).visualise_space_time() # add static=False arg
```

## Visualising a single logical patch

The spacetime view shows logical operations; to look inside one patch, use
`visualise_logical_patch`, which renders the plaquettes of that patch in a given round
(`round_no`, defaulting to the first). It takes the builder rather than a built program, so a
patch can be inspected while the program around it is still being written.

```{code-cell} ipython3
P_DISTANCE = 5
\
builder = LogAsmBuilder()
p0 = builder.declare_patch(
    RotatedPlanarPatch(P_DISTANCE, P_DISTANCE, location=(0, 0))
)

p0.prepare("Z")
p0.measure_stabilisers(4)

visualise_logical_patch(builder, p0)
```

`visualise_logical_patch` on the visualiser opens the same 2D patch view on its own, without
the server. It runs the patch pipeline over a built program, whereas the
`visualise_logical_patch` imported from `deltakit_visualise.utils.patch` earlier takes a builder
and one patch out of it.

+++

## Stepping into the surface code from the spacetime view

`visualise` combines both views into one interactive page. It runs the spacetime and the patch
pipeline over the program and serves the two together from `http://127.0.0.1:8000`, where the 3D
diagram gains a *Surface Code* tool: select the tool, click an operation in the diagram, and the
panel lists the rounds that operation spans. Picking one fetches the patch layout for that round
and draws it beside the diagram.

Because both pipelines run, the program has to be one the patch pipeline supports, such as a
quantum memory experiment.

TODO: the patch lowering pipeline does not support `multi_pauli_measure` yet. Covering the rest
of the operations is already work in progress.

The program below prepares a distance 3 patch in `X`, holds it through 20 rounds of stabiliser
measurements, and measures it out.

```{code-cell} ipython3
builder = LogAsmBuilder()
p = builder.declare_patch(RotatedPlanarPatch(3, 3, location=(0, 0)))
p.prepare("X")
p.measure_stabilisers(20)
p.measure("X")

memory_exp_program = builder.build_program()
```

Unlike `visualise_space_time`, this does not open a browser tab for you, so open
`http://127.0.0.1:8000` yourself once the server is up. In a notebook the server runs in a
background thread, so the kernel is free to carry on; run from a plain script it blocks instead.
The port is fixed, so only one of these servers can run at a time.

```{code-cell} ipython3
LogicalAssemblyVisualiser.from_log_asm_ir(memory_exp_program).visualise()
```

The 20 rounds of stabiliser measurements are one operation in the diagram, so clicking it lists
the rounds it covers.

```{image} images/visualise_rounds.png
:alt: The Surface Code panel listing the rounds of a memory experiment
:width: 520px
```

Choosing a round, round 20 in this case, marks that height in the diagram and draws the patch as
it stands in that round.

```{image} images/visualise_surface_code.png
:alt: The Surface Code panel showing the distance 3 patch at round 20
:width: 520px
```

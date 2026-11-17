---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  display_name: Python (dkx)
  language: python
  name: dkx
---

*Want to follow along? {nb-download}`Download this notebook.<template_notebook.ipynb>`*

```{code-cell} ipython3
from __future__ import annotations

from deltakit.circuit.gates import PauliBasis
from deltakit.explorer import Client, enums, types

client = Client.get_instance()
```

```{code-cell} ipython3
# your code here, e.g. circuit generation:

result = client.generate_circuit(
    types.QECExperimentDefinition(
        experiment_type=enums.QECExperimentType.QUANTUM_MEMORY,
        code_type=enums.QECECodeType.REPETITION,
        num_rounds=1,
        observable_basis=PauliBasis.Z,
        parameters=types.CircuitParameters(sizes=types.Sizes([2]))
    )
)
```

---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Simulation of STIM-like circuits

*Want to follow along? {nb-download}`Download this notebook.<server_simulation.ipynb>`*

Deltakit provides both client-side and server-side simulation capabilities.
For typical simulations involving only features provided by STIM, simulation can be performed locally; this has the advantage of avoiding data transfer and network latency for small circuits.
If the simulation involves advanced features, such as leakage support, simulations will be performed on a Deltakit server.

+++

## Simple simulation example

In this example we run a circuit to prepare Bell's state. Measurements are random, but they should be perfectly correlated.

```{code-cell} ipython3
from deltakit.explorer.simulation import simulate_with_stim

bell_circuit = """
R 0 1
H 0
CX 0 1
M 0 1
"""
```

By default, simulations run locally.

```{code-cell} ipython3
measurements, _ = simulate_with_stim(bell_circuit, shots=8)
measurements
```

When a `Client` instance is provided using the `client` parameter, simulations are run on the cloud.

```{code-cell} ipython3
from deltakit.explorer import Client
Client.set_token('<your token here>')  # see "Setup" documentation
client = Client.get_instance()
measurements, _ = simulate_with_stim(bell_circuit, shots=8, client=client)
measurements
```

## Leakage simulation

Server-side simulator supports additional (not included in STIM) instructions that enable simulation of leakage. `LEAKAGE(probability) [list of targets]` defines a probabilistic leakage event.
This line will generate leakage event of 1% probability on qubits 0, 1 and 2. 

```
LEAKAGE(0.01) 0 1 2
```

The `HERALD_LEAKAGE_EVENT [list of targets]` operator works very similarly to a measurement instruction and records bits of information about the leakage state of the qubits.
They should always accompany measurement instructions. This is how instructions work together:

```
R 0 1

LEAKAGE(1.0) 1          # leakage with 100% probability
M 0                     # measurement value for qubit 0
HERALD_LEAKAGE_EVENT 0  # leakage state for qubit 0, will be 0
M 1                     # measurement value for qubit 1, unpredictable
HERALD_LEAKAGE_EVENT 1  # leakage state for qubit 1, will be 1
```

The code below runs an example with probabilistic leakage. The `client` argument must be provided when running simulations involving leakage.

```{code-cell} ipython3
leakage_bell_stim = """
R 0 1
LEAKAGE(0.5) 1   # qubit 1 leaks 50% of the time
M 0 1
HERALD_LEAKAGE_EVENT 0 1
"""
measurements, leakage = simulate_with_stim(
    stim_circuit=leakage_bell_stim,
    shots=10,
    client=client,
)
```

```{code-cell} ipython3
measurements
```

```{code-cell} ipython3
leakage
```

# Deltakit

[![docs - here!][docs-badge]][docs-link]
[![PyPI][pypi-badge]][pypi-link]
[![Python versions][python-badge]][pypi-link]
[![Pixi][pixi-badge]][pixi-link]
[![Typing][typing-badge]][typing-link]
[![License: Apache 2.0][license-badge]][license-link]
[![SemVer][semver-badge]][semver-link]
[![SPEC 0][spec0-badge]][spec0-link]
[![Issues][issues-badge]][issues-link]
[![Discussions][discussions-badge]][discussions-link]

[docs-badge]: https://readthedocs.org/projects/deltakit/badge/?version=latest
[docs-link]: https://deltakit.readthedocs.io/en/latest/

[pypi-badge]: https://img.shields.io/pypi/v/deltakit.svg
[pypi-link]: https://pypi.org/project/deltakit/

[python-badge]: https://img.shields.io/pypi/pyversions/deltakit

[pixi-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json
[pixi-link]: https://pixi.sh

[typing-badge]: https://img.shields.io/pypi/types/deltakit
[typing-link]: https://typing.python.org/

[license-badge]: https://img.shields.io/badge/License-Apache_2.0-blue.svg
[license-link]: https://www.apache.org/licenses/LICENSE-2.0

[semver-badge]: https://img.shields.io/badge/semver-2.0.0-blue
[semver-link]: https://semver.org/spec/v2.0.0.html

[spec0-badge]: https://img.shields.io/badge/SPEC-0-forestgreen
[spec0-link]: https://scientific-python.org/specs/spec-0000/

[issues-badge]: https://img.shields.io/github/issues/Deltakit/deltakit?logo=github
[issues-link]: https://github.com/Deltakit/deltakit/issues

[discussions-badge]: https://img.shields.io/badge/discussions-join-blue?logo=github
[discussions-link]: https://github.com/Deltakit/deltakit/discussions

Deltakit allows you to create and run quantum error correction (QEC) experiments with features
including circuit generation, simulation, decoding and results analysis.

Whether you're a seasoned QEC researcher or just starting out, Deltakit supports you
in exploring new ways to implement QEC logic all the way to running complex
QEC circuits on QPU hardware.

## Quick Start

### Installation
Install Deltakit with `pip`.

```bash
pip install deltakit
```

### Performing a QEC experiment

`deltakit` provides a full pipeline to help you run quantum error correction experiments.

```python
from deltakit.circuit.gates import PauliBasis
from deltakit.decode import PyMatchingDecoder
from deltakit.decode.analysis import run_decoding_on_circuit
from deltakit.explorer.analysis import calculate_lep_and_lep_stddev
from deltakit.explorer.codes import RotatedPlanarCode, css_code_memory_circuit
from deltakit.explorer.qpu import QPU, ToyNoise

# Creating a noisy memory circuit with the rotated planar code
d = 3
rplanar = RotatedPlanarCode(width=d, height=d)
circuit = css_code_memory_circuit(rplanar, num_rounds=d, logical_basis=PauliBasis.Z)
qpu = QPU(circuit.qubits, noise_model=ToyNoise(p=0.01))
noisy_circuit = qpu.compile_and_add_noise_to_circuit(circuit)

# Perform simulation and correct the measured observable flips with a decoder
num_shots, batch_size = 100_000, 10_000
decoder, noisy_circuit = PyMatchingDecoder.construct_decoder_and_stim_circuit(noisy_circuit)
result = run_decoding_on_circuit(
    noisy_circuit, num_shots, decoder, batch_size, min_fails=100
)

# Print the results
fails = result["fails"]
lep, lep_stddev = calculate_lep_and_lep_stddev(fails, num_shots)
print(f"LEP = {lep:.5g} ± {lep_stddev:.5g}")
```

### Performing a QEC experiment (online)

#### Authentication

The `deltakit` library also allows you to access advanced simulation capabilities that
are not yet available in the open-source local code.

To access them, you need to obtain an authentication token on the
[Deltakit website](https://deltakit.riverlane.com/dashboard/token).

You can register your token by executing the following code once. You do not have to call
`set_token` again, except if you need to change your token.

```python
from deltakit.explorer import Client

Client.set_token("<your token>")
```

#### Experimentation
Generate a QEC experiment by calling the cloud API:

```python
from deltakit.explorer.codes import css_code_stability_circuit, RotatedPlanarCode
from deltakit.circuit.gates import PauliBasis
from deltakit.explorer import Client

# Get a client instance. You need to register your token first.
client = Client.get_instance()
# Generate a stability experiment with the rotated planar code.
circuit = css_code_stability_circuit(
    RotatedPlanarCode(3, 3),
    num_rounds=3,
    logical_basis=PauliBasis.X,
    client=client
)
# Display the resulting circuit
print(circuit)
```

Learn more by reading the [Deltakit docs](https://deltakit.readthedocs.io/en/latest/)!

## Support

- Found a bug? Need a feature? File an [issue](https://github.com/Deltakit/deltakit/issues).
- Usage questions? Visit our [Q&A forum](https://github.com/Deltakit/deltakit/discussions/categories/q-a).
- Have a security concern? See our [security policy](SECURITY.md).

## Development
Help us make Deltakit better! Check out [Contributor guide](CONTRIBUTING.md)

## License
This project is distributed under the [Apache 2.0 License](LICENSE).

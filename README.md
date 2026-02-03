[![docs - here!][docs-badge]][docs-link]
[![PyPI][pypi-badge]][pypi-link]
[![Python versions][python-badge]][pypi-link]
[![Typing][typing-badge]][typing-link]
[![License: Apache 2.0][license-badge]][license-link]
[![SemVer][semver-badge]][semver-link]
[![SPEC 0][spec0-badge]][spec0-link]
[![Issues][issues-badge]][issues-link]
[![Discussions][discussions-badge]][discussions-link]
[![DOI][doi-badge]][doi-link]

[docs-badge]: https://readthedocs.org/projects/deltakit/badge/?version=latest
[docs-link]: https://deltakit.readthedocs.io/

[pypi-badge]: https://img.shields.io/pypi/v/deltakit.svg
[pypi-link]: https://pypi.org/project/deltakit/

[python-badge]: https://img.shields.io/pypi/pyversions/deltakit

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

[doi-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.17145113.svg
[doi-link]: https://doi.org/10.5281/zenodo.17145113

---

<p align="center">
  <picture>
    <img alt="Deltakit logo" src="./docs/logo/Deltakit_Blue_SecondaryLogo_RGB.png" width="75%">
  </picture>
</p>

---

**Deltakit** is a Python package providing a toolkit to create, execute, analyse and benchmark quantum error correction (QEC) experiments. You can use Deltakit to design new high-level QEC logic, and to compile and run it on real hardware and simulators.

<p align="center">
  <a href="https://deltakit.readthedocs.io/en/docs/api.html#deltakit-explorer-codes">
	<img src="https://i.imgur.com/bK3T7RM.png" width="250" style="background-color: white;">
  </a>
  <a href="https://deltakit.readthedocs.io/en/docs/api.html#deltakit-explorer-qpu">
	<img src="https://i.imgur.com/1GN8eRg.png" width="250" style="background-color: white;">
  </a>
<br>
  <a href="https://deltakit.readthedocs.io/en/docs/api.html#deltakit-explorer">
	<img src="https://i.imgur.com/YIVuaGr.png" width="250" style="background-color: white;">
  </a>
  <a href="https://deltakit.readthedocs.io/en/docs/api.html#deltakit-decode">
	<img src="https://i.imgur.com/ngXPlgF.png" width="250" style="background-color: white;">
  </a>
</p>

</tr>
</table>

For more detailed information, check out the [Deltakit documentation](https://deltakit-docs.riverlane.com/en/docs/).

For any usage questions or comments, visit our [Q&A forum](https://github.com/Deltakit/deltakit/discussions/categories/q-a).

## Feature highlights

Standard QEC experiments proceed through several fundamental stages, each of which is facilitated by the functionality provided in the `deltakit` package:

* **Circuit generation:** brings together a representative noisy circuit for the experiment of choice and a quantum error correcting code.
* **Simulation:** generates measurement results by executing and sampling the circuit on the numerical simulator [Stim](https://github.com/quantumlib/Stim).
* **Decoding & analysis:** uses either a standard decoder of choice or Riverlane's proprietary ones to decode measurement samples, apply corrections and produce interpretable relevant metrics.

## Installation guide

`deltakit` is publicly available on [PyPI](https://pypi.org/project/deltakit/) and can be locally installed with `pip`:

```bash
pip install deltakit
```

## Quick Start - Performing a QEC memory experiment on a local machine

`deltakit` helps the design and execution of complete QEC experiments. The first step is to define an encoding process from a quantum circuit and a parametrisable code chosen from a standard family. For instance, here we use the [rotated surface code](https://errorcorrectionzoo.org/c/rotated_surface) from the [Calderbank–Shor–Steane](https://en.wikipedia.org/wiki/CSS_code) (CSS) family. The next step is to declare a QPU instance together with a noise model and a native gate set to compile the circuit to. This produces a QPU compliant noisy circuit that can be executed either on numerical simulators (Stim) or physical hardware to generate noisy bitstring samples. The final step is to apply the decoding process on these bitstrings and correct the circuit. In the following example, a [Minimum Weight Perfect Matching](https://en.wikipedia.org/wiki/Matching_(graph_theory))-based decoder publicly available from the [PyMatching](https://github.com/oscarhiggott/PyMatching) library is used and the logical error probability (LEP) is generated for interpretation.

```python
from deltakit.circuit.gates import PauliBasis
from deltakit.decode import PyMatchingDecoder
from deltakit.decode.analysis import run_decoding_on_circuit
from deltakit.explorer.analysis import calculate_lep_and_lep_stddev
from deltakit.explorer.codes import RotatedPlanarCode, css_code_memory_circuit
from deltakit.explorer.qpu import QPU, ToyNoise

# Step 1. Encoding
# Create a noisy memory circuit with the rotated planar code
d = 3  # Code distance.
rplanar = RotatedPlanarCode(width=d, height=d)
circuit = css_code_memory_circuit(rplanar, num_rounds=d, logical_basis=PauliBasis.Z)

# Step 2. Declare a noisy QPU instance
qpu = QPU(circuit.qubits, noise_model=ToyNoise(p=0.01))
noisy_circuit = qpu.compile_and_add_noise_to_circuit(circuit)

# Step 3. Perform simulation and correct the measured observable flips with a decoder
num_shots, batch_size = 100_000, 10_000
decoder, noisy_circuit = PyMatchingDecoder.construct_decoder_and_stim_circuit(noisy_circuit)
result = run_decoding_on_circuit(
    noisy_circuit, num_shots, decoder, batch_size, min_fails=100
)

# Step 4. Print relevant results
fails = result["fails"]
lep, lep_stddev = calculate_lep_and_lep_stddev(fails, num_shots)
print(f"LEP = {lep:.5g} ± {lep_stddev:.5g}")
```

## Performing an online and remote QEC experiment

`deltakit` allows access to advanced simulation capabilities on the cloud platform via token authentication. To generate a personal access token, please follow the steps described on the [Deltakit website](https://deltakit.riverlane.com/dashboard/token). The generated token can be registered locally by executing the following code *once*.

```python
from deltakit.explorer import Client

Client.set_token("<your-token>")
```

_N.B._`set_token` does not need to be called again, except in case of token revocation.

An instance of the cloud API is required to execute a remote experiment:

```python
from deltakit.explorer.codes import css_code_stability_circuit, RotatedPlanarCode
from deltakit.circuit.gates import PauliBasis
from deltakit.explorer import Client

# Instantiate a cloud client.
# A token needs to be registered first.
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

## Contributing

There are various ways to contribute to `deltakit`:

- **Submitting issues:** To submit bug reports or feature requests, please use our [issue tracker](https://github.com/Deltakit/deltakit/issues).
- **Developing in `deltakit`:** To learn more about how to develop within `deltakit`, please refer to [contributing guidelines](./docs/CONTRIBUTING.md).
- **Security:** For any security concern, please see our [security policy](./docs/SECURITY.md).

> [!NOTE]
> Any contribution will require a Contribution Licence Agreement signature when a Pull Request is created. The recommended contributing workflow is detailed in our [contributing guidelines](./docs/CONTRIBUTING.md).

## License

This project is distributed under the [Apache 2.0 License](LICENSE).

## Citation

If you find this toolkit useful, please consider citing it:

```bibtex
@software{deltakit,
  author = {Prawiroatmodjo, Guen and Burton, Angela and Suau, Adrien and Nnadi, Chidi and Bracken Ziad, Abbas and Melvin, Adam and Richardson, Adam and Walayat, Adnaan and Moylett, Alex and Virbule, Alise and Safehian, AmirReza and Patterson, Andrew and Buyskikh, Anton and Ruben, Archi and Barber, Ben and Reid, Brendan and Manuel, Cai Rees and Seremet, Dan and Byfield, David and Matekole, Elisha and Gallardo, Gabriel and Geher, Gyorgy and Turner, Jack and Lal, Jatin and Camps, Joan and Majaniemi, Joonas and Yates, Joseph and Johar, Kauser and Barnes, Kenton and Caune, Laura and Zigante, Lewis and Skoric, Luka and Jastrzebski, Marcin and Ghibaudi, Marco and Turner, Mark and Haberland, Matt and Stafford, Matthew and Blunt, Nick and Gillett, Nicole and Crawford, Ophelia and McBrien, Philip and Ishtiaq, Samin and Protasov, Stanislav and Wolanski, Stasiu and Hartley, Tom},
  title        = {Deltakit},
  month        = sep,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17145113},
  url          = {https://doi.org/10.5281/zenodo.17145113},
}
```

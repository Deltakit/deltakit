# Google surface code experiment data (subset)

This folder contains a small subset of the experimental data published by the
Google Quantum AI team alongside the paper *"Suppressing quantum errors by
scaling a surface code logical qubit"*. It is used by the
`docs/examples/notebooks/google_qmem_experiment/` notebooks (`decode.md` and
`noise_analysis.md`).

## Source

- **Dataset:** Google Quantum AI Team, *Data for "Suppressing quantum errors by
  scaling a surface code logical qubit"*, Zenodo, 2022.
  DOI: [10.5281/zenodo.6804040](https://doi.org/10.5281/zenodo.6804040)
- **Archive:** `google_qec3v5_experiment_data.zip`
  (MD5 `a7fd8b481c3087090093106382dc217d`)
- **Paper:** Google Quantum AI, "Suppressing quantum errors by scaling a surface
  code logical qubit", *Nature* 614, 676–681 (2023).
  DOI: [10.1038/s41586-022-05434-1](https://doi.org/10.1038/s41586-022-05434-1)

## License

The dataset is licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
This license applies to the files in `input/` only. It does not apply to the
rest of the Deltakit repository, which is licensed under Apache 2.0.

## Changes from the original

The original archive is 315 MB (548 MB unpacked). Only the 20 files that the
notebooks read have been kept. They are unmodified copies of the originals.

| Folder | Files |
|---|---|
| `surface_code_bZ_d3_r01_center_3_5` | `circuit_ideal.stim`, `circuit_noisy.stim`, `measurements.b8`, `sweep.b8` |
| `surface_code_bZ_d3_r03_center_3_5` | same four files |
| `surface_code_bZ_d3_r05_center_3_5` | same four files |
| `surface_code_bZ_d3_r07_center_3_5` | same four files |
| `surface_code_bX_d3_r07_center_3_5` | same four files |

`input/README.txt` is the original README from the root of the archive. It
describes the folder naming scheme and all file formats, including files that
are not part of this subset.

## Getting the full dataset

Download the full archive from
[Zenodo](https://doi.org/10.5281/zenodo.6804040), or run this from the
repository root:

```bash
docs/examples/scripts/get_google_data.sh docs/examples/data
```

The script downloads the archive, checks its MD5 checksum and extracts the
surface code data into `full_dataset/`, next to `input/`. The committed subset
in `input/` is left untouched, and `full_dataset/` is ignored by git.

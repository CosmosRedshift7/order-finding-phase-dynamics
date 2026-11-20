# Order Finding from Phase Dynamics in Topological Acoustics

This repository contains the analysis code and experimental phase data used for the manuscript:

**Order Finding from Phase Dynamics in Topological Acoustics**

## Repository contents

- `main.py` — main analysis script for the calibrated phase-dynamics figures.
- `frequency_control.py` — shared sequential affine frequency controller (Eqs. 27–29).
- `heatmap.py` — period-detection heatmap sweep for the two cases $N=55$ and $N=135$.
- `phases_1.mat` — experimental phase data used by the scripts.
- `results/` — generated figures, heatmaps, and calibration parameters.
- `requirements.txt` — Python dependencies.

## Requirements

The code requires Python 3.10 or newer.

## Installation

Create a local virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

On Linux/macOS with bash or zsh:

```bash
source .venv/bin/activate
```

On Linux/macOS with fish:

```fish
source .venv/bin/activate.fish
```

Then install the required Python packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To confirm that the virtual environment is being used, run:

```bash
python -c "import sys; print(sys.executable)"
```

The printed path should point to `.venv/bin/python` inside this repository.

## Reproducing the results

From the repository root, with the virtual environment activated, run:

```bash
python main.py
```

Alternatively, run the script explicitly with the virtual-environment Python:

```bash
.venv/bin/python main.py
```

The main script reads `phases_1.mat` and writes the calibrated phase-dynamics figures to the `results/` directory. The expected outputs include:

- `results/N55_a4_W8_Delta7.pdf`
- `results/N135_a4_W8_Delta10.pdf`
- `results/N55_a4_W8_Delta7.txt` (Table II)
- `results/N135_a4_W8_Delta10.txt` (Table III)
- `results/calibration_params_phi_12.txt`

To reproduce the period-detection heatmaps, run:

```bash
python heatmap.py
```

The heatmap script writes:

- `results/N55_a4_x0_1_heatmap.pdf`
- `results/N135_a4_x0_1_heatmap.pdf`

### Heatmap categories and colours

Each grid point is classified using its first detected phase-return step `k`
and the reference period `r` already computed by the original heatmap script:

| Category | Meaning |
| --- | --- |
| Exact order | `k == r`. |
| Valid multiple | `k > r` and `k % r == 0`; a valid nonminimal period. |
| Invalid period | A positive return step that is not a multiple of `r`. |
| No return | No threshold return before the frequency band or step budget is exhausted. |

For the configured cases, both the base and the initial state are coprime to
`N`, so `r` is the multiplicative order and a valid period satisfies
`pow(a, k, N) == 1`. The four labels are shown on a discrete colourbar.
Both scripts generate operating frequencies from the aligned affine model.
They interpolate the calibrated recorded phase at each selected frequency
and use its circular distance from the initial phase to detect a return.

The default palette samples four colours from Fabio Crameri's **vik** map.
See [Scientific colour maps](https://www.fabiocrameri.ch/colourmaps/) and
[categorical colour maps](https://www.fabiocrameri.ch/categorical-colour-maps/).
Set `CRAMERI_CMAP_NAME` in `heatmap.py` to choose another palette, or set
`USE_CRAMERI_CMAP = False` for a four-shade grayscale version.

## Citation

If you use this code or data, please cite the associated manuscript and the archived Zenodo release of this repository.

## License

This repository is released under the MIT License.

# Order Finding from Phase Dynamics in Topological Acoustics

This repository contains the analysis code and experimental phase data used for the manuscript:

**Order Finding from Phase Dynamics in Topological Acoustics**

## Repository contents

- `main.py` — main analysis script.
- `phases_1.mat` — experimental phase data used by the script.
- `results/` — generated figures and calibration parameters.
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

The script reads `phases_1.mat` and writes the output files to the `results/` directory. The expected outputs include:

- `results/N55_a4_W8_Delta7.pdf`
- `results/N135_a4_W8_Delta10.pdf`
- `results/calibration_params_phi_12.txt`

## Citation

If you use this code or data, please cite the associated manuscript and the archived Zenodo release of this repository.

## License

This repository is released under the MIT License.

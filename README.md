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

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Reproducing the results

From the repository root, run:

```bash
python main.py
```

The script reads `phases_1.mat` and writes the output files to the `results/` directory. The expected outputs include:

- `results/N55_a4_W8_Delta7.pdf`
- `results/N135_a4_W8_Delta10.pdf`
- `results/calibration_params_phi_12.txt`

## Citation

If you use this code or data, please cite the associated manuscript and the archived Zenodo release of this repository.

## License

This repository is released under the MIT License.

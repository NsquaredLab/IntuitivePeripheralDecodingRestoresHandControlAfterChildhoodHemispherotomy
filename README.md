# Intuitive peripheral decoding restores hand control after childhood hemispherotomy — analysis & figure code

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21269465.svg)](https://doi.org/10.5281/zenodo.21269465)

Analysis and figure-generation code accompanying the study:

> **Intuitive peripheral decoding restores hand control after childhood hemispherotomy**
>
> Dominik I. Braun<sup>1†</sup>, Pauline Wittermann<sup>1†</sup>, Nico G. M. Weber<sup>2</sup>, Dörte Wartke<sup>1</sup>, Pınar Güneş<sup>1</sup>, Felix Wachter<sup>3</sup>, Paula Corcosa<sup>3</sup>, Lina Tan<sup>3,4,5</sup>, Henriette Grieshaber-Bouyer Mandelbaum<sup>3</sup>, Jonas Walter<sup>2</sup>, Jörg Franke<sup>2</sup>, Ferdinand Knieling<sup>3</sup>, and Alessandro Del Vecchio<sup>1\*</sup>
>
> Manuscript under review (2026). Article DOI to be added on publication.
>
> Analysis & figure code archived on Zenodo: [10.5281/zenodo.21269465](https://doi.org/10.5281/zenodo.21269465).
>
> † These authors contributed equally. \* Corresponding author: <alessandro.del.vecchio@fau.de>
>
> **Affiliations**
> 1. Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Department of Artificial Intelligence in Biomedical Engineering, Professur für Neurophysiology and Neural Interfacing; Erlangen, Germany.
> 2. Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Department of Mechanical Engineering, Institute for Factory Automation and Production Systems; Erlangen, Germany.
> 3. Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Department of Pediatrics and Adolescent Medicine, Pediatric Experimental and Translational Imaging Laboratory; Erlangen, Germany.
> 4. Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Department of Medicine 3 – Rheumatology and Immunology; Erlangen, Germany.
> 5. Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Deutsches Zentrum für Immuntherapie (DZI); Erlangen, Germany.

This repository reproduces the quantitative panels of Figures 2 and 3 from the
high-density surface-electromyography (EMG) recordings analysed in the study.

---

## Data availability (please read first)

The EMG recordings analysed in this study were acquired from children who
underwent hemispherotomy — a vulnerable participant group. To protect their
privacy and to honour the consent given by the families and the approval by our research ethics
board, no participant data — neither the raw recordings nor any derived
features — are distributed with this repository. All data are available
from the authors on reasonable request for legitimate scientific use, under a data-use agreement.

This repository provides the complete analysis and figure code so that the
methodology is fully transparent and the published results can be reproduced
once data access has been granted.

**Contact for data requests:** Alessandro Del Vecchio (corresponding author) — <alessandro.del.vecchio@fau.de>

---

## Repository structure

```
.
├── figure-two/
│   ├── two-b-c-d/
│   │   └── figure_two-b-c-d.py         # Figure 2B/2C/2D: raw EMG, RMS + MU spike
│   │                                   #   trains + firing rates, STA MUAP maps
│   └── two-e/                          # Figure 2E: gesture-classification pipeline
│       ├── _pipeline.py                #   import shim + shared raw-data guard
│       ├── _cv.py                      #   shared CatBoost cross-validation engine
│       ├── 01_labeler.py               #   step 1  interactive gesture labelling
│       ├── 02_dataset_creation.py      #   step 2  build labelled raw datasets
│       ├── 03_feature_extraction.py    #   step 3  RMS feature extraction
│       ├── 04_classification.py        #   step 4  Figure 2E classifier (RMS features)
│       ├── 05_visualize_predictions.py #  step 5  Figure 2E confusion-matrix figures
│       ├── 06_supplementary_figure.py  #  step 6  supplementary combined figure
│       └── results/                    #   generated confusion matrices (SVG) + JSON
└── figure-three/
    └── three-b-c/
        └── figure_three-b-c.py         # Figure 3B/3C: myocontrol reaction latency
                                        #   + event-based confusion matrix
```

The Figure 2E scripts are numbered `01`–`06` to make their run order explicit.
They import shared definitions (the canonical gesture order/colours, the
cross-validation engine in [`_cv.py`](figure-two/two-e/_cv.py), …) from one
another through the tiny [`_pipeline.py`](figure-two/two-e/_pipeline.py) shim,
since a module name cannot start with a digit.

---

## Installation

Requires **Python 3.11 or 3.12**.

### With [uv](https://docs.astral.sh/uv/) (recommended)

```bash
uv sync
```

### With pip

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install "numpy>=1.24,<2" "scipy>=1.11,<2" "matplotlib>=3.9,<4" \
            "scikit-learn>=1.3,<2" "catboost>=1.2,<2"
```

---

## Reproducing the results

All analysis steps operate on the participant data, which is not distributed
with this repository but is available from the authors on reasonable request
(see the data note above). Every script also documents its exact inputs
and usage in its module docstring (top of the file), and prints an explanatory
message if a required input is missing.

| Script | Produces |
| --- | --- |
| `figure-two/two-b-c-d/figure_two-b-c-d.py` | Figure 2B/2C/2D panels |
| `figure-three/three-b-c/figure_three-b-c.py` | Figure 3B/3C/3D panels |
| `figure-two/two-e/01_labeler.py` | gesture labels (`labels.json`) — Fig. 2E data prep; supp. panel A |
| `figure-two/two-e/02_dataset_creation.py` | labelled raw datasets (`data/<P>.npz`) — supp. panels A/B |
| `figure-two/two-e/03_feature_extraction.py` | RMS features (`data/<P>_rms_features.npz`) — supp. panel C |
| `figure-two/two-e/04_classification.py` | **Figure 2E** cross-validation scores (RMS features) |
| `figure-two/two-e/05_visualize_predictions.py` | **Figure 2E** confusion-matrix panels |
| `figure-two/two-e/06_supplementary_figure.py` | supplementary classification figure (panels A–D) |

The Figure 2E pipeline is designed to run in numeric order: steps 01→02 prepare
the labelled datasets, step 03 derives the RMS features, and steps 04/05 produce
the published Figure 2E cross-validation scores and confusion matrices from
those features. The shared CatBoost cross-validation engine used by steps 04–06
lives in [`_cv.py`](figure-two/two-e/_cv.py).

```bash
# e.g. reproduce the Figure 2E confusion matrices once the data is in place
uv run python figure-two/two-e/04_classification.py
uv run python figure-two/two-e/05_visualize_predictions.py
```

(Use `python …` instead of `uv run python …` if you installed with pip.)

---

## How to cite

If you use this software, please cite both the article and
this software archive. Citation metadata is provided in
[`CITATION.cff`](./CITATION.cff).

The software archive is deposited on Zenodo and can be cited by its DOI:

> Braun, D. I., Wittermann, P., Weber, N. G. M., Wartke, D., Güneş, P., Wachter, F.,
> Corcosa, P., Tan, L., Grieshaber-Bouyer Mandelbaum, H., Walter, J., Franke, J.,
> Knieling, F., & Del Vecchio, A. (2026). *Analysis and figure code for "Intuitive
> peripheral decoding restores hand control after childhood hemispherotomy"*.
> Zenodo. https://doi.org/10.5281/zenodo.21269465

DOI: [10.5281/zenodo.21269465](https://doi.org/10.5281/zenodo.21269465)

---

## License

Source code in this repository is released under the [MIT License](./LICENSE).
The license covers the code only — the raw human EMG data are not part of
this repository and are governed by a separate data-use agreement (see the data
note above).

---

## Authors and contact

Developed in the [n-squared lab](https://www.nsquared.tf.fau.de/), Department of
Artificial Intelligence in Biomedical Engineering, Friedrich-Alexander-Universität
Erlangen-Nürnberg (FAU).

- Code & analysis: **Pauline Wittermann** and **Dominik I. Braun**
- Data requests / correspondence: **Alessandro Del Vecchio** (corresponding author) — <alessandro.del.vecchio@fau.de>
- Principal investigator: **Alessandro Del Vecchio**

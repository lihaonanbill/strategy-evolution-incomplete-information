# Strategy Evolution Under Incomplete Information

This repository contains the Python code used to reproduce the computational
experiments for a manuscript on strategy evolution under incomplete information
in networked evolutionary games.

No external dataset is required. All pair-wise, hypergraph, and mixed
interaction structures used by the experiments are generated programmatically
from the specified parameters.

## Overview

The experiments study deterministic profile transitions in networked
evolutionary games with incomplete information. The main independent variables
are:

- `d2`: pair-wise degree
- `d3`: 3-uniform hypergraph degree
- `rate` or `R`: information rate
- `r_over_n` or `m`: marginal per capita return in the public goods game
- `update_rule`: strategy updating rule

The main dependent variables are:

- cooperation percentage
- deviate score
- mean convergent steps

## Repository Structure

- `neg_incomplete_info/generate.py`: generates network structures and payoff tables.
- `neg_incomplete_info/transition_matrix.py`: computes payoffs, strategy updates, and transition matrices.
- `neg_incomplete_info/graph_operation.py`: computes attractors, basins, cooperation rates, deviate scores, and convergence metrics.
- `experiments/experiment_cooperation.py`: runs and plots cooperation-percentage parameter sweeps.
- `experiments/experiment_dynamics.py`: runs and plots deviate-score and convergence-step parameter sweeps.
- `examples/run_demo.py`: runs a small demo without external data.
- `results/`: stores generated figures and cached experiment outputs.

## System Requirements

The code is written in Python and uses:

- `numpy`
- `pandas`
- `matplotlib`
- `networkx`

The code has been prepared for Python 3.10 or later. No non-standard hardware
is required. A normal desktop or laptop is sufficient for the demo. Full
parameter sweeps are more expensive because the profile state space grows as
`2^N`.

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Typical installation time on a normal desktop computer is less than a few
minutes, depending on the Python environment and package cache.

## Quick Demo

```bash
python examples/run_demo.py
```

Expected output includes:

- the number of generated local games
- the number of global states
- cooperation percentage
- deviate score
- mean convergent steps

The demo uses a small generated mixed network and should finish quickly on a
normal desktop computer.

## Reproducing the Manuscript Experiments

The full experiments are implemented in `experiments/experiment_cooperation.py`
and `experiments/experiment_dynamics.py`. After installing the dependencies,
run the cooperation sweep with:

```bash
python - <<'PY'
from experiments.experiment_cooperation import run_and_plot_cached

run_and_plot_cached(
    save_pickle_path="results/cache_full.pkl",
    save_csv_path="results/cache_full.csv",
)
PY
```

Run the dynamics sweep with:

```bash
python - <<'PY'
from experiments.experiment_dynamics import run_and_plot_cached

run_and_plot_cached(
    save_pickle_path="results/cache_metrics.pkl",
    save_csv_path="results/cache_metrics.csv",
)
PY
```

The experiment functions are parameterized inside the scripts. The default
manuscript-scale setting uses `N = 10` players and repeated random selections
of accessible neighbors for each parameter point. Full sweeps can take much
longer than the demo.

The cooperation experiment scans combinations of `d2`, `d3`, `r_over_n`, and
`rate`, then computes the expected cooperation percentage.

The dynamics experiment computes deviate score and mean convergent steps by
comparing transition structures under different information rates.

## Inputs and Outputs

No external input dataset is needed. The core generated input is a `games`
dictionary:

```python
games = {
    "game1": ["1", "2"],
    "game2": ["1", "3", "5"],
}
```

Each entry is a local interaction. A list of length 2 represents a pair-wise
edge, and a list of length 3 represents a 3-uniform hyperedge.

Generated outputs include CSV cache files, optional pickle cache files, and
figures saved by the plotting utilities. These outputs should be placed in
`results/` or another user-specified output directory.

## License

This software is released under the MIT License. See `LICENSE`.

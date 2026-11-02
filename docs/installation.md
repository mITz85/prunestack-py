# Installation

## Requirements

* Python **3.15**
* numpy ≥ 2.5, pandas ≥ 3.0, scikit-learn ≥ 1.9, scipy ≥ 1.18, PyYAML ≥ 6.0.3, tqdm ≥ 4.70
* matplotlib ≥ 3.11 (optional, for plots in the HTML report)

## From source

```bash
git clone https://github.com/mITz85/prunestack-py.git
cd prunestack-py
python3.15 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[plots]"
```

Extras:

| Extra | Installs |
|-------|----------|
| `plots` | matplotlib |
| `dev` | pytest, pytest-cov, pytest-timeout, matplotlib |

## Exact environment of the manuscript

```bash
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

## Verify

```bash
prunestack --version
pytest                             # requires the dev extra
```

`python -m prunestack.cli` is equivalent to the `prunestack` command.

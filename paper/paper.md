---
title: "pyqueueing: An Open-Source Python Library for Analytical Queueing Models and Capacity Planning"
tags:
  - Python
  - queueing theory
  - operations research
  - capacity planning
  - Erlang
authors:
  - name: Keitaro Kaida
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher, Japan
    index: 1
date: 18 April 2026
bibliography: paper.bib
---

# Summary

`pyqueueing` is an open-source Python package that provides analytical
queueing models and capacity planning tools through a consistent API.
It implements eleven models—M/M/1, M/M/$c$, M/M/1/$K$, M/M/$c$/$K$,
M/M/$\infty$, M/G/1, Erlang B, Erlang C, Erlang A, and Quasi-Birth-Death
(QBD) processes—along with a call centre staffing planner and parameter
sensitivity analysis utilities. The library targets education, practical
capacity planning, and research prototyping in operations research and
stochastic modelling.

# Statement of Need

Queueing theory provides closed-form and iterative methods for computing
server utilisation, waiting times, queue lengths, and blocking
probabilities in service systems [@gross2008]. These analytical results
are widely used in telecommunications, call centre operations, healthcare,
and cloud computing [@koole2013].

Despite Python's dominance in scientific computing, no maintained package
offers a unified API for standard analytical queueing models.
Practitioners typically rely on ad-hoc scripts, spreadsheets, or
discrete-event simulation frameworks such as SimPy. The table below
compares `pyqueueing` with existing tools.

| Package | Analytical | Simulation | QBD | Staffing |
|---------|:----------:|:----------:|:---:|:--------:|
| **pyqueueing** (this work) | ✓ | — | ✓ | ✓ |
| queueing-tool [@queueingtool] | — | ✓ | — | — |
| SimPy | — | ✓ | — | — |
| Custom scripts / spreadsheets | partial | — | — | — |

`pyqueueing` fills this gap by providing:

- Eleven queueing models behind a common `BaseQueueModel` abstract base class.
- Log-space numerical implementations stable for systems with $c \ge 500$
  servers, avoiding factorial overflow.
- Workforce planning tools built on Erlang A/C models.
- Sensitivity analysis and optional plotting utilities.

# Features

All classical models expose a uniform interface: `utilization()`,
`mean_queue_length()`, `mean_system_size()`, `mean_wait()`,
`mean_system_time()`, `summary()`, and `to_dict()`.

| Class | Notation | Description |
|-------|----------|-------------|
| `MM1` | M/M/1 | Single server, CDF/PDF of $W_q$ |
| `MMC` | M/M/$c$ | Multi-server (log-space stable) |
| `MM1K` | M/M/1/$K$ | Finite capacity, blocking probability |
| `MMcK` | M/M/$c$/$K$ | Multi-server finite (log-space) |
| `MMInf` | M/M/$\infty$ | Infinite servers, Poisson size |
| `MG1` | M/G/1 | General service (Pollaczek–Khinchine) |
| `ErlangB` | M/M/$c$/$c$ | Loss system, trunk sizing |
| `ErlangC` | M/M/$c$ | Wait probability, capacity planning |
| `ErlangA` | M/M/$c$+M | Customer impatience / abandonment |
| `QBD` | — | Matrix-geometric solver |

The `CallCenterPlanner` computes the minimum number of agents for a
target service level and generates staffing tables. The `sensitivity`
module provides `sweep()` and `sweep2d()` for one- and two-dimensional
parameter sweeps with results collected into NumPy arrays.

# Implementation Notes

`pyqueueing` is pure Python, depending on NumPy (≥ 1.24) and
SciPy (≥ 1.10). Matplotlib is an optional dependency for visualisation.

**Numerical stability.** All expressions involving $a^k / k!$ are
computed as $\exp(k \ln a - \ln\Gamma(k+1))$ via
`scipy.special.gammaln`, preventing overflow for $k \ge 171$. This
approach has been verified with $c = 500$ servers.

**QBD solver.** The rate matrix $R$ for QBD processes [@neuts1981] is
computed by successive substitution or logarithmic reduction (cyclic
reduction on a uniformised DTMC) [@latouche1999]. Results have been
validated against known M/M/1 solutions expressed as QBD.

**Package layout.** The `src` layout separates models, solvers,
utilities, plotting, and planning modules. Full type annotations with a
`py.typed` marker (PEP 561) are provided.

# Example Usage

```python
from pyqueueing import MM1, ErlangC
from pyqueueing.sensitivity import sweep

# M/M/1 queue
q = MM1(arrival_rate=2.0, service_rate=3.0)
q.mean_wait()          # 0.6667
q.mean_queue_length()  # 1.3333

# Call centre staffing: 80 % answered within 20 s
ec = ErlangC(arrival_rate=100, service_rate=12)
ec.required_servers(
    target_service_level=(20/3600, 0.80))  # 12

# Sensitivity analysis
result = sweep(MM1, {"service_rate": 3.0},
               "arrival_rate", [1.0, 2.0, 2.5])
result["utilization"]  # array([0.333, 0.667, 0.833])
```

# Testing and Documentation

The test suite comprises 125 unit tests across 15 files, executed via
`pytest`. Tests cover analytical correctness against known results,
cross-model consistency (e.g., M/M/1 as a special case of M/M/$c$ with
$c=1$), and edge cases including large server counts ($c = 500$) and
near-saturation ($\rho \to 1$).

API documentation is provided through comprehensive docstrings on all
public classes and methods. Three Jupyter notebook examples are included
in the repository demonstrating basic models, call centre staffing, and
cross-model comparison.

# Availability

- **Repository:** <https://github.com/kaitaro06/pyqueueing>
- **PyPI:** <https://pypi.org/project/pyqueueing/>
- **License:** MIT

# Acknowledgements

The author acknowledges the academic queueing theory community and prior
literature in stochastic service systems, teletraffic engineering, and
matrix-analytic methods.

# References

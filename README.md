# pyqueueing

A Python library for queueing theory — analytical models, capacity planning, and performance evaluation.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-125%20passed-brightgreen)]()

## Installation

```bash
pip install pyqueueing
```

For development (with test/lint tools):

```bash
pip install pyqueueing[dev]
```

For plotting support:

```bash
pip install pyqueueing[plot]
```

## Quick Start — 30 seconds

```python
from pyqueueing import MM1, MMC, ErlangC

# M/M/1: single server queue
q = MM1(arrival_rate=2.0, service_rate=3.0)
print(f"Utilization: {q.utilization():.2%}")       # 66.67%
print(f"Mean wait time: {q.mean_wait():.4f}")      # 0.6667
print(f"Mean queue length: {q.mean_queue_length():.4f}")  # 1.3333

# M/M/c: multi-server queue
q = MMC(arrival_rate=10, service_rate=3, servers=4)
print(f"P(wait): {q.prob_wait():.4f}")             # Erlang C probability

# Erlang C: "How many agents do I need?"
ec = ErlangC(arrival_rate=100, service_rate=12)
c = ec.required_servers(target_service_level=(20, 0.80))  # 80% answered within 20s
print(f"Required servers: {c}")
```

## Supported Models

| Class | Kendall Notation | Description |
|-------|-----------------|-------------|
| `MM1` | M/M/1 | Single server, infinite capacity |
| `MMC` | M/M/c | Multi-server, infinite capacity (log-space stable for c≥500) |
| `MM1K` | M/M/1/K | Single server, finite capacity |
| `MMcK` | M/M/c/K | Multi-server, finite capacity (log-space stable) |
| `MMInf` | M/M/∞ | Infinite servers (Poisson system size) |
| `MG1` | M/G/1 | General service time (Pollaczek–Khinchine) |
| `ErlangB` | M/M/c/c | Loss system — blocking probability & trunk sizing |
| `ErlangC` | M/M/c | Wait probability + capacity planning |
| `ErlangA` | M/M/c+M | Impatient customers (abandonment) |

### Advanced

| Class | Description |
|-------|-------------|
| `QBD` | Quasi-Birth-Death process (matrix geometric solver) |
| `CallCenterPlanner` | Multi-interval staffing, cost optimization (Erlang A based) |

## Common API

All models share a consistent interface:

```python
q.utilization()        # Server utilization ρ
q.mean_wait()          # Mean waiting time Wq
q.mean_system_time()   # Mean time in system W
q.mean_queue_length()  # Mean queue length Lq
q.mean_system_size()   # Mean number in system L
q.summary()            # Dict of all metrics
q.to_dict()            # Serialize parameters
```

## Use Cases

### Call Center Staffing (Erlang C)

```python
from pyqueueing import ErlangC

# 100 calls/hour, average handle time 5 minutes (12 calls/hour per agent)
ec = ErlangC(arrival_rate=100, service_rate=12)

# How many agents for 80/20 service level? (80% answered within 20 seconds)
agents = ec.required_servers(target_service_level=(20/3600, 0.80))

# How many agents for <5% probability of waiting?
agents = ec.required_servers(target_wait_prob=0.05)

# How many agents for average wait < 10 seconds?
agents = ec.required_servers(target_mean_wait=10/3600)
```

### Network Buffer Sizing (M/M/1/K)

```python
from pyqueueing import MM1K

# Packets arrive at 800/s, processed at 1000/s, buffer holds 50 packets
q = MM1K(arrival_rate=800, service_rate=1000, capacity=50)
print(f"Packet loss rate: {q.prob_block():.6f}")
print(f"Effective throughput: {q.effective_arrival_rate():.1f} pkt/s")
```

### Trunk Line Provisioning (Erlang B)

```python
from pyqueueing import ErlangB

# 100 Erlangs offered, target < 1% blocking
eb = ErlangB(arrival_rate=100, service_rate=1, servers=1)
trunks = eb.required_servers(target_block_prob=0.01)
print(f"Required trunk lines: {trunks}")
```

### General Service Time (M/G/1)

```python
from pyqueueing import MG1

# Exponential service (CV=1) vs deterministic service (CV=0)
q_exp = MG1(arrival_rate=2.0, service_rate=3.0, service_cv=1.0)
q_det = MG1(arrival_rate=2.0, service_rate=3.0, service_cv=0.0)
print(f"M/M/1 Lq: {q_exp.mean_queue_length():.4f}")  # 1.3333
print(f"M/D/1 Lq: {q_det.mean_queue_length():.4f}")  # 0.6667 (half!)
```

### Impatient Customers (Erlang A)

```python
from pyqueueing import ErlangA

# Customers abandon after avg 60s patience
q = ErlangA(arrival_rate=120, service_rate=10, servers=10, patience_rate=1/60)
print(f"Abandon rate: {q.prob_abandon():.2%}")
print(f"ASA (avg speed of answer): {q.mean_wait_answered():.1f}s")
```

### Call Center Staffing Planner

```python
from pyqueueing import CallCenterPlanner

planner = CallCenterPlanner(
    arrival_rate=100, service_rate=12, patience_rate=1/60
)
# Minimum agents for 80/20 service level
agents = planner.required_agents(target_service_level=(20, 0.80))

# Cost-optimal staffing across intervals
table = planner.staffing_table(
    arrival_rates=[80, 100, 120, 90],
    interval_minutes=30,
    target_service_level=(20, 0.80),
)
```

### QBD / Matrix Geometric Method

```python
from pyqueueing import QBD
import numpy as np

# Define transition rate sub-matrices
A0 = np.array([[2.0]])   # upward transitions
A1 = np.array([[-5.0]])  # level-internal
A2 = np.array([[3.0]])   # downward transitions

qbd = QBD(A0=A0, A1=A1, A2=A2)
print(f"Stable: {qbd.is_stable()}")
print(f"Mean level: {qbd.mean_level():.4f}")
```

### Sensitivity Analysis & Plotting

```python
from pyqueueing import MM1
from pyqueueing.sensitivity import sweep
from pyqueueing.plotting import plot_sensitivity

results = sweep(MM1, "arrival_rate", [1, 2, 3, 4],
                fixed={"service_rate": 5.0},
                metrics=["utilization", "mean_wait"])

plot_sensitivity(results, xlabel="Arrival Rate λ")
```

## Key Formulas

### M/M/1
$$\rho = \lambda/\mu, \quad L_q = \frac{\rho^2}{1-\rho}, \quad W_q = \frac{\rho}{\mu(1-\rho)}$$

### M/M/c (Erlang C)
$$C(c, a) = \frac{a^c / ((c-1)!(c-a))}{\sum_{k=0}^{c-1} a^k/k! + a^c/((c-1)!(c-a))}$$

### M/G/1 (Pollaczek–Khinchine)
$$L_q = \frac{\rho^2(1 + C_s^2)}{2(1-\rho)}$$

### Little's Law
$$L = \lambda W, \quad L_q = \lambda W_q$$

## Roadmap

- [x] **v0.1** — Core models (MM1, MMC, MM1K, MMcK, MG1, MMInf, ErlangB, ErlangC, ErlangA)
- [x] **v0.1** — Wait time CDF/PDF, sensitivity analysis, plotting
- [x] **v0.1** — QBD / matrix geometric, CallCenterPlanner
- [ ] **v0.2** — G/G/c approximations, priority queues
- [ ] **v0.3** — Queueing networks, time-varying arrival rates

## License

MIT

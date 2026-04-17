"""Plotting utilities for pyqueueing.

Provides ready-made charts for common queueing analysis tasks.
Requires matplotlib (install via ``pip install pyqueueing[plot]``).
"""

from __future__ import annotations

from typing import Any, Sequence, Type

import numpy as np

from pyqueueing.models.base import BaseQueueModel
from pyqueueing.sensitivity import sweep


def _import_plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install pyqueueing[plot]"
        )


def plot_sensitivity(
    model_cls: Type[BaseQueueModel],
    params: dict[str, Any],
    sweep_param: str,
    sweep_values: Sequence[float] | np.ndarray,
    metrics: Sequence[str] | None = None,
    *,
    title: str | None = None,
    ax=None,
):
    """Plot performance metrics as a function of one parameter.

    Args:
        model_cls: Queueing model class.
        params: Base parameters.
        sweep_param: Parameter to vary.
        sweep_values: Values for the sweep parameter.
        metrics: Metrics to plot (default: mean_wait, mean_queue_length).
        title: Optional plot title.
        ax: Optional matplotlib Axes.

    Returns:
        matplotlib Figure.
    """
    plt = _import_plt()

    if metrics is None:
        metrics = ["mean_wait", "mean_queue_length"]

    result = sweep(model_cls, params, sweep_param, sweep_values, metrics)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    for m in metrics:
        ax.plot(result[sweep_param], result[m], label=m, linewidth=2)

    ax.set_xlabel(sweep_param)
    ax.set_ylabel("Value")
    ax.set_title(title or f"{model_cls.__name__} Sensitivity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_wait_cdf(
    model: BaseQueueModel,
    t_max: float | None = None,
    n_points: int = 200,
    *,
    title: str | None = None,
    ax=None,
):
    """Plot the waiting-time CDF P(Wq ≤ t).

    The model must have a ``wait_time_cdf`` method.

    Args:
        model: A queueing model instance with ``wait_time_cdf``.
        t_max: Maximum time value. Auto-calculated if None.
        n_points: Number of points to plot.
        title: Optional plot title.
        ax: Optional matplotlib Axes.

    Returns:
        matplotlib Figure.
    """
    plt = _import_plt()

    if not hasattr(model, "wait_time_cdf"):
        raise AttributeError(f"{type(model).__name__} has no wait_time_cdf method")

    if t_max is None:
        t_max = model.mean_wait() * 5 if model.mean_wait() > 0 else 1.0

    t = np.linspace(0, t_max, n_points)
    cdf = np.array([model.wait_time_cdf(ti) for ti in t])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.plot(t, cdf, linewidth=2)
    ax.set_xlabel("Wait time t")
    ax.set_ylabel("P(Wq ≤ t)")
    ax.set_title(title or f"{type(model).__name__} Wait Time CDF")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    return fig

"""Sensitivity analysis — vectorized parameter sweeps for queueing models.

Allows passing arrays of parameters to compute performance metrics
across a range of values, useful for capacity planning and what-if analysis.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Type

import numpy as np

from pyqueueing.models.base import BaseQueueModel


def sweep(
    model_cls: Type[BaseQueueModel],
    params: dict[str, Any],
    sweep_param: str,
    sweep_values: Sequence[float] | np.ndarray,
    metrics: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Sweep a single parameter and compute metrics for each value.

    Args:
        model_cls: Queueing model class (e.g., MM1, MMC).
        params: Base parameters dict (e.g., ``{"arrival_rate": 2, "service_rate": 3}``).
        sweep_param: Name of the parameter to sweep.
        sweep_values: Array of values for the sweep parameter.
        metrics: List of method names to evaluate (default: all from ``summary()``).

    Returns:
        Dict mapping metric name to numpy array of results.
        Also includes the sweep parameter values under its name.

    Examples:
        >>> from pyqueueing import MM1
        >>> from pyqueueing.sensitivity import sweep
        >>> result = sweep(MM1, {"service_rate": 3.0}, "arrival_rate", [1.0, 2.0, 2.5])
        >>> result["utilization"]
        array([0.33333333, 0.66666667, 0.83333333])
    """
    sweep_values = np.asarray(sweep_values, dtype=float)

    if metrics is None:
        metrics = [
            "utilization",
            "mean_queue_length",
            "mean_system_size",
            "mean_wait",
            "mean_system_time",
        ]

    results: dict[str, list[float]] = {m: [] for m in metrics}

    for val in sweep_values:
        p = {**params, sweep_param: float(val)}
        try:
            model = model_cls(**p)
            for m in metrics:
                results[m].append(getattr(model, m)())
        except (ValueError, ZeroDivisionError):
            for m in metrics:
                results[m].append(float("nan"))

    out = {sweep_param: sweep_values}
    for m in metrics:
        out[m] = np.array(results[m])
    return out


def sweep2d(
    model_cls: Type[BaseQueueModel],
    params: dict[str, Any],
    sweep_x: tuple[str, Sequence[float] | np.ndarray],
    sweep_y: tuple[str, Sequence[float] | np.ndarray],
    metric: str = "mean_wait",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-dimensional parameter sweep.

    Args:
        model_cls: Queueing model class.
        params: Base parameters dict.
        sweep_x: (param_name, values) for x-axis.
        sweep_y: (param_name, values) for y-axis.
        metric: Method name to evaluate.

    Returns:
        (X, Y, Z) arrays suitable for contour plots.
    """
    x_name, x_vals = sweep_x
    y_name, y_vals = sweep_y
    x_arr = np.asarray(x_vals)
    y_arr = np.asarray(y_vals)
    x_is_int = np.issubdtype(x_arr.dtype, np.integer)
    y_is_int = np.issubdtype(y_arr.dtype, np.integer)

    X, Y = np.meshgrid(x_arr, y_arr)
    Z = np.full(X.shape, float("nan"))

    for i, yv in enumerate(y_arr):
        for j, xv in enumerate(x_arr):
            p = {
                **params,
                x_name: int(xv) if x_is_int else float(xv),
                y_name: int(yv) if y_is_int else float(yv),
            }
            try:
                model = model_cls(**p)
                Z[i, j] = getattr(model, metric)()
            except (ValueError, ZeroDivisionError):
                pass

    return X, Y, Z

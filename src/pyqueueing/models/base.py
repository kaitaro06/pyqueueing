"""Base class defining the common interface for all queueing models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseQueueModel(ABC):
    """Abstract base class for queueing models.

    All concrete queueing models must implement the abstract methods defined here
    to ensure a consistent API across the library.
    """

    @abstractmethod
    def utilization(self) -> float:
        """Return the server utilization ρ."""
        ...

    @abstractmethod
    def mean_queue_length(self) -> float:
        """Return the mean number of customers waiting in queue (Lq)."""
        ...

    @abstractmethod
    def mean_system_size(self) -> float:
        """Return the mean number of customers in the system (L)."""
        ...

    @abstractmethod
    def mean_wait(self) -> float:
        """Return the mean waiting time in queue (Wq)."""
        ...

    @abstractmethod
    def mean_system_time(self) -> float:
        """Return the mean time in the system (W)."""
        ...

    def summary(self) -> dict[str, float]:
        """Return a dict of all standard performance measures."""
        return {
            "utilization": self.utilization(),
            "mean_queue_length_Lq": self.mean_queue_length(),
            "mean_system_size_L": self.mean_system_size(),
            "mean_wait_Wq": self.mean_wait(),
            "mean_system_time_W": self.mean_system_time(),
        }

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize the model parameters to a dict."""
        ...

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.to_dict().items() if k != "model")
        return f"{self.__class__.__name__}({params})"

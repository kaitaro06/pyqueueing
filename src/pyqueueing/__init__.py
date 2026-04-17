"""pyqueueing — A Python library for queueing theory models and analysis.

Provides analytical solutions for classical queueing models including
M/M/1, M/M/c, M/M/1/K, M/M/c/K, M/G/1, M/M/∞, Erlang A/B/C,
and advanced QBD (Quasi-Birth-Death) process analysis.
"""

from pyqueueing.models.mm1 import MM1
from pyqueueing.models.mmc import MMC
from pyqueueing.models.mm1k import MM1K
from pyqueueing.models.mmck import MMcK
from pyqueueing.models.mg1 import MG1
from pyqueueing.models.mminf import MMInf
from pyqueueing.models.erlang_a import ErlangA
from pyqueueing.models.erlang_c import ErlangC
from pyqueueing.models.erlang_b import ErlangB
from pyqueueing.models.qbd import QBD
from pyqueueing.planner import CallCenterPlanner

__version__ = "0.1.0"

__all__ = [
    "MM1",
    "MMC",
    "MM1K",
    "MMcK",
    "MG1",
    "MMInf",
    "ErlangA",
    "ErlangC",
    "ErlangB",
    "QBD",
    "CallCenterPlanner",
]

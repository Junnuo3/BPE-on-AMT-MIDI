"""
Evaluation metric registry. Importing this package registers every built-in
metric (CLAP, FAD, chroma, groove) into METRICS; scripts/run_evaluation.py
picks metrics by name from there. Add a new metric by dropping a module in
this package that calls @register("name") on a Metric subclass and importing
it below -- run_evaluation.py needs no changes.
"""

from .base import METRICS, EvalContext, Metric, register  # noqa: F401
from . import clap   # noqa: F401,E402
from . import fad    # noqa: F401,E402
from . import chroma  # noqa: F401,E402
from . import groove  # noqa: F401,E402
from . import groove_bar  # noqa: F401,E402

__all__ = ["METRICS", "EvalContext", "Metric", "register"]

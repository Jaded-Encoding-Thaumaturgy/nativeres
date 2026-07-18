import importlib
import threading
from collections.abc import Iterator
from inspect import isabstract
from logging import getLogger

from jetpytools import get_subclasses
from vsmasktools import EdgeDetect
from vsview.api import run_in_background

logger = getLogger(__name__)

QCHART_IMPORTED = False
LOCK = threading.Lock()


@run_in_background(name="WarmupImportPlots")
def warmup_plots() -> None:
    """Warmup import of QChart to avoid long startup times."""
    global QCHART_IMPORTED

    with LOCK:
        if not QCHART_IMPORTED:
            QCHART_IMPORTED = True
            logger.debug("Importing Plots")
            importlib.import_module("nativeres.plotting")
            logger.debug("Importing Plots done")


def get_edge_detect_classes() -> Iterator[type[EdgeDetect]]:
    for s in sorted(get_subclasses(EdgeDetect), key=lambda x: x.__name__):  # type: ignore[type-abstract]
        if not isabstract(s) and s.__module__.split(".")[-1] != "_abstract":
            yield s

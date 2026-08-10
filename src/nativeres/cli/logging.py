import logging
import warnings

from rich.console import Console
from rich.logging import RichHandler

console = Console(stderr=True)


def setup_logging() -> None:
    warnings.filterwarnings("always")
    logging.captureWarnings(True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[RichHandler(console=console)],
        format="{name}: {message}",
        style="{",
    )

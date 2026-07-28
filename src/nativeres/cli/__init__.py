"""CLI module"""

import sys
import warnings
from itertools import zip_longest
from logging import INFO, basicConfig, captureWarnings, getLogger
from math import ceil
from typing import Annotated, Any, Literal, assert_never, cast

from jetpytools import SPath, mod2
from rich.console import Console
from rich.logging import RichHandler
from rich.pretty import pretty_repr
from rich.progress import BarColumn, Progress, TextColumn
from rich.style import Style
from rich.table import Table
from vskernels import ComplexKernel
from vsmasktools import EdgeDetect
from vssource import Indexer
from vstools import vs

from .. import funcs
from ..constants import HIGH_RATE, LOW_RATE
from ..kernels import default_kernels
from .components import (
    app,
    base_dim_opt,
    base_parity_opt,
    crop_opt,
    cull_rate_opt,
    debug_opt,
    dim_mode_opt,
    dim_opt,
    frame_opt,
    global_debug_opt,
    indexer_opt,
    input_file_arg,
    kernel_opt,
    linear_opt,
    mask_opt,
    metric_mode_opt,
    radius_opt,
    range_dim_opt,
    show_default_kernels_opt,
    show_vskernels_opt,
    step_opt,
)
from .helpers import get_progress, get_videonode_from_input

console = Console(stderr=True)

warnings.filterwarnings("always")
captureWarnings(True)
basicConfig(
    level=INFO,
    handlers=[RichHandler(console=console)],
    format="{name}: {message}",
    style="{",
)

logger = getLogger(__name__)


@app.callback()
def callback(
    show_kernels: Annotated[bool, show_default_kernels_opt] = False,
    show_vskernels: Annotated[bool, show_vskernels_opt] = False,
    debug: Annotated[bool, debug_opt] = False,
    global_debug: Annotated[bool, global_debug_opt] = False,
) -> None:
    """Descale analysis tools for VapourSynth."""


@app.command(
    help="[bold]Determine the native resolution of upscaled material.[/]\n\n"
    "Analyzes a range of dimensions to find which one produces the lowest error when inverse scaled.\n"
    "Primary use case is finding the native resolution of upscaled anime.",
    no_args_is_help=True,
)
def getnative(
    input_file: Annotated[SPath, input_file_arg],
    range_dim: Annotated[tuple[int, int] | None, range_dim_opt] = None,
    dim_mode: Annotated[Literal["height", "width"], dim_mode_opt] = "height",
    kernel: Annotated[ComplexKernel, kernel_opt] = cast(ComplexKernel, "bilinear"),  # noqa: B008
    linear: Annotated[bool, linear_opt] = False,
    frame: Annotated[int, frame_opt] = 0,
    step: Annotated[float, step_opt] = 1,
    base_parity: Annotated[Literal["odd", "even"], base_parity_opt] = "even",
    crop: Annotated[tuple[int, int, int, int] | None, crop_opt] = None,
    metric_mode: Annotated[funcs.MetricMode, metric_mode_opt] = "MAE",
    indexer: Annotated[Indexer, indexer_opt] = cast(Indexer, "bs"),  # noqa: B008
) -> None:
    import numpy as np
    from PySide6.QtWidgets import QApplication, QMainWindow, QStyle

    from ..plotting import RescalePlotWidget

    clip = get_videonode_from_input(input_file, indexer)

    if linear:
        clip = clip.resize.Point(transfer=vs.TRANSFER_LINEAR)

    # Resolve dimension and the range of dimensions to check
    if range_dim:
        start, stop = range_dim
    else:
        match dim_mode:
            case "height":
                dim = clip.height
            case "width":
                dim = clip.width
            case _:
                assert_never(dim_mode)

        start, stop = int(dim * LOW_RATE), int(dim * HIGH_RATE)

    # Build the list of dims (int or fractional)
    step_f = float(step)
    if step_f.is_integer():
        dims = range(start, stop + 1, int(step_f))
        x_label_fmt = "%.0f"
    else:
        num = int((stop - start) / step_f) + 1
        dims = np.linspace(start, start + step_f * (num - 1), num).tolist()
        x_label_fmt = f"%.{str(step_f)[::-1].find('.') + 1}f"

        if base_parity == "odd":
            dims = [(d, mod2(ceil(d)) + 1) for d in dims]

    # Pair with the fixed dimension
    match dim_mode:
        case "height":
            dimensions = zip_longest([clip.width], dims, fillvalue=clip.width)
        case "width":
            dimensions = zip_longest(dims, [clip.height], fillvalue=clip.height)
        case _:
            assert_never(dim_mode)

    # Pretty progress
    progress = get_progress(console)
    gtask_id = progress.add_task("Gathering data...", total=None)

    logger.debug(kernel)

    with progress:
        results = funcs.getnative(
            clip,
            frame,
            dimensions,  # type: ignore[arg-type]
            kernel,
            crop,
            metric_mode=metric_mode,
            progress_cb=lambda curr, total: progress.update(gtask_id, completed=curr, total=total, visible=True),
        )
        progress.update(gtask_id, total=100, completed=100, refresh=True)

    dims, errors = zip(*results)

    # Show the plot window
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Native Resolution Analysis")
    win.setWindowIcon(win.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView))
    win.resize(1000, 600)

    plot = RescalePlotWidget(
        f"Error plot - {kernel.pretty_string} on {dim_mode}",
        [getattr(d, dim_mode) for d in dims],
        errors,
        dim_mode.title(),
    )
    plot.axis_x.setLabelFormat(x_label_fmt)

    win.setCentralWidget(plot)

    win.show()
    app.exec()


@app.command(
    help="[bold]Identify the best inverse scaler for a given resolution.[/]\n\n"
    "Compares multiple kernels against a specific target resolution to determine which one "
    "was likely used for the original upscaling.",
    epilog="""
[dim]Notes:

 - getscaler gives heuristic results; it's not infallible.

 - Always visually verify the suggested scaler and parameters on multiple frames before trusting them.[/dim]
""",
    no_args_is_help=True,
)
def getscaler(
    input_file: Annotated[SPath, input_file_arg],
    dim: Annotated[float, dim_opt],
    dim_mode: Annotated[Literal["height", "width"], dim_mode_opt] = "height",
    base_dim_opt: Annotated[int | None, base_dim_opt] = None,
    kernels: Annotated[list[ComplexKernel], kernel_opt] = [],  # noqa: B006
    linear: Annotated[bool, linear_opt] = False,
    frame: Annotated[int, frame_opt] = 0,
    crop: Annotated[tuple[int, int, int, int] | None, crop_opt] = None,
    metric_mode: Annotated[funcs.MetricMode, metric_mode_opt] = "MAE",
    mask: Annotated[type[EdgeDetect] | None, mask_opt] = None,
    indexer: Annotated[Indexer, indexer_opt] = cast(Indexer, "bs"),  # noqa: B008
) -> None:
    clip = get_videonode_from_input(input_file, indexer)

    if linear:
        clip = clip.resize.Point(transfer=vs.TRANSFER_LINEAR)

    # Resolve dimension to check
    scaler_args: dict[str, Any] = {
        "width": clip.width,
        "height": clip.height,
        dim_mode: dim,
        f"base_{dim_mode}": base_dim_opt,
    }

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
        transient=True,
    )
    task = progress.add_task("Gathering data...", total=None)

    with progress:
        ress = funcs.getscaler(
            clip,
            frame,
            kernels=(*default_kernels, *kernels),
            crop=crop,
            metric_mode=metric_mode,
            mask=mask,
            **scaler_args,
        )
    progress.update(task, completed=100, total=100, visible=False, refresh=True)

    # Results are sorted and displayed to the CLI for the user
    sorted_ress = sorted(ress, key=lambda r: r.error)
    best = sorted_ress[0]

    logger.debug("%s", pretty_repr(sorted_ress, max_width=200, indent_size=2))

    width, height = scaler_args["width"], scaler_args["height"]

    dwidth = f"{width:.0f}" if float(width).is_integer() else f"{width:.3f}"
    dheight = f"{height:.0f}" if float(height).is_integer() else f"{height:.3f}"

    table = Table(
        title=f"Results for frame {frame} — Resolution: {dwidth}x{dheight}",
        title_style=Style(bold=True),
        caption=f"Smallest error archieved by {best.kernel.pretty_string}: {best.error:.13f}",
        caption_style=Style(bold=True, dim=True),
        caption_justify="left",
        min_width=80,
    )
    table.add_column("Kernel")
    table.add_column("Error %", justify="center")
    table.add_column(metric_mode, justify="right")

    for res in sorted_ress:
        table.add_row(
            res.kernel.pretty_string, f"{res.error * 100 / best.error if best.error else 0:.2f} %", f"{res.error:.13f}"
        )

    console.rule()
    console.print(table, new_line_start=True)
    console.rule()
    console.print(
        "Getfscaler is not infallible!\n"
        "Always visually verify the suggested scaler and parameters on multiple frames before trusting them.",
        style=Style(color="yellow", dim=True),
    )


@app.command(
    help="[bold]Visualize the frequency distribution of a frame.[/]\n\n"
    "Calculates the Discrete Cosine Transform (DCT) of the image rows/columns "
    "to identify spikes that may indicate the native resolution or scaling artifacts.",
    no_args_is_help=True,
)
def getfreq(
    input_file: Annotated[SPath, input_file_arg],
    frame: Annotated[int, frame_opt] = 0,
    cull_rate: Annotated[float, cull_rate_opt] = 3.0,
    radius: Annotated[int, radius_opt] = 50,
    linear: Annotated[bool, linear_opt] = False,
    indexer: Annotated[Indexer, indexer_opt] = cast(Indexer, "bs"),  # noqa: B008
) -> None:
    from PySide6.QtWidgets import QApplication, QMainWindow, QStyle

    from ..funcs import get_dct_distribution
    from ..plotting import FrequencyPlotWidget

    clip = get_videonode_from_input(input_file, indexer)

    if linear:
        clip = clip.resize.Point(transfer=vs.TRANSFER_LINEAR)

    progress = get_progress(console)
    task = progress.add_task("Calculating DCT distribution...", total=None)

    with progress:
        dct_h, dct_v = get_dct_distribution(clip, frame, cull_rate=cull_rate)
        progress.update(task, completed=100, total=100, visible=False, refresh=True)

    min_val_h, max_val_h = int(clip.width * LOW_RATE), int(clip.width * HIGH_RATE)
    min_val_v, max_val_v = int(clip.height * LOW_RATE), int(clip.height * HIGH_RATE)

    # Show the plot window
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Frequency Analysis")
    win.setWindowIcon(win.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView))
    win.resize(1000, 600)

    plot = FrequencyPlotWidget(
        f"DCT Frequency - {input_file.name}",
        dct_h,
        dct_v,
        min_val_h,
        max_val_h,
        min_val_v,
        max_val_v,
        check_radius=radius,
    )

    win.setCentralWidget(plot)

    win.show()
    app.exec()

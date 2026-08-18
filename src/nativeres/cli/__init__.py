"""CLI module"""

import sys
from itertools import zip_longest
from logging import DEBUG, getLogger
from math import ceil
from typing import Annotated, Any, Literal, assert_never

from cyclopts import App, Group, Parameter
from jetpytools import mod2
from rich.pretty import pretty_repr
from rich.progress import BarColumn, Progress, TextColumn
from rich.style import Style
from rich.table import Table
from vskernels import Bilinear, SampleGridModel
from vsmasktools import EdgeDetect
from vssource import BestSource
from vstools import vs

from .. import funcs
from ..constants import HIGH_RATE, LOW_RATE
from ..kernels import default_kernels
from .components import (
    CleanHelpFormatter,
    CropOpt,
    DimModeOpt,
    FrameOpt,
    IndexerOpt,
    InputFileArg,
    KernelOpt,
    KernelsOpt,
    LinearOpt,
    MetricModeOpt,
    SampleGridModelOpt,
    helpers_group,
)
from .helpers import (
    get_progress,
    get_videonode_from_input,
    resolve_dimension,
    show_default_kernels,
    show_masks,
    show_vskernels,
)
from .logging import console, setup_logging

logger = getLogger(__name__)

app = App(
    name="nativeres",
    console=console,
    help_formatter=CleanHelpFormatter.with_newline_metadata(),  # type: ignore[no-untyped-call]
    default_parameter=Parameter(negative=()),
)
exclusive_group = Group("Command Options", sort_key=5)


@app.meta.default
def main_meta(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    show_kernels: Annotated[bool, Parameter(group=helpers_group, show_default=False)] = False,
    show_vskernels_flag: Annotated[
        bool,
        Parameter(name="show-vskernels", group=helpers_group, show_default=False),
    ] = False,
    show_masks_flag: Annotated[bool, Parameter(name="show-masks", group=helpers_group, show_default=False)] = False,
    debug: Annotated[bool, Parameter(show=False)] = False,
    global_debug: Annotated[bool, Parameter(show=False)] = False,
) -> None:
    """
    Descale analysis tools for VapourSynth.

    Args:
        show_kernels: Show the default checked kernels for getscaler and exit.
        show_vskernels_flag: Show the builtin supported kernels from vskernels and exit.
        show_masks_flag: Show the supported edge masks from vsmasktools and exit.
        debug: Enable debug output.
        global_debug: Enable global debug output.
    """
    setup_logging()

    if show_kernels:
        show_default_kernels()
    if show_vskernels_flag:
        show_vskernels()
    if show_masks_flag:
        show_masks()
    if debug:
        getLogger((__package__ or "").split(".")[0]).setLevel(DEBUG)
    if global_debug:
        getLogger().setLevel(DEBUG)

    app(tokens)


@app.command(help_format="rich")
def getnative(
    input_file: InputFileArg,
    /,
    range_dim: Annotated[
        tuple[int, int] | None,
        Parameter(alias="-rd", help="The inclusive range of resolutions to test (START END).", group=exclusive_group),
    ] = None,
    dim_mode: DimModeOpt = "height",
    kernel: KernelOpt = Bilinear(),  # noqa: B008
    linear: LinearOpt = False,
    sample_grid_model: SampleGridModelOpt = "edges",
    frame: FrameOpt = 0,
    step: Annotated[
        float,
        Parameter(
            short_alias=True,
            help="The increment step between resolutions in the tested range.",
            group=exclusive_group,
        ),
    ] = 1,
    base_parity: Annotated[
        Literal["odd", "even"],
        Parameter(alias="-bp", help="Base dimension parity for fractional descales.", group=exclusive_group),
    ] = "even",
    crop: CropOpt = None,
    metric_mode: MetricModeOpt = "MAE",
    indexer: IndexerOpt = BestSource,
) -> None:
    """
    Determine the native resolution of upscaled material

    Analyzes a range of dimensions to find which one produces the lowest error when inverse scaled.
    Primary use case is finding the native resolution of upscaled anime.
    """
    progress = get_progress(console, transient=True)

    with progress:
        task = progress.add_task("Initializing imports...", total=None)

        import numpy as np
        from PySide6.QtWidgets import QApplication, QMainWindow, QStyle

        from ..plotting import RescalePlotWidget

        progress.update(task, visible=False)

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

    sgm = (
        SampleGridModel[f"MATCH_{sample_grid_model.upper()}"]
        if isinstance(sample_grid_model, str)
        else sample_grid_model
    )

    # Pretty progress
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
            sample_grid_model=sgm,
            progress_cb=lambda curr, total: progress.update(
                gtask_id, completed=curr, total=total, refresh=True, visible=True
            ),
            func=getnative,
        )
        progress.update(gtask_id, total=100, completed=100, refresh=True)

    dims, errors = zip(*results)

    # Show the plot window
    qapp = QApplication(sys.argv)
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

    raise SystemExit(qapp.exec())


@app.command(
    help_format="rich",
    help_epilogue="""[dim]Notes:
- getscaler gives heuristic results; it's not infallible.
- Always visually verify the suggested scaler and parameters on multiple frames before trusting them.
[/dim]
""",
)
def getscaler(
    input_file: InputFileArg,
    dim: Annotated[float, Parameter(name="NUMBER", converter=lambda type_, tokens: resolve_dimension(tokens[0].value))],
    /,
    dim_mode: DimModeOpt = "height",
    base_dim: Annotated[int | None, Parameter(alias="-b", group=exclusive_group)] = None,
    kernels: KernelsOpt = [],  # noqa: B006
    linear: LinearOpt = False,
    sample_grid_model: SampleGridModelOpt = "edges",
    frame: FrameOpt = 0,
    crop: CropOpt = None,
    metric_mode: MetricModeOpt = "MAE",
    mask: Annotated[
        type[EdgeDetect] | None,
        Parameter(
            alias="-m",
            converter=lambda type_, tokens: EdgeDetect.from_param(tokens[0].value),
            group=exclusive_group,
        ),
    ] = None,
    indexer: IndexerOpt = BestSource,
) -> None:
    """
    Identify the best inverse scaler for a given resolution.

    Compares multiple kernels against a specific target resolution to determine which one
    was likely used for the original upscaling.

    Args:
        dim: The suspected native resolution to verify.
        base_dim: Base integer dimension if checking for fractional resolution.
        mask: Edge-detection mask to reduce noise influence on the metric.
    """
    clip = get_videonode_from_input(input_file, indexer)

    if linear:
        clip = clip.resize.Point(transfer=vs.TRANSFER_LINEAR)

    # Resolve dimension to check
    scaler_args: dict[str, Any] = {
        "width": clip.width,
        "height": clip.height,
        dim_mode: dim,
        f"base_{dim_mode}": base_dim,
    }

    sgm = (
        SampleGridModel[f"MATCH_{sample_grid_model.upper()}"]
        if isinstance(sample_grid_model, str)
        else sample_grid_model
    )

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
            sample_grid_model=sgm,
            func=getscaler,
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


@app.command
def getfreq(
    input_file: InputFileArg,
    /,
    frame: FrameOpt = 0,
    cull_rate: Annotated[float, Parameter(alias="-cr", group=exclusive_group)] = 3.0,
    radius: Annotated[int, Parameter(short_alias=True, group=exclusive_group)] = 50,
    linear: LinearOpt = False,
    indexer: IndexerOpt = BestSource,
) -> None:
    """
    Visualize the frequency distribution of a frame.

    Calculates the Discrete Cosine Transform (DCT) of the image rows/columns to identify spikes
    that may indicate the native resolution or scaling artifacts.

    Args:
        cull_rate: Cull the sides/top of the frame to focus on the center.
        radius: Radius for finding peaks/spikes in the frequency plot.
    """
    progress = get_progress(console, transient=True)

    with progress:
        task = progress.add_task("Initializing imports...", total=None)

        from PySide6.QtWidgets import QApplication, QMainWindow, QStyle

        from ..funcs import get_dct_distribution
        from ..plotting import FrequencyPlotWidget

        progress.update(task, visible=False)

    clip = get_videonode_from_input(input_file, indexer)

    if linear:
        clip = clip.resize.Point(transfer=vs.TRANSFER_LINEAR)

    task = progress.add_task("Calculating DCT distribution...", total=None)

    with progress:
        dct_h, dct_v = get_dct_distribution(clip, frame, cull_rate=cull_rate)
        progress.update(task, completed=100, total=100, visible=False, refresh=True)

    min_val_h, max_val_h = int(clip.width * LOW_RATE), int(clip.width * HIGH_RATE)
    min_val_v, max_val_v = int(clip.height * LOW_RATE), int(clip.height * HIGH_RATE)

    # Show the plot window
    qapp = QApplication(sys.argv)
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
    qapp.exec()


def main() -> None:
    app.meta()

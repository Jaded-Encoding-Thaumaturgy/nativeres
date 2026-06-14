from jetpytools import SPath
from typer import Argument, BadParameter, Option, Typer
from vsmasktools import EdgeDetect

from ..funcs import resolve_kernel
from .helpers import (
    resolve_dimension,
    resolve_dimension_mode,
    resolve_idx,
    set_debug,
    set_global_debug,
    show_default_kernels,
    show_vskernels,
)

# DEBUG
debug_opt = Option(
    "--debug",
    help="Enable debug output.",
    hidden=True,
    is_eager=True,
    callback=set_debug,
)
global_debug_opt = Option(
    "--global-debug",
    help="Enable global debug output.",
    hidden=True,
    is_eager=True,
    callback=set_global_debug,
)


# App
app = Typer(
    name="nativeres",
    rich_markup_mode="rich",
    pretty_exceptions_enable=True,
    add_completion=False,
    no_args_is_help=True,
)

# Commons
input_file_arg = Argument(
    help="Path to the source material to analyze. "
    "Supports videos, images, or VapourSynth scripts. For scripts, the first output is used.",
    metavar="INPUT",
    resolve_path=True,
    parser=SPath,
)
frame_opt = Option(
    "--frame",
    "-f",
    help="The specific frame number to extract and analyze from video inputs. Ignored for images.",
    metavar="INTEGER",
    rich_help_panel="Common",
)
kernel_opt = Option(
    "--kernel",
    "-k",
    help="The kernel(s) to use for inverse scaling.\n\n"
    "Can be a kernel name or a class call with parameters (e.g., 'Bicubic(b=0, c=0.5)').\n\n"
    "Use --show-kernels for a list of available kernels.",
    metavar="Kernel|Kernel(arg0=..., arg1=...)",
    parser=lambda v: resolve_kernel(v, BadParameter),
    rich_help_panel="Common",
)

dim_mode_opt = Option(
    "--dim-mode",
    "-dm",
    help="Specifies whether to analyze based on the [bold]height[/] or [bold]width[/] of the frame.",
    metavar="height|width|h|w",
    parser=resolve_dimension_mode,
    rich_help_panel="Common",
)
crop_opt = Option(
    "--crop",
    "-c",
    help="Crop the input frame before analysis to remove black bars.\n\n"
    "Format: [bold]LEFT RIGHT TOP BOTTOM[/] (e.g., '0 0 240 240').",
    metavar="L R T B",
    rich_help_panel="Common",
)
metric_mode_opt = Option(
    "--metric-mode",
    "-mm",
    help="The mathematical metric used to compare scaling results.\n\n"
    "- [bold]MAE[/] (Mean Absolute Error)\n\n"
    "- [bold]MSE[/] (Mean Squared Error)\n\n"
    "- [bold]RMSE[/] (Root Mean Squared Error)",
    metavar="MAE|MSE|RMSE",
    parser=lambda value: value.upper(),
    rich_help_panel="Common",
)
indexer_opt = Option(
    "--indexer",
    "-idx",
    help="The VapourSynth indexer used to load files.\n\nSpecifying the plugin namespace is also allowed (e.g., 'bs').",
    metavar="STRING",
    parser=resolve_idx,
    show_default="BestSource",
    rich_help_panel="Common",
)

# Helpers
show_default_kernels_opt = Option(
    "--show-kernels",
    help="Show the default checked kernels for getscaler and exit.",
    is_eager=True,
    show_default=False,
    callback=show_default_kernels,
    rich_help_panel="Helpers",
)
show_vskernels_opt = Option(
    "--show-vskernels",
    help="Show the builtin supported kernels from vskernels and exit.",
    is_eager=True,
    show_default=False,
    callback=show_vskernels,
    rich_help_panel="Helpers",
)

# getnative exclusive
range_dim_opt = Option(
    "--range-dim",
    "-rd",
    help="The inclusive range of resolutions to test.\n\nSpecify as [bold]START END[/] (e.g., '500 1080').",
    metavar="INTEGER INTEGER",
    show_default=False,
)
step_opt = Option("--step", "-s", help="The increment step between resolutions in the tested range.", metavar="NUMBER")


# getscaler exclusive
dim_opt = Argument(
    help="The suspected native resolution to verify. "
    "Use an integer for exact pixels (e.g., 720) or a float for sub-pixel dimensions (e.g., 719.8).",
    metavar="NUMBER",
    parser=resolve_dimension,
)
base_dim_opt = Option(
    "--base-dim",
    "-b",
    help="Base integer dimension if checking for fractional resolution.",
)
mask_opt = Option(
    "--mask",
    "-m",
    help="Edge-detection mask to reduce noise influence on the metric. "
    "Pass a mask name (e.g., 'Prewitt') or a class name from vsmasktools.",
    metavar="EDGEDETECT",
    parser=EdgeDetect.from_param,
)


# Frequency exclusive
cull_rate_opt = Option(
    "--cull-rate",
    "-cr",
    help="Cull the sides/top of the frame to focus on the center.",
    metavar="NUMBER",
    show_default=True,
)
radius_opt = Option(
    "--radius",
    "-r",
    help="Radius for finding peaks/spikes in the frequency plot.",
    metavar="INTEGER",
    show_default=True,
)

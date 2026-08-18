from typing import Annotated, Any, Literal

from cyclopts import Group, Parameter, Token
from cyclopts.help import DefaultFormatter, HelpPanel
from jetpytools import SPath
from rich.console import Console, ConsoleOptions
from vskernels import ComplexKernel

from .. import funcs
from ..funcs import resolve_kernel
from .helpers import get_all_idx, resolve_dimension_mode, resolve_idx

# Groups
common_group = Group("Common Options", sort_key=10)
helpers_group = Group("Helper Options", sort_key=20)


# Custom Converters
def parse_kernels(type_: type[Any], tokens: list[Token]) -> list[ComplexKernel]:
    res = list[ComplexKernel]()

    for token in tokens:
        for s in token.value.split(","):
            res.append(resolve_kernel(s.strip(), ValueError))
    return res


def parse_dim_mode(type_: type[Any], tokens: list[Token]) -> Literal["height", "width"]:
    return resolve_dimension_mode(tokens[0].value)


def parse_crop(type_: type[Any], tokens: list[Token]) -> tuple[int, ...]:
    raw_vals = [t.value for t in tokens]

    match len(raw_vals):
        case 4:
            return tuple(int(v) for v in raw_vals)
        case 1:
            if len(raw_vals[0].split()) == 4:
                return tuple(int(v) for v in raw_vals)

    raise ValueError(f"Invalid crop parameters: {raw_vals}. Expected 4 integers (LEFT RIGHT TOP BOTTOM).")


# Reusable Annotated Parameters
InputFileArg = Annotated[
    SPath,
    Parameter(
        name="INPUT",
        help="Path to the source material to analyze. Supports videos, images, or VapourSynth scripts.",
        show_default=False,
        converter=lambda t, tokens: SPath(tokens[0].value),
    ),
]

FrameOpt = Annotated[
    int,
    Parameter(
        short_alias=True,
        help="The specific frame number to extract and analyze from video inputs. Ignored for images.",
        group=common_group,
    ),
]

KernelOpt = Annotated[
    ComplexKernel,
    Parameter(
        short_alias=True,
        help="The kernel to use for inverse scaling. Can be a kernel name or a class call with parameters.",
        converter=lambda type_, tokens: resolve_kernel(tokens[0].value if tokens else "", ValueError),
        group=common_group,
    ),
]

KernelsOpt = Annotated[
    list[ComplexKernel],
    Parameter(
        name="kernel",
        alias="-k",
        help="The kernel(s) to use for inverse scaling. Can be a kernel name or a class call with parameters.",
        converter=parse_kernels,
        show_default=False,
        group=common_group,
    ),
]

DimModeOpt = Annotated[
    Literal["height", "width"],
    Parameter(
        alias="-dm",
        help="Specifies whether to analyze based on the height or width of the frame.",
        converter=parse_dim_mode,
        group=common_group,
    ),
]

CropOpt = Annotated[
    tuple[int, int, int, int] | None,
    Parameter(
        alias="-c",
        help="Crop the input frame before analysis to remove black bars (LEFT RIGHT TOP BOTTOM).",
        converter=parse_crop,
        n_tokens=4,
        group=common_group,
    ),
]

MetricModeOpt = Annotated[
    funcs.MetricMode,
    Parameter(
        alias="-mm",
        help="The mathematical metric used to compare scaling results (MAE, MSE, RMSE).",
        converter=lambda t, tokens: tokens[0].value.upper(),
        group=common_group,
    ),
]

IdxChoice: Any = Literal[*get_all_idx()]

IndexerOpt = Annotated[
    IdxChoice,
    Parameter(
        alias="-idx",
        help="The VapourSynth indexer used to load files.",
        converter=lambda type_, tokens: resolve_idx(tokens[0].value if tokens else "bs"),
        show_choices=True,
        show_default=lambda s: s.__name__,
        group=common_group,
    ),
]

LinearOpt = Annotated[
    bool,
    Parameter(
        short_alias=True,
        help="Whether to process rescale in linear light.",
        negative="",
        group=common_group,
    ),
]

SampleGridModelOpt = Annotated[
    Literal["edges", "centers", 0, 1],
    Parameter(help="Sampling grid alignment model.", group=common_group),
]


class CleanHelpFormatter(DefaultFormatter):
    def __call__(self, console: Console, options: ConsoleOptions, panel: HelpPanel) -> None:
        panel.entries = [
            entry.copy(positive_names=entry.positive_names[1:])  # type: ignore[no-untyped-call]
            if len(entry.positive_names) > 1 and not entry.positive_names[0].startswith("-")
            else entry
            for entry in panel.entries
        ]
        super().__call__(console, options, panel)

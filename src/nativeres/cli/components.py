from dataclasses import dataclass
from typing import Annotated, Any, Literal, Self

from cyclopts import Group, Parameter, Token
from cyclopts.help import DefaultFormatter, HelpPanel
from jetpytools import SPath
from rich.console import Console, ConsoleOptions
from vskernels import ComplexKernel, SampleGridModel
from vssource import BestSource

from ..funcs import MetricMode, resolve_kernel
from .helpers import get_all_idx, resolve_dimension_mode, resolve_idx

# Groups
common_group = Group("Common Options", sort_key=10)
helpers_group = Group("Helper Options", sort_key=20)


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


@Parameter(alias="-c", converter="parse")
class CropOpt(tuple[int, int, int, int]):
    @Parameter(n_tokens=4, accepts_keys=False)
    @classmethod
    def parse(cls, tokens: list[Token]) -> Self:
        raw_vals = [t.value for t in tokens]

        match len(raw_vals):
            case 4:
                return cls(int(v) for v in raw_vals)
            case 1:
                if len(raw_vals[0].split()) == 4:
                    return cls(int(v) for v in raw_vals)

        raise ValueError(f"Invalid crop parameters: {raw_vals}. Expected 4 integers (LEFT RIGHT TOP BOTTOM).")


KernelOpt = Annotated[
    ComplexKernel,
    Parameter(short_alias=True, converter=lambda type_, tokens: resolve_kernel(tokens[0].value if tokens else "")),
]


@Parameter(name="kernel", alias="-k", show_default=False, converter="parse")
class KernelsOpt(list[ComplexKernel]):
    @classmethod
    def parse(cls, tokens: list[Token]) -> Self:
        res = cls()

        for token in tokens:
            for s in token.value.split(","):
                res.append(resolve_kernel(s.strip(), ValueError))
        return res


IdxChoice: Any = Literal[*get_all_idx()]


@Parameter(name="*", group=common_group)
@dataclass(kw_only=True, frozen=True)
class CommonOpts:
    frame: Annotated[int, Parameter(short_alias=True)] = 0
    """The specific frame number to extract and analyze from video inputs. Ignored for images."""

    linear: Annotated[bool, Parameter(short_alias=True, negative="")] = False
    """Whether to process rescale in linear light."""

    indexer: Annotated[
        IdxChoice,  # pyright: ignore[reportInvalidTypeForm]
        Parameter(
            alias="-idx",
            converter=lambda type_, tokens: resolve_idx(tokens[0].value if tokens else "bs"),
            show_choices=True,
            show_default=lambda s: s.__name__,
        ),
    ] = BestSource
    """The VapourSynth indexer used to load files."""


@Parameter(name="*", group=common_group)
@dataclass(kw_only=True, frozen=True)
class RescaleOpts(CommonOpts):
    dim_mode: Annotated[
        Literal["height", "width"],
        Parameter(alias="-dm", converter=lambda type_, tokens: resolve_dimension_mode(tokens[0].value)),
    ] = "height"
    """Specifies whether to analyze based on the height or width of the frame."""

    sample_grid_model: Literal["edges", "centers", 0, 1] = "edges"
    """Sampling grid alignment model."""

    crop: CropOpt | None = None
    """Crop the input frame before analysis to remove black bars (LEFT RIGHT TOP BOTTOM)."""

    metric_mode: Annotated[
        MetricMode,
        Parameter(alias="-mm", converter=lambda t, tokens: tokens[0].value.upper()),
    ] = "MAE"
    """The mathematical metric used to compare scaling results (MAE, MSE, RMSE)."""

    @property
    def resolved_sample_grid_model(self) -> SampleGridModel:
        return (
            SampleGridModel[f"MATCH_{self.sample_grid_model.upper()}"]
            if isinstance(self.sample_grid_model, str)
            else SampleGridModel(self.sample_grid_model)
        )


class CleanHelpFormatter(DefaultFormatter):
    def __call__(self, console: Console, options: ConsoleOptions, panel: HelpPanel) -> None:
        panel.entries = [
            entry.copy(positive_names=entry.positive_names[1:])  # type: ignore[no-untyped-call]
            if len(entry.positive_names) > 1 and not entry.positive_names[0].startswith("-")
            else entry
            for entry in panel.entries
        ]
        super().__call__(console, options, panel)

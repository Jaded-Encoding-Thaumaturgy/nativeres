from dataclasses import dataclass
from typing import Annotated, Literal, Self

from cyclopts import Group, Parameter, Token
from jetpytools import SPath
from vskernels import ComplexKernel, SampleGridModel
from vssource import BestSource, Indexer

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


KernelOpt = Annotated[
    ComplexKernel,
    Parameter(
        short_alias=True,
        converter=lambda type_, tokens: resolve_kernel(tokens[0].value if tokens else ""),
        metavar="",
    ),
]


class KernelsOpt(list[ComplexKernel]):
    @classmethod
    def parse(cls, tokens: list[Token]) -> Self:
        res = cls()

        for token in tokens:
            for s in token.value.split(","):
                res.append(resolve_kernel(s.strip(), ValueError))
        return res


class CropOpt(tuple[int, int, int, int]):
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


@Parameter(name="*", group=common_group)
@dataclass(kw_only=True, frozen=True)
class CommonOpts:
    frame: Annotated[int, Parameter(short_alias=True)] = 0
    """The specific frame number to extract and analyze from video inputs. Ignored for images."""

    linear: Annotated[bool, Parameter(short_alias=True, negative="")] = False
    """Whether to process rescale in linear light."""

    indexer: Annotated[
        Indexer,
        Parameter(
            alias="-idx",
            accepts_keys=False,
            converter=lambda type_, tokens: resolve_idx(tokens[0].value if tokens else "bs"),
            choices=get_all_idx(),
            show_default=lambda s: s.__class__.__name__,
        ),
    ] = BestSource()  # noqa: RUF009
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

    crop: Annotated[
        CropOpt | None,
        Parameter(
            name="crop",
            alias="-c",
            converter=CropOpt.parse,
            n_tokens=4,
            accepts_keys=False,
            metavar="LEFT RIGHT TOP BOTTOM",
        ),
    ] = None
    """Crop the input frame before analysis to remove black bars."""

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

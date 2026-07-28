"""Python API."""

import ast
import gc
import re
from collections.abc import Callable, Iterable, Sequence
from logging import getLogger
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from jetpytools import FuncExcept, cachedproperty, mod2, to_arr
from vsexprtools import ExprOp, norm_expr
from vskernels import ComplexKernel, ComplexKernelLike, Kernel, LeftShift, Point, TopShift
from vsmasktools import MaskLike, normalize_mask
from vsscale import Rescale
from vsscale.helpers import BottomCrop, CropRel, LeftCrop, RightCrop, TopCrop
from vstools import clip_data_gather, core, depth, get_prop, get_y, padder, vs

if TYPE_CHECKING:
    import numpy as np

type NpFloatArray1D = np.ndarray[tuple[Literal[1]], np.dtype[np.floating[Any]]]
type MetricMode = Literal["MAE", "MSE", "RMSE"]

logger = getLogger(__name__)

_point_resize = Point()


class ResolutionFrac(NamedTuple):
    """Fractional resolution expressed as floating-point width and height."""

    width: float
    """Horizontal size (may be fractional)."""

    height: float
    """Vertical size (may be fractional)."""

    base_width: int | None = None
    """Integer width to contain the clip within."""

    base_height: int | None = None
    """Integer height to contain the clip within."""


class GetNativeResult(NamedTuple):
    """Result for `getnative`, pairing a fractional resolution with an error metric."""

    dim: ResolutionFrac
    """The tested fractional resolution as an `ResolutionFrac`."""

    error: float
    """Computed error for this resolution (higher means worse)."""


def getnative(
    clip: vs.VideoNode,
    frame_num: int,
    dimensions: Iterable[tuple[float | tuple[float, int], float | tuple[float, int]]],
    kernel: ComplexKernelLike,
    crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] | None = None,
    shift: tuple[TopShift, LeftShift] = (0, 0),
    metric_mode: MetricMode = "MAE",
    borders_aware: bool | int | tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] = 8,
    progress_cb: Callable[[int, int], None] | None = None,
    *,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> list[GetNativeResult]:
    """
    Determine the best (fractional) native resolution for a selected frame.

    This checks a list of candidate resolutions by descaling the selected frame using `kernel`
    and comparing the result to the source frame using `metric_mode`.

    Args:
        clip: Source clip.
        frame_num: Frame index in `clip` to evaluate.
        dimensions: Iterable of candidate resolutions to test.
            Each item may be a `(width, height)` tuple or `((width, base_width), (height, base_height))` tuple.
        kernel: Kernel used to perform each descale attempt.
        crop: Optional crop to apply before descaling. Aspect ratio is preserved.
        shift: Optional pixel shift applied during descaling: `(top_shift, left_shift)`.
        metric_mode: Error metric to use: `"MSE"`, `"MAE"`, or `"RMSE"`.
        borders_aware: Amount (or crop tuple) to ignore at image borders to avoid edge noise when computing the metric.
        progress_cb: Optional progress callback called as `progress_cb(current, total)`.
        func: Function returned for custom error handling.
        kwargs: Additional arguments passed to the Rescale class.

    Returns:
        A list of `GetNativeResult` items.
        Each item contains the tested fractional resolution (`dim`) and its associated `error`.
    """
    func = func or getnative

    clip_frame = depth(get_y(clip), 32)[frame_num]

    kernel = ComplexKernel.ensure_obj(kernel, func)
    crops = _norm_border_crops(borders_aware, kernel)

    dimensions = list(dimensions)

    rescale_list = list[Rescale]()
    ress = list[ResolutionFrac]()

    for w, h in dimensions:
        if isinstance(w, tuple):
            w, bw = w
        else:
            bw = None
        if isinstance(h, tuple):
            h, bh = h
        else:
            bh = None

        ress.append(ResolutionFrac(w, h, bw, bh))

        r = Rescale(
            clip_frame,
            h,
            kernel,
            _point_resize,
            width=w,
            base_height=bh,
            base_width=bw,
            crop=crop,
            shift=shift,
            **kwargs,
        )
        rescale_list.append(r)

    def eval_func(n: int) -> vs.VideoNode:
        res = rescale_list[n]
        try:
            return res.rescale
        finally:
            cachedproperty.clear_cache(res)

    rescaled = clip_frame.std.BlankClip(length=len(dimensions)).std.FrameEval(eval_func)
    rescaled = (
        norm_expr([rescaled, clip_frame], getattr(ExprOp, metric_mode.lower())(clip_frame), func=func)
        .std.CropRel(*crops)
        .std.PlaneStats()
    )

    errors = clip_data_gather(rescaled, progress_cb, lambda _, f: get_prop(f, "PlaneStatsAverage", float))

    try:
        return [GetNativeResult(d, e) for d, e in zip(ress, errors)]
    finally:
        for r in rescale_list:
            r.__vs_del__(-1)
        del clip_frame, rescaled
        gc.collect()


getfnative = getnative


class GetScalerResult(NamedTuple):
    """Result for `getscaler`, pairing a kernel with its error."""

    kernel: ComplexKernel
    """The `ComplexKernel` instance that was evaluated."""

    error: float
    """Computed error for that kernel (lower is better)."""


def getscaler(
    clip: vs.VideoNode,
    frame_num: int,
    width: float,
    height: float,
    kernels: ComplexKernelLike | Sequence[ComplexKernelLike],
    base_width: int | None = None,
    base_height: int | None = None,
    crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] | None = None,
    shift: tuple[TopShift, LeftShift] = (0, 0),
    metric_mode: MetricMode = "MAE",
    mask: MaskLike | None = None,
    borders_aware: bool | int | tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] = 8,
    *,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> list[GetScalerResult]:
    """
    Find the best inverse scaler (kernel) for a given frame of a clip.

    Each supplied kernel is tested by descaling the selected frame to the provided
    `(width, height)` and computing an error metric.

    If `width`/`height` are floats, fractional descaling is performed.

    Args:
        clip: Source clip.
        frame_num: Frame index in `clip` to evaluate.
        width: Width to be descaled to. If passed as a float, a fractional descale is performed.
        height: Height to be descaled to. If passed as a float, a fractional descale is performed.
        kernels: A single kernel or a sequence of kernels to evaluate.
        base_width: Optional integer height to contain the clip within.
        base_height: Optional integer width to contain the clip within.
        crop: Optional crop to apply before descaling. Aspect ratio is preserved.
        shift: Pixel shifts applied during descaling: `(top_shift, left_shift)`.
        metric_mode: Error metric to use: `"MSE"`, `"MAE"`, or `"RMSE"`.
        mask: Optional edge-detection mask to reduce noise influence on the metric.
        borders_aware: Amount (or crop tuple) to ignore at image borders to avoid edge noise when computing the metric.
        func: Function returned for custom error handling.
        kwargs: Additional arguments passed to the Rescale class.

    Returns:
        A list of `GetScalerResult` items. Each item contains the evaluated `kernel` and its associated `error`.
    """
    func = func or getscaler

    resolved_kernels = {ComplexKernel.ensure_obj(k, func) for k in to_arr(kernels)}  # type:ignore[arg-type]

    return [
        GetScalerResult(
            kernel,
            get_descale_error(
                clip,
                frame_num,
                width,
                height,
                kernel,
                base_width,
                base_height,
                crop,
                shift,
                metric_mode,
                mask,
                borders_aware,
                **kwargs,
            ),
        )
        for kernel in resolved_kernels
    ]


getfscaler = getscaler


def get_descale_error(
    clip: vs.VideoNode,
    frame_num: int,
    width: float,
    height: float,
    kernel: ComplexKernelLike,
    base_width: int | None = None,
    base_height: int | None = None,
    crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] | None = None,
    shift: tuple[TopShift, LeftShift] = (0, 0),
    metric_mode: MetricMode = "MAE",
    mask: MaskLike | None = None,
    borders_aware: bool | int | tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] = 8,
    *,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> float:
    """
    Compute the descale error for a selected frame using a specific kernel.

    Args:
        clip: Source clip.
        frame_num: Frame index in `clip` to evaluate.
        width: Width to be descaled to. If passed as a float, a fractional descale is performed.
        height: Height to be descaled to. If passed as a float, a fractional descale is performed.
        kernel: Kernel used for the descale operation.
        base_width: Optional integer height to contain the clip within.
        base_height: Optional integer width to contain the clip within.
        crop: Optional crop to apply before descaling. Aspect ratio is preserved.
        shift: Pixel shifts applied during descaling: `(top_shift, left_shift)`.
        metric_mode: Error metric to use: `"MSE"`, `"MAE"`, or `"RMSE"`.
        mask: Optional edge-detection mask to reduce noise influence on the metric.
        borders_aware: Amount (or crop tuple) to ignore at image borders to avoid edge noise when computing the metric.
        func: Function returned for custom error handling.
        kwargs: Additional arguments passed to the Rescale class.

    Returns:
        The computed error as a `float`. Lower values indicate a better descale match.
    """

    func = func or get_descale_error

    clip_frame = depth(get_y(clip), 32)[frame_num]

    kernel = ComplexKernel.ensure_obj(kernel, func)

    rs = Rescale(
        clip_frame,
        height,
        kernel,
        width=width,
        upscaler=_point_resize,
        base_height=base_height,
        base_width=base_width,
        crop=crop or CropRel(),
        shift=shift,
        **kwargs,
    )

    rescaled = rs.rescale

    if mask:
        mask = normalize_mask(mask, clip_frame, clip_frame, func=func)

        rescaled = core.std.MaskedMerge(clip_frame, rescaled, mask)

    crops = _norm_border_crops(borders_aware, kernel)

    rescaled = (
        norm_expr([rescaled, clip_frame], getattr(ExprOp, metric_mode.lower())(clip_frame), func=func)
        .std.CropRel(*crops)
        .std.PlaneStats()
    )

    try:
        return get_prop(rescaled, "PlaneStatsAverage", float, func=func)
    finally:
        rs.__vs_del__(-1)


def _norm_border_crops(
    value: bool | int | tuple[LeftCrop, RightCrop, TopCrop, BottomCrop], kernel: Kernel
) -> tuple[LeftCrop, RightCrop, TopCrop, BottomCrop]:
    match value:
        case True:
            return (kernel.kernel_radius, kernel.kernel_radius, kernel.kernel_radius, kernel.kernel_radius)
        case False:
            return (0, 0, 0, 0)
        case int():
            return (value, value, value, value)
        case _:
            return value


def get_dct_distribution(
    clip: vs.VideoNode,
    frame_num: int,
    cull_rate: float = 3.0,
    func: FuncExcept | None = None,
) -> tuple[NpFloatArray1D, NpFloatArray1D]:
    """
    Calculate DCT frequency distribution for both horizontal and vertical dimensions of a selected frame.

    Args:
        clip: Source clip.
        frame_num: Frame index in `clip` to analyze.
        cull_rate: Cull rate for DCT coefficients.
        func: Function returned for custom error handling.

    Returns:
        A tuple of (dct_h, dct_v).
    """
    import numpy as np
    import scipy.fft

    func = func or get_dct_distribution

    clip_frame = get_y(clip)[frame_num]

    def get_dct(c: vs.VideoNode) -> NpFloatArray1D:
        top_cut = 20 if c.height > 720 else 10
        side_cut = 20

        if cull_rate:
            side_cut = mod2(c.width / (2 + cull_rate))

        padded = padder.MIRROR(
            c.std.Crop(side_cut, side_cut, top_cut, top_cut),
            side_cut // 2,
            side_cut // 2,
            top_cut,
            top_cut,
        ).std.Transpose()

        rows = np.asarray(padded.get_frame(0)[0], dtype=np.float64)
        rows_dct = scipy.fft.dct(rows, axis=-1)
        return np.mean(np.abs(rows_dct), axis=0)

    return get_dct(clip_frame.std.Transpose()), get_dct(clip_frame)


def resolve_kernel(value: str, exc_type: type[Exception] = ValueError) -> ComplexKernel:
    matched = re.match(r"^([A-Za-z_]\w*)(?:\((.*)\))?$", value.strip(), re.DOTALL)

    if not matched:
        raise exc_type("expected KernelName(...) or KernelName")

    name, args_text = matched.group(1), (matched.group(2) or "").strip()

    kernel = ComplexKernel.from_param(name, exc_type)

    if not args_text:
        return kernel()

    expr = f"{name}({args_text})"

    node = ast.parse(expr, mode="eval")
    call = node.body

    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == name):
        raise ValueError

    return kernel(
        *(ast.literal_eval(a) for a in call.args),
        **{kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg},
    )

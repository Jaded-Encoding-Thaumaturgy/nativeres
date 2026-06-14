# nativeres

`nativeres` is a set of descale analysis tools for VapourSynth.

It combines and supersedes the following tools:

- Original **getscaler** by **cN3rd** https://gist.github.com/cN3rd/51077b6abf45b684bf9a3c657d859b43
- Original **getnative** by **Infiziert90** https://github.com/Infiziert90/getnative
- **GetFnative** by **YomikoR** https://github.com/YomikoR/GetFnative
- **getfscaler** https://github.com/Jaded-Encoding-Thaumaturgy/getscaler

The package provides:

- A CLI for frame analysis
- A Python API for VapourSynth scripts and tooling
- An optional **[VSView](https://github.com/Jaded-Encoding-Thaumaturgy/vs-view)** plugin

## What it does

`nativeres` helps answer three common descale questions:

- `getnative`: What native resolution was this frame likely upscaled from?
- `getscaler`: Which scaler most likely produced this upscale?
- `getfreq`: Does the frame's frequency distribution show likely scaling artifacts or native-resolution clues?

## Installation

```bash
# Install the package with `pip`
pip install nativeres
# Or with `uv`
uv tool install nativeres
```

### VSView plugin

This repository also contains a **[VSView](https://github.com/Jaded-Encoding-Thaumaturgy/vs-view)** plugin package.

```bash
pip install nativeres[plugin]
# Or with uv
uv add nativeres[plugin]
```

The plugin registers as a VSView tool panel and tool dock under `Native Resolution`.

Usage can be found in the **[plugin README](https://github.com/Jaded-Encoding-Thaumaturgy/nativeres/blob/master/src/plugin/README.md)**.

## Usage

### CLI

`nativeres` accepts regular media files, images, and VapourSynth scripts (`.vpy` / `.py`). For scripts, the first VapourSynth output is used.

- Displays a plot to determine the native resolution height for the frame 15000 with the Bilinear kernel:

  ```bash
  nativeres getnative file.m2ts --frame 15000 --dim-mode height --kernel bilinear
  ```

  ![`nativeres getnative`](https://raw.githubusercontent.com/Jaded-Encoding-Thaumaturgy/nativeres/master/assets/getnative_plot.svg)

  Right-click the plot to open the context menu, where you can reset zoom and copy or export the plot as `PNG`, `SVG`, `JSON`, or `CSV`.

- Displays a table showing the errors of the inverse scalers at height 800 for frame 15000.

  ```bash
  nativeres getscaler file.m2ts 800 --frame 15000 --dim-mode height
  ```

  ![`nativeres getscaler`](https://raw.githubusercontent.com/Jaded-Encoding-Thaumaturgy/nativeres/master/assets/getscaler_table.svg)

- Compute and display a DCT-based frequency plot for the selected frame.

  ```bash
  nativeres getfreq file.m2ts --frame 15000
  ```

  ![`nativeres getfreq`](https://raw.githubusercontent.com/Jaded-Encoding-Thaumaturgy/nativeres/master/assets/getfreq_plot.svg)

  The frequency plot uses the same right-click context menu for reset zoom plus copy/export actions.

### Python

```python
from nativeres import getnative
from vskernels import Bilinear

clip = ...  # Your VapourSynth clip

results = getnative(
    clip,
    frame_num=15000,
    dimensions=((clip.width, h) for h in range(540, 901)),
    kernel=Bilinear(),
)

best = min(results, key=lambda result: result.error)
print(best.dim.height, best.error)
```

## CLI Reference

Main help:

![`nativeres --help`](https://raw.githubusercontent.com/Jaded-Encoding-Thaumaturgy/nativeres/master/assets/nativeres_help.svg)

`getnative` help:

![`nativeres getnative --help`](https://raw.githubusercontent.com/Jaded-Encoding-Thaumaturgy/nativeres/master/assets/getnative_help.svg)

`getscaler` help:

![`nativeres getscaler --help`](https://raw.githubusercontent.com/Jaded-Encoding-Thaumaturgy/nativeres/master/assets/getscaler_help.svg)

`getfreq` help:

![`nativeres getfreq --help`](https://raw.githubusercontent.com/Jaded-Encoding-Thaumaturgy/nativeres/master/assets/getfreq_help.svg)

## Python API

<!-- markdownlint-disable -->

<a href="https://github.com/Jaded-Encoding-Thaumaturgy/nativeres/blob/master/src/nativeres/funcs.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `getnative`

```python
getnative(
    clip: VideoNode,
    frame_num: int,
    dimensions: Iterable[tuple[float, float]],
    kernel: ComplexKernelLike,
    crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] | None = None,
    shift: tuple[TopShift, LeftShift] = (0, 0),
    metric_mode: MetricMode = 'MAE',
    borders_aware: bool | int | tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] = 8,
    progress_cb: Callable[[int, int], None] | None = None,
    func: FuncExcept | None = None,
    **kwargs: Any
) -> list[GetNativeResult]
```

Determine the best (fractional) native resolution for a selected frame.

This checks a list of candidate resolutions by descaling the selected frame using `kernel` and comparing the result to the source frame using `metric_mode`.

**Args:**

- <b>`clip`</b>: Source clip.
- <b>`frame_num`</b>: Frame index in `clip` to evaluate.
- <b>`dimensions`</b>: Iterable of candidate resolutions to test. Each item may be a `(width, height)` tuple.
- <b>`kernel`</b>: Kernel used to perform each descale attempt.
- <b>`crop`</b>: Optional crop to apply before descaling. Aspect ratio is preserved.
- <b>`shift`</b>: Optional pixel shift applied during descaling: `(top_shift, left_shift)`.
- <b>`metric_mode`</b>: Error metric to use: `"MSE"`, `"MAE"`, or `"RMSE"`.
- <b>`borders_aware`</b>: Amount (or crop tuple) to ignore at image borders to avoid edge noise when computing the metric.
- <b>`progress_cb`</b>: Optional progress callback called as `progress_cb(current, total)`.
- <b>`func`</b>: Function returned for custom error handling.
- <b>`kwargs`</b>: Additional arguments passed to the Rescale class.

**Returns:**
A list of `GetNativeResult` items. Each item contains the tested fractional resolution (`dim`) and its associated `error`.

---

<a href="https://github.com/Jaded-Encoding-Thaumaturgy/nativeres/blob/master/src/nativeres/funcs.py#L160"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `getscaler`

```python
getscaler(
    clip: VideoNode,
    frame_num: int,
    width: float,
    height: float,
    kernels: ComplexKernelLike | Sequence[ComplexKernelLike],
    base_width: int | None = None,
    base_height: int | None = None,
    crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] | None = None,
    shift: tuple[TopShift, LeftShift] = (0, 0),
    metric_mode: MetricMode = 'MAE',
    mask: MaskLike | None = None,
    borders_aware: bool | int | tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] = 8,
    func: FuncExcept | None = None,
    **kwargs: Any
) → list[GetScalerResult]
```

Find the best inverse scaler (kernel) for a given frame of a clip.

Each supplied kernel is tested by descaling the selected frame to the provided `(width, height)` and computing an error metric.

**Args:**

- <b>`clip`</b>: Source clip.
- <b>`frame_num`</b>: Frame index in `clip` to evaluate.
- <b>`width`</b>: Width to be descaled to. If passed as a float, a fractional descale is performed.
- <b>`height`</b>: Height to be descaled to. If passed as a float, a fractional descale is performed.
- <b>`kernels`</b>: A single kernel or a sequence of kernels to evaluate.
- <b>`base_width`</b>: Optional integer height to contain the clip within.
- <b>`base_height`</b>: Optional integer width to contain the clip within.
- <b>`crop`</b>: Optional crop to apply before descaling. Aspect ratio is preserved.
- <b>`shift`</b>: Pixel shifts applied during descaling: `(top_shift, left_shift)`.
- <b>`metric_mode`</b>: Error metric to use: `"MSE"`, `"MAE"`, or `"RMSE"`.
- <b>`mask`</b>: Optional edge-detection mask to reduce noise influence on the metric.
- <b>`borders_aware`</b>: Amount (or crop tuple) to ignore at image borders to avoid edge noise when computing the metric.
- <b>`func`</b>: Function returned for custom error handling.
- <b>`kwargs`</b>: Additional arguments passed to the Rescale class.

**Returns:**
A list of `GetScalerResult` items. Each item contains the evaluated `kernel` and its associated `error`.

---

<a href="https://github.com/Jaded-Encoding-Thaumaturgy/nativeres/blob/master/src/nativeres/funcs.py#L234"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_descale_error`

```python
get_descale_error(
    clip: VideoNode,
    frame_num: int,
    width: float,
    height: float,
    kernel: ComplexKernelLike,
    base_width: int | None = None,
    base_height: int | None = None,
    crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] | None = None,
    shift: tuple[TopShift, LeftShift] = (0, 0),
    metric_mode: MetricMode = 'MAE',
    mask: MaskLike | None = None,
    borders_aware: bool | int | tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] = 8,
    func: FuncExcept | None = None,
    **kwargs: Any
) → float
```

Compute the descale error for a selected frame using a specific kernel.

**Args:**

- <b>`clip`</b>: Source clip.
- <b>`frame_num`</b>: Frame index in `clip` to evaluate.
- <b>`width`</b>: Width to be descaled to. If passed as a float, a fractional descale is performed.
- <b>`height`</b>: Height to be descaled to. If passed as a float, a fractional descale is performed.
- <b>`kernel`</b>: Kernel used for the descale operation.
- <b>`base_width`</b>: Optional integer height to contain the clip within.
- <b>`base_height`</b>: Optional integer width to contain the clip within.
- <b>`crop`</b>: Optional crop to apply before descaling. Aspect ratio is preserved.
- <b>`shift`</b>: Pixel shifts applied during descaling: `(top_shift, left_shift)`.
- <b>`metric_mode`</b>: Error metric to use: `"MSE"`, `"MAE"`, or `"RMSE"`.
- <b>`mask`</b>: Optional edge-detection mask to reduce noise influence on the metric.
- <b>`borders_aware`</b>: Amount (or crop tuple) to ignore at image borders to avoid edge noise when computing the metric.
- <b>`func`</b>: Function returned for custom error handling.
- <b>`kwargs`</b>: Additional arguments passed to the Rescale class.

**Returns:**
The computed error as a `float`. Lower values indicate a better descale match.

---

<a href="https://github.com/Jaded-Encoding-Thaumaturgy/nativeres/blob/master/src/nativeres/funcs.py#L328"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_dct_distribution`

```python
get_dct_distribution(
    clip: VideoNode,
    frame_num: int,
    cull_rate: float = 3.0,
    func: FuncExcept | None = None
) → tuple[NpFloatArray1D, NpFloatArray1D]
```

Calculate DCT frequency distribution for both horizontal and vertical dimensions of a selected frame.

**Args:**

- <b>`clip`</b>: Source clip.
- <b>`frame_num`</b>: Frame index in `clip` to analyze.
- <b>`cull_rate`</b>: Cull rate for DCT coefficients.
- <b>`func`</b>: Function returned for custom error handling.

**Returns:**
A tuple of (dct_h, dct_v).

---

## <kbd>class</kbd> `GetNativeResult`

Result for `getnative`, pairing a fractional resolution with an error metric.

---

## <kbd>class</kbd> `GetScalerResult`

Result for `getscaler`, pairing a kernel with its error.

---

## <kbd>class</kbd> `ResolutionFrac`

Fractional resolution expressed as floating-point width and height.

## Notes

- You should always visually verify the results on multiple frames before committing to a descale setup.

from inspect import isabstract
from typing import Any, Literal, NoReturn

from jetpytools import CustomValueError, SPath, get_subclasses
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from vsengine import Policy
from vskernels import ComplexKernel
from vsmasktools import EdgeDetect
from vssource import BestSource, CacheIndexer, Indexer
from vstools import Matrix, vs

from .logging import console


# Callbacks
def resolve_dimension(value: str) -> float:
    nb = float(value)

    if nb.is_integer():
        return int(nb)

    return nb


def get_all_idx() -> list[str]:
    all_indexers = list[str]()

    with Policy() as policy, policy.new_environment() as env, env.use():
        for s in get_subclasses(Indexer):
            if isabstract(s) or not hasattr(s, "_source_func"):
                continue
            all_indexers.append(s.__name__.lower())

            source_func = getattr(s, "_source_func", None)
            plugin = getattr(source_func, "plugin", None)
            plugin_ns = getattr(plugin, "namespace", None)

            if plugin_ns:
                all_indexers.append(plugin_ns)

    return all_indexers


def resolve_idx(idx: str) -> Indexer:
    indexer = Indexer.from_param(idx, ValueError)

    args = dict[str, Any]()

    if issubclass(indexer, CacheIndexer):
        # Set cache path to None
        args[indexer._cache_arg_name] = None

        if issubclass(indexer, BestSource):
            args["show_pretty_progress"] = True

    return indexer(**args)


def resolve_dimension_mode(mode: str) -> Literal["height", "width"]:
    match mode:
        case "height" | "h":
            return "height"
        case "width" | "w":
            return "width"
        case _:
            raise ValueError("Unknown dimension passed")


def show_default_kernels() -> NoReturn:
    from ..kernels import default_kernels

    for kernel in default_kernels:
        console.print(str(kernel))

    raise SystemExit(0)


def show_vskernels() -> NoReturn:
    all_kernels = {k for k in get_subclasses(ComplexKernel) if not k.is_abstract}

    for kernel in sorted(all_kernels, key=lambda k: k.__name__):
        console.print(kernel.__name__)

    raise SystemExit(0)


def show_masks() -> NoReturn:
    all_masks = {
        s
        for s in get_subclasses(EdgeDetect)  # type: ignore[type-abstract]
        if not isabstract(s) and s.__module__.split(".")[-1] != "_abstract"
    }

    for kernel in sorted(all_masks, key=lambda k: k.__name__):
        console.print(kernel.__name__)

    raise SystemExit(0)


# Helpers
def get_videonode_from_input(path: SPath, indexer: Indexer) -> vs.VideoNode:
    if not path.exists():
        raise ValueError(f"{path.to_str()!r} doesn't exist.")

    if path.suffix in (".py", ".vpy"):
        from vsengine import load_script

        load_script(path, module="__nativeres__").result()
        out = next(iter(vs.get_outputs().values()))

        if not isinstance(out, vs.VideoOutputTuple):
            raise CustomValueError("Unknown VapourSynth output", get_videonode_from_input, type(out))

        return out.clip

    if isinstance(indexer, BestSource):
        from signal import SIG_DFL, SIGINT, signal

        signal(SIGINT, SIG_DFL)

    clip = indexer.source(path, 32, idx_props=False)
    return clip.resize.Bilinear(format=vs.GRAYS, matrix=Matrix.BT709, matrix_in=Matrix.from_video(clip))


def get_progress(console: Console, **kwargs: Any) -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        **kwargs,
    )

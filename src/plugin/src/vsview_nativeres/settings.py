from __future__ import annotations

from collections.abc import Iterator
from contextlib import suppress
from logging import getLogger
from typing import TYPE_CHECKING, Annotated, Literal

from jetpytools import get_subclasses
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer, ValidationError
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget
from vskernels import ComplexKernel
from vsview.api import ColorPicker, Dropdown, ListEdit, ListEditWidget

from nativeres.funcs import resolve_kernel
from nativeres.kernels import default_kernels

if TYPE_CHECKING:
    from PySide6.QtCharts import QChart

logger = getLogger(__name__)


def get_kernel_names() -> Iterator[str]:
    for kernel in get_subclasses(ComplexKernel):
        if not kernel.is_abstract:
            yield kernel.__name__

            with suppress(Exception):
                yield kernel().pretty_string


KERNEL_NAMES = sorted(get_kernel_names())


class KernelListEditWidget(ListEditWidget[str]):
    def validate_text(self, text: str) -> None:
        try:
            self.adapter.validate_python(text)
            resolve_kernel(text, ValidationError)
            self.list_widget.addItem(text)
        except Exception as e:
            logger.error("Invalid value: %s", e)


class KernelsListEdit(ListEdit[str]):
    def create_widget(self, parent: QWidget | None = None) -> KernelListEditWidget:
        w = KernelListEditWidget(
            self.value_type,
            parent,
            self.default_value,
            "Enter kernel name, class or instance:",
            KERNEL_NAMES,
        )
        w.list_widget.setMaximumHeight(200)
        return w


def _freq_dim_color_to_ui(c: QColor) -> str:
    return c.name() if isinstance(c, QColor) else c


def _freq_dim_color_from_ui(s: str) -> str:
    return QColor(s).name()


PydanticQColorMetadata = (BeforeValidator(lambda v: QColor(v)), PlainSerializer(lambda c: c.name()))

PydanticKernel = Annotated[
    ComplexKernel,
    BeforeValidator(lambda v: resolve_kernel(v)),
    PlainSerializer(lambda k: k.pretty_string, return_type=str),
]


class GlobalSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chart_theme: Annotated[
        str,
        Dropdown(
            label="Chart Theme",
            tooltip="The theme of the chart.",
            items=[
                ("Default", "Default"),
                ("Light", "Light"),
                ("BlueCerulean", "BlueCerulean"),
                ("Dark", "Dark"),
                ("BrownSand", "BrownSand"),
                ("BlueNcs", "BlueNcs"),
                ("HighContrast", "HighContrast"),
                ("BlueIcy", "BlueIcy"),
                ("Qt", "Qt"),
            ],
        ),
    ] = "Default"
    frequency_width_color: Annotated[
        QColor,
        *PydanticQColorMetadata,
        ColorPicker(
            label="Frequency Width Color",
            tooltip=(
                "The color of the width distribution in the frequency dimension plot.\n"
                "Only applies for the default theme"
            ),
            to_ui=_freq_dim_color_to_ui,
            from_ui=_freq_dim_color_from_ui,
        ),
    ] = QColor(Qt.GlobalColor.green)
    frequency_height_color: Annotated[
        QColor,
        *PydanticQColorMetadata,
        ColorPicker(
            label="Frequency Height Color",
            tooltip=(
                "The color of the height distribution in the frequency dimension plot\n"
                "Only applies for the default theme"
            ),
            to_ui=_freq_dim_color_to_ui,
            from_ui=_freq_dim_color_from_ui,
        ),
    ] = QColor(Qt.GlobalColor.cyan)
    kernels: Annotated[
        list[PydanticKernel],
        KernelsListEdit(
            label="Kernels",
            value_type=str,
            tooltip="The list of kernels to check in getnative and getscaler mode",
            default_value=[k.pretty_string for k in default_kernels],
        ),
    ] = list(default_kernels)

    def get_chart_theme(self) -> Literal[False] | QChart.ChartTheme:
        from PySide6.QtCharts import QChart

        return False if self.chart_theme == "Default" else getattr(QChart.ChartTheme, f"ChartTheme{self.chart_theme}")


class GetNativeLocalSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    last_dimension: int | None = None
    last_base_parity: int | None = None
    last_min_range: int | None = None
    last_max_range: int | None = None
    last_step: float | None = None
    last_kernel: PydanticKernel | None = None
    last_metric: str | None = None


class GetScalerLocalSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    last_dimension: int | None = None
    last_target_dimension: float | None = None
    last_metric: str | None = None
    last_mask: str | None = None


class LocalSettings(BaseModel):
    getnative: GetNativeLocalSettings = Field(default_factory=GetNativeLocalSettings)
    getscaler: GetScalerLocalSettings = Field(default_factory=GetScalerLocalSettings)

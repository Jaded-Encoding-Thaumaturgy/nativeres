from __future__ import annotations

import csv
import json
from itertools import zip_longest
from logging import getLogger
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import vapoursynth as vs
from jetpytools import fallback, mod2
from PySide6.QtCore import QSignalBlocker, QTimer, Signal
from PySide6.QtGui import QPalette, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from vstools import get_h, get_w
from vsview.api import (
    Accordion,
    IconName,
    IconReloadMixin,
    PluginAPI,
    PluginSettings,
    SegmentedControl,
    VideoOutputProxy,
    WidgetPluginBase,
    run_in_background,
    run_in_loop,
)

from nativeres.constants import HIGH_RATE, LOW_RATE
from nativeres.funcs import (
    GetNativeResult,
    GetScalerResult,
    MetricMode,
    NpFloatArray1D,
    get_dct_distribution,
    getnative,
    getscaler,
    resolve_kernel,
)

from .components import GetNativeImportList, ProgressBar, RangeSpinStack, TabContainer
from .settings import GlobalSettings, LocalSettings
from .utils import get_edge_detect_classes, warmup_plots

if TYPE_CHECKING:
    # Lazy import to avoid long startup times because of QChart
    # Import is done in warmup_plots
    from .plotting import CustomFrequencyPlotWidget, CustomRescalePlotWidget

logger = getLogger(__name__)


class GetNativeTab(TabContainer, IconReloadMixin):
    IMPORT_PLOT_ROLE = Qt.ItemDataRole.UserRole
    IMPORT_FRAME_ROLE = Qt.ItemDataRole.UserRole + 1

    computedPlotAdded = Signal(QGraphicsView)

    def __init__(
        self,
        parent: QWidget,
        api: PluginAPI,
        settings: PluginSettings[GlobalSettings, LocalSettings],
    ) -> None:
        super().__init__(parent)

        self.api = api
        self.settings = settings

        self.controls_section = Accordion("Controls", self)
        controls = self.controls_section.add_hlayout()

        self.dimension = SegmentedControl(["Width", "Height"], self.controls_section)
        self.dimension.current_layout.setContentsMargins(0, 0, 0, 0)
        self.dimension.current_layout.setSpacing(0)
        self.dimension.setToolTip("Choose whether to scan candidate source widths or source heights.")
        self.dimension.segmentChanged.connect(self.on_dimension_segment_changed)
        self._last_dimension: int | None = None
        dimension_layout = self.make_vgroup("Dimension", self.dimension, parent=self.controls_section)

        self.base_parity = SegmentedControl(["Odd", "Even"], self.controls_section)
        self.base_parity.current_layout.setContentsMargins(0, 0, 0, 0)
        self.base_parity.current_layout.setSpacing(0)
        self.base_parity.setToolTip(
            "Switches base dimension parity for fractional descales,\n"
            "introducing a 0.5-pixel subpixel shift to the source offset."
        )
        self.base_parity.segmentChanged.connect(self.on_base_parity_segment_changed)
        self._last_base_parity: int | None = None
        base_parity_layout = self.make_vgroup("Base Parity", self.base_parity, parent=self.controls_section)

        self.reset_values_btn = QPushButton("Reset Controls", self.controls_section)
        self.reset_values_btn.setToolTip("Restore the default range, step, kernel, and metric for the current clip.")
        self.reset_values_btn.clicked.connect(self._set_default_values)

        dimension_base_parity = QVBoxLayout()
        dimension_base_parity.addLayout(dimension_layout)
        dimension_base_parity.addLayout(base_parity_layout)
        dimension_base_parity.addStretch()
        dimension_base_parity.addWidget(self.reset_values_btn)

        controls.addLayout(dimension_base_parity, 1)

        self.range_min_stack = RangeSpinStack(self.controls_section)
        self.range_max_stack = RangeSpinStack(self.controls_section)
        self.range_min_spin_h = QSpinBox(self.controls_section, suffix=" px", minimum=0, maximum=99999, singleStep=1)
        self.range_max_spin_h = QSpinBox(self.controls_section, suffix=" px", minimum=0, maximum=99999, singleStep=1)
        self.range_min_spin_w = QSpinBox(self.controls_section, suffix=" px", minimum=0, maximum=99999, singleStep=1)
        self.range_max_spin_w = QSpinBox(self.controls_section, suffix=" px", minimum=0, maximum=99999, singleStep=1)
        self.range_min_spin_h.setToolTip("Lowest candidate dimension to test.")
        self.range_max_spin_h.setToolTip("Highest candidate dimension to test.")
        self.range_min_spin_w.setToolTip("Lowest candidate dimension to test.")
        self.range_max_spin_w.setToolTip("Highest candidate dimension to test.")
        self.range_min_spin_h.valueChanged.connect(self.on_range_min_h_changed)
        self.range_max_spin_h.valueChanged.connect(self.on_range_max_h_changed)
        self.range_min_spin_w.valueChanged.connect(self.on_range_min_w_changed)
        self.range_max_spin_w.valueChanged.connect(self.on_range_max_w_changed)

        self.range_min_stack.addWidget(self.range_min_spin_h)
        self.range_min_stack.addWidget(self.range_min_spin_w)
        self.range_min_stack.setCurrentWidget(self.range_min_spin_h)

        self.range_max_stack.addWidget(self.range_max_spin_h)
        self.range_max_stack.addWidget(self.range_max_spin_w)
        self.range_max_stack.setCurrentWidget(self.range_max_spin_h)

        range_layout = self.make_vgroup(
            "Range",
            self.range_min_stack,
            self.range_max_stack,
            parent=self.controls_section,
            stretch=False,
        )

        self.step_spin = QDoubleSpinBox(
            self.controls_section,
            decimals=3,
            minimum=0,
            maximum=1,
            stepType=QDoubleSpinBox.StepType.AdaptiveDecimalStepType,
            value=1.0,
        )
        self.step_spin.setToolTip("Distance between tested dimensions.\nFractional values are also supported.")
        step_layout = self.make_vgroup("Step", self.step_spin, parent=self.controls_section, stretch=False)

        self.range_step_layout = QVBoxLayout()
        self.range_step_layout.addLayout(range_layout)
        self.range_step_layout.addLayout(step_layout)
        self.range_step_layout.addStretch()

        controls.addLayout(self.range_step_layout, 1)

        self.kernels_cb = QComboBox(self.controls_section)
        self.kernels_cb.setToolTip("Descale kernel used for each candidate resolution test.")
        kernels_layout = self.make_vgroup("Kernel", self.kernels_cb, parent=self.controls_section, stretch=False)
        self.metrics_cb = QComboBox(self.controls_section)
        self.metrics_cb.addItems(MetricMode.__value__.__args__)
        self.metrics_cb.setToolTip(
            "Error metric used to compare the descaled frame against the source.\n"
            "MAE is generally the most stable, while MSE tends to produce steeper slopes."
        )
        metrics_layout = self.make_vgroup("Metric", self.metrics_cb, parent=self.controls_section, stretch=False)

        self.kernels_metrics_layout = QVBoxLayout()
        self.kernels_metrics_layout.addLayout(kernels_layout)
        self.kernels_metrics_layout.addSpacing(self.range_max_stack.height() + 4)
        self.kernels_metrics_layout.addLayout(metrics_layout)
        self.kernels_metrics_layout.addStretch()

        controls.addLayout(self.kernels_metrics_layout, 1)

        self.import_btn = QPushButton("Import...", self)
        self.import_btn.setToolTip("Import saved getnative result plots from JSON or CSV files.")
        self.import_btn.clicked.connect(self.on_import_btn_clicked)
        self.imported_results = GetNativeImportList(self)
        self.imported_results.setToolTip(
            "Previously computed or imported plots.\nDouble-click a computed result to jump back to its frame."
        )
        self.imported_results.itemClicked.connect(self.on_import_item_clicked)
        self.imported_results.itemDoubleClicked.connect(self.on_import_item_double_clicked)
        self.imported_results.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        import_layout = self.make_vgroup(
            "History", self.import_btn, self.imported_results, parent=self.controls_section, stretch=False
        )

        controls.addLayout(import_layout, 1)

        self.calculate_btn = QPushButton("Calculate", self)
        self.calculate_btn.setToolTip("Run getnative on the current frame using the selected range and kernel.")
        self.calculate_btn.clicked.connect(self.on_calculate_clicked)

        self.btn_layout = QHBoxLayout()
        self.btn_layout.addStretch(3)
        self.btn_layout.addWidget(self.calculate_btn, 2)
        self.btn_layout.addStretch(3)

        self.add_section(self.controls_section)
        self.add_section(self.btn_layout)

        self.progress_bar = ProgressBar(self)
        self.progress_bar.setFixedHeight(24)

        self.progress_container = QWidget(self)
        progress_layout = QVBoxLayout(self.progress_container)
        progress_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.progress_bar)

        self.plot_stack = QStackedWidget(self)
        self.plot_stack.setFrameShape(QFrame.Shape.StyledPanel)
        self.plot_stack.setFrameShadow(QFrame.Shadow.Sunken)

        self.canvas = QStackedWidget(self)
        self.canvas.setFrameShape(QFrame.Shape.StyledPanel)
        self.canvas.setFrameShadow(QFrame.Shadow.Sunken)
        self.canvas.addWidget(self.progress_container)
        self.canvas.addWidget(self.plot_stack)
        self.canvas.setCurrentWidget(self.progress_container)

        self.progress_container.hide()

        self.add_section(self.canvas, 1)

        self.register_icon_callback(self._reload_icons)
        self._reload_icons()
        self._set_default_values()
        self._set_saved_values()
        self.api.aboutToSaveLocal.connect(self.snapshot_ui_values)
        self.api.globalSettingsChanged.connect(self.on_global_settings_changed)

    def _reload_icons(self) -> None:
        icon = (IconName.FILE_IMPORT, self.palette().color(QPalette.ColorGroup.Normal, QPalette.ColorRole.ButtonText))
        self.import_btn.setIcon(self.make_icon(icon))

    def _set_default_values(self) -> None:
        self.dimension.index = 1
        self.base_parity.index = 1
        self._update_dimension_stack(1)
        self._last_base_parity = 1
        with (
            QSignalBlocker(self.range_min_stack),
            QSignalBlocker(self.range_max_stack),
            QSignalBlocker(self.range_min_spin_h),
            QSignalBlocker(self.range_max_spin_h),
            QSignalBlocker(self.range_min_spin_w),
            QSignalBlocker(self.range_max_spin_w),
            QSignalBlocker(self.step_spin),
            QSignalBlocker(self.kernels_cb),
            QSignalBlocker(self.metrics_cb),
        ):
            self.range_min_spin_h.setValue(int(self.api.current_voutput.vs_output.clip.height * LOW_RATE))
            self.range_max_spin_h.setValue(int(self.api.current_voutput.vs_output.clip.height * HIGH_RATE))
            self.range_min_spin_w.setValue(int(self.api.current_voutput.vs_output.clip.width * LOW_RATE))
            self.range_max_spin_w.setValue(int(self.api.current_voutput.vs_output.clip.width * HIGH_RATE))
            self.step_spin.setValue(fallback(self.settings.local_.getnative.last_step, 1.0))
            for k in self.settings.global_.kernels:
                self.kernels_cb.addItem(k.pretty_string, k)
            self.kernels_cb.setCurrentText("Bilinear()")
            self.metrics_cb.setCurrentText("MAE")

        self.update_limits()

    def _set_saved_values(self) -> None:
        with (
            QSignalBlocker(self.dimension),
            QSignalBlocker(self.range_min_spin_h),
            QSignalBlocker(self.range_max_spin_h),
            QSignalBlocker(self.range_min_spin_w),
            QSignalBlocker(self.range_max_spin_w),
            QSignalBlocker(self.step_spin),
            QSignalBlocker(self.kernels_cb),
            QSignalBlocker(self.metrics_cb),
        ):
            if (d := self.settings.local_.getnative.last_dimension) is not None:
                self.dimension.index = d
                self._update_dimension_stack(d)

            if (d := self.settings.local_.getnative.last_base_parity) is not None:
                self.base_parity.index = d

            if (range_min := self.settings.local_.getnative.last_min_h_range) is not None:
                self.range_min_spin_h.setValue(range_min)
            if (range_max := self.settings.local_.getnative.last_max_h_range) is not None:
                self.range_max_spin_h.setValue(range_max)

            if (range_min := self.settings.local_.getnative.last_min_w_range) is not None:
                self.range_min_spin_w.setValue(range_min)
            if (range_max := self.settings.local_.getnative.last_max_w_range) is not None:
                self.range_max_spin_w.setValue(range_max)

            if (step := self.settings.local_.getnative.last_step) is not None:
                self.step_spin.setValue(step)

            if (kernel := self.settings.local_.getnative.last_kernel) is not None:
                self.kernels_cb.setCurrentText(kernel.pretty_string)

            if (metric := self.settings.local_.getnative.last_metric) is not None:
                self.metrics_cb.setCurrentText(metric)

        self.update_limits()

    def snapshot_ui_values(self) -> None:
        self.settings.local_.getnative.last_dimension = self.dimension.index
        self.settings.local_.getnative.last_base_parity = self.base_parity.index
        self.settings.local_.getnative.last_max_h_range = self.range_max_spin_h.value()
        self.settings.local_.getnative.last_min_h_range = self.range_min_spin_h.value()
        self.settings.local_.getnative.last_max_w_range = self.range_max_spin_w.value()
        self.settings.local_.getnative.last_min_w_range = self.range_min_spin_w.value()
        self.settings.local_.getnative.last_step = self.step_spin.value()
        self.settings.local_.getnative.last_kernel = resolve_kernel(self.kernels_cb.currentText())
        self.settings.local_.getnative.last_metric = self.metrics_cb.currentText()

    def on_global_settings_changed(self) -> None:
        kernel = self.kernels_cb.currentText()
        self.kernels_cb.clear()

        for k in self.settings.global_.kernels:
            self.kernels_cb.addItem(k.pretty_string, k)

        self.kernels_cb.setCurrentText(kernel)

        from .plotting import CustomRescalePlotWidget

        for i in range(self.plot_stack.count()):
            if isinstance((plot := self.plot_stack.widget(i)), CustomRescalePlotWidget):
                plot.set_theme(self.settings.global_.get_chart_theme())

    def _get_max_dim(self) -> int:
        clip = self.api.current_voutput.vs_output.clip
        return clip.height if self.dimension.index == 1 else clip.width

    def update_limits(self) -> None:
        clip = self.api.current_voutput.vs_output.clip
        with (
            QSignalBlocker(self.range_min_spin_h),
            QSignalBlocker(self.range_max_spin_h),
            QSignalBlocker(self.range_min_spin_w),
            QSignalBlocker(self.range_max_spin_w),
        ):
            self.range_min_spin_h.setRange(1, clip.height - 1)
            self.range_max_spin_h.setRange(2, clip.height)
            self.range_min_spin_w.setRange(1, clip.width - 1)
            self.range_max_spin_w.setRange(2, clip.width)

            self.range_max_spin_h.setMinimum(self.range_min_spin_h.value() + 1)
            self.range_min_spin_h.setMaximum(self.range_max_spin_h.value() - 1)

            self.range_max_spin_w.setMinimum(self.range_min_spin_w.value() + 1)
            self.range_min_spin_w.setMaximum(self.range_max_spin_w.value() - 1)

    def on_range_min_h_changed(self, value: int) -> None:
        clip = self.api.current_voutput.vs_output.clip
        w_val = get_w(value, clip, 1)
        self.range_max_spin_h.setMinimum(value + 1)
        with QSignalBlocker(self.range_min_spin_w), QSignalBlocker(self.range_max_spin_w):
            self.range_min_spin_w.setValue(w_val)
            self.range_max_spin_w.setMinimum(w_val + 1)

    def on_range_max_h_changed(self, value: int) -> None:
        clip = self.api.current_voutput.vs_output.clip
        w_val = get_w(value, clip, 1)
        self.range_min_spin_h.setMaximum(value - 1)
        with QSignalBlocker(self.range_min_spin_w), QSignalBlocker(self.range_max_spin_w):
            self.range_max_spin_w.setValue(w_val)
            self.range_min_spin_w.setMaximum(w_val - 1)

    def on_range_min_w_changed(self, value: int) -> None:
        clip = self.api.current_voutput.vs_output.clip
        h_val = get_h(value, clip, 1)
        self.range_max_spin_w.setMinimum(value + 1)
        with QSignalBlocker(self.range_min_spin_h), QSignalBlocker(self.range_max_spin_h):
            self.range_min_spin_h.setValue(h_val)
            self.range_max_spin_h.setMinimum(h_val + 1)

    def on_range_max_w_changed(self, value: int) -> None:
        clip = self.api.current_voutput.vs_output.clip
        h_val = get_h(value, clip, 1)
        self.range_min_spin_w.setMaximum(value - 1)
        with QSignalBlocker(self.range_min_spin_h), QSignalBlocker(self.range_max_spin_h):
            self.range_max_spin_h.setValue(h_val)
            self.range_min_spin_h.setMaximum(h_val - 1)

    def on_dimension_segment_changed(self, index: int) -> None:
        if self._last_dimension == index:
            return
        self._update_dimension_stack(index)

    def on_base_parity_segment_changed(self, index: int) -> None:
        self._last_base_parity = index

    def on_calculate_clicked(self) -> None:
        self.calculate_btn.setDisabled(True)

        clip = self.api.current_voutput.vs_output.clip

        start = self.range_min_stack.current.value()
        stop = self.range_max_stack.current.value()
        step_f = self.step_spin.value()

        if step_f.is_integer():
            dims = range(start, stop + 1, int(step_f))
            x_label_fmt = "%.0f"
        else:
            num = int((stop - start) / step_f) + 1
            dims = np.linspace(start, start + step_f * (num - 1), num).tolist()
            x_label_fmt = f"%.{str(step_f)[::-1].find('.') + 1}f"

            if self.base_parity.index == 0:
                dims = [(d, mod2(ceil(d)) + 1) for d in dims]

        if len(dims) > 2000:
            res = QMessageBox.warning(
                self,
                "Large Number of Dimensions",
                f"You are about to calculate {len(dims)} dimensions. "
                "This may take a long time and use significant memory.\n\n"
                "Are you sure you want to proceed?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )

            if res == QMessageBox.StandardButton.Cancel:
                self.calculate_btn.setEnabled(True)
                return

        match self.dimension.index:
            case 1:
                dimensions = zip_longest([clip.width], dims, fillvalue=clip.width)
                dim_mode = "Height"
            case 0:
                dimensions = zip_longest(dims, [clip.height], fillvalue=clip.height)
                dim_mode = "Width"
            case _:
                raise ValueError("Invalid dimension")

        frame = self.api.current_frame
        kernel = self.kernels_cb.currentData()
        metric_mode = cast(MetricMode, self.metrics_cb.currentText())
        title = f"{self.api.current_voutput.vs_name} - {kernel.pretty_string} on {dim_mode.lower()} - frame {frame}"

        self.canvas.setCurrentWidget(self.progress_container)
        self.progress_bar.update_progress(fmt="Gathering data %v / %m", value=0)
        self.progress_container.show()

        @run_in_background(name="GetNativeResults")
        def get_results() -> list[GetNativeResult]:
            with self.api.vs_context():
                return getnative(
                    self.api.current_voutput.vs_output.clip,
                    frame,
                    dimensions,  # type: ignore[arg-type]
                    kernel,
                    metric_mode=metric_mode,
                    progress_cb=(
                        lambda current, total: self.progress_bar.update_progress(
                            value=current,
                            range=(0, total),
                        )
                    ),
                    func=self.on_calculate_clicked,
                )

        def on_done(_: Any) -> None:
            self.progress_bar.reset_progress()
            self.calculate_btn.setEnabled(True)

        def on_success(results: list[GetNativeResult]) -> None:
            plot = self.create_rescale_plot(title, results, dim_mode, x_label_fmt)
            self.plot_stack.addWidget(plot)
            self.plot_stack.setCurrentWidget(plot)
            self.canvas.setCurrentWidget(self.plot_stack)

            result_item = QListWidgetItem(title, self.imported_results, QListWidgetItem.ItemType.UserType)
            result_item.setData(self.IMPORT_PLOT_ROLE, plot)
            result_item.setData(self.IMPORT_FRAME_ROLE, frame)

            self.computedPlotAdded.emit(plot)

        def on_fail(e: BaseException) -> None:
            logger.exception("Failed to get native results", exc_info=e)
            e.__traceback__ = None

        blocker = self.api.blocker()
        blocker.acquire()
        (
            get_results()
            .add_loop_callback(on_done)
            .then(on_success, on_fail, on_loop=True)
            .add_done_callback(lambda _: blocker.release())
        )

    def create_rescale_plot(
        self,
        title: str,
        results: list[GetNativeResult],
        dim_mode: str,
        x_label_fmt: str,
    ) -> CustomRescalePlotWidget:
        from .plotting import CustomRescalePlotWidget

        dims, errors = zip(*results)

        dims = np.fromiter((getattr(d, dim_mode.lower()) for d in dims), dtype=np.float64)

        plot = CustomRescalePlotWidget(title, dims, errors, dim_mode.title(), self.plot_stack)
        plot.set_theme(self.settings.global_.get_chart_theme())
        plot.axis_x.setLabelFormat(x_label_fmt)

        return plot

    def on_import_btn_clicked(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Results",
            "",
            "JSON & CSV Files (*.json *.csv)",
        )

        if not files:
            return

        from .plotting import CustomRescalePlotWidget

        for file_path in files:
            try:
                if file_path.endswith(".json"):
                    dim_mode, dims, errors = self.parse_json(file_path)
                elif file_path.endswith(".csv"):
                    dim_mode, dims, errors = self.parse_csv(file_path)
                else:
                    raise NotImplementedError(f"Unsupported file format: {file_path}")
            except Exception:
                logger.exception("Failed to import results from %s", file_path)
                QMessageBox.critical(self, "Import Error", f"Failed to import results from {file_path}")
                break

            title = f"Imported - {Path(file_path).name}"
            plot = CustomRescalePlotWidget(title, dims, errors, dim_mode.title(), self.plot_stack)
            plot.set_theme(self.settings.global_.get_chart_theme())

            self.plot_stack.addWidget(plot)
            self.plot_stack.setCurrentWidget(plot)
            self.canvas.setCurrentWidget(self.plot_stack)

            result_item = QListWidgetItem(title, self.imported_results, QListWidgetItem.ItemType.UserType)
            result_item.setData(self.IMPORT_PLOT_ROLE, plot)
            result_item.setData(self.IMPORT_FRAME_ROLE, None)

    def on_import_item_clicked(self, item: QListWidgetItem) -> None:
        if plot := item.data(self.IMPORT_PLOT_ROLE):
            self.plot_stack.setCurrentWidget(plot)
            self.canvas.setCurrentWidget(self.plot_stack)

    def on_import_item_double_clicked(self, item: QListWidgetItem) -> None:
        self.on_import_item_clicked(item)

        if (frame := item.data(self.IMPORT_FRAME_ROLE)) is not None:
            self.api.playback.seek(frame)

    def parse_json(self, file_path: str) -> tuple[str, list[float], list[float]]:
        with open(file_path) as f:
            data = json.load(f)
        dim_mode = data["dimension_mode"]
        dims = [p["x"] for p in data["data"]]
        errors = [p["y"] for p in data["data"]]
        return dim_mode, dims, errors

    def parse_csv(self, file_path: str) -> tuple[str, list[float], list[float]]:
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            dim_mode = header[0]
            data = list(reader)
        return dim_mode, [float(row[0]) for row in data], [float(row[1]) for row in data]

    def _update_dimension_stack(self, index: int) -> None:
        self._last_dimension = index

        match index:
            case 1:
                self.range_max_stack.setCurrentWidget(self.range_max_spin_h)
                self.range_min_stack.setCurrentWidget(self.range_min_spin_h)
            case 0:
                self.range_max_stack.setCurrentWidget(self.range_max_spin_w)
                self.range_min_stack.setCurrentWidget(self.range_min_spin_w)
            case _:
                raise ValueError("Invalid dimension")


class GetScalerTab(TabContainer):
    def __init__(
        self,
        parent: QWidget,
        api: PluginAPI,
        settings: PluginSettings[GlobalSettings, LocalSettings],
    ) -> None:
        super().__init__(parent)

        self.api = api
        self.settings = settings

        self.controls_section = Accordion("Controls", self)
        controls = self.controls_section.add_hlayout()

        self.dimension = SegmentedControl(["Width", "Height"], self.controls_section)
        self.dimension.current_layout.setContentsMargins(0, 0, 0, 0)
        self.dimension.current_layout.setSpacing(0)
        self.dimension.setToolTip("Choose whether the entered target value represents width or height.")
        self.dimension.segmentChanged.connect(self.on_segment_changed)
        self._last_dimension: int | None = None

        self.target_dimension = QDoubleSpinBox(
            self.controls_section,
            minimum=1,
            maximum=99999,
            decimals=3,
            suffix=" px",
            singleStep=1,
            stepType=QDoubleSpinBox.StepType.AdaptiveDecimalStepType,
        )
        self.target_dimension.setToolTip("Target descale dimension to test against every configured kernel.")

        dimension_layout = self.make_vgroup(
            "Dimension",
            self.dimension,
            self.target_dimension,
            parent=self.controls_section,
        )

        controls.addLayout(dimension_layout, 1)

        self.metrics_cb = QComboBox(self.controls_section)
        self.metrics_cb.addItems(MetricMode.__value__.__args__)
        self.metrics_cb.setToolTip("Error metric used when ranking kernels.")
        metrics_layout = self.make_vgroup("Metric", self.metrics_cb, parent=self.controls_section)

        self.mask_cb = QComboBox(self.controls_section)
        self.mask_cb.setToolTip("Edge-detection mask to reduce noise influence on the metric.")
        mask_layout = self.make_vgroup("Mask", self.mask_cb, parent=self.controls_section)

        controls.addLayout(metrics_layout)
        controls.addLayout(mask_layout, 1)

        self.calculate_btn = QPushButton("Calculate", self)
        self.calculate_btn.setToolTip("Run getscaler on the current frame with the selected target dimension.")
        self.calculate_btn.clicked.connect(self.on_calculate_clicked)

        self.btn_layout = QHBoxLayout()
        self.btn_layout.addStretch()
        self.btn_layout.addWidget(self.calculate_btn)
        self.btn_layout.addStretch()

        self.table = QTableWidget(self.controls_section, columnCount=3)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setHorizontalHeaderLabels(["Kernel", "Error %", "Error"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(300)
        self.table.setToolTip("Kernel ranking for the current frame.\nLower error means a better descale match.")

        self.add_section(self.controls_section)
        self.add_section(self.btn_layout)
        self.add_section(self.table, 1)

        self._set_default_values()
        self._set_saved_values()
        self.api.aboutToSaveLocal.connect(self.snapshot_ui_values)

    def _set_default_values(self) -> None:
        self.dimension.index = 1
        self._last_dimension = 1

        with (
            QSignalBlocker(self.dimension),
            QSignalBlocker(self.target_dimension),
            QSignalBlocker(self.metrics_cb),
            QSignalBlocker(self.mask_cb),
        ):
            self.metrics_cb.setCurrentText("MAE")
            self.target_dimension.setValue(self._get_max_dim() / 1.5)

            for s in get_edge_detect_classes():
                self.mask_cb.addItem(s.__name__, s)
            self.mask_cb.insertItem(0, "", None)
            self.mask_cb.setCurrentText("")

    def _set_saved_values(self) -> None:
        with QSignalBlocker(self.dimension), QSignalBlocker(self.target_dimension), QSignalBlocker(self.metrics_cb):
            if (dim := self.settings.local_.getscaler.last_dimension) is not None:
                self.dimension.index = dim
                self._last_dimension = dim

            if (dim := self.settings.local_.getscaler.last_target_dimension) is not None:
                self.target_dimension.setValue(dim)

            if (metric := self.settings.local_.getscaler.last_metric) is not None:
                self.metrics_cb.setCurrentText(metric)

            if (mask := self.settings.local_.getscaler.last_mask) is not None:
                self.mask_cb.setCurrentText(mask)

    def snapshot_ui_values(self) -> None:
        self.settings.local_.getscaler.last_dimension = self.dimension.index
        self.settings.local_.getscaler.last_target_dimension = self.target_dimension.value()
        self.settings.local_.getscaler.last_metric = self.metrics_cb.currentText()
        self.settings.local_.getscaler.last_mask = self.mask_cb.currentText()

    def _get_max_dim(self) -> int:
        clip = self.api.current_voutput.vs_output.clip
        return clip.height if self.dimension.index == 1 else clip.width

    def update_limits(self) -> None:
        max_dim = self._get_max_dim()

        with QSignalBlocker(self.target_dimension):
            self.target_dimension.setMaximum(max_dim)

    def on_segment_changed(self, index: int) -> None:
        if self._last_dimension == index:
            return

        clip = self.api.current_voutput.vs_output.clip
        self._last_dimension = index

        match self.dimension.index:
            case 0:
                func = get_w
            case 1:
                func = get_h
            case _:
                raise ValueError("Invalid dimension")

        with QSignalBlocker(self.target_dimension):
            v = self.target_dimension.value()
            self.update_limits()
            self.target_dimension.setValue(func(v, clip, 1))

    def on_calculate_clicked(self) -> None:
        self.calculate_btn.setDisabled(True)
        self.table.clearContents()

        clip = self.api.current_voutput.vs_output.clip
        frame = self.api.current_frame
        kernels = self.settings.global_.kernels
        metric_mode = cast(MetricMode, self.metrics_cb.currentText())
        mask = self.mask_cb.currentData()

        match self.dimension.index:
            case 0:
                width, height = self.target_dimension.value(), clip.height
            case 1:
                width, height = clip.width, self.target_dimension.value()
            case _:
                raise ValueError("Invalid dimension")

        @run_in_background(name="GetScalerResults")
        def get_results() -> list[GetScalerResult]:
            with self.api.vs_context():
                return getscaler(
                    clip,
                    frame,
                    width,
                    height,
                    kernels,
                    metric_mode=metric_mode,
                    mask=mask,
                    func=self.on_calculate_clicked,
                )

        def on_success(results: list[GetScalerResult]) -> None:
            if not results:
                return

            sorted_ress = sorted(results, key=lambda r: r.error)
            best = sorted_ress[0]

            self.table.setUpdatesEnabled(False)
            self.table.setRowCount(len(sorted_ress))

            try:
                for i, res in enumerate(sorted_ress):
                    self.table.setItem(i, 0, QTableWidgetItem(res.kernel.pretty_string))
                    self.table.setItem(
                        i, 1, QTableWidgetItem(f"{res.error * 100 / best.error if best.error else 0:.2f} %")
                    )
                    self.table.setItem(i, 2, QTableWidgetItem(f"{res.error:.13f}"))
            finally:
                self.table.setUpdatesEnabled(True)

        blocker = self.api.blocker()
        blocker.acquire()
        (
            get_results()
            .add_loop_callback(lambda f: self.calculate_btn.setEnabled(True))
            .map(on_success, on_loop=True)
            .catch(lambda e: logger.exception("Failed to get scaler results", exc_info=e))
            .add_done_callback(lambda _: blocker.release())
        )


class GetFreqTab(TabContainer):
    def __init__(
        self,
        parent: QWidget,
        api: PluginAPI,
        settings: PluginSettings[GlobalSettings, LocalSettings],
    ) -> None:
        super().__init__(parent)

        self.api = api
        self.settings = settings

        self.plot: CustomFrequencyPlotWidget | None = None

        self.calc_timer = QTimer(self, singleShot=True, interval=100)
        self.calc_timer.timeout.connect(lambda: self.on_calculate_clicked() if self.plot else None)

        self.controls_section = Accordion("Controls", self)
        controls = self.controls_section.add_hlayout()

        self.cull_rate_spin = QDoubleSpinBox(
            self.controls_section,
            decimals=2,
            minimum=0.01,
            maximum=10.0,
            value=3.0,
            suffix=" x",
            singleStep=0.1,
            stepType=QDoubleSpinBox.StepType.AdaptiveDecimalStepType,
        )
        self.cull_rate_spin.setToolTip(
            "Cull the outer region of the frame before building the DCT distribution to focus on the center."
        )
        self.cull_rate_spin.valueChanged.connect(lambda _: self.calc_timer.start())
        controls.addLayout(self.make_vgroup("Cull Rate", self.cull_rate_spin, parent=self.controls_section))

        self.radius_spin = QSpinBox(self.controls_section, minimum=1, maximum=100, value=50, suffix=" px")
        self.radius_spin.setToolTip("Neighborhood radius used to detect spikes\nand candidate native-resolution peaks.")
        self.radius_spin.valueChanged.connect(self.on_radius_changed)
        controls.addLayout(self.make_vgroup("Radius", self.radius_spin, parent=self.controls_section))

        self.calculate_btn = QPushButton("Calculate", self)
        self.calculate_btn.clicked.connect(self.on_calculate_clicked)
        calculate_layout = self.make_vgroup("", parent=self.controls_section)
        calculate_layout.addWidget(self.calculate_btn)
        controls.addLayout(calculate_layout)

        controls.addStretch()

        self.add_section(self.controls_section)

        self.canvas = QStackedWidget(self)
        self.canvas.setFrameShape(QFrame.Shape.StyledPanel)
        self.canvas.setFrameShadow(QFrame.Shadow.Sunken)
        self.canvas.setMinimumHeight(400)

        self._last_request_id = 0

        self.add_section(self.canvas, 1)

        self.api.globalSettingsChanged.connect(self.on_global_settings_changed)
        self.api.register_on_destroy(self.on_destroy)

    def on_global_settings_changed(self) -> None:
        if self.plot:
            self.plot.set_theme(
                self.settings.global_.get_chart_theme(),
                hpen_color=self.settings.global_.frequency_width_color,
                vpen_color=self.settings.global_.frequency_height_color,
            )

    def on_destroy(self) -> None:
        self.calc_timer.stop()
        if self.plot:
            self.canvas.removeWidget(self.plot)
            self.plot.deleteLater()
            self.plot = None

    def on_radius_changed(self, value: int) -> None:
        if self.plot:
            self.plot.check_radius = value
            self.plot.set_spikes_h()
            self.plot.set_spikes_v()

    def on_calculate_clicked(self) -> None:
        clip = self.api.current_voutput.vs_output.clip
        frame = self.api.current_frame
        cull_rate = self.cull_rate_spin.value()
        title = f"DCT Distribution - {self.api.current_voutput.vs_name} frame {frame}"

        self._last_request_id += 1
        request_id = self._last_request_id

        @run_in_background(name="GetDCTDistribution")
        def get_results() -> tuple[NpFloatArray1D, NpFloatArray1D]:
            with self.api.vs_context():
                return get_dct_distribution(clip, frame, cull_rate)

        def on_success(results: tuple[NpFloatArray1D, NpFloatArray1D]) -> None:
            if request_id != self._last_request_id:
                return

            new_plot = self.create_freq_plot(title, results, clip)

            if self.plot:
                self.canvas.removeWidget(self.plot)
                self.plot.deleteLater()

            self.canvas.addWidget(new_plot)
            self.canvas.setCurrentWidget(new_plot)
            self.plot = new_plot

        blocker = self.api.blocker()
        blocker.acquire()
        (
            get_results()
            .map(on_success, on_loop=True)
            .catch(lambda e: logger.exception("Failed to get DCT distribution", exc_info=e))
            .add_done_callback(lambda _: blocker.release())
        )

    def create_freq_plot(
        self, title: str, results: tuple[NpFloatArray1D, NpFloatArray1D], clip: vs.VideoNode
    ) -> CustomFrequencyPlotWidget:
        from .plotting import CustomFrequencyPlotWidget

        min_val_h, max_val_h = int(clip.width * LOW_RATE), int(clip.width * HIGH_RATE)
        min_val_v, max_val_v = int(clip.height * LOW_RATE), int(clip.height * HIGH_RATE)

        plot = CustomFrequencyPlotWidget(
            title,
            results[0],
            results[1],
            min_val_h,
            max_val_h,
            min_val_v,
            max_val_v,
            self.radius_spin.value(),
            self.canvas,
        )
        plot.set_theme(
            self.settings.global_.get_chart_theme(),
            hpen_color=self.settings.global_.frequency_width_color,
            vpen_color=self.settings.global_.frequency_height_color,
        )

        return plot


class NativeResPlugin(WidgetPluginBase[GlobalSettings, LocalSettings]):
    identifier = "jet_vsview_nativeres"
    display_name = "Native Resolution"

    def __init__(self, parent: QWidget, api: PluginAPI) -> None:
        super().__init__(parent, api)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.tabs = QTabWidget(self, movable=False, tabsClosable=False)

        self.tab_getnative = GetNativeTab(self.tabs, self.api, self.settings)
        self.tab_getnative.computedPlotAdded.connect(self.dump_plot_results)
        self.tabs.addTab(self.tab_getnative, "Get Native")

        self.tab_getscaler = GetScalerTab(self.tabs, self.api, self.settings)
        self.tabs.addTab(self.tab_getscaler, "Get Scaler")

        self.tab_getfreq = GetFreqTab(self.tabs, self.api, self.settings)
        self.tabs.addTab(self.tab_getfreq, "Get Frequencies")

        main_layout.addWidget(self.tabs)
        warmup_plots()

    # Plugin hooks
    def on_current_voutput_changed(self, voutput: VideoOutputProxy, tab_index: int) -> None:
        self.update_ui()

    def on_current_frame_changed(self, n: int) -> None:
        self.update_ui()

    @run_in_loop(return_future=False)
    def update_ui(self) -> None:
        if self.api.is_playing:
            return

        # GetNative Tab
        self.tab_getnative.update_limits()

        # GetScaler Tab
        self.tab_getscaler.update_limits()

        # GetFreq Tab
        if self.tab_getfreq.plot:
            self.tab_getfreq.calc_timer.start()

    def on_playback_stopped(self) -> None:
        self.update_ui()

    @run_in_background(name="DumpPlotResults")
    def dump_plot_results(self, plot: CustomRescalePlotWidget) -> None:
        if path := self.api.get_local_storage(self):
            with (path / plot.chart().title()).open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerows(plot.serialize_csv())
            logger.info("Dumped plot results to %s", path)
        else:
            logger.debug("No local storage path found")

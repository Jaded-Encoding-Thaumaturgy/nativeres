import csv
import json
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from logging import getLogger
from typing import Any, Literal

import numpy as np
from jetpytools import clamp
from PySide6.QtCharts import QCategoryAxis, QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis
from PySide6.QtCore import QBuffer, QIODevice, QMargins, QMimeData, QPointF, Qt
from PySide6.QtGui import (
    QAction,
    QColor,
    QContextMenuEvent,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPalette,
    QPen,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtSvg import QSvgGenerator
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsTextItem,
    QMenu,
    QWidget,
)
from scipy.signal import argrelextrema

type FloatArray1D = Sequence[float] | np.ndarray[tuple[int], np.dtype[np.floating]]

logger = getLogger(__name__)


def get_color_scheme() -> Qt.ColorScheme:
    if not isinstance(app := QApplication.instance(), QApplication):
        raise NotImplementedError("QApplication instance is not available")
    return app.styleHints().colorScheme()


def get_chart_theme() -> QChart.ChartTheme:
    match get_color_scheme():
        case Qt.ColorScheme.Dark:
            return QChart.ChartTheme.ChartThemeDark
        case Qt.ColorScheme.Light:
            return QChart.ChartTheme.ChartThemeLight
        case _:
            raise NotImplementedError("Unsupported color scheme")


class LabelText(QGraphicsTextItem):
    def set_html_text(self, text: str, palette: QPalette) -> None:
        bg = palette.color(QPalette.ColorRole.Window).name()
        fg = palette.color(QPalette.ColorRole.WindowText).name()
        border = palette.color(QPalette.ColorRole.Highlight).name()

        self.setHtml(
            f"<div style='background-color: {bg}; color: {fg}; border: 1px solid {border}; "
            "border-radius: 4px; padding: 4px; font-family: sans-serif;'>"
            f"{text}</div>"
        )


class CustomHorizontalTitle(QGraphicsTextItem):
    def __init__(self, title: str, parent: QGraphicsItem, font: QFont) -> None:
        super().__init__(title, parent)
        self.setZValue(5)
        self.setFont(font)

    def update_from_chart(self, chart: QChart) -> None:
        if not (area := chart.plotArea()).isValid():
            return

        # Position the horizontal Y-axis title in the left margin
        # Center x horizontally in the left margin (area.left())
        # Center y vertically relative to the plot area
        rect = self.boundingRect()
        self.setPos(
            (area.left() - rect.width()) / 3,
            area.top() + (area.height() - rect.height()) / 2,
        )


class BasePlotWidget(QChartView):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        chart = QChart(
            theme=get_chart_theme(),
            title=title,
            margins=QMargins(50, 10, 10, 10),
        )
        super().__init__(chart, parent)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)

        self._is_panning = False
        self._last_mouse_pos = QPointF()

        # Crosshair overlays
        cross_color = self.palette().color(QPalette.ColorRole.ToolTipText)
        cross_color.setAlphaF(0.8)
        cross_pen = QPen(cross_color, 1, Qt.PenStyle.DashLine)
        self.v_line = QGraphicsLineItem(chart)
        self.v_line.setPen(cross_pen)
        self.v_line.setZValue(10)

        self.h_line = QGraphicsLineItem(chart)
        self.h_line.setPen(cross_pen)
        self.h_line.setZValue(10)

        # Tooltip-style hover label
        self.label = LabelText(chart)
        self.label.setZValue(11)

        self.set_overlays_visible(False)

        self.menu = QMenu(self)
        self.reset_action = QAction("Reset Zoom", self)
        self.reset_action.triggered.connect(self.reset_zoom)
        self.menu.addAction(self.reset_action)

        self.menu.addSeparator()

        self.copy_menu = self.menu.addMenu("Copy")

        self.copy_png_action = QAction("Copy as PNG", self)
        self.copy_png_action.triggered.connect(self.copy_png)
        self.copy_menu.addAction(self.copy_png_action)

        self.copy_svg_action = QAction("Copy as SVG", self)
        self.copy_svg_action.triggered.connect(self.copy_svg)
        self.copy_menu.addAction(self.copy_svg_action)

        self.copy_json_action = QAction("Copy as JSON", self)
        self.copy_json_action.triggered.connect(self.copy_json)
        self.copy_menu.addAction(self.copy_json_action)

        self.copy_csv_action = QAction("Copy as CSV", self)
        self.copy_csv_action.triggered.connect(self.copy_csv)
        self.copy_menu.addAction(self.copy_csv_action)

        self.export_menu = self.menu.addMenu("Export")

        self.png_action = QAction("Export as PNG", self)
        self.png_action.triggered.connect(self.on_export_png)
        self.export_menu.addAction(self.png_action)

        self.svg_action = QAction("Export as SVG", self)
        self.svg_action.triggered.connect(self.on_export_svg)
        self.export_menu.addAction(self.svg_action)

        self.json_action = QAction("Export as JSON", self)
        self.json_action.triggered.connect(self.on_export_json)
        self.export_menu.addAction(self.json_action)

        self.csv_action = QAction("Export as CSV", self)
        self.csv_action.triggered.connect(self.on_export_csv)
        self.export_menu.addAction(self.csv_action)

    @property
    def is_panning(self) -> bool:
        return self._is_panning

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        self.menu.exec(event.globalPos())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = True
            self._last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.set_overlays_visible(False)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            # Re-trigger move event to restore crosshair immediately
            self.mouseMoveEvent(event)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._is_panning:
            pos = event.position()
            delta = pos - self._last_mouse_pos
            self.chart().scroll(-delta.x(), delta.y())
            self._last_mouse_pos = pos
            event.accept()
            return
        super().mouseMoveEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.angleDelta().y() > 0:
            self.chart().zoom(1.25)
        else:
            self.chart().zoom(1 / 1.25)
        event.accept()

    @abstractmethod
    def reset_zoom(self) -> None: ...

    @abstractmethod
    def serialize_json(self) -> Any: ...

    @abstractmethod
    def serialize_csv(self) -> Iterable[Iterable[Any]]: ...

    def on_export_json(self, checked: bool = False, filename: str | None = None) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export JSON",
            filename or self.chart().title(),
            "JSON Files (*.json)",
        )
        if path:
            with open(path, "w") as f:
                json.dump(self.serialize_json(), f, indent=2)
            logger.info("Exported JSON to %s", path)

    def on_export_csv(self, checked: bool = False, filename: str | None = None) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            filename or self.chart().title(),
            "CSV Files (*.csv)",
        )
        if path:
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.serialize_csv())
            logger.info("Exported CSV to %s", path)

    def on_export_png(self, checked: bool = False, filename: str | None = None) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export PNG",
            filename or self.chart().title(),
            "PNG Files (*.png)",
        )
        if path:
            self.render_to_image().save(path, "PNG")  # type: ignore[call-overload]
            logger.info("Exported PNG to %s", path)

    def on_export_svg(self, checked: bool = False, filename: str | None = None) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export SVG",
            filename or self.chart().title(),
            "SVG Files (*.svg)",
        )
        if path:
            self.render_to_svg(file=path)
            logger.info("Exported SVG to %s", path)

    def copy_json(self) -> None:
        QApplication.clipboard().setText(json.dumps(self.serialize_json(), indent=2))
        logger.info("Copied JSON to clipboard")

    def copy_csv(self) -> None:
        QApplication.clipboard().setText("\n".join(",".join(map(str, row)) for row in self.serialize_csv()))
        logger.info("Copied CSV to clipboard")

    def copy_png(self) -> None:
        image = self.render_to_image()

        buffer = QBuffer(self)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        image.save(buffer, "PNG")  # type: ignore[call-overload]
        buffer.close()

        mime_data = QMimeData()
        mime_data.setData("image/png", buffer.data())
        mime_data.setImageData(image)
        mime_data.setText(self.chart().title())

        QApplication.clipboard().setMimeData(mime_data)
        logger.info("Copied PNG to clipboard")

    def copy_svg(self) -> None:
        buffer = QBuffer(self)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        self.render_to_svg(output=buffer)
        buffer.close()

        svg_bytes = buffer.data()
        mime_data = QMimeData()
        mime_data.setData("image/svg+xml", svg_bytes)
        mime_data.setText(bytes(svg_bytes.data()).decode("utf-8"))

        QApplication.clipboard().setMimeData(mime_data)
        logger.info("Copied SVG to clipboard")

    def render_to_image(self) -> QImage:
        self.set_overlays_visible(False)

        dpr = self.screen().devicePixelRatio()
        image = QImage(self.size() * dpr, QImage.Format.Format_ARGB32)
        image.setDevicePixelRatio(dpr)
        image.fill(Qt.GlobalColor.transparent)

        with QPainter(image) as painter:
            painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
            self.render(painter)

        return image

    def render_to_svg(self, file: str | None = None, output: QIODevice | None = None) -> None:
        self.set_overlays_visible(False)

        generator = QSvgGenerator()
        generator.setSize(self.size())
        generator.setViewBox(self.rect())
        generator.setTitle(self.chart().title())

        if file:
            generator.setFileName(file)
        if output:
            generator.setOutputDevice(output)

        with QPainter(generator) as painter:
            painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
            self.render(painter)

    def set_overlays_visible(self, visible: bool) -> None:
        self.v_line.setVisible(visible)
        self.h_line.setVisible(visible)
        self.label.setVisible(visible)


class RescalePlotWidget(BasePlotWidget):
    def __init__(
        self,
        title: str,
        dims: FloatArray1D,
        errors: FloatArray1D,
        dimension_mode: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title, parent)
        self.chart().legend().hide()

        margins = self.chart().margins()
        margins.setLeft(75)
        self.chart().setMargins(margins)

        self.dims = np.asarray(dims, np.float64)
        self.errors = np.asarray(errors, np.float64)
        self.errors_log = np.log10(self.errors.clip(1e-15, None))
        self.dimension_mode = dimension_mode
        self._last_snap_idx = -1

        self.series = QLineSeries(self)
        self.series.setMarkerSize(2.5)
        self.series.setPointsVisible(True)
        if self.dims.size > 0:
            self.series.appendNp(self.dims, self.errors_log)  # type: ignore[arg-type]
        self.chart().addSeries(self.series)

        # Setup X Axis (Linear)
        self.axis_x = QValueAxis(self)
        self.axis_x.setTitleText(dimension_mode)
        self.axis_x.setLabelFormat("%.0f")
        self.axis_x.setTickCount(11)
        self.chart().addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.series.attachAxis(self.axis_x)

        if self.dims.size > 0:
            x_min, x_max = self.dims.min(), self.dims.max()
            pad = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
            self.initial_x_range = (x_min - pad, x_max + pad)
            self.axis_x.setRange(*self.initial_x_range)
        else:
            self.initial_x_range = (0.0, 10.0)

        # Setup Y Axis (Linear mapping log10 values with true value labels)
        self.axis_y = QCategoryAxis(self, labelsPosition=QCategoryAxis.AxisLabelsPosition.AxisLabelsPositionOnValue)
        self.axis_y.setTickCount(11)
        self.axis_y.setTitleVisible(False)
        self.chart().addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        self.series.attachAxis(self.axis_y)
        self.axis_y.rangeChanged.connect(self._update_y_ticks)

        if self.errors_log.size > 0:
            y_min = self.errors_log.min()
            y_max = self.errors_log.max()
            pad = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
            self.initial_y_range = (y_min - pad, y_max + pad)
            self.axis_y.setRange(*self.initial_y_range)
        else:
            self.initial_y_range = (-15.0, 0.0)

        # Custom horizontal Y title
        self.axis_y_title = CustomHorizontalTitle("Error", self.chart(), self.axis_y.titleFont())

        logger.debug("RescalePlotWidget %r initialized", title)

    @property
    def default_series_pen(self) -> QPen:
        return QPen(
            self.palette().color(QPalette.ColorRole.BrightText),
            1.0,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.SquareCap,
            Qt.PenJoinStyle.MiterJoin,
        )

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.is_panning:
            return super().mouseMoveEvent(event)

        pos = event.position()
        area = self.chart().plotArea()

        if not area.contains(pos) or self.dims.size <= 0:
            return self.set_overlays_visible(False)

        val = self.chart().mapToValue(pos)

        # Lookup for nearest data point
        idx = self.dims.searchsorted(val.x()).clip(1, self.dims.size - 1)
        best_idx = int(idx if abs(self.dims[idx] - val.x()) < abs(self.dims[idx - 1] - val.x()) else idx - 1)

        if best_idx == self._last_snap_idx and self.v_line.isVisible():
            return

        self._last_snap_idx = best_idx

        x, y = self.dims[best_idx], self.errors[best_idx]
        y_log = self.errors_log[best_idx]
        point = self.chart().mapToPosition(QPointF(x, y_log))

        # Update crosshairs to snap to the data point
        self.v_line.setLine(point.x(), area.top(), point.x(), area.bottom())
        self.h_line.setLine(area.left(), point.y(), area.right(), point.y())

        self.label.set_html_text(f"<b>{self.dimension_mode}:</b> {x:g}<br><b>Error:</b> {y:.10e}", self.palette())

        # Avoid tooltip occlusion at chart edges
        lx, ly = point.x() + 15, point.y() - 45
        if lx + 160 > area.right():  # 160 is estimated tooltip width
            lx = point.x() - 165
        if ly < area.top():
            ly = point.y() + 15

        self.label.setPos(lx, ly)
        self.set_overlays_visible(True)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.axis_y_title.update_from_chart(self.chart())

    def reset_zoom(self) -> None:
        self.axis_x.setRange(*self.initial_x_range)
        self.axis_y.setRange(*self.initial_y_range)

    def serialize_json(self) -> Any:
        return {
            "dimension_mode": self.dimension_mode,
            "data": [{"x": float(x), "y": float(y)} for x, y in zip(self.dims, self.errors)],
        }

    def serialize_csv(self) -> Iterable[Iterable[Any]]:
        yield [f"{self.dimension_mode}", "Error"]
        yield from zip(self.dims, self.errors)

    def _update_y_ticks(self, y_min: float, y_max: float) -> None:
        for label in self.axis_y.categoriesLabels():
            self.axis_y.remove(label)

        if y_min == y_max:
            return

        seen_labels = set[str]()

        for val in np.linspace(y_min, y_max, self.axis_y.tickCount()):
            label = f"{10**val:.3e}"
            # Ensure the label is unique by appending a space if we zoom really far in
            while label in seen_labels:
                label += " "

            seen_labels.add(label)
            self.axis_y.append(label, float(val))


class FrequencyPlotWidget(BasePlotWidget):
    H_PEN = QPen(Qt.GlobalColor.green, 1.0)
    V_PEN = QPen(Qt.GlobalColor.cyan, 1.0)
    GRAY_PEN = QPen(QColor.fromRgbF(0.5, 0.5, 0.5, 0.75))
    TRANSPARENT_COLOR = QColor("transparent")

    def __init__(
        self,
        title: str,
        dct_h: FloatArray1D,
        dct_v: FloatArray1D,
        min_val_h: int,
        max_val_h: int,
        min_val_v: int,
        max_val_v: int,
        check_radius: int = 50,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title, parent)

        chart = self.chart()

        margins = chart.margins()
        margins.setLeft(80)
        chart.setMargins(margins)

        self.gray_pen = QPen(self.GRAY_PEN)
        self.dct_h = np.log10(np.asarray(dct_h, np.float64).clip(1e-4, None))
        self.dct_v = np.log10(np.asarray(dct_v, np.float64).clip(1e-4, None))
        self.min_val_h = min_val_h
        self.max_val_h = max_val_h
        self.min_val_v = min_val_v
        self.max_val_v = max_val_v
        self.check_radius = check_radius
        self._last_snap_idx = -1
        self._last_snap_dim = ""
        self._current_focus: Literal["h", "v", ""] = ""

        y_min = min(self.dct_h.min(), self.dct_v.min())
        y_max = max(self.dct_h.max(), self.dct_v.max())
        y_pad = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
        self.initial_y_range = (y_min - y_pad, y_max + y_pad)
        self.initial_xh_range = (0.0, max(1, len(self.dct_h) - 1))
        self.initial_xv_range = (0.0, max(1, len(self.dct_v) - 1))

        # Horizontal Series
        self.series_h = QLineSeries(self)
        self.series_h.setName("Width")
        self.series_h.appendNp(np.arange(len(self.dct_h), dtype=np.float64), self.dct_h)  # type: ignore[arg-type]
        chart.addSeries(self.series_h)

        # Vertical Series
        self.series_v = QLineSeries(self)
        self.series_v.setName("Height")
        self.series_v.appendNp(np.arange(len(self.dct_v), dtype=np.float64), self.dct_v)  # type: ignore[arg-type]
        chart.addSeries(self.series_v)

        self.h_pen = self.series_h.pen()
        self.v_pen = self.series_v.pen()

        # Spikes Horizontal Series
        self.series_spikes_h = QScatterSeries(
            self,
            color=self.series_h.pen().color(),
            borderColor=self.TRANSPARENT_COLOR,
            markerSize=10,
        )
        self.series_spikes_h.setName("Spikes Width")
        chart.addSeries(self.series_spikes_h)
        self.set_spikes_h()

        # Spikes Vertical Series
        self.series_spikes_v = QScatterSeries(
            self,
            color=self.series_v.pen().color(),
            borderColor=self.TRANSPARENT_COLOR,
            markerSize=10,
        )
        self.series_spikes_v.setName("Spikes Height")
        chart.addSeries(self.series_spikes_v)
        self.set_spikes_v()

        # Horizontal Axis (Bottom)
        self.axis_x_h = QValueAxis(self, labelFormat="%.0f", tickCount=11)
        self.axis_x_h.setTitleText("Width")
        self.axis_x_h.setRange(*self.initial_xh_range)
        chart.addAxis(self.axis_x_h, Qt.AlignmentFlag.AlignBottom)
        self.series_h.attachAxis(self.axis_x_h)
        self.series_spikes_h.attachAxis(self.axis_x_h)

        # Vertical Axis (Bottom)
        self.axis_x_v = QValueAxis(self, labelFormat="%.0f", tickCount=11)
        self.axis_x_v.setTitleText("Height")
        self.axis_x_v.setRange(*self.initial_xv_range)
        chart.addAxis(self.axis_x_v, Qt.AlignmentFlag.AlignBottom)
        self.series_v.attachAxis(self.axis_x_v)
        self.series_spikes_v.attachAxis(self.axis_x_v)

        # Shared Y Axis
        self.axis_y = QValueAxis(self)
        self.axis_y.setTitleVisible(False)
        self.axis_y.setRange(*self.initial_y_range)
        chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        self.series_h.attachAxis(self.axis_y)
        self.series_v.attachAxis(self.axis_y)
        self.series_spikes_h.attachAxis(self.axis_y)
        self.series_spikes_v.attachAxis(self.axis_y)

        self.axis_y_title = CustomHorizontalTitle("DCT Value", chart, self.axis_y.titleFont())

        logger.debug("FrequencyPlotWidget %r initialized", title)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.is_panning:
            return super().mouseMoveEvent(event)

        pos = event.position()
        chart = self.chart()
        area = chart.plotArea()

        if not area.contains(pos):
            self.reset_focus_series()
            self.set_overlays_visible(False)
            self._last_snap_idx = -1
            self._last_snap_dim = ""
            return

        # Map mouse to Width coordinate
        val_h = chart.mapToValue(pos, self.series_h)
        idx_h = int(clamp(val_h.x(), 0, len(self.dct_h) - 1))
        y_log_h = float(self.dct_h[idx_h])

        # Map mouse to Height coordinate
        val_v = chart.mapToValue(pos, self.series_v)
        idx_v = int(clamp(val_v.x(), 0, len(self.dct_v) - 1))
        y_log_v = float(self.dct_v[idx_v])

        # Map mouse Y to logical value
        val_y = chart.mapToValue(pos, self.series_h).y()

        # Decide which one to snap to based on Y distance
        dist_h = abs(y_log_h - val_y)
        dist_v = abs(y_log_v - val_y)

        if dist_h < dist_v:
            dim = "h"
            snap_idx, snap_y = idx_h, y_log_h
            snap_series = self.series_h
            label_text = f"<b>Width:</b> {idx_h} (Val: {10**y_log_h:.10f})"
        else:
            dim = "v"
            snap_idx, snap_y = idx_v, y_log_v
            snap_series = self.series_v
            label_text = f"<b>Height:</b> {idx_v} (Val: {10**y_log_v:.10f})"

        if snap_idx == self._last_snap_idx and dim == self._last_snap_dim and self.v_line.isVisible():
            return

        self._last_snap_idx = snap_idx
        self._last_snap_dim = dim
        self.focus_series(dim)  # type: ignore[arg-type]

        # Snap crosshair to selected point
        point = chart.mapToPosition(QPointF(snap_idx, snap_y), snap_series)
        self.v_line.setLine(point.x(), area.top(), point.x(), area.bottom())
        self.h_line.setLine(area.left(), point.y(), area.right(), point.y())

        self.label.set_html_text(label_text, self.palette())

        lx, ly = point.x() + 15, point.y() - 45
        if lx + 180 > area.right():
            lx = point.x() - 185
        if ly < area.top():
            ly = point.y() + 15
        self.label.setPos(lx, ly)
        self.set_overlays_visible(True)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.axis_y_title.update_from_chart(self.chart())

    def reset_zoom(self) -> None:
        self.axis_x_h.setRange(*self.initial_xh_range)
        self.axis_x_v.setRange(*self.initial_xv_range)
        self.axis_y.setRange(*self.initial_y_range)

    def serialize_json(self) -> Any:
        return {
            "width": [{"idx": i, "val": v} for i, v in enumerate(self.dct_h)],
            "height": [{"idx": i, "val": v} for i, v in enumerate(self.dct_v)],
        }

    def serialize_csv(self) -> Iterable[Iterable[Any]]:
        yield ["Index", "Width (log10)", "Height (log10)"]

        for i in range(max(len(self.dct_h), len(self.dct_v))):
            h = self.dct_h[i] if i < len(self.dct_h) else ""
            v = self.dct_v[i] if i < len(self.dct_v) else ""
            yield [i, h, v]

    def set_spikes_h(self) -> None:
        spikes_h = self._get_spikes(self.dct_h, self.min_val_h, self.max_val_h)

        if len(spikes_h):
            sh_x = spikes_h.astype(np.float64)
            sh_y = self.dct_h[spikes_h].astype(np.float64)
            self.series_spikes_h.replaceNp(sh_x, sh_y)  # type: ignore[arg-type]
        else:
            self.series_spikes_h.clear()

    def set_spikes_v(self) -> None:
        spikes_v = self._get_spikes(self.dct_v, self.min_val_v, self.max_val_v)

        if len(spikes_v):
            sv_x = spikes_v.astype(np.float64)
            sv_y = self.dct_v[spikes_v].astype(np.float64)
            self.series_spikes_v.replaceNp(sv_x, sv_y)  # type: ignore[arg-type]
        else:
            self.series_spikes_v.clear()

    def focus_series(self, dimension: Literal["h", "v", ""], /) -> None:
        if self._current_focus == dimension:
            return

        self._current_focus = dimension
        self.apply_focus()

    def reset_focus_series(self) -> None:
        if self._current_focus == "":
            return

        self._current_focus = ""
        self.apply_focus()

    def apply_focus(self) -> None:
        match self._current_focus:
            case "h":
                h_pen, v_pen = self.h_pen, self.gray_pen
                order = [self.series_v, self.series_spikes_v, self.series_h, self.series_spikes_h]
            case "v":
                h_pen, v_pen = self.gray_pen, self.v_pen
                order = [self.series_h, self.series_spikes_h, self.series_v, self.series_spikes_v]
            case _:
                h_pen, v_pen = self.h_pen, self.v_pen
                order = [self.series_h, self.series_v, self.series_spikes_h, self.series_spikes_v]

        self.series_h.setPen(h_pen)
        self.series_v.setPen(v_pen)

        self.series_spikes_h.setPen(h_pen)
        self.series_spikes_h.setColor(h_pen.color())

        self.series_spikes_v.setPen(v_pen)
        self.series_spikes_v.setColor(v_pen.color())

        # Re-add series to change Z-order (bring focused to front)
        for s in order:
            self.chart().removeSeries(s)
            self.chart().addSeries(s)

        # Re-attach axes after removal
        self.series_h.attachAxis(self.axis_x_h)
        self.series_h.attachAxis(self.axis_y)
        self.series_v.attachAxis(self.axis_x_v)
        self.series_v.attachAxis(self.axis_y)
        self.series_spikes_h.attachAxis(self.axis_x_h)
        self.series_spikes_h.attachAxis(self.axis_y)
        self.series_spikes_v.attachAxis(self.axis_x_v)
        self.series_spikes_v.attachAxis(self.axis_y)

    def _get_spikes(self, dct: np.ndarray, min_v: int, max_v: int) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
        (max_idx,) = argrelextrema(dct, np.less, order=self.check_radius)
        (min_idx,) = argrelextrema(dct, np.greater, order=self.check_radius)

        idx = np.concatenate((max_idx, min_idx))
        return idx[(idx > min_v) & (idx < max_v)]

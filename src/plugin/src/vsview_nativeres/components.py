from jetpytools import CustomNotImplementedError
from PySide6.QtCore import QSize
from PySide6.QtGui import Qt
from PySide6.QtWidgets import (
    QLabel,
    QLayout,
    QListWidget,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from vsview.api import run_in_loop


class TabContainer(QScrollArea):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QScrollArea.Shape.NoFrame)

        self.container = QWidget(self)
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(8, 8, 8, 8)
        self.container_layout.setSpacing(8)

        self.setWidget(self.container)

    def add_section(self, section: QWidget | QLayout, stretch: int = 0) -> None:
        if isinstance(section, QLayout):
            self.container_layout.addLayout(section, stretch)
        else:
            self.container_layout.addWidget(section, stretch)

    def finalize(self) -> None:
        self.container_layout.addStretch(1)

    @staticmethod
    def make_vgroup(title: str, *widgets: QWidget, parent: QWidget, stretch: bool = True) -> QVBoxLayout:
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(8)

        if title:
            vbox.addWidget(QLabel(title, parent))
        for w in widgets:
            vbox.addWidget(w)
        if stretch:
            vbox.addStretch()

        return vbox


class GetNativeImportList(QListWidget):
    def sizeHint(self) -> QSize:
        return QSize(256, 0)


class ProgressBar(QProgressBar):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(24)
        self.setRange(0, 100)
        self.setTextVisible(True)

    @run_in_loop(return_future=False)
    def update_progress(
        self,
        *,
        value: int | None = None,
        range: tuple[int, int] | None = None,
        fmt: str | None = None,
        increment: int | None = None,
    ) -> None:
        if range:
            self.setRange(*range)
        if fmt:
            self.setFormat(fmt)
        if value is not None:
            self.setValue(value)
        if increment is not None:
            self.setValue(self.value() + increment)

    @run_in_loop(return_future=False)
    def reset_progress(self) -> None:
        self.reset()
        self.setFormat("%p%")


class RangeSpinStack(QStackedWidget):
    @property
    def current(self) -> QSpinBox:
        if isinstance(w := self.currentWidget(), QSpinBox):
            return w
        raise CustomNotImplementedError

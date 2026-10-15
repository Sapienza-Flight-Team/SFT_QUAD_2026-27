"""Run with: PYTHONPATH=src python -m tools.config_file_generation.gui"""

from pathlib import Path
from runpy import run_path
import sys

import yaml
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.options import Options
from core.stage_catalog import build_stage_catalog


def make_field(value):
    """Return an editor and a function that reads its current value."""
    if isinstance(value, Options):
        items = value.items
        if isinstance(items, (tuple, dict)):
            choices = list(items.values()) if isinstance(items, dict) else items
            widget = QComboBox()
            widget.addItems(value.alias)
            widget.setCurrentIndex(
                list(items).index(value.default)
                if isinstance(items, dict)
                else value.default
            )
            return widget, lambda: choices[widget.currentIndex()]
        if isinstance(items, range):
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, len(items) - 1)
            slider.setValue(items.index(value.default))
            label = QLabel(str(value.default))
            slider.valueChanged.connect(lambda index: label.setText(str(items[index])))
            layout.addWidget(slider)
            layout.addWidget(label)
            return widget, lambda: items[slider.value()]
        value = value.get_default()

    if isinstance(value, dict):
        widget = ConfigForm(value)
        return widget, widget.values
    if isinstance(value, bool):
        widget = QCheckBox()
        widget.setChecked(value)
        return widget, widget.isChecked
    if isinstance(value, (str, int, float)):
        widget = QLineEdit(str(value))
        return widget, lambda: type(value)(widget.text())
    widget = QLineEdit(
        yaml.safe_dump(value, default_flow_style=True).strip().removesuffix("\n...")
    )
    return widget, lambda: yaml.safe_load(widget.text())


class ConfigForm(QWidget):
    def __init__(self, definition):
        super().__init__()
        self.readers = {}
        layout = QFormLayout(self)
        for name, value in definition.items():
            widget, self.readers[name] = make_field(value)
            widget.setObjectName(str(name))
            layout.addRow(str(name), widget)

    def values(self):
        return {name: read() for name, read in self.readers.items()}


class ConfigFormStack(QStackedWidget):
    """Size the stack using only the selected implementation's form."""

    def __init__(self):
        super().__init__()
        self.currentChanged.connect(self.updateGeometry)

    def sizeHint(self):
        widget = self.currentWidget()
        return widget.sizeHint().expandedTo(QSize(0, 0)) if widget else QSize(0, 0)

    def minimumSizeHint(self):
        widget = self.currentWidget()
        return (
            widget.minimumSizeHint().expandedTo(QSize(0, 0)) if widget else QSize(0, 0)
        )


class ConfigEditor(QWidget):
    def __init__(self, catalog):
        super().__init__()
        self.stages = {}
        self.setWindowTitle("Pipeline configuration")
        self.resize(800, 700)
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        stage_layout = QVBoxLayout(content)
        for stage, implementations in catalog.items():
            group = QGroupBox(stage)
            group.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
            )
            group_layout = QVBoxLayout(group)
            selector = QComboBox()
            selector.addItem("none", None)
            forms = ConfigFormStack()
            forms.addWidget(QWidget())
            for implementation, entry in implementations.items():
                selector.addItem(implementation, implementation)
                forms.addWidget(ConfigForm(run_path(str(entry["config"]))["CONFIG"]))
            selector.currentIndexChanged.connect(forms.setCurrentIndex)
            group_layout.addWidget(selector)
            group_layout.addWidget(forms)
            stage_layout.addWidget(group)
            self.stages[stage] = selector, forms
        stage_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        save = QPushButton("Save YAML")
        save.clicked.connect(self.save)
        layout.addWidget(save)

    def configuration(self):
        return [
            {
                stage: [
                    {
                        selector.currentData(): [
                            {name: value}
                            for name, value in forms.currentWidget().values().items()
                        ]
                    }
                ]
            }
            for stage, (selector, forms) in self.stages.items()
            if selector.currentData() is not None
        ]

    def save(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save configuration", "stage_config.yaml", "YAML (*.yaml *.yml)"
        )
        if filename:
            Path(filename).write_text(
                yaml.safe_dump(self.configuration(), sort_keys=False), encoding="utf-8"
            )


def main():
    app = QApplication(sys.argv)
    editor = ConfigEditor(build_stage_catalog())
    editor.show()
    app.exec()


if __name__ == "__main__":
    main()

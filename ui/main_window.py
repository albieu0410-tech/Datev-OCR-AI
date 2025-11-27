from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QApplication,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from PIL.ImageQt import ImageQt

from .. import core
from ..automation import AutomationController
from ..config import load_config, save_config
from ..core import CFG_FILE, DEFAULTS


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    kind: str = "text"  # text, path, int, float, bool
    tooltip: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    decimals: int = 2
    dialog: str | None = None  # "file" or "directory"
    dialog_filter: str | None = None
    placeholder: str | None = None
    options: Sequence[tuple[str, str]] | None = None


OCR_ENGINE_OPTIONS: Sequence[tuple[str, str]] = (
    ("tesseract", "Tesseract"),
    ("paddle", "PaddleOCR"),
)


CAL_POINT_ACTIONS = [
    (
        "search_point",
        "Search Point",
        "Position your mouse over the DATEV search field inside the RDP window.",
    ),
    (
        "pdf_search_point",
        "PDF Search Point",
        "Position your mouse over the PDF search box.",
    ),
    (
        "pdf_hits_point",
        "Primary PDF Result Button",
        "Position your mouse over the primary PDF result button.",
    ),
    (
        "pdf_hits_second_point",
        "Secondary PDF Result Button",
        "Position your mouse over the secondary PDF result button (if available).",
    ),
    (
        "pdf_hits_third_point",
        "Tertiary PDF Result Button",
        "Position your mouse over the third PDF result button (if available).",
    ),
    (
        "doc_view_point",
        "View Button",
        "Position your mouse over the 'View' button for documents.",
    ),
    (
        "pdf_close_point",
        "PDF Close Button",
        "Position your mouse over the close button of the PDF viewer.",
    ),
    (
        "sw_gg_close_point",
        "SW GG Close Button",
        "Position your mouse over the close button for the Stammdaten window.",
    ),
]

CAL_BOX_ACTIONS = [
    (
        "result_region",
        "Result Region",
        "Position your mouse over the TOP-LEFT of the result area.",
        "Now position your mouse over the BOTTOM-RIGHT of the result area.",
    ),
    (
        "doclist_region",
        "Doc List Region",
        "Position your mouse over the TOP-LEFT of the document list.",
        "Now position your mouse over the BOTTOM-RIGHT of the document list.",
    ),
    (
        "pdf_text_region",
        "PDF Text Region",
        "Position your mouse over the TOP-LEFT of the PDF text area.",
        "Now position your mouse over the BOTTOM-RIGHT of the PDF text area.",
    ),
    (
        "rechnungen_region",
        "Rechnungen Region",
        "Position your mouse over the TOP-LEFT of the Rechnungen area.",
        "Now position your mouse over the BOTTOM-RIGHT of the Rechnungen area.",
    ),
    (
        "rechnungen_gg_region",
        "GG Region",
        "Position your mouse over the TOP-LEFT of the GG area.",
        "Now position your mouse over the BOTTOM-RIGHT of the GG area.",
    ),
    (
        "fees_file_search_region",
        "Fees File Search Region",
        "Position your mouse over the TOP-LEFT of the Fees file search strip.",
        "Now position your mouse over the BOTTOM-RIGHT of the Fees file search strip.",
    ),
    (
        "fees_seiten_region",
        "Fees Seiten Region",
        "Position your mouse over the TOP-LEFT of the Seiten region.",
        "Now position your mouse over the BOTTOM-RIGHT of the Seiten region.",
    ),
    (
        "akten_document_filter_region",
        "Akten Document Filter Region",
        "Position your mouse over the TOP-LEFT of the Akten document filter.",
        "Now position your mouse over the BOTTOM-RIGHT of the Akten document filter.",
    ),
    (
        "instance_region",
        "Instance Table Region",
        "Position your mouse over the TOP-LEFT of the instance table.",
        "Now position your mouse over the BOTTOM-RIGHT of the instance table.",
    ),
    (
        "sw_gg_region",
        "SW GG Region",
        "Position your mouse over the TOP-LEFT of the Stammdaten SW GG text area.",
        "Now position your mouse over the BOTTOM-RIGHT of the Stammdaten SW GG text area.",
    ),
]


class WorkerSignals(QObject):
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    progress = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()


class CountdownDialog(QDialog):
    def __init__(self, title: str, message_template: str, seconds: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.label = QLabel("", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        self._message_template = message_template
        self._remaining = max(1, seconds)
        self._update_text()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)

    def _update_text(self):
        try:
            text = self._message_template.format(n=self._remaining)
        except Exception:
            text = self._message_template
        self.label.setText(text)

    def _tick(self):
        self._remaining -= 1
        if self._remaining <= 0:
            self._timer.stop()
            self.accept()
            return
        self._update_text()


class TaskWorker(QRunnable):
    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = dict(kwargs)
        self.signals = WorkerSignals()

    def run(self) -> None:  # pragma: no cover - background thread
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            self.signals.error.emit(str(exc))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


GENERAL_SECTIONS: Sequence[tuple[str, Sequence[FieldSpec]]] = (
    (
        "RDP Connection",
        (
            FieldSpec("rdp_title_regex", "Window Title Regex"),
            FieldSpec(
                "type_delay",
                "Type Delay (s)",
                kind="float",
                minimum=0.0,
                maximum=5.0,
                decimals=3,
            ),
            FieldSpec(
                "post_search_wait",
                "Post-search Wait (s)",
                kind="float",
                minimum=0.0,
                maximum=10.0,
                decimals=2,
            ),
            FieldSpec(
                "search_point",
                "Search Point (rel x, y)",
                tooltip="Comma-separated relative coordinates",
            ),
            FieldSpec(
                "result_region",
                "Result Region (rel l, t, w, h)",
                tooltip="Comma-separated relative rectangle",
            ),
        ),
    ),
    (
        "Excel Input",
        (
            FieldSpec(
                "excel_path",
                "Workbook Path",
                kind="path",
                dialog="file",
                dialog_filter="Excel Files (*.xlsx *.xls)",
            ),
            FieldSpec("excel_sheet", "Sheet Name/Index"),
            FieldSpec("input_column", "Input Column"),
            FieldSpec("start_cell", "Start Cell"),
            FieldSpec(
                "max_rows",
                "Max Rows",
                kind="int",
                minimum=0,
                maximum=10_000,
            ),
            FieldSpec(
                "results_csv",
                "Results CSV",
                kind="path",
                dialog="file",
                dialog_filter="CSV Files (*.csv)",
            ),
        ),
    ),
    (
        "OCR Settings",
        (
            FieldSpec(
                "tesseract_path",
                "Tesseract Binary",
                kind="path",
                dialog="file",
            ),
            FieldSpec("tesseract_lang", "Languages"),
            FieldSpec(
                "upscale_x",
                "Upscale Factor",
                kind="int",
                minimum=1,
                maximum=10,
            ),
            FieldSpec("keyword", "Keyword"),
            FieldSpec("typing_test_text", "Typing Test Text"),
            FieldSpec("color_ocr", "Enable Color OCR", kind="bool"),
            FieldSpec(
                "use_full_region_parse",
                "Use Full-region Parsing",
                kind="bool",
            ),
            FieldSpec("normalize_ocr", "Normalize OCR Output", kind="bool"),
        ),
    ),
)

CALIBRATION_SECTIONS: Sequence[tuple[str, Sequence[FieldSpec]]] = (
    (
        "Primary Regions",
        (
            FieldSpec("result_region", "Result Region"),
            FieldSpec("doclist_region", "Doc List Region"),
            FieldSpec("pdf_text_region", "PDF Text Region"),
            FieldSpec("instance_region", "Instance Table Region"),
        ),
    ),
    (
        "Streitwert Controls",
        (
            FieldSpec("pdf_search_point", "PDF Search Point"),
            FieldSpec("pdf_hits_point", "Primary Hits Button"),
            FieldSpec("pdf_hits_second_point", "Secondary Hits Button"),
            FieldSpec("pdf_hits_third_point", "Tertiary Hits Button"),
            FieldSpec("pdf_close_point", "PDF Close Button"),
            FieldSpec("doc_view_point", "Document View Button"),
        ),
    ),
    (
        "Fees Calibration",
        (
            FieldSpec("fees_file_search_region", "File-search Region"),
            FieldSpec("fees_seiten_region", "Seiten Region"),
        ),
    ),
)

STREITWERT_SECTIONS: Sequence[tuple[str, Sequence[FieldSpec]]] = (
    (
        "Filtering",
        (
            FieldSpec("includes", "Include Tokens"),
            FieldSpec("excludes", "Exclude Tokens"),
            FieldSpec(
                "exclude_prefix_k",
                "Exclude Entries Starting With 'K'",
                kind="bool",
            ),
            FieldSpec("streitwert_term", "PDF Search Term"),
        ),
    ),
    (
        "Timing",
        (
            FieldSpec(
                "doc_open_wait",
                "Doc Open Wait (s)",
                kind="float",
                minimum=0.0,
                maximum=10.0,
                decimals=2,
            ),
            FieldSpec(
                "pdf_hit_wait",
                "PDF Hit Wait (s)",
                kind="float",
                minimum=0.0,
                maximum=10.0,
                decimals=2,
            ),
            FieldSpec(
                "pdf_view_extra_wait",
                "PDF View Extra Wait (s)",
                kind="float",
                minimum=0.0,
                maximum=10.0,
                decimals=2,
            ),
            FieldSpec(
                "streitwert_overlay_skip_waits",
                "Skip manual waits when overlay detected",
                kind="bool",
            ),
            FieldSpec(
                "ignore_top_doc_row",
                "Ignore top Streitwert match",
                kind="bool",
            ),
        ),
    ),
    (
        "Outputs",
        (
            FieldSpec(
                "streitwert_results_csv",
                "Results CSV",
                kind="path",
                dialog="file",
                dialog_filter="CSV Files (*.csv)",
            ),
            FieldSpec(
                "log_extract_results_csv",
                "Log Extract CSV",
                kind="path",
                dialog="file",
                dialog_filter="CSV Files (*.csv)",
            ),
        ),
    ),
)

RECHNUNGEN_SECTIONS: Sequence[tuple[str, Sequence[FieldSpec]]] = (
    (
        "Regions",
        (
            FieldSpec("rechnungen_region", "Primary Region"),
            FieldSpec("rechnungen_gg_region", "GG Region"),
        ),
    ),
    (
        "Timing",
        (
            FieldSpec(
                "rechnungen_search_wait",
                "Search Wait (s)",
                kind="float",
                minimum=0.0,
                maximum=10.0,
                decimals=2,
            ),
            FieldSpec(
                "rechnungen_region_wait",
                "Region Wait (s)",
                kind="float",
                minimum=0.0,
                maximum=10.0,
                decimals=2,
            ),
            FieldSpec(
                "rechnungen_overlay_skip_waits",
                "Skip waits on overlay",
                kind="bool",
            ),
        ),
    ),
    (
        "Outputs",
        (
            FieldSpec(
                "rechnungen_results_csv",
                "Rechnungen CSV",
                kind="path",
                dialog="file",
                dialog_filter="CSV Files (*.csv)",
            ),
            FieldSpec(
                "rechnungen_only_results_csv",
                "Rechnungen-only CSV",
                kind="path",
                dialog="file",
                dialog_filter="CSV Files (*.csv)",
            ),
            FieldSpec(
                "rechnungen_gg_results_csv",
                "GG Results CSV",
                kind="path",
                dialog="file",
                dialog_filter="CSV Files (*.csv)",
            ),
        ),
    ),
)

FEES_SECTIONS: Sequence[tuple[str, Sequence[FieldSpec]]] = (
    (
        "Options",
        (
            FieldSpec("fees_search_token", "Search Token"),
            FieldSpec("fees_bad_prefixes", "Bad Prefixes"),
            FieldSpec(
                "fees_pages_max_clicks",
                "Max Page Clicks",
                kind="int",
                minimum=1,
                maximum=100,
            ),
            FieldSpec(
                "fees_overlay_skip_waits",
                "Skip waits based on overlay",
                kind="bool",
            ),
        ),
    ),
    (
        "Outputs",
        (
            FieldSpec(
                "fees_csv_path",
                "Fees CSV",
                kind="path",
                dialog="file",
                dialog_filter="CSV Files (*.csv)",
            ),
        ),
    ),
)

AKTEN_SECTIONS: Sequence[tuple[str, Sequence[FieldSpec]]] = (
        (
            "Document Filter",
            (
                FieldSpec("akten_document_filter_region", "Filter Region"),
                FieldSpec("akten_search_term", "Search Term"),
                FieldSpec("akten_ignore_tokens", "Ignore Tokens"),
                FieldSpec(
                    "akten_results_csv",
                    "Results CSV",
                    kind="path",
                    dialog="file",
                    dialog_filter="CSV Files (*.csv)",
                    tooltip="Path where Akten workflow results will be written.",
                ),
                FieldSpec(
                    "akten_filter_term",
                    "Filter Typing Term",
                    tooltip="Optional text to type directly into the Akten filter field before searching.",
                ),
            FieldSpec(
                "akten_filter_wait",
                "Filter Wait (s)",
                kind="float",
                minimum=0.0,
                maximum=5.0,
                decimals=2,
                tooltip="Delay after typing into the filter field to allow OCR to pick up the new text.",
            ),
        ),
    ),
)

LOG_SECTIONS: Sequence[tuple[str, Sequence[FieldSpec]]] = (
    (
        "Storage",
        (
            FieldSpec(
                "log_folder",
                "Log Folder",
                kind="path",
                dialog="directory",
            ),
            FieldSpec(
                "log_extract_results_csv",
                "Extract CSV",
                kind="path",
                dialog="file",
                dialog_filter="CSV Files (*.csv)",
            ),
        ),
    ),
)

PROFILE_SECTIONS: Sequence[tuple[str, Sequence[FieldSpec]]] = (
    (
        "Profiles",
        (
            FieldSpec(
                "use_amount_profile",
                "Restrict to amount profile",
                kind="bool",
            ),
            FieldSpec("active_amount_profile", "Active Profile"),
            FieldSpec(
                "amount_profiles",
                "Profiles JSON",
                kind="text",
                tooltip="Serialized representation of configured profiles.",
            ),
            FieldSpec(
                "program_ocr_engine",
                "Program OCR Engine",
                options=OCR_ENGINE_OPTIONS,
                tooltip="Engine for interface/doc list OCR (restart may be required when switching).",
            ),
            FieldSpec(
                "program_ocr_lang",
                "Program OCR Language",
                tooltip="OCR language(s) for interface/doc list recognition.",
            ),
            FieldSpec(
                "document_ocr_engine",
                "Document OCR Engine",
                options=OCR_ENGINE_OPTIONS,
                tooltip="Engine for PDF/document extraction.",
            ),
            FieldSpec(
                "document_ocr_lang",
                "Document OCR Language",
                tooltip="OCR language(s) for PDF/document extraction.",
            ),
        ),
    ),
)

GG_SECTIONS: Sequence[tuple[str, Sequence[FieldSpec]]] = (
    (
        "Workflow",
        (
            FieldSpec(
                "sw_gg_results_csv",
                "Results CSV",
                kind="path",
                dialog="file",
                dialog_filter="CSV Files (*.csv)",
            ),
            FieldSpec(
                "sw_gg_keyword",
                "Keyword",
                tooltip="Line fragment that must be present when extracting the Wert amount.",
            ),
            FieldSpec(
                "sw_gg_value_prefix",
                "Value Prefix",
                tooltip="Prefix that appears before the desired amount (e.g., 'Wert:').",
            ),
            FieldSpec(
                "sw_gg_open_wait",
                "Open Wait (s)",
                kind="float",
                minimum=0.0,
                maximum=999.0,
                decimals=2,
                tooltip="Delay after opening Stammdaten before capturing.",
            ),
            FieldSpec(
                "sw_gg_capture_wait",
                "Capture Wait (s)",
                kind="float",
                minimum=0.0,
                maximum=999.0,
                decimals=2,
                tooltip="Delay before capturing screenshot (after window opens).",
            ),
            FieldSpec(
                "sw_gg_close_wait",
                "Close Wait (s)",
                kind="float",
                minimum=0.0,
                maximum=999.0,
                decimals=2,
                tooltip="Delay after closing Stammdaten before moving to the next invoice.",
            ),
        ),
    ),
)

TAB_DEFINITION: Sequence[tuple[str, Sequence[tuple[str, Sequence[FieldSpec]]]]] = (
    ("General", GENERAL_SECTIONS),
    ("Calibration", CALIBRATION_SECTIONS),
    ("Streitwert", STREITWERT_SECTIONS),
    ("Rechnungen", RECHNUNGEN_SECTIONS),
    ("Fees", FEES_SECTIONS),
    ("Akten", AKTEN_SECTIONS),
    ("GG", GG_SECTIONS),
    ("Logs", LOG_SECTIONS),
    ("OCR & Profiles", PROFILE_SECTIONS),
)


class ConfigBinder:
    def __init__(self, config: Mapping[str, Any] | MutableMapping[str, Any]):
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = dict(config)
        self._updaters: dict[str, list[Callable[[Any], None]]] = {}

    def _set(self, key: str, value: Any) -> None:
        self.config[key] = value

    def _register(self, key: str, updater: Callable[[Any], None]) -> None:
        self._updaters.setdefault(key, []).append(updater)

    def refresh(self) -> None:
        for key, updaters in self._updaters.items():
            value = self.config.get(key)
            for update in updaters:
                update(value)

    @staticmethod
    def _with_blocked_signals(widget, setter):
        widget.blockSignals(True)
        try:
            setter()
        finally:
            widget.blockSignals(False)

    def bind_line_edit(self, key: str, widget: QLineEdit) -> QLineEdit:
        widget.setText(self._text_value(self.config.get(key)))
        widget.textChanged.connect(lambda text, k=key: self._set(k, text))
        self._register(
            key,
            lambda value, w=widget: self._with_blocked_signals(
                w, lambda: w.setText(self._text_value(value))
            ),
        )
        return widget

    def bind_checkbox(self, key: str, widget: QCheckBox) -> QCheckBox:
        widget.setChecked(bool(self.config.get(key)))
        widget.toggled.connect(lambda checked, k=key: self._set(k, bool(checked)))
        self._register(
            key,
            lambda value, w=widget: self._with_blocked_signals(
                w, lambda: w.setChecked(bool(value))
            ),
        )
        return widget

    def bind_combo_box(self, key: str, widget: QComboBox) -> QComboBox:
        def _index_for(value):
            for idx in range(widget.count()):
                if widget.itemData(idx) == value:
                    return idx
            return -1

        current_value = self.config.get(key)
        index = _index_for(current_value)
        widget.setCurrentIndex(index if index >= 0 else 0)

        def _on_change(idx):
            self._set(key, widget.itemData(idx))

        widget.currentIndexChanged.connect(_on_change)

        def _update(value):
            self._with_blocked_signals(
                widget,
                lambda: widget.setCurrentIndex(
                    _index_for(value) if _index_for(value) >= 0 else 0
                ),
            )

        self._register(key, _update)
        return widget

    def bind_spin_box(
        self, key: str, widget: QSpinBox, *, value_type: type[int] | type[float] = int
    ) -> QSpinBox:
        value = self.config.get(key)
        if value is not None:
            widget.setValue(int(value))
        widget.valueChanged.connect(lambda val, k=key: self._set(k, value_type(val)))
        self._register(
            key,
            lambda v, w=widget: self._with_blocked_signals(
                w, lambda: w.setValue(int(v) if v is not None else w.value())
            ),
        )
        return widget

    def bind_double_spin_box(
        self, key: str, widget: QDoubleSpinBox
    ) -> QDoubleSpinBox:
        value = self.config.get(key)
        if value is not None:
            widget.setValue(float(value))
        widget.valueChanged.connect(lambda val, k=key: self._set(k, float(val)))
        self._register(
            key,
            lambda v, w=widget: self._with_blocked_signals(
                w, lambda: w.setValue(float(v) if v is not None else w.value())
            ),
        )
        return widget

    @staticmethod
    def _text_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return ", ".join(str(part) for part in value)
        return str(value)


class MainWindow(QMainWindow):
    def __init__(self, *, config: Mapping[str, Any] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("DATEV OCR Automation (PyQt6)")
        self.resize(1280, 860)
        self.cfg = dict(config) if config is not None else load_config()
        self.binder = ConfigBinder(self.cfg)
        self.thread_pool = QThreadPool(self)
        self.automation = AutomationController(self.binder.config)
        self.preview_image = None
        self._build_ui()

    # --- UI construction -------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)
        self.tabs = QTabWidget()
        for title, sections in TAB_DEFINITION:
            tab_widget, tab_layout = self._build_tab(sections)
            if title == "OCR & Profiles":
                self._augment_ocr_tab(tab_layout)
            elif title == "Streitwert":
                self._augment_streitwert_tab(tab_layout)
            elif title == "Calibration":
                self._augment_calibration_tab(tab_layout)
            elif title == "Logs":
                self._augment_log_tab(tab_layout)
            elif title == "Rechnungen":
                self._augment_rechnungen_tab(tab_layout)
            elif title == "Fees":
                self._augment_fees_tab(tab_layout)
            elif title == "Akten":
                self._augment_akten_tab(tab_layout)
            elif title == "GG":
                self._augment_sw_gg_tab(tab_layout)
            elif title == "Streitwert":
                self._augment_streitwert_tab(tab_layout)
            tab_layout.addStretch(1)
            self.tabs.addTab(tab_widget, title)
        left_layout.addWidget(self.tabs)
        left_layout.addWidget(self._build_action_bar())
        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.addWidget(self._build_preview_group())
        right_layout.addWidget(self._build_log_group())
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

    def _build_tab(
        self, sections: Sequence[tuple[str, Sequence[FieldSpec]]]
    ) -> tuple[QScrollArea, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)
        for title, fields in sections:
            layout.addWidget(self._build_section(title, fields))
        scroll.setWidget(container)
        return scroll, layout

    def _augment_ocr_tab(self, layout: QVBoxLayout) -> None:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        test_full = QPushButton("Test Parse (Full Region)")
        test_full.clicked.connect(self.handle_test_parse_full)
        test_profile = QPushButton("Test Parse (Profile Region)")
        test_profile.clicked.connect(self.handle_test_parse_profile)
        row_layout.addWidget(test_full)
        row_layout.addWidget(test_profile)
        row_layout.addStretch(1)
        layout.addWidget(row)

    def _augment_streitwert_tab(self, layout: QVBoxLayout) -> None:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        run_btn = QPushButton("Run Streitwert")
        run_btn.clicked.connect(self.handle_run_streitwert)
        run_with_rechn_btn = QPushButton("Run Streitwert + Rechnungen")
        run_with_rechn_btn.clicked.connect(self.handle_run_streitwert_with_rechnungen)
        capture_doc_btn = QPushButton("Capture Doclist Snapshot")
        capture_doc_btn.clicked.connect(self.handle_capture_doclist)
        row_layout.addWidget(run_btn)
        row_layout.addWidget(run_with_rechn_btn)
        row_layout.addWidget(capture_doc_btn)
        row_layout.addStretch(1)
        layout.addWidget(row)

    def _augment_rechnungen_tab(self, layout: QVBoxLayout) -> None:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        run_rechnungen_btn = QPushButton("Run Rechnungen Only")
        run_rechnungen_btn.clicked.connect(self.handle_run_rechnungen_only)
        run_gg_btn = QPushButton("Run GG Extraction")
        run_gg_btn.clicked.connect(self.handle_run_rechnungen_gg)
        row_layout.addWidget(run_rechnungen_btn)
        row_layout.addWidget(run_gg_btn)
        row_layout.addStretch(1)
        layout.addWidget(row)

    def _augment_akten_tab(self, layout: QVBoxLayout) -> None:
        cal_row = QWidget()
        cal_layout = QHBoxLayout(cal_row)
        cal_layout.setContentsMargins(0, 0, 0, 0)
        cal_layout.setSpacing(8)
        pick_btn = QPushButton("Pick Filter Region")
        pick_btn.clicked.connect(self.handle_pick_akten_filter_region)
        cal_layout.addWidget(pick_btn)
        cal_layout.addStretch(1)
        layout.addWidget(cal_row)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        run_btn = QPushButton("Run Akten")
        run_btn.clicked.connect(self.handle_run_akten)
        row_layout.addWidget(run_btn)
        run_filtered_btn = QPushButton("Run Akten (Filtered)")
        run_filtered_btn.clicked.connect(self.handle_run_akten_filtered)
        row_layout.addWidget(run_filtered_btn)
        test_btn = QPushButton("Test Akten Setup")
        test_btn.clicked.connect(self.handle_test_akten_setup)
        row_layout.addWidget(test_btn)
        capture_btn = QPushButton("Capture Akten Doclist")
        capture_btn.clicked.connect(self.handle_capture_akten_doclist)
        row_layout.addWidget(capture_btn)
        row_layout.addStretch(1)
        layout.addWidget(row)

    def _augment_sw_gg_tab(self, layout: QVBoxLayout) -> None:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        run_btn = QPushButton("Run SW GG Extraction")
        run_btn.clicked.connect(self.handle_run_sw_gg)
        test_btn = QPushButton("Test SW GG Setup")
        test_btn.clicked.connect(self.handle_test_sw_gg)
        line_btn = QPushButton("Test SW GG Line Preview")
        line_btn.clicked.connect(self.handle_test_sw_gg_line_preview)
        row_layout.addWidget(run_btn)
        row_layout.addWidget(test_btn)
        row_layout.addWidget(line_btn)
        row_layout.addStretch(1)
        layout.addWidget(row)

    def _augment_calibration_tab(self, layout: QVBoxLayout) -> None:
        point_box = QGroupBox("Point Pickers")
        point_layout = QHBoxLayout(point_box)
        point_layout.setContentsMargins(6, 6, 6, 6)
        point_layout.setSpacing(6)
        for key, label, prompt in CAL_POINT_ACTIONS:
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _, cfg_key=key, lbl=label, msg=prompt: self.handle_pick_point(cfg_key, lbl, msg)
            )
            point_layout.addWidget(btn)
        point_layout.addStretch(1)
        layout.addWidget(point_box)

        region_box = QGroupBox("Region Pickers")
        region_layout = QHBoxLayout(region_box)
        region_layout.setContentsMargins(6, 6, 6, 6)
        region_layout.setSpacing(6)
        for key, label, msg1, msg2 in CAL_BOX_ACTIONS:
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _, cfg_key=key, lbl=label, a=msg1, b=msg2: self.handle_pick_region(cfg_key, lbl, a, b)
            )
            region_layout.addWidget(btn)
        region_layout.addStretch(1)
        layout.addWidget(region_box)

    def _augment_log_tab(self, layout: QVBoxLayout) -> None:
        button = QPushButton("Extract Streitwert from Logs")
        button.clicked.connect(self.handle_run_log_extraction)
        layout.addWidget(button)
        layout.addStretch(1)

    def _augment_fees_tab(self, layout: QVBoxLayout) -> None:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        run_btn = QPushButton("Run Fees")
        run_btn.clicked.connect(self.handle_run_fees)
        row_layout.addWidget(run_btn)
        row_layout.addStretch(1)
        layout.addWidget(row)

    def _ensure_rdp_connection(self) -> bool:
        try:
            if not self.automation.current_rect:
                self.automation.connect_rdp()
            return bool(self.automation.current_rect)
        except Exception as exc:
            self.log_message(f"RDP connection failed: {exc}")
            QMessageBox.warning(self, "RDP Connection", f"Failed to connect:\n{exc}")
            return False

    def _show_capture_countdown(self, title: str, prompt: str, seconds: int = 3) -> None:
        dialog = CountdownDialog(title, f"{prompt}\nCapturing in {{n}}…", seconds, self)
        dialog.exec()

    def _capture_absolute_point(self, title: str, prompt: str, seconds: int = 3):
        if not self._ensure_rdp_connection():
            return None
        QMessageBox.information(
            self,
            title,
            f"{prompt}\n\nClick OK when ready. Countdown starts immediately.",
        )
        self._show_capture_countdown(title, prompt, seconds)
        return self.automation.get_mouse_position()

    def _capture_relative_point(self, title: str, prompt: str, seconds: int = 3):
        abs_point = self._capture_absolute_point(title, prompt, seconds=seconds)
        if abs_point is None:
            return None
        rect = self.automation.current_rect
        if not rect:
            return None
        return core.abs_to_rel(rect, abs_point=abs_point)

    def _capture_relative_box(
        self, title: str, prompt1: str, prompt2: str, seconds: int = 3
    ):
        first = self._capture_absolute_point(f"{title} (Step 1)", prompt1, seconds=seconds)
        if first is None:
            return None
        second = self._capture_absolute_point(
            f"{title} (Step 2)", prompt2, seconds=seconds
        )
        if second is None:
            return None
        x1, y1 = first
        x2, y2 = second
        left, top = min(x1, x2), min(y1, y2)
        width, height = abs(x2 - x1), abs(y2 - y1)
        rect = self.automation.current_rect
        if not rect:
            return None
        rel_box = core.abs_to_rel(rect, abs_box=(left, top, width, height))
        return rel_box

    def handle_pick_point(self, config_key: str, label: str, prompt: str) -> None:
        self._sync_controller_config()
        rel = self._capture_relative_point(label, prompt)
        if rel is None:
            return
        self.binder.config[config_key] = rel
        self.binder.refresh()
        self.log_message(
            f"{label} set to ({rel[0]:.3f}, {rel[1]:.3f}). Save the configuration to persist."
        )

    def handle_pick_region(
        self, config_key: str, label: str, prompt1: str, prompt2: str
    ) -> None:
        self._sync_controller_config()
        rel_box = self._capture_relative_box(label, prompt1, prompt2)
        if rel_box is None:
            return
        self.binder.config[config_key] = rel_box
        self.binder.refresh()
        self.log_message(
            f"{label} set to ({rel_box[0]:.3f}, {rel_box[1]:.3f}, {rel_box[2]:.3f}, {rel_box[3]:.3f}). Save the configuration to persist."
        )

    def handle_run_log_extraction(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "Log Extraction",
            self.automation.run_log_extraction,
            on_success=self._on_log_extract_success,
            progress_handler=self.log_message,
        )

    def handle_pick_akten_filter_region(self) -> None:
        self.handle_pick_region(
            "akten_document_filter_region",
            "Akten Document Filter Region",
            "Position your mouse over the TOP-LEFT of the Akten document filter.",
            "Now position your mouse over the BOTTOM-RIGHT of the Akten document filter.",
        )

    def handle_run_fees(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "Fees Extraction",
            self.automation.run_fees,
            on_success=self._on_fees_success,
            progress_handler=self.log_message,
        )

    def _build_section(self, title: str, fields: Sequence[FieldSpec]) -> QGroupBox:
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for spec in fields:
            widget, label_override = self._create_field_widget(spec)
            label_text = label_override if label_override is not None else (
                "" if spec.kind == "bool" else spec.label
            )
            if spec.kind == "bool":
                form.addRow("", widget)
            else:
                form.addRow(label_text, widget)
        return group

    def _build_action_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        save_btn = QPushButton("Save Config")
        save_btn.clicked.connect(self.handle_save_config)
        load_btn = QPushButton("Load Config")
        load_btn.clicked.connect(self.handle_load_config)
        reset_btn = QPushButton("Reset Defaults")
        reset_btn.clicked.connect(self.handle_reset_config)
        connect_btn = QPushButton("Connect RDP")
        connect_btn.clicked.connect(self.handle_connect_rdp)
        capture_btn = QPushButton("Capture Result Region")
        capture_btn.clicked.connect(self.handle_capture_result)
        capture_rechnungen_btn = QPushButton("Capture Rechnungen List")
        capture_rechnungen_btn.clicked.connect(self.handle_capture_rechnungen)
        test_typing_btn = QPushButton("Test Typing")
        test_typing_btn.clicked.connect(self.handle_test_typing)
        run_btn = QPushButton("Run Batch")
        run_btn.clicked.connect(self.handle_run_batch)
        layout.addWidget(save_btn)
        layout.addWidget(load_btn)
        layout.addWidget(reset_btn)
        layout.addWidget(connect_btn)
        layout.addWidget(capture_btn)
        layout.addWidget(capture_rechnungen_btn)
        layout.addWidget(test_typing_btn)
        layout.addStretch(1)
        layout.addWidget(run_btn)
        return bar

    def _build_preview_group(self) -> QGroupBox:
        group = QGroupBox("Preview")
        layout = QVBoxLayout(group)
        self.preview_label = QLabel("Capture a region to preview OCR input.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(220)
        self.preview_label.setStyleSheet(
            "border: 1px solid #999; background-color: #1f1f1f; color: #f0f0f0;"
        )
        layout.addWidget(self.preview_label)
        return group

    def _build_log_group(self) -> QGroupBox:
        group = QGroupBox("Log")
        layout = QVBoxLayout(group)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        layout.addWidget(self.log_output)
        button_bar = QWidget()
        button_layout = QHBoxLayout(button_bar)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)
        copy_btn = QPushButton("Copy Log")
        copy_btn.clicked.connect(self.handle_copy_log)
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.handle_clear_log)
        button_layout.addWidget(copy_btn)
        button_layout.addWidget(clear_btn)
        button_layout.addStretch(1)
        layout.addWidget(button_bar)
        return group

    # --- Automation hooks ------------------------------------------------
    def _sync_controller_config(self) -> None:
        self.automation.update_config(self.binder.config)

    def handle_connect_rdp(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "Connect RDP",
            self.automation.connect_rdp,
            on_success=self._on_connect_success,
        )

    def handle_capture_result(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "Capture Result Region",
            self.automation.capture_result_region,
            on_success=self._on_capture_success,
        )

    def handle_capture_rechnungen(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "Capture Rechnungen List",
            self.automation.capture_rechnungen_region,
            on_success=self._on_capture_rechnungen_success,
        )

    def handle_run_batch(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "Batch Extraction",
            self.automation.run_batch,
            on_success=self._on_batch_success,
            progress_handler=self.log_message,
        )

    def handle_run_streitwert_with_rechnungen(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "Streitwert + Rechnungen",
            self.automation.run_streitwert,
            on_success=self._on_streitwert_success,
            progress_handler=self.log_message,
            include_rechnungen=True,
        )

    def handle_run_streitwert(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "Streitwert Extraction",
            self.automation.run_streitwert,
            on_success=self._on_streitwert_success,
            progress_handler=self.log_message,
        )

    def handle_capture_doclist(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "Capture Doclist Snapshot",
            self.automation.capture_doclist_snapshot,
            progress_handler=self.log_message,
        )

    def handle_run_rechnungen_only(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "Rechnungen Extraction",
            self.automation.run_rechnungen_only,
            on_success=self._on_rechnungen_only_success,
            progress_handler=self.log_message,
        )

    def handle_run_rechnungen_gg(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "GG Extraction",
            self.automation.run_rechnungen_gg,
            on_success=self._on_rechnungen_gg_success,
            progress_handler=self.log_message,
        )

    def handle_run_akten(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "Akten Extraction",
            self.automation.run_akten,
            on_success=self._on_akten_success,
            progress_handler=self.log_message,
        )

    def handle_run_akten_filtered(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "Akten Extraction (Filtered)",
            self.automation.run_akten_with_filtering,
            on_success=self._on_akten_filtered_success,
            progress_handler=self.log_message,
        )

    def handle_capture_akten_doclist(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "Akten Doclist Capture",
            self.automation.test_akten_doclist_capture,
            on_success=self._on_test_akten_capture_success,
            progress_handler=self.log_message,
        )

    def handle_run_sw_gg(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "SW GG Extraction",
            self.automation.run_sw_gg_extraction,
            on_success=self._on_sw_gg_success,
            progress_handler=self.log_message,
        )

    def handle_test_typing(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "Typing Test",
            self.automation.test_typing,
            on_success=self._on_test_typing_success,
            progress_handler=self.log_message,
        )

    def handle_test_akten_setup(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "Akten Setup Test",
            self.automation.test_akten_setup,
            on_success=self._on_test_akten_setup_success,
            progress_handler=self.log_message,
        )

    def handle_test_sw_gg(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "SW GG Setup Test",
            self.automation.test_sw_gg_setup,
            on_success=self._on_test_sw_gg_success,
            progress_handler=self.log_message,
        )

    def handle_test_sw_gg_line_preview(self) -> None:
        # Show warning to user
        reply = QMessageBox.question(
            self,
            "SW GG Line Preview",
            "⚠ IMPORTANT:\n\n"
            "Before running this test, you must:\n"
            "1. Manually open a SW GG document in the RDP window\n"
            "2. Make sure the Stammdaten text area is visible\n"
            "3. Ensure it contains the keyword you're searching for\n\n"
            "The test will capture whatever is currently shown in the sw_gg_region.\n\n"
            "Ready to proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            self.log_message("SW GG Line Preview cancelled by user.")
            return

        self.log_message("⚠ Starting SW GG Line Preview - ensure SW GG document is open!")
        self._sync_controller_config()
        self.run_background(
            "SW GG Line Preview",
            self.automation.test_sw_gg_line_preview,
            on_success=self._on_test_sw_gg_line_preview_success,
            progress_handler=self.log_message,
        )

    def handle_test_parse_full(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "Test Parse (Full Region)",
            lambda: self.automation.parse_result_region(use_profile=False),
            on_success=self._on_capture_success,
        )

    def handle_test_parse_profile(self) -> None:
        self._sync_controller_config()
        self.run_background(
            "Test Parse (Profile Region)",
            lambda: self.automation.parse_result_region(use_profile=True),
            on_success=self._on_capture_success,
        )

    def run_background(
        self,
        description: str,
        fn: Callable,
        *fn_args,
        on_success: Callable[[object], None] | None = None,
        progress_handler: Callable[[str], None] | None = None,
        **fn_kwargs,
    ) -> None:
        worker = TaskWorker(fn, *fn_args, **fn_kwargs)
        if on_success is not None:
            worker.signals.result.connect(on_success)
        if progress_handler is not None:
            worker.signals.progress.connect(progress_handler)
            worker.kwargs.setdefault("progress_callback", worker.signals.progress.emit)
        worker.signals.error.connect(lambda err, desc=description: self._handle_task_error(desc, err))
        worker.signals.finished.connect(
            lambda desc=description: self.log_message(f"{desc} finished.")
        )
        self.log_message(f"{description} started…")
        self.thread_pool.start(worker)

    def _on_connect_success(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            result = {}
        rect = result.get("rect")
        title = result.get("window_title", "")
        rect_text = f"{rect}" if rect else "unknown region"
        self.log_message(f"Connected to '{title}' with client rect {rect_text}.")
        QMessageBox.information(self, "RDP Connection", "Connected to RDP client.")

    def _on_capture_success(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            result = {}
        image = result.get("image")
        text = (result.get("text") or "").strip()
        box = result.get("box")
        if image is not None:
            self.show_preview_image(image)
        if box:
            self.log_message(f"Captured result region at {box}.")
        amount = result.get("amount")
        keyword = result.get("keyword")
        profile_used = result.get("profile_used")
        if text:
            self.log_message(f"OCR Preview:\n{text}")
        else:
            self.log_message("No OCR text detected.")
        if profile_used:
            self.log_message(f"Profile used: {profile_used} (keyword: {keyword})")
        if amount:
            self.log_message(f"Extracted amount: {amount}")
        elif keyword:
            self.log_message(f"No amount detected near keyword '{keyword}'.")

    def _on_capture_rechnungen_success(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            result = {}

        region_key = result.get("region_key", "rechnungen_region")
        box = result.get("box")
        entries = result.get("entries", [])
        entry_count = result.get("entry_count", 0)
        summary_text = result.get("text", "")
        status = result.get("status", "")
        preview_image = result.get("image")
        ocr_line_count = result.get("ocr_line_count", 0)

        # Show preview image with bounding boxes
        if preview_image is not None:
            self.show_preview_image(preview_image)

        # Log the capture details
        if box:
            self.log_message(f"Captured {region_key} at {box}.")

        self.log_message(f"OCR detected {ocr_line_count} raw lines.")

        # Display the entries
        if entry_count > 0:
            self.log_message(f"\n{'='*60}")
            self.log_message(f"Found {entry_count} Rechnungen entries:")
            self.log_message(f"{'='*60}")
            self.log_message(summary_text)
            self.log_message(f"{'='*60}\n")

            # Show detailed breakdown
            gg_entries = [e for e in entries if e.get("is_gg")]
            non_gg_entries = [e for e in entries if not e.get("is_gg")]

            if gg_entries:
                self.log_message(f"GG Entries (red boxes): {len(gg_entries)}")
            if non_gg_entries:
                self.log_message(f"Non-GG Entries (blue boxes): {len(non_gg_entries)}")

            # Show message box with summary
            msg = f"Captured {entry_count} invoice entries.\n\n"
            if gg_entries:
                msg += f"• {len(gg_entries)} GG entries (red boxes)\n"
            if non_gg_entries:
                msg += f"• {len(non_gg_entries)} other entries (blue boxes)\n"
            msg += f"\nSee preview image and log for details."

            QMessageBox.information(self, "Rechnungen List", msg)
        else:
            self.log_message(f"No Rechnungen entries detected. Status: {status}")
            raw_lines = result.get("raw_lines", [])
            if raw_lines:
                self.log_message(f"Raw OCR lines detected ({len(raw_lines)}):")
                for i, line in enumerate(raw_lines[:15], 1):  # Show more lines for debugging
                    self.log_message(f"  {i}. {line}")
            QMessageBox.warning(
                self,
                "Rechnungen List",
                f"No invoice entries detected.\n\nOCR Lines: {ocr_line_count}\nStatus: {status}\n\nCheck the preview image and log for OCR details."
            )

    def _on_batch_success(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            result = {}
        rows = result.get("rows", 0)
        csv_path = result.get("output_csv")
        message = f"Processed {rows} rows."
        if csv_path:
            message += f" Output saved to {csv_path}."
        self.log_message(message)
        QMessageBox.information(self, "Batch Extraction", message)

    def _on_streitwert_success(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            result = {}
        rows = result.get("rows", 0)
        csv_path = result.get("output_csv")
        message = f"Processed {rows} Streitwert entries."
        if csv_path:
            message += f" Results saved to {csv_path}."
        rechn_csv = result.get("rechnungen_csv")
        if rechn_csv:
            message += f" Rechnungen summary saved to {rechn_csv}."
        self.log_message(message)
        QMessageBox.information(self, "Streitwert Extraction", message)

    def _on_rechnungen_only_success(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            result = {}
        rows = result.get("rows", 0)
        csv_path = result.get("output_csv")
        message = f"Processed {rows} Rechnungen entries."
        if csv_path:
            message += f" Results saved to {csv_path}."
        self.log_message(message)
        QMessageBox.information(self, "Rechnungen Extraction", message)

    def _on_rechnungen_gg_success(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            result = {}
        rows = result.get("rows", 0)
        csv_path = result.get("output_csv")
        message = f"Processed {rows} GG entries."
        if csv_path:
            message += f" Results saved to {csv_path}."
        self.log_message(message)
        QMessageBox.information(self, "GG Extraction", message)

    def _on_log_extract_success(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            result = {}
        rows = result.get("rows", 0)
        csv_path = result.get("output_csv")
        message = f"Processed {rows} log entries."
        if csv_path:
            message += f" Results saved to {csv_path}."
        self.log_message(message)
        QMessageBox.information(self, "Log Extraction", message)

    def _on_fees_success(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            result = {}
        rows = result.get("rows", 0)
        csv_path = result.get("output_csv")
        message = f"Processed {rows} Fees entries."
        if csv_path:
            message += f" Results saved to {csv_path}."
        self.log_message(message)
        QMessageBox.information(self, "Fees Extraction", message)

    def _on_akten_success(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            result = {}
        rows = result.get("rows", 0)
        captured = result.get("captured", 0)
        csv_path = result.get("output_csv")
        run_log = result.get("run_log")
        message = f"Opened {rows} Akten document(s); captured {captured} date(s)."
        if csv_path:
            message += f" Results saved to {csv_path}."
        if run_log:
            self.log_message(f"Akten run log: {run_log}")
        self.log_message(message)
        QMessageBox.information(self, "Akten Extraction", message)

    def _on_akten_filtered_success(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            result = {}
        rows = result.get("rows", 0)
        dates_extracted = result.get("captured", result.get("dates_extracted", 0))
        csv_path = result.get("output_csv")
        run_log = result.get("run_log")
        message = f"Processed {rows} Akten document(s); captured {dates_extracted} date(s)."
        if csv_path:
            message += f" Results saved to {csv_path}."
        if run_log:
            self.log_message(f"Akten run log: {run_log}")
        self.log_message(message)
        QMessageBox.information(self, "Akten Extraction (Filtered)", message)

    def _on_sw_gg_success(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            result = {}
        rows = result.get("rows", 0)
        csv_path = result.get("output_csv")
        message = f"Captured {rows} SW GG amount(s)."
        if csv_path:
            message += f" Results saved to {csv_path}."
        self.log_message(message)
        QMessageBox.information(self, "SW GG Extraction", message)

    def _on_test_typing_success(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            result = {}
        text = result.get("text", "")
        point = result.get("point")
        point_text = f" at {point}" if point else ""
        message = f"Typed '{text}'{point_text}."
        self.log_message(message)
        QMessageBox.information(self, "Typing Test", message)

    def _on_test_akten_setup_success(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            result = {}
        az_value = result.get("aktenzeichen") or "(not detected)"
        doclist_log_path = result.get("doclist_log_path") or ""
        doclist_preview_path = result.get("doclist_preview_path") or ""
        session_log_path = result.get("session_log_path") or ""

        message_lines = [f"Akten filter snapshot: {az_value}."]
        if doclist_log_path:
            info = f"Doclist OCR log saved to {doclist_log_path}"
            self.log_message(info)
            message_lines.append(info)
        if doclist_preview_path:
            info = f"Doclist screenshot saved to {doclist_preview_path}"
            self.log_message(info)
            message_lines.append(info)
        if session_log_path:
            info = f"Session log saved to {session_log_path}"
            self.log_message(info)
            message_lines.append(info)

        message = "\n\n".join(message_lines)
        self.log_message(f"Akten filter snapshot: {az_value}.")
        QMessageBox.information(self, "Akten Setup Test", message)

    def _on_test_akten_capture_success(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            result = {}
        aktenzeichen = result.get("aktenzeichen") or "(unknown)"
        matched_entry = result.get("matched_entry") or "(no match)"
        matched_index = result.get("matched_index", 0)
        preview_path = result.get("doclist_preview_path") or ""
        log_path = result.get("doclist_log_path") or ""
        row_count = result.get("row_count", 0)

        if preview_path:
            self.log_message(f"Akten capture screenshot: {preview_path}")
        if log_path:
            self.log_message(f"Akten capture OCR log: {log_path}")

        message = (
            f"Akten doclist capture complete.\n\n"
            f"Aktenzeichen: {aktenzeichen}\n"
            f"Rows detected: {row_count}\n"
            f"Matched entry #{matched_index}: {matched_entry}"
        )
        self.log_message(message)
        QMessageBox.information(self, "Akten Doclist Capture", message)

    def _on_test_sw_gg_success(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            result = {}

        amount = result.get("amount") or "(not detected)"
        entry = result.get("entry") or "(no entry)"
        aktenzeichen = result.get("aktenzeichen") or "(unknown)"
        total_entries = result.get("total_entries", 0)
        gg_count = result.get("gg_count", 0)
        entry_list = result.get("entry_list", [])
        details = result.get("results") or []

        # Log detailed results
        self.log_message(f"\n{'='*60}")
        self.log_message(f"SW GG Setup Test Results")
        self.log_message(f"{'='*60}")
        self.log_message(f"Aktenzeichen: {aktenzeichen}")
        self.log_message(f"Total entries detected: {total_entries}")
        self.log_message(f"GG entries: {gg_count}")
        self.log_message(f"Other entries: {total_entries - gg_count}")
        self.log_message(f"\nTested entry: {entry}")
        self.log_message(f"Extracted Wert: {amount}")

        if details:
            self.log_message(f"\nEntry breakdown:")
            for info in details:
                self.log_message(
                    f"  {info.get('entry', '(n/a)')} -> {info.get('amount') or '(none)'} ({info.get('status', 'n/a')})"
                )
        elif entry_list:
            self.log_message(f"\nAll detected entries:")
            for idx, e in enumerate(entry_list, 1):
                self.log_message(f"  {idx}. {e}")
        self.log_message(f"{'='*60}\n")

        # Show message box
        message = f"SW GG Test Complete!\n\n"
        message += f"Aktenzeichen: {aktenzeichen}\n"
        message += f"Detected {total_entries} invoice entries\n"
        if gg_count > 0:
            message += f"• {gg_count} GG entries\n"
        if total_entries - gg_count > 0:
            message += f"• {total_entries - gg_count} other entries\n"
        message += f"\nTested: {entry}\n"
        message += f"Wert: {amount}\n"
        message += f"\nSee log for full entry list."

        QMessageBox.information(self, "SW GG Setup Test", message)

    def _on_test_sw_gg_line_preview_success(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            result = {}
        line_text = result.get("line_text") or "(no line detected)"
        amount = result.get("amount") or "(not extracted)"
        preview_path = result.get("preview_path", "")
        preview_image = result.get("preview_image")
        found = bool(result.get("found"))

        # Show the preview image in the GUI
        if preview_image is not None:
            self.show_preview_image(preview_image)
            self.log_message("SW GG Line Preview image displayed.")

        self.log_message("SW GG Line Preview Result:")
        self.log_message(f"  Found line: {found}")
        self.log_message(f"  Line text: {line_text}")
        self.log_message(f"  Extracted Wert: {amount}")
        if preview_path:
            self.log_message(f"  Preview saved to: {preview_path}")
        QMessageBox.information(
            self,
            "SW GG Line Preview",
            f"{'Matched' if found else 'No match'}.\nLine: {line_text}\nWert: {amount}\nPreview: {preview_path or '(not saved)'}",
        )

    def _handle_task_error(self, description: str, error: str) -> None:
        message = f"{description} failed: {error}"
        self.log_message(message)
        QMessageBox.warning(self, "Automation Error", message)

    # --- Preview helpers -------------------------------------------------
    def show_preview_image(self, image) -> None:
        self.preview_image = image.copy()
        self._update_preview_pixmap()

    def _update_preview_pixmap(self) -> None:
        if self.preview_image is None:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("Capture a region to preview OCR input.")
            return
        image = self.preview_image
        qimage = ImageQt(image.convert("RGB"))
        pixmap = QPixmap.fromImage(qimage)
        target_size = self.preview_label.size()
        scaled = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)
        self.preview_label.setText("")

    def resizeEvent(self, event):  # noqa: D401
        super().resizeEvent(event)
        self._update_preview_pixmap()

    # --- Field helpers ---------------------------------------------------
    def _create_field_widget(self, spec: FieldSpec) -> tuple[QWidget, str | None]:
        widget: QWidget
        if spec.options:
            combo = QComboBox()
            for value, label in spec.options:
                combo.addItem(label, value)
            self.binder.bind_combo_box(spec.key, combo)
            widget = combo
        elif spec.kind == "text":
            edit = QLineEdit()
            if spec.placeholder:
                edit.setPlaceholderText(spec.placeholder)
            self.binder.bind_line_edit(spec.key, edit)
            widget = edit
        elif spec.kind == "path":
            widget = self._build_path_field(spec)
        elif spec.kind == "int":
            spin = QSpinBox()
            if spec.minimum is not None:
                spin.setMinimum(int(spec.minimum))
            if spec.maximum is not None:
                spin.setMaximum(int(spec.maximum))
            self.binder.bind_spin_box(spec.key, spin)
            widget = spin
        elif spec.kind == "float":
            spin = QDoubleSpinBox()
            spin.setDecimals(spec.decimals)
            if spec.minimum is not None:
                spin.setMinimum(float(spec.minimum))
            if spec.maximum is not None:
                spin.setMaximum(float(spec.maximum))
            self.binder.bind_double_spin_box(spec.key, spin)
            widget = spin
        elif spec.kind == "bool":
            checkbox = QCheckBox(spec.label)
            self.binder.bind_checkbox(spec.key, checkbox)
            widget = checkbox
        else:
            edit = QLineEdit()
            self.binder.bind_line_edit(spec.key, edit)
            widget = edit
        if spec.tooltip:
            widget.setToolTip(spec.tooltip)
        return widget, None

    def _build_path_field(self, spec: FieldSpec) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        self.binder.bind_line_edit(spec.key, edit)
        button = QPushButton("Browse…")

        def choose_path() -> None:
            try:
                start_dir = edit.text().strip() or ""
                options = QFileDialog.Option.DontUseNativeDialog
                if spec.dialog == "directory":
                    path = QFileDialog.getExistingDirectory(
                        self, f"Select {spec.label}", start_dir, options=options
                    )
                else:
                    file_filter = spec.dialog_filter or "All Files (*)"
                    path, _ = QFileDialog.getOpenFileName(
                        self,
                        f"Select {spec.label}",
                        start_dir,
                        file_filter,
                        options=options,
                    )
            except Exception as exc:
                self.log_message(f"[UI] Browse dialog failed: {exc}")
                QMessageBox.warning(self, "Browse Error", f"Unable to open dialog: {exc}")
                return
            if not path:
                return
            cleaned = str(path).strip()
            if not cleaned:
                return
            edit.setText(cleaned)

        button.clicked.connect(choose_path)
        layout.addWidget(edit)
        layout.addWidget(button)
        return container

    # --- Actions ---------------------------------------------------------
    def handle_save_config(self) -> None:
        self._sync_controller_config()
        save_config(self.binder.config)
        self.log_message(f"Configuration saved to {CFG_FILE}.")
        QMessageBox.information(self, "Save Config", "Configuration saved successfully.")

    def handle_load_config(self) -> None:
        self.cfg = load_config()
        self.binder.config = self.cfg
        self.binder.refresh()
        self._sync_controller_config()
        self.log_message("Configuration reloaded from disk.")
        QMessageBox.information(self, "Load Config", "Configuration loaded.")

    def handle_reset_config(self) -> None:
        self.cfg = dict(DEFAULTS)
        self.binder.config = self.cfg
        self.binder.refresh()
        self._sync_controller_config()
        self.log_message("Configuration reset to defaults.")

    def _not_implemented(self, action: str) -> None:
        self.log_message(f"{action} is not yet implemented in the PyQt6 interface.")
        QMessageBox.information(
            self,
            "Coming Soon",
            f"{action} is part of the ongoing PyQt6 migration and is not available yet.",
        )

    # --- Logging ---------------------------------------------------------
    def log_message(self, text: str) -> None:
        self.log_output.appendPlainText(text)
        cursor = self.log_output.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_output.setTextCursor(cursor)

    def handle_copy_log(self) -> None:
        clipboard = QApplication.clipboard()
        clipboard.setText(self.log_output.toPlainText())
        self.log_message("[UI] Log copied to clipboard.")

    def handle_clear_log(self) -> None:
        self.log_output.clear()


__all__ = ["MainWindow"]

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from PyQt6.QtWidgets import QApplication

if __package__ in {None, ""}:  # pragma: no cover - script execution fallback
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from datev_ocr.config import load_config
from datev_ocr.ui import MainWindow


def run(argv: Sequence[str] | None = None) -> int:
    app = QApplication(list(argv) if argv is not None else sys.argv)
    window = MainWindow(config=load_config())
    window.show()
    return app.exec()


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())

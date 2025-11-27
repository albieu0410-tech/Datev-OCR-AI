from __future__ import annotations

import csv
import os
import re
import time
import unicodedata
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyautogui
from PIL import Image, ImageEnhance, ImageOps, ImageDraw

from . import core
from . import ocr_preprocessing


class AutomationController:
    """Thin orchestration layer that reuses the existing automation helpers."""

    _KFB_RE = re.compile(r"(?<![0-9A-Za-z])k\s*[-./]?\s*f\s*[-./]?\s*b", re.IGNORECASE)
    _KFB_WORD_RE = re.compile(
        r"kosten\s*festsetzungs\s*beschl(?:uss|uß|\.)?", re.IGNORECASE
    )
    _AMT_NUM_RE = re.compile(r"\b\d{1,3}(?:[\.\s]\d{3})*(?:,\d{2})?\b")
    _WORDS_HINT_RE = re.compile(
        r"\b(?:euro|eur|tausend|hundert|einhundert|zweihundert|dreihundert|vierhundert|fuenf|fünf|sechs|sieben|acht|neun|zehn|elf|zwoelf|zwölf|zwanzig|dreissig|dreißig|vierzig|fuenfzig|fünfzig|sechzig|siebzig|achtzig|neunzig)\b",
        re.IGNORECASE,
    )
    _AKTEN_AZ_RE = re.compile(r"\bAZ\s*[:\-]?\s*([0-9A-Za-z./-]+)", re.IGNORECASE)
    _AKTEN_DATE_RE = re.compile(r"\b([A-Za-zÄÖÜäöüß]+),\s*(\d{2}\.\d{2}\.\d{4})\b", re.UNICODE)

    def __init__(self, config: Mapping[str, Any]):
        self.config: Mapping[str, Any] = config
        self.current_rect: tuple[int, int, int, int] | None = None
        self._rdp_window = None
        self._ocr_log_paths: dict[str, str] = {}
        self._apply_tesseract_path()
        self._preview_dir = Path(self.config.get("__preview_dir__", "img"))
        try:
            self._preview_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def update_config(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self._apply_tesseract_path()

    def _apply_tesseract_path(self) -> None:
        path = (self.config or {}).get("tesseract_path")
        if not path:
            return
        path = str(path).strip().strip('"').replace("/", "\\")
        if not path.lower().endswith("tesseract.exe"):
            if os.path.isdir(path):
                path = os.path.join(path, "tesseract.exe")
            else:
                path = path if path.lower().endswith(".exe") else os.path.join(path, "tesseract.exe")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tesseract not found at: {path}")
        core.pytesseract.pytesseract.tesseract_cmd = path

    # ------------------------------------------------------------------
    def connect_rdp(self) -> dict[str, Any]:
        title_re = self.config.get("rdp_title_regex") or r".*"
        window, rect = core.connect_rdp_window(title_re)
        self.current_rect = rect
        self._rdp_window = window
        rect_tuple = tuple(rect)
        title = ""
        title_getter = getattr(window, "window_text", None)
        if callable(title_getter):
            try:
                title = title_getter() or ""
            except Exception:  # pragma: no cover - best effort logging
                title = ""
        if not title:
            try:
                info = getattr(window, "element_info", None)
                title = getattr(info, "name", "") or ""
            except Exception:
                title = ""
        if not title:
            title = str(window)
        return {
            "rect": rect_tuple,
            "window_title": title,
        }

    def _save_preview(self, image: Image.Image | None, tag: str) -> Path | None:
        if image is None:
            return None
        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            safe_tag = re.sub(r"[^A-Za-z0-9_-]+", "_", tag) or "preview"
            path = self._preview_dir / f"{timestamp}_{safe_tag}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            return path
        except Exception:
            return None

    def _append_ocr_log(self, label: str, context: str, lines, prefix="", log=None):
        """
        Save OCR results to a log file in the log/ folder.

        Args:
            label: Label for the log file (e.g., aktenzeichen, filename)
            context: Context type (e.g., "doclist", "pdf", "filter")
            lines: List of tuples (x, y, w, h, text) from OCR
            prefix: Logging prefix
            log: Optional logging function
        """
        log = log or (lambda msg: None)

        try:
            # Ensure log directory exists
            log_dir = Path(self.config.get("log_folder") or core.LOG_DIR)
            log_dir.mkdir(parents=True, exist_ok=True)

            # Create safe filename
            safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_path = log_dir / f"{timestamp}_{safe_label}_{context}.log"

            # Write OCR results to log file
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=== OCR Log: {label} ({context}) ===\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Context: {context}\n")
                f.write(f"Total lines: {len(lines)}\n")
                f.write(f"\n{'='*60}\n\n")

                for idx, line in enumerate(lines, 1):
                    if len(line) == 5:
                        x, y, w, h, text = line
                        f.write(f"{idx:03d}: [{x:4d}, {y:4d}, {w:4d}x{h:3d}] {text}\n")
                    else:
                        f.write(f"{idx:03d}: {line}\n")

            log(f"{prefix}📝 OCR log saved to: {log_path.name}")
            return str(log_path)

        except Exception as e:
            log(f"{prefix}⚠ Failed to save OCR log: {e}")
            return None

    def _save_session_log(self, label: str, lines: list[str], prefix: str = "", log=None) -> str | None:
        """
        Persist a simple text log into the configured log directory.

        Args:
            label: Identifier used for the filename.
            lines: Sequence of log lines to store.
            prefix: Optional prefix for status messages.
            log: Optional logging callback.
        """
        log = log or (lambda msg: None)
        if not lines:
            return None

        try:
            log_dir = Path(self.config.get("log_folder") or core.LOG_DIR)
            log_dir.mkdir(parents=True, exist_ok=True)
            safe_label = core.sanitize_filename(label) or "session"
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_path = log_dir / f"{timestamp}_{safe_label}.log"

            with open(log_path, "w", encoding="utf-8") as fh:
                for line in lines:
                    fh.write(f"{line.rstrip()}\n")

            log(f"{prefix}📝 Session log saved to: {log_path.name}")
            return str(log_path)
        except Exception as exc:
            log(f"{prefix}⚠ Failed to save session log: {exc}")
            return None

    def _ensure_rect(self) -> tuple[int, int, int, int]:
        if not self.current_rect:
            raise RuntimeError("RDP client not connected. Run 'Connect RDP' first.")
        return self.current_rect

    def capture_result_region(self) -> dict[str, Any]:
        return self.parse_result_region(use_profile=False)

    def capture_doclist_snapshot(self, progress_callback=None) -> dict[str, Any]:
        log = progress_callback or (lambda msg: None)
        self._apply_tesseract_path()
        if not self.current_rect:
            self.connect_rdp()
        doc_rect = self._doclist_abs_rect()
        if not doc_rect:
            raise RuntimeError("Document list region is not calibrated.")
        left, top, right, bottom = doc_rect
        width = max(1, right - left)
        height = max(1, bottom - top)

        log(f"[Doclist Snapshot] Capturing document list region...")
        lines, snapshot, raw_lines = self._capture_window_doc_lines(prefix="[Doclist Snapshot] ", log=log)
        log(f"[Doclist Snapshot] Captured {len(lines)} OCR lines (filtered), {len(raw_lines)} total raw OCR lines")

        preview_path = self._save_preview(snapshot, "doclist_snapshot_full")
        if preview_path:
            log(f"[Doclist Snapshot] Preview image saved: {preview_path}")

        # Filter and log detailed line information
        matches, inc_tokens, exc_tokens, debug_rows = self._filter_streitwert_rows(lines)
        log(f"[Doclist Snapshot] Filtered: {len(matches)} matched, {len(debug_rows)} rejected")

        # Write to log file - ALWAYS log ALL raw OCR lines for debugging
        log_label = "doclist_snapshot"
        log(f"[Doclist Snapshot] Creating OCR log file...")
        try:
            # Always write log file with ALL raw OCR data
            core.ensure_log_dir()
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            safe = core.sanitize_filename(log_label)
            filename = f"{timestamp}_{safe}.log"
            path = os.path.join(core.LOG_DIR, filename)

            with open(path, "w", encoding="utf-8") as f:
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{stamp}] Doclist Snapshot Capture\n")
                f.write(f"Total raw OCR lines: {len(raw_lines)}\n")
                f.write(f"Filtered lines (inside region): {len(lines)}\n")
                f.write(f"Matched lines: {len(matches)}\n")
                f.write(f"Include tokens: {', '.join(inc_tokens) if inc_tokens else '(none)'}\n")
                f.write(f"Exclude tokens: {', '.join(exc_tokens) if exc_tokens else '(none)'}\n")
                f.write("\n")

                # Add diagnostic info about region configuration
                doc_rect = self._doclist_abs_rect()
                if doc_rect and self.current_rect:
                    doc_left, doc_top, doc_w, doc_h = doc_rect
                    win_left, win_top, win_right, win_bottom = self.current_rect
                    win_w = win_right - win_left
                    win_h = win_bottom - win_top
                    offset_x = doc_left - win_left
                    offset_y = doc_top - win_top

                    f.write("="*80 + "\n")
                    f.write("COORDINATE FILTER DIAGNOSTIC:\n")
                    f.write("="*80 + "\n")
                    f.write(f"RDP Window: ({win_left}, {win_top}, {win_w}x{win_h})\n")
                    f.write(f"Doclist Region (absolute): ({doc_left}, {doc_top}, {doc_w}x{doc_h})\n")
                    f.write(f"Doclist Region (relative to window): x=[{offset_x}, {offset_x + doc_w}], y=[{offset_y}, {offset_y + doc_h}]\n")
                    f.write(f"\n")
                    f.write(f"Lines are ONLY included if their center point falls within:\n")
                    f.write(f"  X range: {offset_x} to {offset_x + doc_w}\n")
                    f.write(f"  Y range: {offset_y} to {offset_y + doc_h}\n")
                    f.write(f"\n")

                    # Show which lines would have matched if region was correct
                    potential_matches = []
                    for entry in raw_lines:
                        if isinstance(entry, (list, tuple)) and len(entry) == 5:
                            x, y, w, h, text = entry
                            raw_text = (text or "").lower()
                            # Check if any include token is in the text
                            if any(tok in raw_text for tok in inc_tokens if tok):
                                cx = x + max(w or 0, 1) / 2
                                cy = y + max(h or 0, 1) / 2
                                in_region = (offset_x <= cx <= offset_x + doc_w and
                                           offset_y <= cy <= offset_y + doc_h)
                                potential_matches.append((entry, in_region))

                    if potential_matches:
                        f.write(f"IMPORTANT: Found {len(potential_matches)} line(s) containing include tokens:\n")
                        for idx, (entry, in_region) in enumerate(potential_matches, 1):
                            x, y, w, h, text = entry
                            cx = x + max(w or 0, 1) / 2
                            cy = y + max(h or 0, 1) / 2
                            status = "✓ INSIDE region" if in_region else "✗ OUTSIDE region (FILTERED OUT)"
                            f.write(f"  {idx}. [{status}] ({x},{y},{w},{h}) center=({cx:.1f},{cy:.1f})\n")
                            f.write(f"     Text: {text}\n")
                        f.write(f"\n")
                        if not any(in_region for _, in_region in potential_matches):
                            f.write(f"*** ALL MATCHING LINES ARE OUTSIDE THE DOCLIST REGION ***\n")
                            f.write(f"*** YOU NEED TO RECALIBRATE THE DOCLIST_REGION ***\n")
                            f.write(f"\n")
                    f.write("\n")

                f.write("="*80 + "\n")
                f.write("ALL RAW OCR LINES (entire window):\n")
                f.write("="*80 + "\n")

                if raw_lines:
                    for idx, entry in enumerate(raw_lines, 1):
                        if isinstance(entry, (list, tuple)) and len(entry) == 5:
                            x, y, w, h, text = entry
                        else:
                            x = y = w = h = None
                            text = entry if not isinstance(entry, (list, tuple)) else entry[-1]

                        raw = text or ""
                        norm = core.normalize_line(raw)
                        soft = core.normalize_line_soft(raw)

                        f.write(f"{idx:03d}: ({x},{y},{w},{h}) -> {raw}\n")
                        if norm and norm != raw:
                            f.write(f"      norm: {norm}\n")
                        if soft and soft not in {raw, norm}:
                            f.write(f"      soft: {soft}\n")
                else:
                    f.write("(No OCR lines detected)\n")

                f.write("\n")
                f.write("="*80 + "\n")
                f.write("FILTERED LINES (inside doclist region only):\n")
                f.write("="*80 + "\n")

                if lines:
                    for idx, entry in enumerate(lines, 1):
                        if isinstance(entry, (list, tuple)) and len(entry) == 5:
                            x, y, w, h, text = entry
                        else:
                            x = y = w = h = None
                            text = entry if not isinstance(entry, (list, tuple)) else entry[-1]

                        raw = text or ""
                        norm = core.normalize_line(raw)
                        soft = core.normalize_line_soft(raw)

                        f.write(f"{idx:03d}: ({x},{y},{w},{h}) -> {raw}\n")
                        if norm and norm != raw:
                            f.write(f"      norm: {norm}\n")
                        if soft and soft not in {raw, norm}:
                            f.write(f"      soft: {soft}\n")
                else:
                    f.write("(No lines inside doclist region)\n")

                f.write("\n")
                f.write("="*80 + "\n")
                f.write("MATCHED LINES (passed include/exclude filter):\n")
                f.write("="*80 + "\n")

                if matches:
                    for idx, match in enumerate(matches, 1):
                        token = match.get('token', '')
                        raw = match.get('raw', '')
                        soft = match.get('soft', '')
                        f.write(f"{idx:03d}: [{token}] {raw}\n")
                        if soft and soft != raw.lower():
                            f.write(f"      soft: {soft}\n")
                else:
                    f.write("(No matches after filtering)\n")

            log(f"[Doclist Snapshot] OCR log file for '{log_label}' → {path}")
            log(f"[Doclist Snapshot] OCR log file created successfully")
        except Exception as log_exc:
            log(f"[Doclist Snapshot] ERROR creating log file: {log_exc}")

        # User-friendly console log output
        if log:
            log(f"[Doclist Snapshot] ")
            log(f"[Doclist Snapshot] ╔════════════════════════════════════════════════════════════╗")
            log(f"[Doclist Snapshot] ║              DOCLIST CAPTURE SUMMARY                       ║")
            log(f"[Doclist Snapshot] ╚════════════════════════════════════════════════════════════╝")
            log(f"[Doclist Snapshot] ")
            log(f"[Doclist Snapshot] 📊 OCR Results:")
            log(f"[Doclist Snapshot]    • Total raw lines detected: {len(raw_lines)}")
            log(f"[Doclist Snapshot]    • Lines in doclist region: {len(lines)}")
            log(f"[Doclist Snapshot]    • Lines matched (after filtering): {len(matches)}")
            log(f"[Doclist Snapshot] ")

            if inc_tokens:
                log(f"[Doclist Snapshot] ✓ Include tokens: {', '.join(inc_tokens)}")
            if exc_tokens:
                log(f"[Doclist Snapshot] ✗ Exclude tokens: {', '.join(exc_tokens)}")
            log(f"[Doclist Snapshot] ")

            if matches:
                log(f"[Doclist Snapshot] ✅ MATCHED ENTRIES ({len(matches)} found):")
                log(f"[Doclist Snapshot] ─────────────────────────────────────────────────────────")
                for idx, match in enumerate(matches[:10], 1):
                    token = match.get('token', '').upper()
                    raw = match.get('raw', '')
                    log(f"[Doclist Snapshot]   {idx}. [{token}] {raw[:80]}{'...' if len(raw) > 80 else ''}")
                if len(matches) > 10:
                    log(f"[Doclist Snapshot]   ... and {len(matches) - 10} more")
            else:
                log(f"[Doclist Snapshot] ⚠ NO MATCHES FOUND")

            log(f"[Doclist Snapshot] ")

            if len(lines) == 0 and len(raw_lines) > 0:
                log(f"[Doclist Snapshot] ⚠ WARNING: OCR detected {len(raw_lines)} lines but NONE are in doclist region")
                log(f"[Doclist Snapshot]    → You may need to recalibrate the doclist_region")

            if debug_rows and len(matches) < len(lines):
                log(f"[Doclist Snapshot] ")
                log(f"[Doclist Snapshot] ℹ Filtered out ({len(debug_rows)} lines):")
                shown = 0
                for raw, reason in debug_rows[:5]:
                    log(f"[Doclist Snapshot]   • {raw[:60]}{'...' if len(raw) > 60 else ''} → {reason}")
                    shown += 1
                if len(debug_rows) > 5:
                    log(f"[Doclist Snapshot]   ... and {len(debug_rows) - 5} more (see log file for details)")

            log(f"[Doclist Snapshot] ")
            log(f"[Doclist Snapshot] 📝 Detailed log saved to: {path}")
            log(f"[Doclist Snapshot] ")

        return {
            "box": (left, top, width, height),
            "image_path": str(preview_path) if preview_path else "",
            "line_count": len(lines),
            "matches": [m.get("raw", "") for m in matches],
        }

    def capture_rechnungen_region(self) -> dict[str, Any]:
        """
        Capture and parse the Rechnungen region to list all invoice entries.
        Returns a dict with invoice entries including dates, amounts, and invoice numbers.
        Includes preview image with red bounding boxes around detected rows.
        """
        rect = self._ensure_rect()

        # Try rechnungen_gg_region first, fall back to rechnungen_region
        region_key = "rechnungen_gg_region"
        rel_box = self.config.get(region_key)
        if not rel_box or len(rel_box) != 4:
            region_key = "rechnungen_region"
            rel_box = self.config.get(region_key)

        if not rel_box or len(rel_box) != 4:
            raise RuntimeError("Rechnungen region is not configured. Use 'Pick Rechnungen GG Region' to calibrate.")

        left, top, width, height = core.rel_to_abs(rect, rel_box)
        if width <= 0 or height <= 0:
            raise RuntimeError("Configured Rechnungen region is empty.")

        # Wait for region to load
        try:
            self._wait_for_rechnungen_region(region_key, prefix="", label="Rechnungen", log=None)
        except Exception:
            pass  # Continue even if wait fails

        # Capture base image for preview/snapshot
        base_image = core.grab_xywh(left, top, width, height)
        self._save_preview(base_image, "rechnungen_base")

        # Capture and process the region using enhanced preprocessing
        lines = self._capture_region_lines(region_key, label="Rechnungen", log=None, prefix="")

        if not lines:
            self._save_preview(base_image, "rechnungen_empty")
            return {
                "region_key": region_key,
                "box": (left, top, width, height),
                "image": base_image,
                "entries": [],
                "entry_count": 0,
                "text": "",
                "status": "No OCR lines detected in Rechnungen region",
            }

        # Parse the lines into structured entries
        entries = self._parse_rechnungen_entries(lines, prefix="", log=None)

        # Create preview image with bounding boxes
        preview_image = self._draw_rechnungen_boxes(base_image, entries, lines)
        self._save_preview(preview_image, "rechnungen_preview")

        # Format the entries for display
        formatted_entries = []
        for entry in entries:
            formatted_entries.append({
                "date": entry.get("date", ""),
                "amount": entry.get("amount", ""),
                "invoice": entry.get("invoice", ""),
                "label": entry.get("label", "") or entry.get("label_normalized", ""),
                "raw": entry.get("raw", ""),
                "is_gg": entry.get("label_is_gg", False),
                "y": entry.get("y", 0),
                "h": entry.get("h", 0),
            })

        # Build summary text
        summary_lines = []
        for idx, entry in enumerate(formatted_entries, 1):
            label_tag = " [GG]" if entry["is_gg"] else ""
            summary_lines.append(
                f"{idx}. {entry['date']} | {entry['amount']} | Invoice: {entry['invoice']}{label_tag}"
            )

        summary_text = "\n".join(summary_lines) if summary_lines else "No invoice entries found"

        return {
            "region_key": region_key,
            "box": (left, top, width, height),
            "image": preview_image,
            "base_image": base_image,
            "entries": formatted_entries,
            "entry_count": len(formatted_entries),
            "text": summary_text,
            "status": "success" if formatted_entries else "no_entries",
            "raw_lines": [line[4] for line in lines if len(line) > 4],
            "ocr_line_count": len(lines),
        }

    def parse_result_region(self, use_profile: bool = False) -> dict[str, Any]:
        rect = self._ensure_rect()
        rel_box = self.config.get("result_region")
        if not rel_box or len(rel_box) != 4:
            raise RuntimeError("Result region is not configured.")
        left, top, width, height = core.rel_to_abs(rect, rel_box)
        if width <= 0 or height <= 0:
            raise RuntimeError("Configured result region is empty.")
        base_image = core.grab_xywh(left, top, width, height)
        keyword = (self.config.get("keyword") or "").strip() or "Honorar"
        parse_image = base_image
        profile_used = None
        if (
            use_profile
            and self.config.get("use_amount_profile")
            and self.config.get("amount_profiles")
        ):
            profile = self._get_active_profile()
            if profile:
                parse_image = self._crop_to_profile(parse_image, profile.get("sub_region"))
                profile_keyword = (profile.get("keyword") or "").strip()
                if profile_keyword:
                    keyword = profile_keyword
                profile_used = profile.get("name")

        df = self._ocr_data(parse_image, context="program", psm=6)
        tsv_lines = core.lines_from_tsv(df)
        simple_lines = [(line[1], line[4] or "") for line in tsv_lines]
        if self.config.get("normalize_ocr", True):
            normalized_lines = [
                (y, core.normalize_line_soft(text)) for y, text in simple_lines
            ]
        else:
            normalized_lines = simple_lines
        full_text = "\n".join(text for _, text in normalized_lines if text)
        amount, matched_line = core.extract_amount_from_lines(
            normalized_lines, keyword=keyword
        )
        self._save_preview(parse_image, "result_preview-full" if not use_profile else "result_preview-profile")
        return {
            "image": parse_image,
            "text": full_text,
            "box": (left, top, width, height),
            "amount": amount,
            "matched_line": matched_line,
            "keyword": keyword,
            "profile_used": profile_used,
            "lines": normalized_lines,
        }

    # ------------------------------------------------------------------
    def run_batch(self, progress_callback=None) -> dict[str, Any]:
        announce = progress_callback or (lambda msg: None)
        cfg = self.config
        self._apply_tesseract_path()
        if not self.current_rect:
            self.connect_rdp()
        df = self._load_input_dataframe()
        rows, col_idx = self._prepare_batch_rows(df)
        if rows.empty:
            raise RuntimeError("No input rows available for batch processing.")
        total = len(rows)
        self._focus_rdp_window()
        search_point = cfg.get("search_point")
        if not search_point or len(search_point) != 2:
            raise RuntimeError("Search point is not configured.")
        use_profile = bool(cfg.get("use_amount_profile"))
        post_wait = float(cfg.get("post_search_wait") or 1.0)
        type_delay = float(cfg.get("type_delay") or 0.02)
        results = []
        for idx, (_, row) in enumerate(rows.iterrows(), start=1):
            query = "" if row.iloc[col_idx] is None else str(row.iloc[col_idx])
            announce(f"[{idx}/{total}] Query: {query}")
            self._focus_rdp_window()
            self._click_relative_point(search_point)
            self._clear_search_field()
            self._type_exact_text(query, fallback_interval=type_delay, progress=announce)
            pyautogui.press("enter")
            time.sleep(post_wait)
            parsed = self.parse_result_region(use_profile=use_profile)
            rec = row.to_dict()
            rec["__query__"] = query
            rec["extracted_text"] = parsed.get("text") or ""
            rec["extracted_amount"] = parsed.get("amount")
            rec["extracted_line"] = parsed.get("matched_line") or ""
            results.append(rec)
            announce(
                f" -> {parsed.get('amount') or '(none)'} from profile {parsed.get('profile_used') or 'full region'}"
            )
        output_path = cfg.get("results_csv") or "rdp_results.csv"
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")
        announce(f"Saved {len(results)} rows to {output_path}.")
        return {"output_csv": output_path, "rows": len(results)}

    # ------------------------------------------------------------------
    def run_streitwert(self, include_rechnungen: bool = False, progress_callback=None):
        log = progress_callback or (lambda msg: None)
        self._apply_tesseract_path()
        if not self.current_rect:
            self.connect_rdp()
        doc_rect = self._doclist_abs_rect()
        if not doc_rect:
            raise RuntimeError("Document list region is not calibrated.")
        queries = self._gather_aktenzeichen_queries()
        if not queries:
            raise RuntimeError("No Aktenzeichen entries were found in the configured Excel sheet.")
        self._reset_ocr_log_state()
        skip_waits = self._should_skip_manual_waits()
        list_wait = 0.0 if skip_waits else float(self.config.get("post_search_wait", 1.2))
        doc_wait = 0.0 if skip_waits else float(self.config.get("doc_open_wait", 1.2))
        total = len(queries)
        results = []
        rechnungen_results = [] if include_rechnungen else None
        for idx, (aktenzeichen, row_data) in enumerate(queries, start=1):
            prefix = f"[{idx}/{total}] "
            log(f"{prefix}Searching doc list for Aktenzeichen: {aktenzeichen}")
            if not self._type_doclist_query(aktenzeichen, log=log, prefix=prefix):
                log(f"{prefix}Unable to type Aktenzeichen. Skipping entry.")
                continue
            if list_wait > 0:
                time.sleep(list_wait)
            self._wait_for_doc_search_ready(prefix=prefix, log=log, reason="after Aktenzeichen search")
            self._wait_for_doclist_ready(prefix=prefix, log=log, reason="after Aktenzeichen search")
            if include_rechnungen:
                summary = self._extract_rechnungen_summary(prefix=prefix, log=log)
                if summary is None:
                    log(f"{prefix}Rechnungen capture returned no data; storing defaults.")
                    summary = self._summarize_rechnungen_entries([])
                else:
                    self._log_rechnungen_summary(prefix, summary, log=log)
                if rechnungen_results is not None:
                    row = self._build_rechnungen_result_row(aktenzeichen, summary)
                    row["instance_detected"] = None
                    rechnungen_results.append(row)
            self._wait_for_doc_search_ready(prefix=prefix, log=log, reason="before PDF search")
            pdf_term = (self.config.get("streitwert_term") or "Streitwert").strip() or "Streitwert"
            if not self._type_pdf_search(pdf_term, prefix=prefix, log=log):
                log(f"{prefix}Unable to type Streitwert term in the PDF search box; skipping.")
                continue
            log(f"{prefix}Typed '{pdf_term}' into the PDF search box.")
            self._wait_for_doc_search_ready(prefix=prefix, log=log, reason="after PDF search")
            self._wait_for_doclist_ready(prefix=prefix, log=log, reason="after PDF search")

            doc_rect = self._doclist_abs_rect()
            if not doc_rect:
                log(f"{prefix}Doc list region is not calibrated.")
                continue
            doc_left, doc_top, doc_w, doc_h = doc_rect
            lines, window_img, _ = self._capture_window_doc_lines(prefix=prefix, log=log)

            if not lines:
                log(f"{prefix}Doc list OCR returned no lines after typing Streitwert.")
                continue
            matches, inc_tokens, _, debug_rows = self._filter_streitwert_rows(lines)
            ordered = self._prioritize_streitwert_matches(matches, inc_tokens)
            ordered = self._apply_ignore_top_doc_row(ordered, prefix=prefix, log=log)
            if not ordered:
                log_reason = ", ".join(f"{reason}: {raw}" for raw, reason in debug_rows[:10]) or "no OCR rows"
                self._append_ocr_log(f"{aktenzeichen}_doclist_nomatch", "doclist", lines, prefix=prefix, log=log)
                if window_img is not None:
                    snapshot_path = self._save_preview(window_img, f"doclist_nomatch_{aktenzeichen}")
                    if snapshot_path:
                        log(f"{prefix}Saved raw doclist snapshot to {snapshot_path}")
                if debug_rows:
                    debug_preview = " | ".join(raw for raw, _ in debug_rows[:5])
                    log(f"{prefix}Candidate preview: {debug_preview}")
                log(f"{prefix}No matching rows for '{aktenzeichen}'. Details: {log_reason}")
                continue
            first = ordered[0]
            tag = first.get("token") or "any"
            log_label = f"{aktenzeichen}_{first.get('raw', '')}".strip("_")
            self._append_ocr_log(log_label, "doclist", lines, prefix=prefix, log=log)
            preview = ", ".join(f"{m.get('token') or 'any'} ▸ {m.get('raw')}" for m in ordered[:3])
            log(f"{prefix}Selecting {tag} match: {first.get('raw')} | candidates: {preview}")
            if not self._select_doclist_entry(first, doc_rect, prefix=prefix, log=log):
                log(f"{prefix}Unable to activate doc row: {first.get('raw')}")
                continue
            if not self._click_view_button(prefix=prefix, log=log):
                log(f"{prefix}Skipping entry because the View button click failed.")
                continue
            log(f"{prefix}Clicked View button for the selected row.")
            if doc_wait > 0:
                time.sleep(doc_wait)
            amount = self._process_open_pdf(
                prefix=prefix,
                log_label=log_label,
                log=log,
                search_term=self.config.get("streitwert_term") or "Streitwert",
                retype=True,
            )

            rec = {
                "aktenzeichen": aktenzeichen,
                "row_text": first.get("norm", ""),
                "amount": amount or "",
                "instance_detected": None,
            }
            results.append(rec)
            log(f"{prefix}{aktenzeichen} / {first.get('norm', '')} ▸ {amount or '(none)'}")
            self._close_active_pdf(prefix=prefix, log=log)
            time.sleep(0.5)
        output_csv = self.config.get("streitwert_results_csv") or "streitwert_results.csv"
        if results:
            output_dir = os.path.dirname(output_csv)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")
            log(f"Done. Saved Streitwert results to {output_csv}")
        else:
            log("No Streitwert results were collected from the Excel list.")
        rechnungen_csv = None
        if include_rechnungen and rechnungen_results is not None:
            rechnungen_csv = self.config.get("rechnungen_results_csv") or "Streitwert_Results_Rechnungen.csv"
            if rechnungen_results:
                output_dir = os.path.dirname(rechnungen_csv)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                pd.DataFrame(rechnungen_results).to_csv(rechnungen_csv, index=False, encoding="utf-8-sig")
                log(f"Done. Saved Rechnungen results to {rechnungen_csv}")
            else:
                log("No Rechnungen summaries were captured during Streitwert scan.")
        return {"output_csv": output_csv, "rows": len(results), "rechnungen_csv": rechnungen_csv}

    def run_rechnungen_only(self, progress_callback=None):
        log = progress_callback or (lambda msg: None)
        self._apply_tesseract_path()
        if not self.current_rect:
            self.connect_rdp()
        if not self._doclist_abs_rect():
            raise RuntimeError("Doc list region is not calibrated.")
        queries = self._gather_aktenzeichen_queries()
        if not queries:
            raise RuntimeError("No Aktenzeichen entries were found in the configured Excel sheet.")
        skip_waits = self._should_skip_manual_waits()
        wait_setting = self.config.get("rechnungen_search_wait", self.config.get("post_search_wait", 1.2))
        try:
            list_wait = float(wait_setting)
        except Exception:
            list_wait = float(core.DEFAULTS.get("rechnungen_search_wait", 1.2))
        if skip_waits:
            list_wait = 0.0
        results = []
        total = len(queries)
        for idx, (aktenzeichen, _) in enumerate(queries, start=1):
            prefix = f"[Rechnungen {idx}/{total}] "
            log(f"{prefix}Searching doc list for Aktenzeichen: {aktenzeichen}")
            if not self._type_doclist_query(aktenzeichen, log=log, prefix=prefix):
                log(f"{prefix}Unable to type Aktenzeichen. Skipping entry.")
                continue
            if list_wait > 0:
                time.sleep(list_wait)
            self._wait_for_doc_search_ready(prefix=prefix, log=log, reason="after Aktenzeichen search")
            self._wait_for_doclist_ready(prefix=prefix, log=log, reason="after Aktenzeichen search")
            summary = self._extract_rechnungen_summary(prefix=prefix, log=log)
            if summary is None:
                log(f"{prefix}Rechnungen capture returned no data; storing defaults.")
                summary = self._summarize_rechnungen_entries([])
            else:
                self._log_rechnungen_summary(prefix, summary, log=log)
            row = self._build_rechnungen_result_row(aktenzeichen, summary)
            row["instance_detected"] = None
            results.append(row)
        output_csv = self.config.get("rechnungen_only_results_csv") or "rechnungen_only_results.csv"
        if results:
            output_dir = os.path.dirname(output_csv)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")
            log(f"Done. Saved Rechnungen-only results to {output_csv}")
        else:
            log("No Rechnungen values were captured from the Excel list.")
        return {"output_csv": output_csv, "rows": len(results)}

    def run_rechnungen_gg(self, progress_callback=None):
        log = progress_callback or (lambda msg: None)
        self._apply_tesseract_path()
        if not self.current_rect:
            self.connect_rdp()
        if not self._doclist_abs_rect():
            raise RuntimeError("Doc list region is not calibrated.")
        queries = self._gather_aktenzeichen_queries()
        if not queries:
            raise RuntimeError("No Aktenzeichen entries were found in the configured Excel sheet.")
        skip_waits = self._should_skip_manual_waits()
        wait_setting = self.config.get("rechnungen_search_wait", self.config.get("post_search_wait", 1.2))
        try:
            list_wait = float(wait_setting)
        except Exception:
            list_wait = float(core.DEFAULTS.get("rechnungen_search_wait", 1.2))
        if skip_waits:
            list_wait = 0.0
        results = []
        total = len(queries)
        for idx, (aktenzeichen, _) in enumerate(queries, start=1):
            prefix = f"[GG {idx}/{total}] "
            log(f"{prefix}Searching doc list for Aktenzeichen: {aktenzeichen}")
            if not self._type_doclist_query(aktenzeichen, log=log, prefix=prefix):
                log(f"{prefix}Unable to type Aktenzeichen. Skipping entry.")
                results.append(
                    {
                        "aktenzeichen": aktenzeichen,
                        "gg_detected": False,
                        "gg_count": 0,
                        "gg_amounts": "",
                        "gg_dates": "",
                        "gg_invoices": "",
                        "gg_raw": "",
                    }
                )
                continue
            if list_wait > 0:
                time.sleep(list_wait)
            self._wait_for_doc_search_ready(prefix=prefix, log=log, reason="after Aktenzeichen search")
            self._wait_for_doclist_ready(prefix=prefix, log=log, reason="after Aktenzeichen search")
            entries = self._extract_rechnungen_gg_entries(prefix=prefix, log=log)
            if not entries:
                summary_line = self._build_gg_summary_line(aktenzeichen, [])
                log(f"{prefix}{summary_line}")
                results.append(
                    {
                        "aktenzeichen": aktenzeichen,
                        "gg_detected": False,
                        "gg_count": 0,
                        "gg_amounts": "",
                        "gg_dates": "",
                        "gg_invoices": "",
                        "gg_raw": "",
                    }
                )
                continue
            amounts = [entry.get("amount", "") or "" for entry in entries]
            dates = [entry.get("date", "") or "" for entry in entries]
            invoices = [entry.get("invoice", "") or "" for entry in entries]
            raw_rows = [entry.get("raw", "") or "" for entry in entries]
            summary_line = self._build_gg_summary_line(aktenzeichen, entries)
            log(f"{prefix}{summary_line}")
            results.append(
                {
                    "aktenzeichen": aktenzeichen,
                    "gg_detected": True,
                    "gg_count": len(entries),
                    "gg_amounts": "; ".join(amounts),
                    "gg_dates": "; ".join(dates),
                    "gg_invoices": "; ".join(invoices),
                    "gg_raw": "; ".join(raw_rows),
                }
            )
        output_csv = self.config.get("rechnungen_gg_results_csv") or "rechnungen_gg_results.csv"
        if results:
            output_dir = os.path.dirname(output_csv)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")
            log(f"Done. Saved GG results to {output_csv}")
        else:
            log("No GG entries were captured from the Excel list.")
        return {"output_csv": output_csv, "rows": len(results)}

    def run_sw_gg_extraction(self, progress_callback=None):
        log = progress_callback or (lambda msg: None)
        self._apply_tesseract_path()
        if not self.current_rect:
            self.connect_rdp()
        self._ensure_sw_gg_ready()
        queries = self._gather_aktenzeichen_queries()
        if not queries:
            raise RuntimeError("No Aktenzeichen entries were found in the configured Excel sheet.")
        wait_setting = self.config.get("rechnungen_search_wait", self.config.get("post_search_wait", 1.2))
        try:
            list_wait = float(wait_setting)
        except Exception:
            list_wait = float(core.DEFAULTS.get("rechnungen_search_wait", 1.2))
        if self._should_skip_manual_waits():
            list_wait = 0.0
        rows_out = []
        total = len(queries)

        # Wrap in try-finally to ensure CSV is saved even on timeout/error
        try:
            for idx, (aktenzeichen, _) in enumerate(queries, start=1):
                prefix = f"[SW GG {idx}/{total}] "
                log(f"{prefix}Searching doc list for Aktenzeichen: {aktenzeichen}")

                try:
                    if not self._type_doclist_query(aktenzeichen, log=log, prefix=prefix):
                        log(f"{prefix}Unable to type Aktenzeichen. Skipping entry.")
                        continue
                    if list_wait > 0:
                        time.sleep(list_wait)
                    self._wait_for_doc_search_ready(prefix=prefix, log=log, reason="after Aktenzeichen search")
                    self._wait_for_doclist_ready(prefix=prefix, log=log, reason="after Aktenzeichen search")

                    # Capture all invoice entries (GG and non-GG)
                    entries = self._collect_doclist_entries(prefix=prefix, log=log)
                    if not entries:
                        log(f"{prefix}No Rechnungen entries detected in the calibrated region.")
                        continue

                    log(f"{prefix}Processing {len(entries)} invoice entries...")

                    wert_found = False
                    for entry_idx, entry in enumerate(entries, start=1):
                        try:
                            label = entry.get("raw") or f"Rechnung #{entry_idx}"
                            is_gg = entry.get("is_gg", False)
                            date = entry.get("date", "")
                            invoice_num = entry.get("invoice", "")
                            entry_amount = entry.get("amount", "")

                            gg_indicator = "🔴 GG" if is_gg else "🔵"
                            log(f"{prefix}[{entry_idx}/{len(entries)}] {gg_indicator} Opening '{label}'.")

                            if not self._open_sw_gg_entry(entry, prefix=prefix, log=log):
                                log(f"{prefix}Unable to open '{label}'.")
                                rows_out.append({
                                    "aktenzeichen": aktenzeichen,
                                    "rechnung": label,
                                    "date": date,
                                    "invoice_number": invoice_num,
                                    "entry_amount": entry_amount,
                                    "is_gg": is_gg,
                                    "extracted_wert": "",
                                    "status": "failed_to_open",
                                })
                                continue

                            # Extract Wert from the opened document (with timeout protection)
                            try:
                                extracted_wert, raw_text = self._extract_sw_gg_amount(prefix=prefix, log=log)
                            except Exception as extract_exc:
                                log(f"{prefix}ERROR during extraction: {extract_exc}")
                                extracted_wert = ""
                                raw_text = f"extraction_error: {str(extract_exc)[:200]}"

                            if extracted_wert:
                                log(f"{prefix}{label} -> Wert: {extracted_wert} ✓")
                                wert_found = True
                            else:
                                log(f"{prefix}{label} -> (Wert not found)")

                            rows_out.append({
                                "aktenzeichen": aktenzeichen,
                                "rechnung": label,
                                "date": date,
                                "invoice_number": invoice_num,
                                "entry_amount": entry_amount,
                                "is_gg": is_gg,
                                "extracted_wert": extracted_wert or "",
                                "raw_text": (raw_text or "")[:500],
                                "status": "success" if extracted_wert else "wert_not_found",
                            })

                            self._close_sw_gg_window(prefix=prefix, log=log)
                            time.sleep(0.2)

                            # Skip remaining entries once we found a Wert
                            if wert_found:
                                remaining = len(entries) - entry_idx
                                if remaining > 0:
                                    log(f"{prefix}✓ Wert found! Skipping {remaining} remaining entries for this Aktenzeichen.")
                                break
                        except Exception as entry_exc:
                            log(f"{prefix}ERROR processing entry {entry_idx}: {entry_exc}")
                            rows_out.append({
                                "aktenzeichen": aktenzeichen,
                                "rechnung": f"Entry {entry_idx}",
                                "date": "",
                                "invoice_number": "",
                                "entry_amount": "",
                                "is_gg": False,
                                "extracted_wert": "",
                                "status": f"error: {str(entry_exc)[:100]}",
                            })
                            continue

                except Exception as az_exc:
                    log(f"{prefix}ERROR processing Aktenzeichen: {az_exc}")
                    rows_out.append({
                        "aktenzeichen": aktenzeichen,
                        "rechnung": "",
                        "date": "",
                        "invoice_number": "",
                        "entry_amount": "",
                        "is_gg": False,
                        "extracted_wert": "",
                        "status": f"az_error: {str(az_exc)[:100]}",
                    })
                    continue

        finally:
            # ALWAYS save CSV, even if timeout/error occurred
            output_csv = (self.config.get("sw_gg_results_csv") or "sw_gg_results.csv").strip()
            if rows_out and output_csv:
                try:
                    output_dir = os.path.dirname(output_csv)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    pd.DataFrame(rows_out).to_csv(output_csv, index=False, encoding="utf-8-sig")
                    log(f"[SW GG] ✓ Saved {len(rows_out)} entries to {output_csv}")
                except Exception as save_exc:
                    log(f"[SW GG] ERROR saving CSV: {save_exc}")
            elif rows_out:
                log(f"[SW GG] Collected {len(rows_out)} entries (CSV output disabled).")
            else:
                log("[SW GG] No entries were processed.")

        return {"output_csv": output_csv if rows_out else "", "rows": len(rows_out)}

    def test_sw_gg_setup(self, progress_callback=None):
        log = progress_callback or (lambda msg: None)
        prefix = "[SW GG Test] "
        self._apply_tesseract_path()
        if not self.current_rect:
            self.connect_rdp()
        self._ensure_sw_gg_ready()
        queries = self._gather_aktenzeichen_queries()
        if not queries:
            raise RuntimeError("No Aktenzeichen entries were found in the configured Excel sheet.")
        aktenzeichen, _ = queries[0]
        log(f"{prefix}Using Aktenzeichen '{aktenzeichen}' for the test run.")
        if not self._type_doclist_query(aktenzeichen, log=log, prefix=prefix):
            raise RuntimeError("Unable to type Aktenzeichen into the document list.")
        wait_setting = self.config.get("rechnungen_search_wait", self.config.get("post_search_wait", 1.2))
        try:
            list_wait = float(wait_setting)
        except Exception:
            list_wait = float(core.DEFAULTS.get("rechnungen_search_wait", 1.2))
        if self._should_skip_manual_waits():
            list_wait = 0.0
        if list_wait > 0:
            time.sleep(list_wait)
        self._wait_for_doc_search_ready(prefix=prefix, log=log, reason="after Aktenzeichen search")
        self._wait_for_doclist_ready(prefix=prefix, log=log, reason="after Aktenzeichen search")

        # Capture all invoice entries
        entries = self._collect_doclist_entries(prefix=prefix, log=log)
        if not entries:
            raise RuntimeError("No Rechnungen entries detected in the calibrated region.")

        log(f"{prefix}Found {len(entries)} invoice entries to test:")
        log(f"{prefix}{'='*60}")
        for idx, entry in enumerate(entries, 1):
            gg_indicator = "🔴 GG" if entry.get("is_gg") else "🔵 Other"
            log(f"{prefix}{idx}. {gg_indicator} - {entry.get('raw', 'N/A')}")
        log(f"{prefix}{'='*60}")

        # Test with entries - wrap in try to save results even on error
        results = []
        wert_found = False
        try:
            for idx, entry in enumerate(entries, 1):
                try:
                    label = entry.get("raw") or f"Rechnung #{idx}"
                    log(f"{prefix}Testing entry {idx}/{len(entries)}: {label}")
                    if not self._open_sw_gg_entry(entry, prefix=prefix, log=log):
                        log(f"{prefix}Unable to open '{label}'; skipping.")
                        results.append({"entry": label, "amount": "", "status": "failed_to_open"})
                        continue

                    try:
                        amount, raw_text = self._extract_sw_gg_amount(prefix=prefix, log=log)
                    except Exception as extract_exc:
                        log(f"{prefix}ERROR during extraction: {extract_exc}")
                        amount = ""
                        raw_text = f"extraction_error: {str(extract_exc)[:200]}"

                    self._close_sw_gg_window(prefix=prefix, log=log)

                    if amount:
                        log(f"{prefix}{label} -> Wert: {amount} ✓")
                        wert_found = True
                    else:
                        log(f"{prefix}{label} -> (Wert not detected)")
                        if raw_text:
                            log(f"{prefix}OCR Text sample: {raw_text[:200]}")
                    results.append(
                        {
                            "entry": label,
                            "amount": amount or "",
                            "raw_text": raw_text or "",
                            "status": "success" if amount else "wert_not_found",
                        }
                    )

                    # Skip remaining entries once we found a Wert
                    if wert_found:
                        remaining = len(entries) - idx
                        if remaining > 0:
                            log(f"{prefix}✓ Wert found! Skipping {remaining} remaining test entries.")
                        break
                except Exception as entry_exc:
                    log(f"{prefix}ERROR processing entry {idx}: {entry_exc}")
                    results.append({
                        "entry": f"Entry {idx}",
                        "amount": "",
                        "raw_text": "",
                        "status": f"error: {str(entry_exc)[:100]}",
                    })
                    continue
        finally:
            # Save test results to CSV even on error/timeout
            test_csv = "sw_gg_test_results.csv"
            if results:
                try:
                    pd.DataFrame(results).to_csv(test_csv, index=False, encoding="utf-8-sig")
                    log(f"{prefix}✓ Saved {len(results)} test results to {test_csv}")
                except Exception as save_exc:
                    log(f"{prefix}ERROR saving test results CSV: {save_exc}")

        return {
            "amount": results[0]["amount"] if results else "",
            "entry": results[0]["entry"] if results else "",
            "aktenzeichen": aktenzeichen,
            "total_entries": len(entries),
            "entry_list": [e.get("raw", "") for e in entries],
            "gg_count": sum(1 for e in entries if e.get("is_gg")),
            "results": results,
        }

    def test_akten_setup(self, progress_callback=None):
        """
        Test the Akten workflow setup by:
        1. Get an Aktenzeichen from the Excel file
        2. Type the Aktenzeichen in the document list search
        3. Find files matching the configured search term (e.g., "Aufforderungsschreiben")
        4. Click on the matching file to select it
        5. Click the View button to open the PDF
        6. Extract the date from the PDF

        Returns:
            dict with keys:
                - aktenzeichen: The AZ from Excel
                - file_matched: The filename that matched
                - date_extracted: The extracted date
                - status: "ok" or error message
        """
        log = progress_callback or (lambda msg: None)
        prefix = "[Akten Test] "

        base_log = log
        session_lines: list[str] = []
        session_label = "akten_test_setup"
        session_log_path = ""
        session_log_saved = False
        doclist_log_path = ""
        doclist_preview_path = ""

        def buffered_log(message: str):
            session_lines.append(message)
            base_log(message)

        def ensure_session_log(label_suffix: str = "") -> str:
            nonlocal session_log_saved, session_log_path
            if session_log_saved or not session_lines:
                return session_log_path
            label = session_label
            if label_suffix:
                label = f"{label}_{label_suffix}"
            session_log_path = (
                self._save_session_log(label, session_lines, prefix=prefix, log=base_log) or ""
            )
            session_log_saved = True
            return session_log_path

        def finalize_result(payload: dict[str, Any], label_suffix: str = "") -> dict[str, Any]:
            ensure_session_log(label_suffix=label_suffix)
            if not isinstance(payload, dict):
                payload = {}
            payload.setdefault("session_log_path", session_log_path)
            payload.setdefault("doclist_log_path", doclist_log_path)
            payload.setdefault("doclist_preview_path", doclist_preview_path)
            return payload

        log = buffered_log

        self._apply_tesseract_path()

        # Connect to RDP if needed
        if not self.current_rect:
            self.connect_rdp()
            if not self.current_rect:
                ensure_session_log("rdp_connection_failed")
                raise RuntimeError("Unable to connect to RDP window.")

        # Focus the RDP window
        try:
            self._focus_rdp_window()
        except Exception:
            pass

        # Check all required calibrations
        checks = [
            ("Doc list region", bool(self._doclist_abs_rect())),
            ("PDF text region", bool(self.config.get("pdf_text_region"))),
            ("View button", isinstance(self.config.get("doc_view_point"), (list, tuple))),
            ("PDF close button", isinstance(self.config.get("pdf_close_point"), (list, tuple))),
        ]

        all_ok = True
        for label, ok in checks:
            status = "✓ OK" if ok else "✗ MISSING"
            log(f"{prefix}{label}: {status}")
            all_ok = all_ok and ok

        if not all_ok:
            ensure_session_log("missing_calibration")
            raise RuntimeError(
                "One or more Akten calibration steps are missing. "
                "Please configure Doc list region, PDF text region, "
                "View button, and PDF close button."
            )

        # Get the search term for matching files
        search_term = self.config.get("akten_search_term") or "Aufforderungsschreiben"
        log(f"{prefix}Using search term: '{search_term}'")

        # Get an Aktenzeichen from the Excel file
        log(f"{prefix}Loading Aktenzeichen from Excel...")
        queries = self._gather_aktenzeichen_queries()
        if not queries:
            ensure_session_log("excel_empty")
            raise RuntimeError("No Aktenzeichen entries were found in the configured Excel sheet.")

        aktenzeichen, _ = queries[0]
        safe_az_label = core.sanitize_filename(aktenzeichen or "")
        if safe_az_label:
            session_label = f"{safe_az_label}_akten_test"
        log(f"{prefix}Using Aktenzeichen: '{aktenzeichen}'")

        # Type the Aktenzeichen in the doclist search
        if not self._type_doclist_query(aktenzeichen, prefix=prefix, log=log):
            ensure_session_log("type_doclist_failed")
            raise RuntimeError(f"Unable to type Aktenzeichen '{aktenzeichen}' into document list.")

        # Wait for search to complete
        try:
            wait_after = float(self.config.get("post_search_wait", 1.2))
        except Exception:
            wait_after = 1.2

        if wait_after > 0:
            time.sleep(max(0.2, wait_after))

        # Wait for doc search to be ready
        self._wait_for_doc_search_ready(prefix=prefix, log=log, reason="test setup")
        self._wait_for_doclist_ready(prefix=prefix, log=log, reason="test setup")

        # Get the document list region
        doc_rect = self._doclist_abs_rect()
        if not doc_rect:
            ensure_session_log("doclist_region_missing")
            raise RuntimeError("Doc list region is not calibrated.")

        # Capture the doclist rows to find files matching the search term
        log(f"{prefix}📋 Performing OCR on document list region...")
        rows = self._ocr_doclist_rows_boxes(log=log, prefix=prefix)

        if not rows:
            log(f"{prefix}⚠ No documents found for Aktenzeichen '{aktenzeichen}' (OCR returned 0 rows).")
            return finalize_result(
                {
                    "aktenzeichen": aktenzeichen,
                    "file_matched": None,
                    "date_extracted": None,
                    "status": "no_files_found",
                    "search_term": search_term,
                    "file_count": 0,
                },
                label_suffix="no_files",
            )

        log(f"{prefix}✓ OCR detected {len(rows)} document(s) in the list:")
        log(f"{prefix}{'='*60}")
        for idx, (text, box) in enumerate(rows, 1):
            log(f"{prefix}  {idx}. {text}")
        log(f"{prefix}{'='*60}")

        # Find the file that matches the search term
        log(f"{prefix}🔍 Searching for files containing '{search_term}'...")
        ignore_tokens = self._akten_ignore_tokens()
        matched_idx, matched_file = self._find_doclist_match(
            rows, search_term, ignore_tokens=ignore_tokens, prefix=prefix, log=log
        )

        if not matched_file:
            return finalize_result(
                {
                    "aktenzeichen": aktenzeichen,
                    "file_matched": None,
                    "date_extracted": None,
                    "status": "no_match_for_search_term",
                    "search_term": search_term,
                    "file_count": len(rows),
                },
                label_suffix="no_match",
            )

        # Save OCR results to log file
        lines_for_log = [(box[0], box[1], box[2] - box[0], box[3] - box[1], text) for text, box in rows]
        doclist_log_path = (
            self._append_ocr_log(
                f"{aktenzeichen}_test_doclist", "doclist", lines_for_log, prefix=prefix, log=log
            )
            or doclist_log_path
        )

        # Capture screenshot with bounding boxes
        try:
            log(f"{prefix}📸 Capturing doclist screenshot with bounding boxes...")
            x, y, w, h = doc_rect
            doclist_img = core.grab_xywh(x, y, w, h)

            if doclist_img:
                highlight = doclist_img.convert("RGBA")
                draw = ImageDraw.Draw(highlight)

                for idx, (text, box) in enumerate(rows, 1):
                    left, top, right, bottom = box
                    rel_left = left - x
                    rel_top = top - y
                    rel_right = right - x
                    rel_bottom = bottom - y

                    # Use different color for matched file
                    if idx - 1 == matched_idx:
                        color = (255, 0, 0, 220)  # Red for matched
                    else:
                        color = (0, 255, 0, 220)  # Green for others

                    draw.rectangle(
                        [rel_left, rel_top, rel_right, rel_bottom],
                        outline=color,
                        width=3,
                    )

                    draw.text(
                        (rel_left + 5, rel_top + 5),
                        f"#{idx}",
                        fill=(255, 255, 0, 255),
                    )

                preview_image = highlight.convert("RGB")
                preview_path = self._save_preview(preview_image, f"akten_test_{aktenzeichen}_doclist")
                if preview_path:
                    doclist_preview_path = str(preview_path)
                    log(f"{prefix}✓ Screenshot saved: {preview_path.name}")
        except Exception as e:
            log(f"{prefix}⚠ Failed to save screenshot: {e}")

        # Select the matched file
        file_text, file_box = matched_file
        log(f"{prefix}📄 Selecting matched file: {file_text}")
        log(f"{prefix}🖱 Click position: ({file_box[0]}, {file_box[1]})")

        file_match = {
            "raw": file_text,
            "x": file_box[0],
            "y": file_box[1],
            "w": max(file_box[2] - file_box[0], 1),
            "h": max(file_box[3] - file_box[1], 1),
        }

        if not self._select_doclist_entry(file_match, doc_rect, prefix=prefix, log=log):
            ensure_session_log("doclist_select_failed")
            raise RuntimeError(f"Unable to select the file '{file_text}'.")

        log(f"{prefix}✓ Successfully selected the file.")

        # Wait for selection to register
        per_row_wait = max(0.2, float(self.config.get("rechnungen_region_wait", 0.4)))
        time.sleep(per_row_wait)

        # Click the View button to open the PDF
        log(f"{prefix}👁 Opening PDF with View button...")
        if not self._click_view_button(prefix=prefix, log=log):
            ensure_session_log("view_button_failed")
            raise RuntimeError("View button click failed.")

        # Wait for PDF to open
        doc_wait = max(0.2, float(self.config.get("doc_open_wait", 1.0)))
        extra_wait = max(0.0, float(self.config.get("pdf_view_extra_wait", 0.0) or 0.0))
        log(f"{prefix}⏳ Waiting {doc_wait + extra_wait:.1f}s for PDF to open...")
        time.sleep(doc_wait + extra_wait)

        self._wait_for_pdf_ready(prefix=prefix, log=log, reason="Akten PDF open")

        # Extract the date from the PDF
        log(f"{prefix}📄 Performing OCR on PDF to extract date...")
        date_value, full_text = self._extract_akten_date_from_pdf(prefix=prefix, log=log)

        # Save PDF screenshot with date highlighted
        self._save_pdf_screenshot(date_found=date_value, aktenzeichen=aktenzeichen, prefix=prefix, log=log)

        # Save OCR log for PDF
        if full_text:
            pdf_lines = [(0, idx * 20, 800, 20, line) for idx, line in enumerate(full_text.split('\n'))]
            self._append_ocr_log(f"{aktenzeichen}_test_pdf", "pdf", pdf_lines, prefix=prefix, log=log)

        if date_value:
            log(f"{prefix}✓✓ SUCCESS: Date extracted: {date_value}")
        else:
            log(f"{prefix}⚠⚠ WARNING: Date not detected")
            if full_text:
                log(f"{prefix}  PDF OCR text sample (first 200 chars):")
                log(f"{prefix}  {full_text[:200]}")
            else:
                log(f"{prefix}  PDF OCR returned no text")

        # Close the PDF
        log(f"{prefix}🚪 Closing PDF...")
        self._close_active_pdf(prefix=prefix, log=log)
        time.sleep(0.4)

        log(f"{prefix}{'='*60}")
        log(f"{prefix}✓ Test completed successfully!")
        log(f"{prefix}  Aktenzeichen: {aktenzeichen}")
        log(f"{prefix}  File matched: {file_text}")
        log(f"{prefix}  Date extracted: {date_value or '(not found)'}")
        log(f"{prefix}{'='*60}")

        result_payload = {
            "aktenzeichen": aktenzeichen,
            "file_matched": file_text,
            "date_extracted": date_value,
            "status": "ok" if date_value else "date_not_detected",
            "search_term": search_term,
            "file_count": len(rows),
            "raw_text": full_text[:200] if full_text else "",
            "doclist_log_path": doclist_log_path,
            "doclist_preview_path": doclist_preview_path,
        }
        return finalize_result(result_payload)

    def test_akten_doclist_capture(self, progress_callback=None):
        """
        Capture the entire RDP window after typing an Aktenzeichen and highlight
        the doclist row that matches the configured search term.
        """
        log = progress_callback or (lambda msg: None)
        prefix = "[Akten Capture] "

        self._apply_tesseract_path()
        if not self.current_rect:
            self.connect_rdp()
            if not self.current_rect:
                raise RuntimeError("Unable to connect to RDP window.")

        doc_rect = self._doclist_abs_rect()
        if not doc_rect:
            raise RuntimeError("Doc list region is not calibrated.")

        queries = self._gather_aktenzeichen_queries()
        if not queries:
            raise RuntimeError("No Aktenzeichen entries were found in the configured Excel sheet.")

        aktenzeichen, _ = queries[0]
        search_term = (self.config.get("akten_search_term") or "Aufforderungsschreiben").strip()
        ignore_tokens = self._akten_ignore_tokens()

        log(f"{prefix}Using Aktenzeichen '{aktenzeichen}' for capture test.")
        if not self._type_doclist_query(aktenzeichen, prefix=prefix, log=log):
            raise RuntimeError(f"Unable to type Aktenzeichen '{aktenzeichen}' into document list.")

        wait_after = float(self.config.get("post_search_wait", 1.2))
        if wait_after > 0:
            time.sleep(max(0.2, wait_after))
        self._wait_for_doc_search_ready(prefix=prefix, log=log, reason="capture test")
        self._wait_for_doclist_ready(prefix=prefix, log=log, reason="capture test")

        lines, window_img, _ = self._capture_window_doc_lines(prefix=prefix, log=log)
        if not lines:
            raise RuntimeError("No doclist OCR lines were detected for the capture test.")

        rows = []
        for entry in lines:
            if not (isinstance(entry, (list, tuple)) and len(entry) == 5):
                continue
            x, y, w, h, text = entry
            width = max(1, int(w or 0))
            height = max(1, int(h or 0))
            rows.append((text or "", (x, y, x + width, y + height)))

        log(f"{prefix}✓ OCR detected {len(rows)} row(s) in the doclist.")
        log(f"{prefix}{'='*60}")
        for idx, (text, _) in enumerate(rows, 1):
            log(f"{prefix}  {idx}. {text}")
        log(f"{prefix}{'='*60}")

        lines_for_log = []
        for text, box in rows:
            left, top, right, bottom = box
            lines_for_log.append((left, top, right - left, bottom - top, text))
        doclist_log_path = self._append_ocr_log(
            f"{aktenzeichen}_capture_doclist", "doclist", lines_for_log, prefix=prefix, log=log
        )

        matched_idx, matched_entry = self._find_doclist_match(
            rows, search_term, ignore_tokens=ignore_tokens, prefix=prefix, log=log
        )

        preview_path = ""
        doc_preview_name = f"akten_capture_{aktenzeichen}"
        win_left, win_top, _, _ = self.current_rect
        doc_left, doc_top, _, _ = doc_rect
        offset_x = doc_left - win_left
        offset_y = doc_top - win_top

        try:
            capture_img = window_img
            if capture_img is None:
                left, top, right, bottom = self.current_rect
                capture_img = core.grab_xywh(left, top, right - left, bottom - top)
            if capture_img:
                highlight = capture_img.convert("RGBA")
                draw = ImageDraw.Draw(highlight)
                for idx, (_, box) in enumerate(rows, 1):
                    left, top, right, bottom = box
                    abs_left = left + offset_x
                    abs_top = top + offset_y
                    abs_right = right + offset_x
                    abs_bottom = bottom + offset_y
                    color = (255, 0, 0, 220) if idx - 1 == matched_idx else (0, 255, 0, 160)
                    draw.rectangle([abs_left, abs_top, abs_right, abs_bottom], outline=color, width=3)
                    draw.text((abs_left + 5, abs_top + 5), f"#{idx}", fill=(255, 255, 0, 255))
                preview_image = highlight.convert("RGB")
                saved = self._save_preview(preview_image, doc_preview_name)
                if saved:
                    preview_path = str(saved)
                    log(f"{prefix}✓ Window capture saved: {saved.name}")
        except Exception as exc:
            log(f"{prefix}⚠ Failed to save capture screenshot: {exc}")

        matched_text = matched_entry[0] if matched_entry else ""
        return {
            "aktenzeichen": aktenzeichen,
            "search_term": search_term,
            "matched_entry": matched_text,
            "matched_index": (matched_idx + 1) if matched_idx is not None else 0,
            "row_count": len(rows),
            "doclist_log_path": doclist_log_path or "",
            "doclist_preview_path": preview_path,
            "matched": bool(matched_entry),
        }

    def _safe_crop(self, img: Image.Image, box):
        if img is None:
            return None
        left, top, right, bottom = box
        left = max(0, int(left))
        top = max(0, int(top))
        right = max(left + 1, int(right))
        bottom = max(top + 1, int(bottom))
        return img.crop((left, top, right, bottom))

    def detect_instance(self, prefix: str = ""):
        return {"instance": 1}

    def _current_aktenzeichen_text(self):
        return ""

    def _fees_should_skip(self, line: str) -> bool:
        bad = (self.config.get("fees_bad_prefixes") or "").strip()
        if not bad:
            return False
        tokens = [b.strip().lower() for b in bad.split(";") if b.strip()]
        if not tokens:
            return False
        raw = (line or "").strip()
        line_lower = raw.lower()
        line_norm = core.normalize_line_soft(raw).lower()
        for tok in tokens:
            if not tok:
                continue
            tok_norm = core.normalize_line_soft(tok).lower()
            if line_lower.startswith(tok) or line_norm.startswith(tok_norm):
                return True
        return False

    def _type_fast(self, text: str):
        delay = max(0.002, float(self.config.get("type_delay", 0.02)))
        self._type_exact_text(text, fallback_interval=delay)

    def _click_file_search_and_type_kfb(self):
        region = self.config.get("fees_file_search_region") or [0, 0, 0, 0]
        try:
            ax, ay, aw, ah = core.rel_to_abs(self.current_rect, region)
        except Exception:
            raise RuntimeError("Fees file search region not calibrated.")
        x = int(ax + max(1, aw // 2))
        y = int(ay + max(1, ah // 2))
        self._focus_rdp_window()
        pyautogui.click(x, y)
        time.sleep(0.1)
        token = (self.config.get("fees_search_token") or "KFB").strip()
        self._type_fast(token)
        pyautogui.press("enter")

    def _select_doclist_entry(self, match, doc_rect, focus_first=False, prefix="", log=None):
        """
        Click on a document list entry based on its match coordinates.
        This ensures that the file with the matching token is clicked, not just the view button.

        Args:
            match: Dictionary with keys 'x', 'y', 'w', 'h', 'raw' representing the matched line
            doc_rect: Tuple of (x, y, width, height) for the document list region
            focus_first: If True, click on the doc list region first to ensure focus
            prefix: Logging prefix
            log: Optional logging function

        Returns:
            True if successfully clicked the entry, False otherwise
        """
        if not match or not doc_rect:
            if log:
                log(f"{prefix}No match/doc_rect provided for selection.")
            return False

        rx, ry, rw, rh = doc_rect

        # If focus_first is requested, click in the doc list region to ensure it has focus
        if focus_first:
            focus_x = rx + max(5, rw // 40)
            focus_y = ry + max(5, rh // 40)
            if log:
                log(f"{prefix}Focusing doc list at ({focus_x}, {focus_y}).")
            pyautogui.click(focus_x, focus_y)
            time.sleep(0.2)

        # Calculate the click position based on the match coordinates
        # Use the match's x, y coordinates which are relative to the doc_rect
        local_x = match.get("x", 0) + max(
            12, min(match.get("w", 20) // 2 if match.get("w") else 20, max(match.get("w", 20) - 10, 12))
        )
        local_y = match.get("y", 0) + max(
            10, min(match.get("h", 20) // 2 if match.get("h") else 20, max(match.get("h", 20) - 10, 10))
        )

        # Ensure coordinates are within bounds
        local_x = max(5, local_x)
        local_y = max(5, local_y)

        # Convert to absolute screen coordinates
        click_x = rx + min(local_x, max(rw - 5, 5))
        click_y = ry + min(local_y, max(rh - 5, 5))

        if log:
            log(
                f"{prefix}Clicking row '{match.get('raw', '')}' at ({click_x}, {click_y}) "
                f"size ({match.get('w', 0)}x{match.get('h', 0)})."
            )

        # Move to and click the position
        pyautogui.moveTo(click_x, click_y)
        pyautogui.click(click_x, click_y)
        time.sleep(0.2)

        if log:
            log(f"{prefix}Clicked row '{match.get('raw', '')}'.")
        return True

    def _click_view_button(self, prefix="", log=None):
        """
        Click the View button to open the selected document.

        Args:
            prefix: Logging prefix
            log: Optional logging function

        Returns:
            True if successfully clicked the View button, False otherwise
        """
        if not self.current_rect:
            if log:
                log(f"{prefix}No RDP window rect available for clicking View button.")
            return False

        # Get the configured view button point
        point = self.config.get("doc_view_point")
        if not (isinstance(point, (list, tuple)) and len(point) == 2):
            msg = "View button point is not configured. Please calibrate it."
            if log:
                log(f"{prefix}{msg}")
            else:
                if prefix:
                    print(f"{prefix}{msg}")
            return False

        # Ensure RDP window has focus
        try:
            self._focus_rdp_window()
        except Exception:
            pass

        # Convert relative point to absolute coordinates
        vx, vy = core.rel_to_abs(self.current_rect, point)

        if log:
            log(f"{prefix}Clicking View button at ({vx}, {vy}).")

        # Move to and click the View button
        pyautogui.moveTo(vx, vy)
        pyautogui.click(vx, vy)
        time.sleep(0.1)

        return True

    def _close_active_pdf(self, prefix="", log=None):
        """
        Click the configured PDF close button to dismiss the open viewer.
        """
        log = log or (lambda msg: None)
        if not self.current_rect:
            log(f"{prefix}No RDP window rect available for closing PDF.")
            return False

        point = self.config.get("pdf_close_point")
        if not (isinstance(point, (list, tuple)) and len(point) == 2):
            log(f"{prefix}PDF close button point is not configured.")
            return False

        try:
            self._focus_rdp_window()
        except Exception:
            pass

        try:
            cx, cy = core.rel_to_abs(self.current_rect, point)
        except Exception as exc:
            log(f"{prefix}Unable to resolve PDF close point: {exc}")
            return False

        try:
            pyautogui.click(cx, cy)
            close_wait = max(0.0, float(self.config.get("pdf_close_wait", 0.3) or 0.0))
            if close_wait > 0:
                log(f"{prefix}Waiting {close_wait}s after clicking PDF close button...")
                time.sleep(close_wait)
            log(f"{prefix}Clicked PDF close button at ({cx}, {cy}).")
            return True
        except Exception as exc:
            log(f"{prefix}Failed to click PDF close button: {exc}")
            return False

    def _fees_overlay_wait(self, what: str):
        skip = bool(self.config.get("fees_overlay_skip_waits", True))
        if skip:
            if what == "doclist":
                self._wait_for_doclist_ready(prefix="[Fees] ")
            elif what == "pdf":
                self._wait_for_pdf_ready(prefix="[Fees] ")
        else:
            time.sleep(0.6)

    def _ensure_sw_gg_ready(self):
        required_regions = {
            "rechnungen_region": "Rechnungen region",
            "sw_gg_region": "SW GG region",
        }
        required_points = {
            "sw_gg_close_point": "SW GG close button point",
        }
        missing = [
            label
            for key, label in required_regions.items()
            if not (isinstance(self.config.get(key), (list, tuple)) and len(self.config.get(key)) == 4)
        ]
        missing.extend(
            label
            for key, label in required_points.items()
            if not (isinstance(self.config.get(key), (list, tuple)) and len(self.config.get(key)) == 2)
        )
        if missing:
            raise RuntimeError(f"SW GG setup incomplete: {', '.join(missing)}.")

    def _collect_doclist_entries(self, prefix="", log=None):
        """
        Capture doclist entries and return them in a format suitable for clicking.
        Returns a list of dict entries with 'match' (the filtered streitwert-style match object)
        and additional metadata.
        """
        # Capture doclist lines using the standard OCR flow
        lines, window_img, _ = self._capture_window_doc_lines(prefix=prefix, log=log)
        if not lines:
            if log:
                log(f"{prefix}No doclist OCR lines captured.")
            return []

        # Filter the lines using the same logic as the Streitwert flow
        matches, inc_tokens, _, debug_rows = self._filter_streitwert_rows(lines)

        if not matches:
            if log:
                log(f"{prefix}No matching doclist entries after filtering.")
                if debug_rows:
                    preview = " | ".join(raw for raw, _ in debug_rows[:3])
                    log(f"{prefix}Rejected entries preview: {preview}")
            return []

        # Prioritize matches based on include tokens
        ordered = self._prioritize_streitwert_matches(matches, inc_tokens)
        ordered = self._apply_ignore_top_doc_row(ordered, prefix=prefix, log=log)

        if not ordered:
            if log:
                log(f"{prefix}No doclist entries remaining after priority filtering.")
            return []

        # Parse the matched lines to extract additional metadata (dates, amounts, etc.)
        # This helps identify GG entries and extract invoice numbers
        parsed_entries = self._parse_rechnungen_entries(lines, prefix=prefix, log=log)

        # Build a mapping from raw text to parsed entry for quick lookup
        parsed_map = {}
        for entry in parsed_entries:
            entry_raw = entry.get("label") or entry.get("label_normalized") or entry.get("raw", "")
            if entry_raw:
                parsed_map[entry_raw.lower()] = entry

        # Convert matches to entries with click coordinates and metadata
        entries = []
        for match in ordered:
            raw_text = match.get("raw", "")

            # Try to find corresponding parsed entry for metadata
            parsed_entry = None
            match_key = raw_text.lower()
            if match_key in parsed_map:
                parsed_entry = parsed_map[match_key]

            # Extract metadata from parsed entry if available
            date = parsed_entry.get("date", "") if parsed_entry else ""
            invoice = parsed_entry.get("invoice", "") if parsed_entry else ""
            amount = parsed_entry.get("amount", "") if parsed_entry else ""
            is_gg = parsed_entry.get("label_is_gg", False) if parsed_entry else False

            entries.append({
                "match": match,  # The filtered match object (for clicking)
                "raw": raw_text,
                "date": date,
                "invoice": invoice,
                "amount": amount,
                "is_gg": is_gg,
            })

        if log:
            log(f"{prefix}Collected {len(entries)} doclist entries for processing.")

        return entries

    def _open_sw_gg_entry(self, entry, prefix="", log=None):
        """
        Open a SW GG entry by clicking on the doclist row and then double-clicking to open.

        Args:
            entry: Dict with 'match' key containing the filtered match object from _filter_streitwert_rows
            prefix: Log prefix
            log: Logging callback

        Returns:
            bool: True if successfully opened, False otherwise
        """
        if not isinstance(entry, dict):
            if log:
                log(f"{prefix}Invalid entry type: {type(entry)}")
            return False

        match = entry.get("match")
        if not match:
            if log:
                log(f"{prefix}Entry missing 'match' object.")
            return False

        # Get doclist region for coordinate calculation
        doc_rect = self._doclist_abs_rect()
        if not doc_rect:
            if log:
                log(f"{prefix}Doc list region is not calibrated.")
            return False

        # First, select the doclist entry (single click to focus)
        if not self._select_doclist_entry(match, doc_rect, prefix=prefix, log=log):
            if log:
                log(f"{prefix}Unable to select doc row: {match.get('raw')}")
            return False

        # Now double-click to open the document
        # Calculate the click position from the match coordinates
        rx, ry, rw, rh = doc_rect
        local_x = match.get("x", 0) + max(
            12, min((match.get("w") or 20) // 2, max((match.get("w") or 20) - 10, 12))
        )
        local_y = match.get("y", 0) + max(
            10, min((match.get("h") or 20) // 2, max((match.get("h") or 20) - 10, 10))
        )
        local_x = max(5, local_x)
        local_y = max(5, local_y)
        click_x = rx + min(local_x, max(rw - 5, 5))
        click_y = ry + min(local_y, max(rh - 5, 5))

        try:
            self._focus_rdp_window()
            if log:
                log(f"{prefix}Double-clicking at ({click_x}, {click_y}) to open document.")
            pyautogui.doubleClick(click_x, click_y)

            wait_seconds = max(0.2, float(self.config.get("sw_gg_open_wait", 1.0) or 0.2))
            if log:
                log(f"{prefix}Waiting {wait_seconds}s for SW GG document to open...")
            time.sleep(wait_seconds)
            return True
        except Exception as exc:
            if log:
                log(f"{prefix}Failed to open entry via double-click: {exc}")
            return False

    def _extract_sw_gg_amount(self, prefix="", log=None):
        """
        Extract SW GG amount by OCRing the ENTIRE window and parsing the text.
        This avoids black image issues with region captures.
        """
        if not self.current_rect:
            return None, ""

        # Wait for capture (gives document time to render after opening)
        capture_wait = max(0.0, float(self.config.get("sw_gg_capture_wait", 1.0) or 0.0))
        if capture_wait > 0:
            if log:
                log(f"{prefix}Waiting {capture_wait}s before capturing screenshot...")
            time.sleep(capture_wait)

        try:
            # Capture the ENTIRE RDP window instead of just a region
            left, top, right, bottom = self.current_rect
            w, h = right - left, bottom - top
            if log:
                log(f"{prefix}Capturing entire RDP window at ({left}, {top}, {w}x{h}).")
            img = core.grab_xywh(left, top, w, h)
        except Exception as exc:
            if log:
                log(f"{prefix}Failed to capture RDP window: {exc}")
            return None, ""

        self._save_preview(img, "sw_gg_full_window")

        # OCR the entire window
        text = self._ocr_text(img, context="document")
        if log:
            if text and text.strip():
                lines = text.strip().splitlines()
                preview = " | ".join(lines[:5])
                log(f"{prefix}SW GG OCR preview ({len(lines)} lines): {preview or '(empty)'}")
                # Log keyword being searched
                keyword = (self.config.get("sw_gg_keyword") or core.DEFAULTS.get("sw_gg_keyword", "")).strip()
                log(f"{prefix}Searching for keyword: '{keyword}'")
            else:
                log(f"{prefix}WARNING: OCR returned empty text")

        # Parse amount from the full text (with logging)
        amount = self._sw_gg_parse_amount(text or "", log=log, prefix_msg=prefix)

        if log:
            if amount:
                log(f"{prefix}✓ Extracted amount: {amount}")
            else:
                log(f"{prefix}⚠ No amount found in OCR text")

        return amount, text or ""

    def test_sw_gg_line_preview(self, progress_callback=None):
        """
        Test SW GG line preview with comprehensive error handling and logging.
        Captures the ENTIRE RDP WINDOW, performs OCR, highlights matching lines, and extracts amounts.

        IMPORTANT: You must manually open a SW GG document window BEFORE running this test.
        The test will capture the entire RDP window and search for the keyword in all text.
        """
        log = progress_callback or (lambda msg: None)
        prefix = "[SW GG Line] "

        try:
            # Step 1: Initialize
            log(f"{prefix}Initializing test...")
            log(f"{prefix}⚠ IMPORTANT: Ensure a SW GG document window is open!")
            self._apply_tesseract_path()

            # Step 2: Ensure RDP connection
            if not self.current_rect:
                log(f"{prefix}No RDP connection found, connecting...")
                try:
                    self.connect_rdp()
                    log(f"{prefix}RDP connected successfully.")
                except Exception as e:
                    log(f"{prefix}ERROR: Failed to connect to RDP: {e}")
                    raise RuntimeError(f"RDP connection failed: {e}") from e

            # Step 3: Wait a moment for the screen to settle
            log(f"{prefix}Waiting 0.5s for screen to settle...")
            import time
            time.sleep(0.5)

            # Step 4: Get RDP window bounds
            log(f"{prefix}Config check:")
            log(f"{prefix}  current_rect: {self.current_rect}")

            # Step 5: Capture ENTIRE RDP window (not just a region)
            log(f"{prefix}Capturing ENTIRE RDP window instead of just a region...")
            try:
                left, top, right, bottom = self.current_rect
                w, h = right - left, bottom - top
                log(f"{prefix}Window coordinates: ({left}, {top}, {w}x{h})")
                img = core.grab_xywh(left, top, w, h)
                log(f"{prefix}Screenshot captured: {img.size[0]}x{img.size[1]} pixels")

                # Save the raw screenshot for debugging
                raw_preview_path = self._save_preview(img, "sw_gg_line_raw_capture")
                if raw_preview_path:
                    log(f"{prefix}Raw screenshot saved to: {raw_preview_path}")
            except Exception as e:
                log(f"{prefix}ERROR: Screenshot capture failed: {e}")
                import traceback
                log(f"{prefix}Traceback:\n{traceback.format_exc()}")
                raise RuntimeError(f"Screenshot capture failed: {e}") from e

            # Step 6: Perform OCR on entire window
            log(f"{prefix}Performing OCR on entire window...")
            try:
                text = self._ocr_text(img, context="document")
                if not text or not text.strip():
                    log(f"{prefix}WARNING: OCR returned empty text")
                    lines = []
                else:
                    log(f"{prefix}OCR text length: {len(text)} characters")
                    # Show first 500 chars of OCR text for debugging
                    text_preview = text[:500].replace('\n', ' | ')
                    log(f"{prefix}OCR preview: {text_preview}...")

                    # Parse text into lines manually
                    lines = []
                    for idx, line in enumerate(text.splitlines()):
                        line = line.strip()
                        if line:
                            # Create fake line entries since we don't have position data
                            lines.append((0, idx * 20, img.size[0], 20, line))
                    log(f"{prefix}Parsed {len(lines)} text lines from OCR")

                    # Log first few lines for debugging
                    if lines:
                        log(f"{prefix}First OCR lines:")
                        for idx, (_, _, _, _, txt) in enumerate(lines[:10], 1):
                            log(f"{prefix}  {idx}. {txt}")
            except Exception as e:
                log(f"{prefix}ERROR: OCR failed: {e}")
                import traceback
                log(f"{prefix}Traceback:\n{traceback.format_exc()}")
                raise RuntimeError(f"OCR failed: {e}") from e

            if not lines:
                log(f"{prefix}ERROR: No OCR lines detected")
                raise RuntimeError("No OCR lines detected in the RDP window.")

            # Step 7: Search for keyword match
            keyword = (self.config.get("sw_gg_keyword") or core.DEFAULTS.get("sw_gg_keyword", "")).strip().lower()
            prefix_token = (self.config.get("sw_gg_value_prefix") or core.DEFAULTS.get("sw_gg_value_prefix", "Wert")).strip()

            # Normalize keyword for matching (replace commas with periods, remove extra spaces)
            keyword_normalized = keyword.replace(',', '.').replace('  ', ' ')

            log(f"{prefix}Searching for keyword: '{keyword}' (normalized: '{keyword_normalized}')")
            log(f"{prefix}Value prefix: '{prefix_token}'")

            match = None
            for idx, entry in enumerate(lines):
                x0, y0, w0, h0, text_line = entry

                # Normalize the text line for comparison (replace commas with periods)
                text_normalized = text_line.lower().replace(',', '.').replace('  ', ' ')

                if keyword_normalized and keyword_normalized in text_normalized:
                    match = entry
                    log(f"{prefix}✓ MATCHED line {idx + 1}: {text_line}")
                    log(f"{prefix}  Matched text (normalized): {text_normalized[:100]}")
                    break

            if not match:
                log(f"{prefix}WARNING: No line matched keyword '{keyword_normalized}'")
                log(f"{prefix}  Tried matching in {len(lines)} lines")
                # Show all lines for debugging
                log(f"{prefix}All OCR lines:")
                for idx, (_, _, _, _, txt) in enumerate(lines, 1):
                    log(f"{prefix}  {idx}. {txt[:100]}")

            # Step 8: Extract amount
            amount = ""
            line_text = ""

            if match:
                x0, y0, w0, h0, line_text = match
                log(f"{prefix}Extracting amount from: {line_text}")
                try:
                    amount = self._extract_amount_from_sw_gg_line(line_text, prefix_token) or ""
                    if amount:
                        log(f"{prefix}Extracted amount: {amount}")
                    else:
                        log(f"{prefix}WARNING: No amount found in matched line")
                except Exception as e:
                    log(f"{prefix}ERROR: Amount extraction failed: {e}")
                    amount = ""

            # Step 9: Create preview image
            log(f"{prefix}Creating preview image...")
            preview_image = None
            try:
                highlight = img.convert("RGBA")
                draw = ImageDraw.Draw(highlight)

                if match:
                    lx, ly, lw, lh, _ = match
                    draw.rectangle(
                        [lx, ly, lx + lw, ly + lh],
                        outline=(255, 0, 90, 220),
                        width=4,
                    )
                    log(f"{prefix}Drew highlight box at ({lx}, {ly}, {lw}x{lh})")

                preview_image = highlight.convert("RGB")
                preview_path = self._save_preview(preview_image, "sw_gg_line_preview")
                if preview_path:
                    log(f"{prefix}Preview saved to: {preview_path}")
                else:
                    log(f"{prefix}WARNING: Preview image not saved")
            except Exception as e:
                log(f"{prefix}ERROR: Preview image creation failed: {e}")
                preview_path = None

            # Step 10: Return results
            log(f"{prefix}Test completed successfully.")
            return {
                "line_text": line_text,
                "amount": amount,
                "preview_path": str(preview_path) if preview_path else "",
                "preview_image": preview_image,
                "found": bool(match),
            }

        except RuntimeError:
            # Already logged, just re-raise
            raise
        except Exception as exc:
            # Unexpected error
            log(f"{prefix}UNEXPECTED ERROR: {exc}")
            import traceback
            log(f"{prefix}Full traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"SW GG Line Preview failed: {exc}") from exc

    def _sw_gg_parse_amount(self, text: str, log=None, prefix_msg=""):
        """
        Parse amount from SW GG OCR text by finding the keyword line and extracting the amount.

        Args:
            text: OCR text from entire window
            log: Optional logging function
            prefix_msg: Optional prefix for log messages

        Returns:
            Extracted amount string or None
        """
        if not text:
            if log:
                log(f"{prefix_msg}[Parse] No text provided")
            return None

        keyword = (self.config.get("sw_gg_keyword") or core.DEFAULTS.get("sw_gg_keyword", "")).strip().lower()
        value_prefix = (self.config.get("sw_gg_value_prefix") or core.DEFAULTS.get("sw_gg_value_prefix", "Wert")).strip()

        if log:
            log(f"{prefix_msg}[Parse] Keyword: '{keyword}'")
            log(f"{prefix_msg}[Parse] Value prefix: '{value_prefix}'")

        # Normalize keyword for matching (same as line preview)
        keyword_normalized = keyword.replace(',', '.').replace('  ', ' ')

        if log:
            log(f"{prefix_msg}[Parse] Keyword normalized: '{keyword_normalized}'")

        lines = [line.strip() for line in text.splitlines() if line.strip()]

        if log:
            log(f"{prefix_msg}[Parse] Checking {len(lines)} lines...")

        for idx, line in enumerate(lines, 1):
            # Normalize the text line for comparison (same as line preview)
            text_normalized = line.lower().replace(',', '.').replace('  ', ' ')

            # Check if keyword matches in normalized text
            if keyword_normalized and keyword_normalized not in text_normalized:
                continue

            # Found matching line
            if log:
                log(f"{prefix_msg}[Parse] ✓ Matched line {idx}: {line[:100]}")

            # Extract amount from matched line
            amount = self._extract_amount_from_sw_gg_line(line, value_prefix)
            if amount:
                if log:
                    log(f"{prefix_msg}[Parse] ✓ Extracted amount: {amount}")
                return amount
            else:
                if log:
                    log(f"{prefix_msg}[Parse] ⚠ Line matched but no amount extracted from: {line[:100]}")

        if log:
            log(f"{prefix_msg}[Parse] ✗ No matching line found with keyword '{keyword_normalized}'")
        return None

    def _extract_amount_from_sw_gg_line(self, line: str, prefix: str):
        """
        Extract amount from SW GG line with improved normalization.
        Handles complex formats like: "(4) VV (Wert: 5.912,44 Euro) -253,50 Euro"
        """
        if not line:
            return None

        # Normalize the line
        normalized = core.normalize_line(line)

        # Find the prefix (e.g., "Wert:")
        pattern = prefix or "Wert"
        lower_line = normalized.lower()
        idx = lower_line.find(pattern.lower())

        if idx == -1:
            # Try without normalization as fallback
            idx = line.lower().find(pattern.lower())
            if idx == -1:
                return None
            segment = line[idx:]
        else:
            segment = normalized[idx:]

        # Extract amount after prefix with multiple strategies
        # Strategy 1: Look for pattern like "Wert: 5.912,44" or "Wert: 5.912,44 Euro"
        # Match: optional spaces, colon, spaces, then amount
        match = re.search(
            r"(?::\s*)?(?:€\s*)?([\d]{1,3}(?:\.\d{3})*,\d{2})(?:\s*(?:EUR|Euro|€))?",
            segment,
            re.IGNORECASE
        )

        if match:
            amount_str = match.group(1)
            # Clean and standardize
            return core.clean_amount_display(amount_str)

        # Strategy 2: Broader pattern - any amount-like string
        match = re.search(r"([\d\.\s]+\d,\d{2})", segment)
        if match:
            amount_str = match.group(1).replace(" ", "")
            return core.clean_amount_display(amount_str)

        # Strategy 3: Even broader - just digits and separators
        match = re.search(r"([\d\.,]+)", segment)
        if match:
            return core.clean_amount_display(match.group(1))

        return None

    def _close_sw_gg_window(self, prefix="", log=None):
        point = self.config.get("sw_gg_close_point")
        close_wait = max(0.0, float(self.config.get("sw_gg_close_wait", 0.3) or 0.0))
        try:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                self._click_relative_point(point)
                time.sleep(0.2)
                if log:
                    log(f"{prefix}Waiting {close_wait}s after closing SW GG window...")
                time.sleep(close_wait)
                return True
        except Exception as exc:
            if log:
                log(f"{prefix}Failed to click SW GG close button: {exc}, trying ESC...")
        pyautogui.press("esc")
        time.sleep(0.2)
        if log:
            log(f"{prefix}Waiting {close_wait}s after closing SW GG window (ESC)...")
        time.sleep(close_wait)
        return False

    def _fees_analyze_seiten_region(self, x0, y0, width, height, max_clicks=None, img=None, lang=None):
        result = {"img": None, "token_summary": "", "digit_summary": "", "positions": []}
        if width <= 0 or height <= 0:
            return result
        lang_override = lang
        try:
            preview = img or core.grab_xywh(x0, y0, width, height)
        except Exception:
            return result
        result["img"] = preview
        band_top = int(preview.height * 0.55)
        band = self._safe_crop(preview, (0, band_top, preview.width, preview.height))
        if not band or band.width < 2 or band.height < 2:
            return result
        scale = 3
        proc = band.resize((band.width * scale, band.height * scale), Image.LANCZOS)
        proc = ImageOps.autocontrast(ImageOps.grayscale(proc))
        try:
            df = self._ocr_data(proc, context="program", psm=6, lang=lang_override)
        except Exception:
            return result
        if df is None or "text" not in df.columns:
            return result
        texts = []
        digits = []
        for row in df.itertuples():
            text = str(getattr(row, "text", "")).strip()
            if not text:
                continue
            left = getattr(row, "left", 0) or 0
            top = getattr(row, "top", 0) or 0
            width_raw = getattr(row, "width", 0) or 1
            height_raw = getattr(row, "height", 0) or 1
            left = float(left) / scale
            top = float(top) / scale + band_top
            width_raw = float(width_raw) / scale
            height_raw = float(height_raw) / scale
            texts.append(text)
            if re.fullmatch(r"\d+", text):
                center_x = left + width_raw / 2.0
                center_y = top + height_raw / 2.0
                digits.append((text, center_x, center_y, height_raw))
        if texts:
            summary = " | ".join(texts)
            if len(summary) > 200:
                summary = summary[:197] + "..."
            result["token_summary"] = summary
        digits.sort(key=lambda item: item[1])
        positions = []
        limit = max_clicks or len(digits)
        for idx, (label, cx, cy, h) in enumerate(digits):
            if idx >= limit:
                break
            abs_x = int(round(x0 + cx))
            baseline = y0 + cy
            target_y = baseline - h * 2.0
            target_y = max(y0 + 5, min(y0 + height - 5, target_y))
            positions.append((idx + 1, abs_x, int(round(target_y))))
        result["digit_summary"] = " ".join(label for label, *_ in digits[:limit])
        result["positions"] = positions
        return result

    def _fees_iter_click_pages(self, max_clicks=None, return_positions=False):
        max_clicks = (
            int(self.config.get("fees_pages_max_clicks", 12))
            if max_clicks is None
            else max_clicks
        )
        try:
            x0, y0, width, height = core.rel_to_abs(
                self.current_rect, self.config.get("fees_seiten_region", [0, 0, 0, 0])
            )
        except Exception:
            return None
        analysis = self._fees_analyze_seiten_region(
            x0, y0, width, height, max_clicks=max_clicks
        )
        positions = analysis.get("positions") or []
        if not positions:
            step = max(1, width // max(1, max_clicks))
            positions = [
                (i + 1, x0 + step // 2 + i * step, y0 + height // 2)
                for i in range(max_clicks)
            ]
        for idx, x, y in positions:
            pyautogui.click(x, y)
            time.sleep(0.15)
            self._fees_overlay_wait("pdf")
            if return_positions:
                yield (idx, x, y)
            else:
                yield idx

    def _is_pdf_open(self):
        region = self.config.get("pdf_text_region")
        if not region:
            return False
        try:
            x, y, w, h = core.rel_to_abs(self.current_rect, region)
            if w <= 0 or h <= 0:
                return False
            img = core.grab_xywh(x, y, w, h)
            df = self._ocr_data(img, context="document", psm=6)
            if df is None or "text" not in df.columns:
                return False
            texts = [t for t in df["text"].tolist() if str(t).strip()]
            return len(texts) > 3
        except Exception:
            return False

    def _fees_scan_current_page_amount(self):
        region = self.config.get("pdf_text_region")
        if not region:
            return None
        try:
            x, y, w, h = core.rel_to_abs(self.current_rect, region)
            if w <= 0 or h <= 0:
                return None
            img = core.grab_xywh(x, y, w, h)
            df = self._ocr_data(img, context="document", psm=6)
        except Exception:
            return None
        if df is None or "text" not in df.columns:
            return None
        text = " ".join(t for t in df["text"].tolist() if str(t).strip()).strip()
        if not text:
            return None
        if not self._WORDS_HINT_RE.search(text):
            return None
        match = self._AMT_NUM_RE.search(text)
        if match:
            return match.group(0)
        return None

    def _fees_is_kfb_line(self, text: str) -> bool:
        if not text:
            return False
        if self._KFB_RE.search(text):
            return True
        norm = core.normalize_line_soft(text).lower()
        if not norm:
            return False
        if self._KFB_WORD_RE.search(norm):
            return True
        compact = re.sub(r"[^a-zß]", "", norm)
        return compact.startswith("kostenfestsetzungsbeschl")

    def _fees_collect_kfb_rows(self, log=None):
        lines = self._capture_doclist_lines(log=log, prefix="[Fees] ")
        matches = []
        for idx, entry in enumerate(lines):
            x, y, w, h, text = entry
            text = (text or "").strip()
            if not text:
                continue
            if self._fees_should_skip(text):
                continue
            if not self._fees_is_kfb_line(text):
                continue
            matches.append(
                {
                    "index": idx,
                    "raw": text,
                    "match": {
                        "norm": text,
                        "raw": text,
                        "token": "",
                        "x": x or 0,
                        "y": y or 0,
                        "w": w or 0,
                        "h": h or 0,
                    },
                }
            )
        return matches

    def _fees_open_and_extract_one(self, match, prefix=""):
        doc_rect = self._doclist_abs_rect()
        if not doc_rect:
            raise RuntimeError("Doc list region not calibrated.")
        self._focus_rdp_window()
        if not self._select_doclist_entry(match["match"], doc_rect, focus_first=True, prefix=prefix):
            return None
        self._fees_overlay_wait("doclist")
        if not self._click_view_button(prefix=prefix):
            return None
        self._fees_overlay_wait("pdf")
        wait_seconds = float(self.config.get("doc_open_wait", 1.2))
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        amount = None
        iterator = self._fees_iter_click_pages()
        if iterator is None:
            iterator = []
        for _ in iterator:
            val = self._fees_scan_current_page_amount()
            if val:
                amount = val
                break
        self._close_active_pdf(prefix=prefix)
        self._fees_overlay_wait("doclist")
        return amount

    def run_fees(self, progress_callback=None):
        log = progress_callback or (lambda msg: None)
        self._apply_tesseract_path()
        if not self.current_rect:
            self.connect_rdp()
        inst_info = self.detect_instance(prefix="[Fees] ") or {}
        inst = inst_info.get("instance") or 1
        log(f"[Fees] Instance → open first {inst} KFB file(s).")
        self._click_file_search_and_type_kfb()
        self._fees_overlay_wait("doclist")
        kfb_rows = self._fees_collect_kfb_rows(log=log)
        if not kfb_rows:
            log("[Fees] No KFB entries found.")
            return {"rows": 0, "output_csv": self.config.get("fees_csv_path", "fees_results.csv")}
        log(f"[Fees] Found {len(kfb_rows)} KFB entr{'y' if len(kfb_rows)==1 else 'ies'}.")
        amounts = [None, None, None]
        total = min(inst, len(kfb_rows))
        for j in range(total):
            match = kfb_rows[j]
            log(f"[Fees] Opening KFB {j+1}/{total}: row {match['index']} → {match['raw']}")
            amt = self._fees_open_and_extract_one(match, prefix="[Fees] ")
            amounts[j] = amt
            if amt:
                log(f"[Fees] Amount: {amt}")
            else:
                log("[Fees] Amount not found.")
        aktenzeichen = self._current_aktenzeichen_text()
        row = {
            "aktenzeichen": aktenzeichen,
            "instance_detected": inst,
            "fees_inst1": amounts[0] or "",
            "fees_inst2": amounts[1] or "",
            "fees_inst3": amounts[2] or "",
        }
        path = self.config.get("fees_csv_path", "fees_results.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        log(f"[Fees] Saved → {path}")
        return {"rows": total, "output_csv": path}

    # ------------------------------------------------------------------
    def _get_active_profile(self):
        active = (self.config.get("active_amount_profile") or "").strip()
        if not active:
            return None
        for profile in self.config.get("amount_profiles") or []:
            if profile.get("name") == active:
                return profile
        return None

    @staticmethod
    def _crop_to_profile(image, rel_box):
        if not rel_box or len(rel_box) != 4:
            return image
        l, t, w, h = rel_box
        width, height = image.size
        crop_box = (
            max(0, int(l * width)),
            max(0, int(t * height)),
            max(0, int((l + w) * width)),
            max(0, int((t + h) * height)),
        )
        crop_box = (
            min(crop_box[0], width),
            min(crop_box[1], height),
            min(max(crop_box[2], crop_box[0] + 1), width),
            min(max(crop_box[3], crop_box[1] + 1), height),
        )
        return image.crop(crop_box)

    def _focus_rdp_window(self) -> None:
        if self._rdp_window is None:
            self.connect_rdp()
        try:
            if self._rdp_window is not None:
                self._rdp_window.set_focus()
        except Exception:
            self.connect_rdp()
            if self._rdp_window is not None:
                self._rdp_window.set_focus()

    def get_mouse_position(self) -> tuple[int, int]:
        pos = pyautogui.position()
        return int(pos.x), int(pos.y)

    def _click_relative_point(self, rel_point):
        rect = self._ensure_rect()
        x, y = core.rel_to_abs(rect, rel_point)
        pyautogui.click(x, y)

    def _clear_search_field(self) -> None:
        try:
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("backspace")
        except Exception as exc:
            raise RuntimeError(f"Failed to clear search field: {exc}") from exc

    def _type_exact_text(self, text, *, fallback_interval=0.02, progress=None) -> None:
        message = None
        pause = max(fallback_interval, 0.001)
        try:
            core.send_keys(
                str(text),
                pause=pause,
                with_spaces=True,
                with_tabs=True,
                with_newlines=True,
                turn_off_numlock=False,
                vk_packet=True,
            )
        except Exception as exc:
            message = f"[Type] send_keys failed ({exc}); using simulated typing."
            pyautogui.typewrite(str(text), interval=fallback_interval)
        if message and progress:
            progress(message)

    def _load_input_dataframe(self) -> pd.DataFrame:
        path = (self.config.get("excel_path") or "").strip()
        if not path:
            raise RuntimeError("Excel path is not configured.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Excel file not found: {path}")
        sheet = self.config.get("excel_sheet", "Sheet1")
        try:
            if isinstance(sheet, str) and sheet.strip().isdigit():
                sheet_name = int(sheet.strip())
            else:
                sheet_name = sheet
            df = pd.read_excel(path, sheet_name=sheet_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to read Excel '{path}': {exc}") from exc
        return df

    def _prepare_batch_rows(self, df: pd.DataFrame):
        cfg = self.config
        start_cell = (cfg.get("start_cell") or "").strip()
        max_rows = cfg.get("max_rows")
        max_rows = int(max_rows) if str(max_rows).strip().isdigit() else 0
        if start_cell:
            match = re.match(r"^\s*([A-Za-z]+)\s*([0-9]+)\s*$", start_cell)
            if not match:
                raise ValueError(
                    f"Invalid start cell '{start_cell}'. Use Excel-like references (e.g., B2)."
                )
            col_letters, row_num = match.group(1).upper(), int(match.group(2))
            col_idx = self._column_index_from_letters(col_letters)
            if col_idx < 0 or col_idx >= len(df.columns):
                raise ValueError(
                    f"Start cell column '{col_letters}' is outside the available columns."
                )
            start_idx = max(row_num - 2, 0)
            rows = df.iloc[start_idx:]
        else:
            input_col = cfg.get("input_column") or "query"
            if input_col not in df.columns:
                raise ValueError(
                    f"Input column '{input_col}' not found in sheet ({list(df.columns)})."
                )
            col_idx = df.columns.get_loc(input_col)
            rows = df
        if max_rows > 0:
            rows = rows.head(max_rows)
        return rows, col_idx

    @staticmethod
    def _column_index_from_letters(letters: str) -> int:
        idx = 0
        for ch in letters:
            if not ("A" <= ch <= "Z"):
                raise ValueError(f"Invalid column letter '{ch}' in '{letters}'.")
            idx = idx * 26 + (ord(ch) - 64)
        return idx - 1

    def _resolve_ocr_settings(self, context: str = "document") -> tuple[str, str]:
        ctx = "program" if context == "program" else "document"
        engine_key = "program_ocr_engine" if ctx == "program" else "document_ocr_engine"
        lang_key = "program_ocr_lang" if ctx == "program" else "document_ocr_lang"
        fallback_lang = (self.config.get("tesseract_lang") or "deu+eng").strip() or "deu+eng"
        engine = (self.config.get(engine_key) or "tesseract").strip().lower()
        if engine not in core.AVAILABLE_OCR_ENGINES:
            engine = "tesseract"
        lang = (self.config.get(lang_key) or fallback_lang).strip() or fallback_lang
        return engine, lang

    def _ocr_data(
        self,
        image,
        *,
        context: str = "document",
        psm: int = 6,
        lang: str | None = None,
        engine: str | None = None,
    ):
        eng_name, lang_name = self._resolve_ocr_settings(context)
        if lang:
            lang_name = lang
        if engine:
            eng_name = engine
        try:
            return core.do_ocr_data(image, lang=lang_name, psm=psm, engine=eng_name)
        except Exception:
            # Return empty DataFrame on any error
            return pd.DataFrame(
                columns=[
                    "level",
                    "page_num",
                    "block_num",
                    "par_num",
                    "line_num",
                    "word_num",
                    "left",
                    "top",
                    "width",
                    "height",
                    "conf",
                    "text",
                ]
            )

    def _ocr_text(
        self,
        image,
        *,
        context: str = "document",
        lang: str | None = None,
        engine: str | None = None,
        normalize: bool = True,
    ) -> str:
        if image is None:
            return ""
        try:
            eng_name, lang_name = self._resolve_ocr_settings(context)
            if lang:
                lang_name = lang
            if engine:
                eng_name = engine
            try:
                text = core.do_ocr_color(image, lang=lang_name, engine=eng_name)
                # Apply normalization to fix common OCR artifacts
                if normalize and text:
                    text = ocr_preprocessing.normalize_ocr_text(text)
                return text
            except AttributeError as exc:
                if "_thread._local" in str(exc):
                    core.reset_paddle_engines()
                    text = core.do_ocr_color(image, lang=lang_name, engine=eng_name)
                    if normalize and text:
                        text = ocr_preprocessing.normalize_ocr_text(text)
                    return text
                raise
        except Exception:
            return ""

    def _gather_aktenzeichen_queries(self):
        df = self._load_input_dataframe()
        rows, col_idx = self._prepare_batch_rows(df)
        queries = []
        for _, row in rows.iterrows():
            value = row.iloc[col_idx]
            text = "" if value is None else str(value).strip()
            if text and text.lower() != "nan":
                queries.append((text, row.to_dict()))
        return queries

    def _should_skip_manual_waits(self) -> bool:
        return bool(self.config.get("streitwert_overlay_skip_waits")) or bool(
            self.config.get("rechnungen_overlay_skip_waits")
        )

    def _doclist_abs_rect(self):
        if not self.current_rect:
            return None
        region = self.config.get("doclist_region")
        if not (isinstance(region, (list, tuple)) and len(region) == 4):
            return None
        try:
            return core.rel_to_abs(self.current_rect, region)
        except Exception:
            return None

    def _capture_doclist_lines(self, log=None, prefix=""):
        return self._capture_region_lines("doclist_region", label="Doclist", log=log, prefix=prefix)

    def _capture_window_doc_lines(self, prefix="", log=None):
        if not self.current_rect:
            if log:
                log(f"{prefix}ERROR: No RDP connection (current_rect is None)")
            return [], None, []  # Return empty raw_lines too
        doc_rect = self._doclist_abs_rect()
        if not doc_rect:
            if log:
                log(f"{prefix}ERROR: Doc list region is not calibrated.")
            return [], None, []
        doc_left, doc_top, doc_w, doc_h = doc_rect
        win_left, win_top, win_right, win_bottom = self.current_rect
        win_w = max(1, win_right - win_left)
        win_h = max(1, win_bottom - win_top)

        if log:
            log(f"{prefix}Window rect: ({win_left}, {win_top}, {win_w}x{win_h})")
            log(f"{prefix}Doclist rect: ({doc_left}, {doc_top}, {doc_w}x{doc_h})")

        try:
            window_img = core.grab_xywh(win_left, win_top, win_w, win_h)
            if log:
                log(f"{prefix}Captured window image: {window_img.size[0]}x{window_img.size[1]} pixels")
        except Exception as exc:
            if log:
                log(f"{prefix}ERROR: Unable to capture window region: {exc}")
            return [], None, []

        scale = max(1, int(self.config.get("upscale_x") or 3))
        if log:
            log(f"{prefix}OCR scale: {scale}x")

        df = self._ocr_data(window_img, context="document", psm=6)

        if df is None or df.empty:
            if log:
                log(f"{prefix}ERROR: OCR returned no data (DataFrame is None or empty)")
            return [], window_img, []

        if log:
            log(f"{prefix}OCR DataFrame: {len(df)} rows")

        raw_lines = core.lines_from_tsv(df, scale=scale)

        if log:
            log(f"{prefix}Raw OCR lines: {len(raw_lines)}")

        offset_x = doc_left - win_left
        offset_y = doc_top - win_top

        if log:
            log(f"{prefix}Offset: ({offset_x}, {offset_y})")
            log(f"{prefix}Filter bounds: x=[{offset_x}, {offset_x + doc_w}], y=[{offset_y}, {offset_y + doc_h}]")

        filtered = []
        filtered_out_count = 0
        for entry in raw_lines:
            if not (isinstance(entry, (list, tuple)) and len(entry) == 5):
                continue
            x, y, w, h, text = entry
            cx = x + max(w or 0, 1) / 2
            cy = y + max(h or 0, 1) / 2
            if (
                cx < offset_x
                or cx > offset_x + doc_w
                or cy < offset_y
                or cy > offset_y + doc_h
            ):
                filtered_out_count += 1
                continue
            filtered.append((x - offset_x, y - offset_y, w, h, text))

        if log:
            log(f"{prefix}Filtering: {len(raw_lines)} raw → {len(filtered)} inside region ({filtered_out_count} filtered out)")

        # Return raw_lines in addition to filtered and image
        return filtered, window_img, raw_lines

    def _ocr_doclist_rows_boxes(self, log=None, prefix=""):
        """
        OCR the entire RDP window and return doclist rows (text + bounding boxes)
        limited to the calibrated doclist region.
        """
        lines, _, _ = self._capture_window_doc_lines(prefix=prefix, log=log)
        rows = []
        for entry in lines:
            if not (isinstance(entry, (list, tuple)) and len(entry) == 5):
                continue
            x, y, w, h, text = entry
            width = max(1, int(w or 0))
            height = max(1, int(h or 0))
            left = int(x or 0)
            top = int(y or 0)
            rows.append(
                (
                    text or "",
                    (
                        left,
                        top,
                        left + width,
                        top + height,
                    ),
                )
            )
        return rows

    def _find_doclist_match(self, rows, search_term, ignore_tokens=None, prefix="", log=None):
        """
        Locate the doclist row containing the configured search term, skipping ignore tokens.

        Returns:
            tuple[int, tuple[str, tuple[int, int, int, int]]] | tuple[None, None]
        """
        log = log or (lambda msg: None)
        search_term_raw = (search_term or "").strip()
        if not search_term_raw:
            log(f"{prefix}⚠ Search term is empty; cannot locate doclist entry.")
            return None, None

        search_term_lower = search_term_raw.lower()
        search_term_norm = core.normalize_line_soft(search_term_raw).lower()
        ignore_tokens = [tok for tok in (ignore_tokens or []) if tok]

        for idx, (text, box) in enumerate(rows):
            text_lower = (text or "").lower()
            text_norm = core.normalize_line_soft(text or "").lower()

            if ignore_tokens and (
                any(tok in text_lower for tok in ignore_tokens)
                or any(tok in text_norm for tok in ignore_tokens)
            ):
                matched_tok = next((tok for tok in ignore_tokens if tok in text_lower), "")
                log(f"{prefix}  {idx + 1}. ⏭ Skipped (contains ignore token '{matched_tok}'): {text}")
                continue

            if search_term_lower in text_lower or (search_term_norm and search_term_norm in text_norm):
                log(f"{prefix}  {idx + 1}. ✓ MATCH: {text}")
                return idx, (text, box)

            log(f"{prefix}  {idx + 1}. ✗ No match: {text}")

        log(f"{prefix}⚠ No row contained the term '{search_term}'.")
        return None, None

    def _capture_region_lines(self, cfg_key, label="", log=None, prefix="", force_engine=None):
        if not self.current_rect:
            return []
        region = self.config.get(cfg_key)
        if not (isinstance(region, (list, tuple)) and len(region) == 4):
            if log:
                log(f"{prefix}{label or cfg_key.replace('_', ' ').title()} region is not configured.")
            return []
        try:
            if cfg_key in {"rechnungen_region", "rechnungen_gg_region"}:
                self._wait_for_rechnungen_region(
                    cfg_key, prefix=prefix, label=label or cfg_key, log=log
                )
            x, y, w, h = core.rel_to_abs(self.current_rect, region)
            scale = max(1, int(self.config.get("upscale_x") or 3))
            img = core.grab_xywh(x, y, w, h)

            # Upscale image BEFORE preprocessing for better small text detection
            if scale > 1:
                img = ocr_preprocessing.upscale_image(img, scale)
        except Exception as exc:
            if log:
                log(f"{prefix}Failed to capture {label or cfg_key}: {exc}")
            return []

        variants = self._prepare_ocr_variants(img, label=label)
        lines = []
        seen = set()
        doc_context_keys = {"pdf_text_region"}
        context = "document" if cfg_key in doc_context_keys else "program"

        # Use PSM 6 (uniform text block) for most contexts, PSM 11 (sparse text) for tables
        label_upper = (label or "").strip().upper()
        psm = 11 if label_upper in ("GG", "RECHNUNGEN", "DOCLIST", "TABLE") else 6

        for variant in variants:
            try:
                df = self._ocr_data(variant, context=context, psm=psm, engine=force_engine)
            except AttributeError as exc:
                if "_thread._local" in str(exc):
                    core.reset_paddle_engines()
                    try:
                        df = self._ocr_data(variant, context=context, psm=psm)
                    except Exception:
                        continue
                else:
                    continue
            except Exception:
                continue
            variant_lines = core.lines_from_tsv(df, scale=scale)
            for entry in variant_lines:
                if not (isinstance(entry, (list, tuple)) and len(entry) == 5):
                    continue
                x0, y0, w0, h0, text = entry
                key = (
                    int(round((x0 or 0) / 4)),
                    int(round((y0 or 0) / 4)),
                    int(round((w0 or 0) / 4)),
                    int(round((h0 or 0) / 4)),
                    (text or "").strip().lower(),
                )
                if key in seen:
                    continue
                seen.add(key)
                lines.append(entry)

        lines.sort(key=lambda entry: (entry[1], entry[0]))
        if log:
            log(f"{prefix}{label or cfg_key.replace('_', ' ').title()} OCR lines: {len(lines)} (PSM={psm}).")
        return lines

    def _prepare_ocr_variants(self, img, label=""):
        """
        Prepare OCR variants with enhanced preprocessing for tables/invoices.
        Uses specialized preprocessing for Rechnungen/GG workflows.
        """
        if img is None:
            return []

        label_upper = (label or "").strip().upper()

        # Use specialized preprocessing for invoice/table structures
        if label_upper in ("GG", "RECHNUNGEN"):
            variants = ocr_preprocessing.preprocess_for_invoice_rows(img)
            # Add inverted variant for highlighted rows (especially for GG)
            if label_upper == "GG":
                try:
                    gray = img.convert("L")
                    inverted = ImageOps.invert(gray)
                    variants.append(ImageOps.autocontrast(inverted))
                except Exception:
                    pass
            return variants

        # For other labels, use standard table preprocessing
        if label_upper in ("DOCLIST", "TABLE"):
            return ocr_preprocessing.preprocess_for_table_ocr(img, aggressive=False)

        # Fallback to basic variants for other contexts
        variants = []
        try:
            gray = img.convert("L")
        except Exception:
            try:
                gray = ImageOps.grayscale(img)
            except Exception:
                return [img]

        try:
            base_auto = ImageOps.autocontrast(gray)
        except Exception:
            base_auto = gray
        variants.append(base_auto)

        try:
            contrast_img = ImageEnhance.Contrast(gray).enhance(2.0)
            variants.append(ImageOps.autocontrast(contrast_img))
        except Exception:
            pass

        try:
            bright_img = ImageEnhance.Brightness(gray).enhance(1.2)
            variants.append(ImageOps.autocontrast(bright_img))
        except Exception:
            pass

        # Deduplicate
        unique = []
        seen = set()
        for candidate in variants:
            if candidate is None:
                continue
            try:
                key = (candidate.mode, candidate.size, tuple(candidate.histogram()))
            except Exception:
                key = (candidate.mode, candidate.size)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)

        return unique or [base_auto]

    def _akten_ignore_tokens(self):
        raw = (self.config.get("akten_ignore_tokens") or "").strip()
        if not raw:
            tokens = []
        else:
            tokens = [tok.strip().lower() for tok in re.split(r"[;,]+", raw) if tok.strip()]
        if "anlage" not in tokens:
            tokens.append("anlage")
        return tokens

    def _save_pdf_screenshot(self, date_found=None, aktenzeichen="", prefix="", log=None):
        """
        Capture and save a screenshot of the PDF text region with the date highlighted.

        Args:
            date_found: The date string that was found (to highlight it)
            aktenzeichen: Used for naming the file
            prefix: Logging prefix
            log: Logging function
        """
        log = log or (lambda msg: None)

        try:
            region = self.config.get("pdf_text_region")
            if not (isinstance(region, (list, tuple)) and len(region) == 4):
                return None

            x, y, w, h = core.rel_to_abs(self.current_rect, region)
            img = core.grab_xywh(x, y, w, h)

            if not img:
                return None

            # If we found a date, try to highlight it
            if date_found:
                try:
                    highlight = img.convert("RGBA")
                    draw = ImageDraw.Draw(highlight)

                    # Perform OCR with detailed data to get word positions
                    ocr_data = self._ocr_data(img, context="document", psm=6)

                    # Find the date string in the OCR results
                    for _, row in ocr_data.iterrows():
                        word_text = str(row.get("text", "")).strip()
                        if date_found in word_text:
                            # Draw rectangle around the date
                            left = int(row.get("left", 0))
                            top = int(row.get("top", 0))
                            width = int(row.get("width", 0))
                            height = int(row.get("height", 0))

                            draw.rectangle(
                                [left, top, left + width, top + height],
                                outline=(255, 0, 0, 220),  # Red highlight
                                width=4,
                            )
                            break

                    img = highlight.convert("RGB")
                except Exception as e:
                    log(f"{prefix}  ⚠ Failed to highlight date: {e}")

            # Save the screenshot
            safe_az = re.sub(r"[^A-Za-z0-9_-]+", "_", aktenzeichen or "unknown")
            status = "found" if date_found else "notfound"
            preview_path = self._save_preview(img, f"akten_pdf_{safe_az}_{status}")

            if preview_path:
                log(f"{prefix}  📸 PDF screenshot saved: {preview_path.name}")
                return preview_path

        except Exception as e:
            log(f"{prefix}  ⚠ Failed to save PDF screenshot: {e}")

        return None

    def _extract_akten_filter_info(self, doclist_entry_text="", prefix="", log=None):
        """
        Extract an Aktenzeichen directly from the OCR'd doclist entry text.
        """
        log = log or (lambda msg: None)
        entry_text = (doclist_entry_text or "").strip()

        if not entry_text:
            if log:
                log(f"{prefix}  ⚠ No doclist entry text available for AZ extraction.")
            return None, ""

        match = self._AKTEN_AZ_RE.search(entry_text)
        if match:
            az = match.group(1).strip()
            if log:
                log(f"{prefix}  ✓ Extracted AZ from doclist entry: '{az}'")
            return az, entry_text

        if log:
            log(f"{prefix}  ⚠ No AZ pattern found in doclist entry text.")

        return None, entry_text

    def _extract_akten_date_from_pdf(self, prefix="", log=None):
        """
        Extract date from PDF in DD.MM.YYYY format by OCR'ing the entire window,
        falling back to the configured PDF text region if needed.
        """
        log = log or (lambda msg: None)

        text = ""
        text_source = ""

        # Capture the entire RDP window first
        if self.current_rect:
            left, top, right, bottom = self.current_rect
            w, h = right - left, bottom - top
            try:
                log(f"{prefix}  Capturing entire RDP window for PDF OCR ({w}x{h})...")
                window_img = core.grab_xywh(left, top, w, h)
                text = self._ocr_text(window_img, context="document")
                text_source = "window"
            except Exception as exc:
                log(f"{prefix}  ⚠ Failed to capture RDP window: {exc}")

        # Fallback: restricted PDF region
        if not text:
            region = self.config.get("pdf_text_region")
            if not (isinstance(region, (list, tuple)) and len(region) == 4):
                log(f"{prefix}  PDF text region not configured; cannot extract date.")
                return None, ""
            try:
                x, y, w, h = core.rel_to_abs(self.current_rect, region)
                log(f"{prefix}  Capturing PDF text region ({w}x{h}) for OCR...")
                region_img = core.grab_xywh(x, y, w, h)
                text = self._ocr_text(region_img, context="document")
                text_source = "region"
            except Exception as exc:
                log(f"{prefix}  Failed to capture PDF region: {exc}")
                return None, ""

        if not text:
            log(f"{prefix}  ⚠ OCR returned empty text from both window and region captures.")
            return None, ""

        sample_lines = text.split("\n")[:5]
        log(
            f"{prefix}  OCR text from {text_source or 'unknown source'} "
            f"({len(text)} chars, showing first {len(sample_lines)} line(s)):"
        )
        for line in sample_lines:
            if line.strip():
                log(f"{prefix}    | {line.strip()[:80]}")

        match = self._AKTEN_DATE_RE.search(text)
        if match:
            date_str = match.group(2).strip()
            city = match.group(1).strip()
            log(f"{prefix}  ✓ Found date with city pattern: '{city}, {date_str}'")
            return date_str, text

        log(f"{prefix}  Trying simple date pattern (DD.MM.YYYY)...")
        simple_date_re = re.compile(r"\b([0-3]\d\.[0-1]\d\.\d{4})\b")
        date_matches = simple_date_re.findall(text)

        if date_matches:
            for date_str in date_matches:
                parts = date_str.split(".")
                if len(parts) != 3:
                    continue
                try:
                    day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                except ValueError:
                    log(f"{prefix}  ✗ Failed to parse candidate date '{date_str}'")
                    continue
                if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2099:
                    log(f"{prefix}  ✓ Validated date: {date_str}")
                    return date_str, text
                log(f"{prefix}  ✗ Invalid date values: {date_str}")

        log(f"{prefix}  ⚠ No valid date pattern found in OCR text.")
        return None, text

    def _filter_streitwert_rows(self, lines):
        inc_tokens = [
            token.strip().lower()
            for token in (self.config.get("includes") or "").split(",")
            if token.strip()
        ]
        exc_tokens = [
            token.strip().lower()
            for token in (self.config.get("excludes") or "").split(",")
            if token.strip()
        ]
        inc_match = [(tok, core.normalize_for_token_match(tok)) for tok in inc_tokens]
        exc_match = [(tok, core.normalize_for_token_match(tok)) for tok in exc_tokens]
        exclude_k = bool(self.config.get("exclude_prefix_k", False))
        matches = []
        debug_rows = []
        for x, y, w, h, txt in lines:
            raw = (txt or "").strip()
            if not raw:
                continue
            norm = core.normalize_line(raw)
            low_raw = raw.lower()
            low_norm = norm.lower()
            soft_raw = core.normalize_for_token_match(raw)
            soft_norm = core.normalize_for_token_match(norm)
            forced_skip = False
            for label, pattern in core.FORCED_STREITWERT_EXCLUDES:
                try:
                    if pattern.search(raw):
                        debug_rows.append((raw, f"forced exclude '{label}'"))
                        forced_skip = True
                        break
                except Exception:
                    continue
            if forced_skip:
                continue
            if exclude_k and re.match(r"^\s*k", low_raw):
                debug_rows.append((raw, "excluded prefix 'K'"))
                continue
            if exc_tokens:
                fields = [f for f in (low_raw, low_norm, soft_raw, soft_norm) if f]
                excluded = False
                for tok, tok_soft in exc_match:
                    if any(tok in field for field in fields):
                        excluded = True
                        break
                    if tok_soft and any(tok_soft in field for field in fields):
                        excluded = True
                        break
                if excluded:
                    debug_rows.append((raw, "matched exclude token"))
                    continue
            matched_token = None
            if inc_match:
                fields = [f for f in (low_raw, low_norm, soft_raw, soft_norm) if f]
                for tok, tok_soft in inc_match:
                    if any(tok in field for field in fields):
                        matched_token = tok
                        break
                    if tok_soft and any(tok_soft in field for fields in fields):
                        matched_token = tok
                        break
            if inc_tokens and not matched_token:
                debug_rows.append((raw, "missing include token"))
                continue
            matches.append(
                {
                    "norm": norm,
                    "x": x or 0,
                    "y": y or 0,
                    "w": w or 0,
                    "h": h or 0,
                    "raw": raw,
                    "token": matched_token or "",
                    "soft": soft_raw,
                }
            )
        return matches, inc_tokens, exc_tokens, debug_rows

    def _prioritize_streitwert_matches(self, matches, inc_tokens):
        if not matches:
            return []
        if not inc_tokens:
            return matches
        ordered = []
        used = set()
        for tok in inc_tokens:
            best_idx = None
            for idx, match in enumerate(matches):
                if idx in used:
                    continue
                if match.get("token") == tok:
                    best_idx = idx
                    break
            if best_idx is not None:
                ordered.append(matches[best_idx])
                used.add(best_idx)
        for idx, match in enumerate(matches):
            if idx not in used:
                ordered.append(match)
        return ordered

    def _apply_ignore_top_doc_row(self, ordered, prefix="", log=None):
        if not ordered or not bool(self.config.get("ignore_top_doc_row", False)):
            return ordered
        top_match = min(ordered, key=lambda m: (m.get("y", 0), m.get("x", 0)))
        remaining = [match for match in ordered if match is not top_match]
        if log:
            log(f"{prefix}Ignoring top doc row match: {top_match.get('raw', '')}")
        if not remaining and log:
            log(f"{prefix}No remaining Streitwert matches after ignoring top row.")
        return remaining or ordered

    def _type_doclist_query(self, query, *, log=None, prefix=""):
        if not self.current_rect:
            return False
        search_point = self.config.get("search_point")
        if not (isinstance(search_point, (list, tuple)) and len(search_point) == 2):
            if log:
                log(f"{prefix}Doc list search point is not configured. Please calibrate it.")
            return False
        self._focus_rdp_window()
        x, y = core.rel_to_abs(self.current_rect, search_point)
        if log:
            log(f"{prefix}Clicking search box at ({x}, {y}) and typing '{query}'.")
        pyautogui.click(x, y)
        self._clear_search_field()
        self._type_exact_text(query or "", fallback_interval=float(self.config.get("type_delay", 0.02)))
        if log:
            log(f"{prefix}Pressing Enter after typing query.")
        pyautogui.press("enter")
        return True

    def _wait_for_doc_search_ready(self, prefix="", log=None, timeout=10.0, reason=""):
        rel_box = self._search_overlay_rel_box()
        if not rel_box:
            return
        suffix = f" ({reason})" if reason else ""
        start = time.time()
        notified = False
        last_log = 0.0
        while True:
            overlay = self._detect_overlay_in_rel_box(rel_box)
            if not overlay:
                if notified and log:
                    log(f"{prefix}Document search overlay cleared{suffix}.")
                return
            now = time.time()
            desc = (
                overlay.get("norm")
                or core.normalize_line(overlay.get("raw"))
                or overlay.get("raw")
                or "(overlay text not recognized)"
            )
            coords = (
                overlay.get("abs_x", 0),
                overlay.get("abs_y", 0),
                overlay.get("abs_w", 0),
                overlay.get("abs_h", 0),
            )
            if log and ((not notified) or (now - last_log >= 1.5)):
                log(
                    f"{prefix}Document search overlay detected{suffix}: '{desc}' at ({coords[0]}, {coords[1]}, {coords[2]}x{coords[3]}). Waiting..."
                )
                last_log = now
            notified = True
            if time.time() - start > timeout:
                if log:
                    log(f"{prefix}Timeout waiting for document search overlay to clear{suffix}. Continuing.")
                return
            time.sleep(0.5)

    def _wait_for_doclist_ready(self, prefix="", log=None, timeout=12.0, reason=""):
        if not self.config.get("doclist_region"):
            return
        suffix = f" ({reason})" if reason else ""
        start = time.time()
        notified = False
        last_log = 0.0
        while True:
            overlay = self._detect_overlay_in_rel_box(self.config.get("doclist_region"))
            if not overlay:
                if notified and log:
                    log(f"{prefix}Document list overlay cleared{suffix}.")
                return
            now = time.time()
            desc = (
                overlay.get("norm")
                or core.normalize_line(overlay.get("raw"))
                or overlay.get("raw")
                or "(overlay text not recognized)"
            )
            coords = (
                overlay.get("abs_x", 0),
                overlay.get("abs_y", 0),
                overlay.get("abs_w", 0),
                overlay.get("abs_h", 0),
            )
            if log and ((not notified) or (now - last_log >= 1.5)):
                log(
                    f"{prefix}Document list overlay detected{suffix}: '{desc}' at ({coords[0]}, {coords[1]}, {coords[2]}x{coords[3]}). Waiting..."
                )
                last_log = now
            notified = True
            if time.time() - start > timeout:
                if log:
                    log(f"{prefix}Timeout waiting for document list overlay to clear{suffix}. Continuing.")
                return
            time.sleep(0.5)

    def _wait_for_pdf_ready(self, prefix="", log=None, timeout=12.0, reason=""):
        if not self.current_rect or not self.config.get("pdf_text_region"):
            return
        log = log or (lambda msg: None)
        suffix = f" ({reason})" if reason else ""
        start = time.time()
        notified = False
        last_log = 0.0
        while True:
            overlays = []

            pdf_box = self.config.get("pdf_text_region")
            if pdf_box:
                overlay_pdf = self._detect_overlay_in_rel_box(pdf_box)
                if overlay_pdf:
                    overlay_pdf["area"] = "PDF view"
                    overlays.append(overlay_pdf)

            doc_box = self.config.get("doclist_region")
            if doc_box:
                overlay_doc = self._detect_overlay_in_rel_box(doc_box)
                if overlay_doc:
                    overlay_doc["area"] = "Document list"
                    overlays.append(overlay_doc)

            search_box = self._search_overlay_rel_box()
            if search_box:
                overlay_search = self._detect_overlay_in_rel_box(search_box)
                if overlay_search:
                    overlay_search["area"] = "Document search"
                    overlays.append(overlay_search)

            if not overlays:
                if notified:
                    log(f"{prefix}PDF overlays cleared{suffix}.")
                return

            now = time.time()
            if (not notified) or (now - last_log >= 1.5):
                for entry in overlays:
                    desc = (
                        entry.get("norm")
                        or core.normalize_line(entry.get("raw"))
                        or entry.get("raw")
                        or "(overlay text not recognized)"
                    )
                    coords = (
                        entry.get("abs_x", 0),
                        entry.get("abs_y", 0),
                        entry.get("abs_w", 0),
                        entry.get("abs_h", 0),
                    )
                    area = entry.get("area", "Overlay")
                    log(
                        f"{prefix}{area} overlay detected{suffix}: '{desc}' at ({coords[0]}, {coords[1]}, {coords[2]}x{coords[3]}). Waiting..."
                    )
                last_log = now
            notified = True

            if now - start > timeout:
                log(f"{prefix}Timeout waiting for PDF overlays to clear{suffix}. Continuing.")
                return
            time.sleep(0.4)

    def _detect_overlay_in_rel_box(self, rel_box):
        if not self.current_rect or not rel_box:
            return None
        try:
            abs_left, abs_top, _, _ = core.rel_to_abs(self.current_rect, rel_box)
            img, scale = core._grab_region_color_generic(self.current_rect, rel_box, self.config.get("upscale_x"))
        except Exception:
            return None
        df = self._ocr_data(img, context="program", psm=6)
        lines = core.lines_from_tsv(df, scale=scale)
        overlay = self._find_overlay_entry(lines)
        if overlay:
            entry = overlay.copy()
            entry["abs_x"] = abs_left + overlay["x"]
            entry["abs_y"] = abs_top + overlay["y"]
            return entry
        return None

    def _search_overlay_rel_box(self):
        if not self.current_rect:
            return None
        point = self.config.get("search_point")
        if not (isinstance(point, (list, tuple)) and len(point) == 2):
            return None
        left, top, right, bottom = self.current_rect
        width = max(1, right - left)
        height = max(1, bottom - top)
        target_w = min(420, width)
        target_h = min(220, height)
        rel_w = target_w / width
        rel_h = target_h / height
        rel_left = max(0.0, min(point[0] - rel_w / 2, 1 - rel_w))
        rel_top = max(0.0, min(point[1] - rel_h / 2, 1 - rel_h))
        return [rel_left, rel_top, rel_w, rel_h]

    def _find_overlay_entry(self, lines):
        for x, y, w, h, raw in lines:
            if raw is None:
                continue
            norm = core.normalize_line(raw)
            candidates = [
                str(raw).strip().lower(),
                norm.lower() if norm else "",
            ]
            ascii_candidates = []
            for cand in candidates:
                if not cand:
                    continue
                ascii_candidates.append(
                    unicodedata.normalize("NFKD", cand).encode("ascii", "ignore").decode("ascii")
                )
            candidates.extend(ascii_candidates)
            for candidate in candidates:
                if not candidate:
                    continue
                for pattern in core.DOC_LOADING_PATTERNS:
                    if pattern in candidate:
                        return {"raw": str(raw), "norm": norm, "x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        return None

    def _get_rechnungen_region_wait(self) -> float:
        wait_setting = self.config.get(
            "rechnungen_region_wait", core.DEFAULTS.get("rechnungen_region_wait", 0.0)
        )
        try:
            wait_seconds = float(wait_setting)
        except Exception:
            wait_seconds = float(core.DEFAULTS.get("rechnungen_region_wait", 0.0))
        return max(0.0, wait_seconds)

    def _wait_for_rechnungen_region(self, cfg_key, prefix="", label="", log=None):
        wait_seconds = self._get_rechnungen_region_wait()
        if wait_seconds <= 0 or self._should_skip_manual_waits():
            return
        name = label or cfg_key.replace("_", " ").title()
        if log:
            log(f"{prefix}Waiting {wait_seconds:.2f}s for {name} region.")
        time.sleep(wait_seconds)

    def _draw_rechnungen_boxes(self, base_image, entries, ocr_lines):
        """Draw colored bounding boxes around detected Rechnungen entries."""
        from PIL import Image, ImageDraw

        if not entries:
            return base_image.copy()

        preview = base_image.convert("RGBA")
        dim_overlay = Image.new("RGBA", preview.size, (0, 0, 0, 80))
        preview = Image.alpha_composite(preview, dim_overlay)
        draw = ImageDraw.Draw(preview, "RGBA")

        highlight_colors = {
            True: ((255, 210, 0, 140), (255, 230, 80), (0, 0, 0, 180)),
            False: ((0, 200, 200, 120), (0, 255, 255), (0, 0, 0, 180)),
        }

        scale = max(1, int(self.config.get("upscale_x") or 3))

        summary_lines = []
        for entry in entries:
            y = entry.get("y", 0)
            h = entry.get("h", 20)
            x = entry.get("x", 0)
            w = entry.get("w", preview.width)
            if y >= preview.height:
                y = y // scale
                h = max(1, h // scale)
            if x >= preview.width:
                x = x // scale
                w = max(1, w // scale)

            x = max(0, min(int(x), preview.width - 1))
            y = max(0, min(int(y), preview.height - 1))
            w = max(1, min(int(w), preview.width - x))
            h = max(1, min(int(h), preview.height - y))

            fill_color, outline_color, text_bg = highlight_colors[bool(entry.get("is_gg"))]
            draw.rectangle([x, y, x + w, y + h], fill=fill_color, outline=outline_color, width=3)

            label = entry.get("label", "")
            if label:
                text_width = max(int(len(label) * 7), 40)
                label_top = max(0, y - 18)
                draw.rectangle([x, label_top, x + text_width, label_top + 16], fill=text_bg)
                draw.text((x + 4, label_top + 2), label, fill=(255, 255, 255, 220))

            summary_lines.append(
                f"{entry.get('date', '??.??.????')}  |  {entry.get('amount', '(n/a)') or '(n/a)'}"
            )

        if summary_lines:
            lines = summary_lines[:5]
            panel_height = 20 * len(lines) + 8
            panel = Image.new("RGBA", (preview.width, panel_height), (0, 0, 0, 160))
            preview.paste(panel, (0, preview.height - panel_height), panel)
            for idx, line in enumerate(lines):
                draw.text(
                    (10, preview.height - panel_height + 4 + idx * 18),
                    line,
                    fill=(255, 255, 255, 230),
                )

        return preview.convert("RGB")

    def _merge_ocr_rows(self, lines):
        if not lines:
            return []

        def _safe_number(value, default=0.0):
            try:
                return float(value)
            except Exception:
                return float(default)

        heights = [
            _safe_number(h, 0.0)
            for _, _, _, h, _ in lines
            if isinstance(h, (int, float)) and h and _safe_number(h) > 0
        ]
        heights.sort()
        median_h = heights[len(heights) // 2] if heights else 12.0
        tolerance = max(4.0, median_h * 0.6)

        groups = []
        for entry in sorted(lines, key=lambda x: (x[1], x[0])):
            if not (isinstance(entry, (list, tuple)) and len(entry) == 5):
                continue
            x, y, w, h, text = entry
            raw_text = (text or "").strip()
            if not raw_text:
                continue
            x_val = _safe_number(x)
            y_val = _safe_number(y)
            w_val = max(_safe_number(w), 0.0)
            h_val = max(_safe_number(h, median_h), 0.0) or median_h
            center = y_val + h_val / 2.0

            target = None
            for group in groups:
                if abs(center - group["center"]) <= tolerance:
                    target = group
                    break

            if target is None:
                target = {
                    "items": [],
                    "min_x": x_val,
                    "min_y": y_val,
                    "max_x": x_val + max(w_val, 1.0),
                    "max_y": y_val + max(h_val, 1.0),
                    "center": center,
                }
                groups.append(target)
            else:
                target["min_x"] = min(target["min_x"], x_val)
                target["min_y"] = min(target["min_y"], y_val)
                target["max_x"] = max(target["max_x"], x_val + max(w_val, 1.0))
                target["max_y"] = max(target["max_y"], y_val + max(h_val, 1.0))
                target["center"] = (target["min_y"] + target["max_y"]) / 2.0

            target["items"].append(
                {
                    "x": x_val,
                    "y": y_val,
                    "w": w_val,
                    "h": h_val,
                    "text": raw_text,
                }
            )

        merged = []
        for group in groups:
            items_sorted = sorted(group["items"], key=lambda item: item["x"])
            pieces = [item["text"] for item in items_sorted if item.get("text")]
            if not pieces:
                continue
            combined = " ".join(pieces).strip()
            if not combined:
                continue
            min_x = int(round(group["min_x"]))
            min_y = int(round(group["min_y"]))
            width = int(round(max(1.0, group["max_x"] - group["min_x"])))
            height = int(round(max(1.0, group["max_y"] - group["min_y"])))
            tokens = [
                {
                    "x": int(round(item.get("x", 0.0))),
                    "y": int(round(item.get("y", 0.0))),
                    "w": int(round(max(item.get("w", 0.0), 1.0))),
                    "h": int(round(max(item.get("h", 0.0), 1.0))),
                    "text": item.get("text", ""),
                }
                for item in items_sorted
            ]
            merged.append(
                {
                    "x": min_x,
                    "y": min_y,
                    "w": width,
                    "h": height,
                    "text": combined,
                    "tokens": tokens,
                }
            )

        merged.sort(key=lambda item: (item.get("y", 0), item.get("x", 0)))
        return merged

    def _select_rechnungen_amount_candidate(self, row, norm):
        tokens = row.get("tokens") or []
        row_x = row.get("x", 0)
        row_y = row.get("y", 0)
        if tokens:
            merged_tokens = []
            buffer = None
            for token in sorted(tokens, key=lambda t: (float(t.get("y", row_y)), float(t.get("x", row_x)))):
                text = (token.get("text") or "").strip()
                if not text:
                    continue
                if buffer is None:
                    buffer = token.copy()
                    buffer["text"] = text
                    continue
                same_row = abs(float(token.get("y", row_y)) - float(buffer.get("y", row_y))) <= max(
                    float(token.get("h", 12)), 12
                )
                overlap = (
                    float(token.get("x", row_x))
                    - (float(buffer.get("x", row_x)) + float(buffer.get("w", 0)) + 4)
                    <= 0
                )
                if same_row and overlap:
                    buffer["text"] = f"{buffer.get('text', '')} {text}".strip()
                else:
                    merged_tokens.append(buffer)
                    buffer = token.copy()
                    buffer["text"] = text
            if buffer:
                merged_tokens.append(buffer)
            tokens = merged_tokens
        row_w = row.get("w", 0)
        row_h = row.get("h", 0)
        candidates = []
        seen = set()
        # Identify the leftmost date position to prefer nearby amounts
        date_x = None
        for token in sorted(tokens, key=lambda t: float(t.get("x", row_x))):
            token_text = token.get("text", "")
            if token_text and core.DATE_RE.search(core.normalize_line(token_text)):
                date_x = token.get("x", row_x)
                break

        for token in tokens:
            raw = token.get("text", "")
            if not raw:
                continue
            token_x = token.get("x", row_x)
            key_base = int(round(token_x))
            for amt in core.find_amount_candidates(raw):
                display = core.clean_amount_display(amt.get("display")) if amt else None
                if not display:
                    continue
                key = (display, key_base)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "display": display,
                        "value": amt.get("value"),
                        "x": token_x,
                        "box": (
                            int(round(token.get("x", row_x))),
                            int(round(token.get("y", row_y))),
                            int(round(max(token.get("w", 0) or 1, 1))),
                            int(round(max(token.get("h", 0) or 1, 1))),
                        ),
                        "source": raw.strip(),
                    }
                )

        if not candidates:
            key_base = int(round(row_x))
            for amt in core.find_amount_candidates(norm):
                display = core.clean_amount_display(amt.get("display")) if amt else None
                if not display:
                    continue
                key = (display, key_base)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "display": display,
                        "value": amt.get("value"),
                        "x": row_x,
                        "box": (
                            int(round(row_x)),
                            int(round(row_y)),
                            int(round(max(row_w, 1))),
                            int(round(max(row_h, 1))),
                        ),
                        "source": norm.strip(),
                    }
                )

        if not candidates:
            return {}

        zero = Decimal("0")
        anchor_x = date_x if date_x is not None else row_x

        def sort_key(candidate):
            value = candidate.get("value")
            display = candidate.get("display") or ""
            has_currency = 0 if re.search(r"(EUR|€|�'�)", display, re.IGNORECASE) else 1
            positive = 0 if value is not None and value > zero else 1
            x_coord = candidate.get("x", row_x)
            if anchor_x is None:
                date_side = 0
                distance = abs(x_coord - row_x)
            else:
                date_side = 0 if x_coord >= anchor_x else 1
                distance = abs(x_coord - anchor_x)
            magnitude = -value if value is not None else Decimal("0")
            return (has_currency, date_side, distance, positive, x_coord, magnitude)

        candidates.sort(key=sort_key)
        return candidates[0]

    def _extract_rechnungen_label_info(self, row, norm):
        tokens = row.get("tokens") or []
        row_x = row.get("x", 0)
        row_y = row.get("y", 0)
        row_w = max(row.get("w", 0), 1)
        label_tokens = []

        for token in tokens:
            raw = token.get("text", "")
            if not raw:
                continue
            normalized = core.normalize_line(raw)
            cleaned = re.sub(r"[^A-Z0-9]", "", normalized.upper())
            if not cleaned or not re.search(r"[A-Z]", cleaned):
                continue
            label_tokens.append(
                {
                    "raw": raw.strip(),
                    "clean": cleaned,
                    "x": token.get("x", row_x),
                    "y": token.get("y", row_y),
                    "w": int(round(max(token.get("w", 0), 1))),
                    "h": int(round(max(token.get("h", 0), 1))),
                }
            )

        if not label_tokens:
            return {
                "display": "",
                "normalized": core.normalize_gg_candidate(norm),
                "box": None,
            }

        label_tokens.sort(key=lambda info: info.get("x", row_x))
        cutoff = row_x + row_w * 0.55
        tail = [info for info in label_tokens if info.get("x", row_x) >= cutoff]
        if not tail:
            tail = label_tokens[-3:]

        combined_raw = " ".join(info["raw"] for info in tail if info.get("raw"))
        combined_clean = "".join(info.get("clean", "") for info in tail)
        normalized = core.normalize_gg_candidate(combined_raw or combined_clean)

        min_x = min(info.get("x", row_x) for info in tail)
        min_y = min(info.get("y", row_y) for info in tail)
        max_x = max(info.get("x", row_x) + max(info.get("w", 1), 1) for info in tail)
        max_y = max(info.get("y", row_y) + max(info.get("h", 1), 1) for info in tail)
        box = (
            int(round(min_x)),
            int(round(min_y)),
            int(round(max_x - min_x)),
            int(round(max_y - min_y)),
        )

        return {
            "display": combined_raw.strip() or combined_clean,
            "normalized": normalized or core.normalize_gg_candidate(norm),
            "box": box,
        }

    def _parse_rechnungen_entries(self, lines, prefix="", log=None):
        merged_lines = self._merge_ocr_rows(lines)
        entries = []
        skipped = []
        for row in merged_lines:
            raw = (row.get("text") or "").strip()
            if not raw:
                continue
            norm = core.normalize_line(raw)
            amount_info = self._select_rechnungen_amount_candidate(row, norm)
            amount = amount_info.get("display") if amount_info else None
            date_match = core.DATE_RE.search(norm) if norm else None
            if not amount or not date_match:
                reason = []
                if not amount:
                    reason.append("amount")
                if not date_match:
                    reason.append("date")
                skipped.append((norm, ", ".join(reason) or "missing data"))
                if log:
                    snippet = (norm or raw)[:120]
                    log(
                        f"{prefix}Skipping row '{snippet}' (missing {', '.join(reason) or 'data'})."
                    )
                continue

            # Standardize date format (fix OCR errors like O->0)
            date_text = core.standardize_date(date_match.group(0))

            try:
                date_obj = datetime.strptime(date_text, "%d.%m.%Y")
            except ValueError:
                date_obj = None

            # Standardize invoice number (fix OCR errors)
            invoice_match = core.INVOICE_RE.search(norm)
            invoice = core.standardize_invoice_number(invoice_match.group(0)) if invoice_match else ""
            label_info = self._extract_rechnungen_label_info(row, norm)
            label_display = label_info.get("display", "")
            label_normalized = label_info.get("normalized", "")
            entry = {
                "raw": raw,
                "norm": norm,
                "amount": core.clean_amount_display(amount) if amount else amount,
                "amount_box": amount_info.get("box") if amount_info else None,
                "amount_value": amount_info.get("value") if amount_info else None,
                "amount_source": amount_info.get("source") if amount_info else "",
                "date": date_text,
                "date_obj": date_obj,
                "invoice": invoice,
                "label": label_display,
                "label_normalized": label_normalized,
                "label_is_gg": (
                    core.is_gg_label(label_normalized) or core.is_gg_label(label_display)
                ),
                "label_box": label_info.get("box"),
                "x": row.get("x", 0),
                "y": row.get("y", 0),
                "w": row.get("w", 0),
                "h": row.get("h", 0),
            }
            entries.append(entry)
        entries.sort(
            key=lambda e: (
                e.get("date_obj") or datetime.min,
                e.get("y", 0),
                e.get("x", 0),
            )
        )
        if skipped and log:
            log(f"{prefix}Skipped {len(skipped)} Rechnungen row(s) due to missing data.")
        for idx, entry in enumerate(entries, 1):
            detail = self._format_rechnungen_detail(entry)
            label_display = entry.get("label") or entry.get("label_normalized") or "-"
            amount_txt = entry.get("amount", "") or "(no amount)"
            bounds = entry.get("amount_box")
            bounds_txt = (
                f" | Bounds: ({bounds[0]}, {bounds[1]}, {bounds[2]}, {bounds[3]})"
                if isinstance(bounds, (list, tuple)) and len(bounds) == 4
                else ""
            )
            if log:
                log(f"{prefix}Row {idx}: {amount_txt}{detail} | Label: {label_display}{bounds_txt}")
        for norm, reason in skipped[:6]:
            if log:
                log(f"{prefix}Skipped Rechnungen row '{norm}' ({reason}).")
        return entries

    def _is_gg_entry(self, entry):
        if not entry:
            return False
        if entry.get("label_is_gg"):
            return True
        label_norm = entry.get("label_normalized") or ""
        if core.is_gg_label(label_norm):
            return True
        label_display = entry.get("label") or ""
        if core.is_gg_label(label_display):
            return True
        raw_text = entry.get("raw") or ""
        if raw_text:
            tokens = re.split(r"[^A-Z0-9]+", core.normalize_line(raw_text).upper())
            for token in tokens:
                if not token or len(token) < 2:
                    continue
                if not re.search(r"[A-Z]", token):
                    continue
                if core.is_gg_label(token):
                    return True
        return False

    def _extract_rechnungen_gg_entries(self, prefix="", log=None):
        lines = self._capture_region_lines("rechnungen_gg_region", prefix=prefix, label="GG", log=log)
        if not lines:
            return []
        entries = self._parse_rechnungen_entries(lines, prefix=prefix, log=log)
        gg_entries = [entry for entry in entries if self._is_gg_entry(entry)]
        if log:
            if gg_entries:
                log(f"{prefix}Detected {len(gg_entries)} GG transaction(s).")
            else:
                log(f"{prefix}No GG transactions detected.")
        for idx, entry in enumerate(gg_entries, 1):
            amount = entry.get("amount", "") or "(no amount)"
            detail = self._format_rechnungen_detail(entry)
            label_display = entry.get("label") or entry.get("label_normalized") or "GG"
            bounds = entry.get("amount_box")
            bounds_txt = (
                f" | Bounds: ({bounds[0]}, {bounds[1]}, {bounds[2]}, {bounds[3]})"
                if isinstance(bounds, (list, tuple)) and len(bounds) == 4
                else ""
            )
            if log:
                log(f"{prefix}GG #{idx}: {amount}{detail} | Label: {label_display}{bounds_txt}")
        return gg_entries

    def _summarize_rechnungen_entries(self, entries):
        def _copy(entry):
            if not entry:
                return {"amount": "", "date": "", "invoice": "", "raw": ""}
            return {
                "amount": entry.get("amount", ""),
                "date": entry.get("date", ""),
                "invoice": entry.get("invoice", ""),
                "raw": entry.get("raw", ""),
            }

        no_invoice = [e for e in entries if not e.get("invoice")]
        if no_invoice:
            no_invoice.sort(
                key=lambda e: (
                    e.get("date_obj") or datetime.min,
                    e.get("y", 0),
                    e.get("x", 0),
                )
            )
            total_entry = no_invoice[-1]
        else:
            total_entry = None

        invoice_entries = [e for e in entries if e.get("invoice")]
        invoice_entries.sort(
            key=lambda e: (
                e.get("date_obj") or datetime.min,
                e.get("y", 0),
                e.get("x", 0),
            )
        )
        court_entry = invoice_entries[-1] if invoice_entries else None
        gg_entry = invoice_entries[0] if len(invoice_entries) >= 2 else None

        summary = {
            "total": _copy(total_entry),
            "total_found": bool(total_entry),
            "court": _copy(court_entry),
            "court_found": bool(court_entry),
            "gg": _copy(gg_entry)
            if gg_entry
            else {"amount": "0", "date": "", "invoice": "", "raw": ""},
            "gg_found": bool(gg_entry),
            "entries": [_copy(e) for e in entries],
        }
        if not summary["gg_found"]:
            if len(invoice_entries) == 1:
                summary["gg_missing_reason"] = "only one Rechnungen entry with invoice"
            elif not invoice_entries:
                summary["gg_missing_reason"] = "no Rechnungen entries with invoice"
        summary["invoice_entry_count"] = len(invoice_entries)
        summary["total_entry_count"] = len(no_invoice)
        return summary

    def _format_rechnungen_detail(self, entry):
        parts = []
        date = entry.get("date") if isinstance(entry, dict) else None
        invoice = entry.get("invoice") if isinstance(entry, dict) else None
        if date:
            parts.append(date)
        if invoice:
            parts.append(invoice)
        if not parts:
            return ""
        return f" ({' | '.join(parts)})"

    def _build_gg_summary_line(self, aktenzeichen, entries):
        label = aktenzeichen or "(unbekannt)"
        if not entries:
            return f"{label}: (no GG)"
        summary_parts = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            amount = entry.get("amount") or "(no amount)"
            detail = self._format_rechnungen_detail(entry)
            summary_parts.append(f"{amount}{detail}")
        if not summary_parts:
            return f"{label}: (no GG)"
        return f"{label}: {'; '.join(summary_parts)}"

    def _log_rechnungen_summary(self, prefix, summary, log=None):
        log_fn = log or (lambda msg: None)
        if not summary:
            log_fn(f"{prefix}-Total Fees: (not found)")
            log_fn(f"{prefix}-Received Court Fees: (not found)")
            log_fn(f"{prefix}-Received GG: 0")
            return

        total_txt = summary.get("total", {}).get("amount", "") or "(not found)"
        log_fn(f"{prefix}-Total Fees: {total_txt}")

        court_entry = summary.get("court", {})
        court_amt = court_entry.get("amount", "")
        if court_amt:
            detail = self._format_rechnungen_detail(court_entry)
            log_fn(f"{prefix}-Received Court Fees: {court_amt}{detail}")
        else:
            log_fn(f"{prefix}-Received Court Fees: (not found)")

        gg_entry = summary.get("gg", {})
        if summary.get("gg_found"):
            detail = self._format_rechnungen_detail(gg_entry)
            log_fn(f"{prefix}-Received GG: {gg_entry.get('amount', '')}{detail}")
        else:
            gg_amt = gg_entry.get("amount", "0") or "0"
            log_fn(f"{prefix}-Received GG: {gg_amt}")
            reason = summary.get("gg_missing_reason")
            if reason:
                log_fn(f"{prefix}  ↳ Reason: {reason}.")

    def _build_rechnungen_result_row(self, aktenzeichen, summary):
        if not summary:
            summary = {
                "total": {"amount": "", "date": "", "invoice": ""},
                "court": {"amount": "", "date": "", "invoice": ""},
                "gg": {"amount": "0", "date": "", "invoice": ""},
            }
        total = summary.get("total", {})
        court = summary.get("court", {})
        gg = summary.get("gg", {})
        return {
            "aktenzeichen": aktenzeichen,
            "total_fees_amount": total.get("amount", ""),
            "total_fees_date": total.get("date", ""),
            "total_fees_invoice": total.get("invoice", ""),
            "received_court_amount": court.get("amount", ""),
            "received_court_date": court.get("date", ""),
            "received_court_invoice": court.get("invoice", ""),
            "received_gg_amount": gg.get("amount", ""),
            "received_gg_date": gg.get("date", ""),
            "received_gg_invoice": gg.get("invoice", ""),
        }

    def _extract_rechnungen_summary(self, prefix="", log=None):
        lines = self._capture_region_lines("rechnungen_region", prefix=prefix, label="Rechnungen", log=log)
        if not lines:
            return None
        entries = self._parse_rechnungen_entries(lines, prefix=prefix, log=log)
        return self._summarize_rechnungen_entries(entries)

    def run_log_extraction(self, progress_callback=None):
        log = progress_callback or (lambda msg: None)
        log_dir = (self.config.get("log_folder") or core.LOG_DIR).strip() or core.LOG_DIR
        if not os.path.isdir(log_dir):
            core.ensure_log_dir()
            if not os.path.isdir(log_dir):
                log(f"[Log Extract] Log directory not found: {log_dir}")
                return {"rows": 0, "output_csv": self.config.get("log_extract_results_csv")}
        files = [
            os.path.join(log_dir, name)
            for name in sorted(os.listdir(log_dir))
            if name.lower().endswith(".log")
        ]
        if not files:
            log(f"[Log Extract] No .log files found in {os.path.abspath(log_dir)}")
            return {"rows": 0, "output_csv": self.config.get("log_extract_results_csv")}
        output_csv = (self.config.get("log_extract_results_csv") or "").strip()
        log(
            f"[Log Extract] Processing {len(files)} log file(s) from {os.path.abspath(log_dir)}"
        )
        results = []
        fallback_keywords = core.build_streitwert_keywords(self.config.get("streitwert_term") or "Streitwert")
        for path in files:
            label = self._log_label_from_filename(os.path.basename(path))
            amount, context, section = self._extract_amount_from_log(path, fallback_keywords)
            display_label = label or os.path.basename(path)
            if amount:
                results.append(
                    {
                        "log_file": os.path.basename(path),
                        "label": label,
                        "amount": amount,
                        "section": section or "",
                        "context": context or "",
                    }
                )
                log(f"[Log Extract] {display_label} → {amount} ({section or 'section unknown'})")
            else:
                log(f"[Log Extract] {display_label} → (none)")
        if results and output_csv:
            try:
                pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")
                log(f"[Log Extract] Saved {len(results)} entries to {output_csv}")
            except Exception as exc:
                log(f"[Log Extract] Failed to write CSV '{output_csv}': {exc}")
        elif results:
            log(f"[Log Extract] Collected {len(results)} entries (CSV output disabled).")
        else:
            log("[Log Extract] No Streitwert amounts were detected in the logs.")
        return {"rows": len(results), "output_csv": output_csv}

    def run_akten(self, progress_callback=None):
        log = progress_callback or (lambda msg: None)
        prefix = "[Akten] "
        self._apply_tesseract_path()
        if not self.current_rect:
            self.connect_rdp()
            if not self.current_rect:
                return {"rows": 0, "captured": 0}
        doc_rect = self._doclist_abs_rect()
        if not doc_rect:
            log(f"{prefix}Doc list region is not calibrated.")
            return {"rows": 0, "captured": 0}
        if not self.config.get("pdf_text_region"):
            log(f"{prefix}PDF text region is not calibrated.")
            return {"rows": 0, "captured": 0}
        if "doc_view_point" not in self.config:
            log(f"{prefix}View button point is not configured.")
            return {"rows": 0, "captured": 0}
        if "pdf_close_point" not in self.config:
            log(f"{prefix}PDF close button point is not configured.")
            return {"rows": 0, "captured": 0}
        queries = self._gather_aktenzeichen_queries()
        if not queries:
            log(f"{prefix}No Aktenzeichen entries were found in the configured Excel sheet.")
            return {"rows": 0, "captured": 0}
        search_term = (self.config.get("akten_search_term") or "Aufforderungsschreiben").strip()
        ignore_tokens = self._akten_ignore_tokens()
        wait_after = float(self.config.get("post_search_wait", 1.2))
        doc_wait = max(0.2, float(self.config.get("doc_open_wait", 1.0)))
        per_row_wait = max(0.2, float(self.config.get("rechnungen_region_wait", 0.4)))
        extra_wait = max(0.0, float(self.config.get("pdf_view_extra_wait", 0.0) or 0.0))
        base_log = log
        run_log_file = None
        run_log_path = ""
        try:
            log_dir = Path(self.config.get("log_folder") or core.LOG_DIR)
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_log_path = log_dir / f"{timestamp}_akten_run.log"
            run_log_file = open(run_log_path, "w", encoding="utf-8")

            def live_log(message: str):
                base_log(message)
                try:
                    run_log_file.write(message + "\n")
                    run_log_file.flush()
                except Exception:
                    pass

            log = live_log
        except Exception:
            run_log_file = None
            run_log_path = ""
            log = base_log

        opened = 0
        captured = 0
        total = len(queries)
        rows_out: list[dict[str, str]] = []

        try:
            for idx, (aktenzeichen, _) in enumerate(queries, start=1):
                step_prefix = f"[Akten {idx}/{total}] "

                def record_result(status, matched="", date="", note=""):
                    rows_out.append(
                        {
                            "aktenzeichen": aktenzeichen,
                            "search_term": search_term,
                            "matched_entry": matched,
                            "date_extracted": date,
                            "status": status,
                            "note": note or "",
                            "run_log": str(run_log_path),
                        }
                    )

                log(f"{step_prefix}Typing Aktenzeichen '{aktenzeichen}' into the document list...")
                if not self._type_doclist_query(aktenzeichen, log=log, prefix=step_prefix):
                    log(f"{step_prefix}Unable to type Aktenzeichen. Skipping entry.")
                    record_result("type_failed")
                    continue

                if wait_after > 0:
                    time.sleep(max(0.2, wait_after))
                self._wait_for_doc_search_ready(prefix=step_prefix, log=log, reason="after Aktenzeichen search")
                self._wait_for_doclist_ready(prefix=step_prefix, log=log, reason="after Aktenzeichen search")

                log(f"{step_prefix}📋 Performing OCR on document list region...")
                rows = self._ocr_doclist_rows_boxes(log=log, prefix=step_prefix)
                if not rows:
                    log(f"{step_prefix}⚠ No documents detected (OCR returned 0 rows).")
                    record_result("no_rows")
                    continue

                log(f"{step_prefix}✓ OCR detected {len(rows)} document(s):")
                log(f"{step_prefix}{'='*60}")
                for row_idx, (text, _) in enumerate(rows, 1):
                    log(f"{step_prefix}  {row_idx}. {text}")
                log(f"{step_prefix}{'='*60}")

                matched_idx, matched_file = self._find_doclist_match(
                    rows, search_term, ignore_tokens=ignore_tokens, prefix=step_prefix, log=log
                )
                if not matched_file:
                    log(f"{step_prefix}Skipping Aktenzeichen '{aktenzeichen}' (no match for '{search_term}').")
                    record_result("no_match", note="search term not found")
                    continue

                file_text, file_box = matched_file
                match = {
                    "raw": file_text,
                    "x": file_box[0],
                    "y": file_box[1],
                    "w": max(file_box[2] - file_box[0], 1),
                    "h": max(file_box[3] - file_box[1], 1),
                }

                log(f"{step_prefix}🖱 Clicking matched file at ({file_box[0]}, {file_box[1]})...")
                if not self._select_doclist_entry(match, doc_rect, prefix=step_prefix, log=log):
                    log(f"{step_prefix}✗ Failed to click matched file. Skipping entry.")
                    record_result("click_failed", matched=file_text)
                    continue

                time.sleep(per_row_wait)

                az_detected, az_raw_text = self._extract_akten_filter_info(
                    doclist_entry_text=file_text, prefix=step_prefix, log=log
                )
                az_display = az_detected or aktenzeichen
                if not az_detected and az_raw_text:
                    log(f"{step_prefix}⚠ AZ not detected in OCR; falling back to spreadsheet value.")

                log(f"{step_prefix}👁 Opening PDF with View button...")
                if not self._click_view_button(prefix=step_prefix, log=log):
                    log(f"{step_prefix}✗ View button click failed; skipping entry.")
                    record_result("view_failed", matched=file_text)
                    continue

                opened += 1
                time.sleep(doc_wait + extra_wait)
                self._wait_for_pdf_ready(prefix=step_prefix, log=log, reason="Akten PDF open")

                log(f"{step_prefix}📄 Performing OCR on PDF to extract date...")
                date_value, full_text = self._extract_akten_date_from_pdf(prefix=step_prefix, log=log)

                if date_value:
                    captured += 1
                    log(f"{step_prefix}✓ {az_display} -> {date_value}")
                    record_result("ok", matched=file_text, date=date_value)
                else:
                    log(f"{step_prefix}⚠ {az_display} -> (date not found)")
                    if full_text:
                        log(f"{step_prefix}  PDF OCR text sample: {full_text[:150]}")
                    else:
                        log(f"{step_prefix}  PDF OCR returned no text")
                    record_result("date_not_found", matched=file_text)

                self._close_active_pdf(prefix=step_prefix, log=log)
                time.sleep(0.4)
        finally:
            if run_log_file:
                try:
                    run_log_file.close()
                except Exception:
                    pass

        output_csv = (self.config.get("akten_results_csv") or "akten_results.csv").strip()
        if rows_out and output_csv:
            try:
                output_dir = os.path.dirname(output_csv)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                pd.DataFrame(rows_out).to_csv(output_csv, index=False, encoding="utf-8-sig")
                log(f"{prefix}✓ Saved {len(rows_out)} Akten entries to {output_csv}")
            except Exception as exc:
                log(f"{prefix}ERROR saving Akten results CSV: {exc}")
        elif rows_out:
            log(f"{prefix}Collected {len(rows_out)} Akten entries (CSV output disabled).")

        if opened == 0:
            log(f"{prefix}Finished without opening any documents.")
        else:
            log(f"{prefix}Opened {opened} document(s); captured {captured} date(s).")
        if run_log_file:
            run_log_file.close()
        return {
            "rows": opened,
            "captured": captured,
            "output_csv": output_csv if rows_out else "",
            "run_log": str(run_log_path) if run_log_path else "",
        }

    def run_akten_with_filtering(self, progress_callback=None):
        """
        Alias for the main Akten workflow. The filtered button now runs the
        same document-selection process as Run Akten.
        """
        return self.run_akten(progress_callback=progress_callback)

    def _log_label_from_filename(self, filename):
        base = os.path.splitext(filename)[0]
        match = re.match(r"^\d{8}-\d{6}_(.+)$", base)
        if match:
            base = match.group(1)
        label = base.replace("__", " – ").replace("_", " ")
        return label.strip()

    def _parse_log_file_sections(self, path):
        sections = []
        current_section = ""
        current_entries = []
        current_keywords = []
        last_coords = (0, 0, 0, 0)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.rstrip("\n")
                    stripped = line.strip()
                    if not stripped:
                        continue
                    sec_match = core.LOG_SECTION_RE.match(stripped)
                    if sec_match:
                        if current_entries:
                            sections.append(
                                {
                                    "section": current_section,
                                    "entries": list(current_entries),
                                    "keywords": list(current_keywords),
                                }
                            )
                        current_section = sec_match.group(1).strip()
                        current_entries = []
                        current_keywords = []
                        last_coords = (0, 0, 0, 0)
                        continue
                    kw_match = core.LOG_KEYWORD_RE.match(stripped)
                    if kw_match:
                        keywords = [token.strip() for token in kw_match.group(1).split(",") if token.strip()]
                        current_keywords = keywords
                        continue
                    entry_match = core.LOG_ENTRY_RE.match(line)
                    if entry_match:
                        coords_text = entry_match.group(2)
                        coords_parts = [c.strip() for c in coords_text.split(",")]
                        if len(coords_parts) >= 4:
                            try:
                                x = int(coords_parts[0])
                                y = int(coords_parts[1])
                                w = int(coords_parts[2])
                                h = int(coords_parts[3])
                            except Exception:
                                x = y = w = h = 0
                        else:
                            x = y = w = h = 0
                        text = entry_match.group(3).strip()
                        current_entries.append((x, y, w, h, text))
                        last_coords = (x, y, w, h)
                        continue
                    soft_match = core.LOG_SOFT_RE.match(line)
                    if soft_match and current_entries:
                        text = soft_match.group(1).strip()
                        if text:
                            x, y, w, h = last_coords
                            current_entries.append((x, y, w, h, text))
                        continue
                    norm_match = core.LOG_NORM_RE.match(line)
                    if norm_match and current_entries:
                        text = norm_match.group(1).strip()
                        if text:
                            x, y, w, h = last_coords
                            current_entries.append((x, y, w, h, text))
                        continue
        except Exception:
            return []
        if current_entries:
            sections.append(
                {
                    "section": current_section,
                    "entries": list(current_entries),
                    "keywords": list(current_keywords),
                }
            )
        return sections

    def _extract_amount_from_log(self, path, fallback_keywords):
        sections = self._parse_log_file_sections(path)
        if not sections:
            return None, None, None
        for info in sections:
            entries = info.get("entries") or []
            if not entries:
                continue
            keywords = info.get("keywords") or fallback_keywords
            amount, line = core.extract_amount_from_lines(
                entries,
                keyword=keywords,
                min_value=core.STREITWERT_MIN_AMOUNT,
            )
            if not amount and keywords is not fallback_keywords:
                amount, line = core.extract_amount_from_lines(
                    entries,
                    keyword=fallback_keywords,
                    min_value=core.STREITWERT_MIN_AMOUNT,
                )
            if amount:
                return core.clean_amount_display(amount), line, info.get("section")
        combined_entries = []
        for info in sections:
            combined_entries.extend(info.get("entries") or [])
        if combined_entries:
            amount, line = core.extract_amount_from_lines(
                combined_entries,
                keyword=fallback_keywords,
                min_value=core.STREITWERT_MIN_AMOUNT,
            )
            if amount:
                return core.clean_amount_display(amount), line, "combined"
        return None, None, None

    def _ensure_region_min_size(self, x, y, w, h, *, min_width=80, min_height=24):
        """Expand tiny relative regions so OCR/clicks can work reliably."""
        left = int(round(x))
        top = int(round(y))
        width = max(1, int(round(w)))
        height = max(1, int(round(h)))
        min_width = max(1, int(round(min_width)))
        min_height = max(1, int(round(min_height)))
        if width >= min_width and height >= min_height:
            return left, top, width, height
        target_w = max(width, min_width)
        target_h = max(height, min_height)
        pad_x = max(0, (target_w - width) // 2)
        pad_y = max(0, (target_h - height) // 2)
        try:
            screen_w, screen_h = pyautogui.size()
        except Exception:
            screen_w = screen_h = 0
        max_x = max(0, screen_w - target_w) if screen_w else left
        max_y = max(0, screen_h - target_h) if screen_h else top
        left = max(0, min(left - pad_x, max_x))
        top = max(0, min(top - pad_y, max_y))
        return left, top, target_w, target_h


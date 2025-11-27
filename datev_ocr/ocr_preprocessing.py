"""
Enhanced OCR preprocessing for table/tree views with invoice rows.
Focuses on improving text detection for light backgrounds and small fonts.
"""

import re
import unicodedata
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    _HAS_CV2 = False


def normalize_ocr_text(text: str) -> str:
    """
    Normalize OCR output text to fix common artifacts and standardize format.

    Handles:
    - Unicode normalization (NFKC)
    - Common OCR misreads (0/O, 1/l, 5/S, etc.)
    - Extra spaces and line breaks
    - Currency symbols
    - Date formats
    - Punctuation issues

    Args:
        text: Raw OCR text

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Fix common currency symbol issues
    text = text.replace("\u0080", "€")
    text = text.replace("EUR0", "EURO")
    text = text.replace("0EUR", "OEUR")

    # Fix common OCR misreads in amounts
    # Be careful - only apply to amount-like patterns
    amount_pattern = r'\b\d+[.,\s]+\d+\s*(?:EUR|€|Euro)?\b'

    def fix_amount_chars(match):
        amount_text = match.group(0)
        # In amounts, fix common misreads
        fixed = amount_text.replace("O", "0")  # O -> 0 in amounts
        fixed = fixed.replace("o", "0")
        fixed = fixed.replace("l", "1")  # l -> 1
        fixed = fixed.replace("I", "1")  # I -> 1
        fixed = fixed.replace("S", "5")  # S -> 5 in amounts
        return fixed

    text = re.sub(amount_pattern, fix_amount_chars, text, flags=re.IGNORECASE)

    # Standardize spacing
    text = text.replace("\t", " ")
    text = re.sub(r' +', ' ', text)  # Multiple spaces -> single space

    # Fix common date format issues
    # e.g., "25.O7.2024" -> "25.07.2024"
    date_pattern = r'\b(\d{1,2})\s*\.\s*([O0o]\d|\d[O0o]|\d{2})\s*\.\s*(\d{4})\b'

    def fix_date(match):
        day, month, year = match.groups()
        # Fix O/o to 0 in month part
        month = month.replace('O', '0').replace('o', '0')
        return f"{day}.{month}.{year}"

    text = re.sub(date_pattern, fix_date, text)

    # Fix invoice numbers - preserve mixed alphanumeric
    # e.g., "2O24007445" -> "2024007445"
    invoice_pattern = r'\b20[O0o]4\d{6,}\b'

    def fix_invoice(match):
        inv = match.group(0)
        inv = inv.replace('O', '0').replace('o', '0')
        return inv

    text = re.sub(invoice_pattern, fix_invoice, text)

    # Fix common word misreads
    text = re.sub(r'\bRechnungen\b', 'Rechnungen', text, flags=re.IGNORECASE)
    text = re.sub(r'\bWert\b', 'Wert', text, flags=re.IGNORECASE)
    text = re.sub(r'\bEUR\b', 'EUR', text, flags=re.IGNORECASE)

    # Clean up line breaks
    text = re.sub(r'\n\s*\n', '\n', text)  # Multiple blank lines -> single

    # Strip each line
    lines = [line.strip() for line in text.splitlines()]
    text = '\n'.join(line for line in lines if line)

    return text.strip()


def normalize_ocr_line(line: str) -> str:
    """
    Normalize a single line of OCR text.
    Similar to normalize_ocr_text but optimized for single lines.

    Args:
        line: Single line of OCR text

    Returns:
        Normalized line
    """
    if not line:
        return ""

    # Use the full normalization
    normalized = normalize_ocr_text(line)

    # For single lines, also collapse all whitespace to single spaces
    normalized = ' '.join(normalized.split())

    return normalized


def preprocess_for_table_ocr(img: Image.Image, aggressive: bool = False) -> list[Image.Image]:
    """
    Generate multiple preprocessed variants optimized for table/tree OCR.

    Args:
        img: Input PIL Image
        aggressive: If True, includes more aggressive preprocessing variants

    Returns:
        List of preprocessed PIL Images to try with OCR
    """
    variants = []

    # Convert to grayscale
    try:
        gray = img.convert("L")
    except Exception:
        gray = ImageOps.grayscale(img)

    # Variant 1: Basic autocontrast
    try:
        base_auto = ImageOps.autocontrast(gray)
        variants.append(base_auto)
    except Exception:
        variants.append(gray)

    # Variant 2: Enhanced contrast
    try:
        contrast_enhanced = ImageEnhance.Contrast(gray).enhance(2.5)
        variants.append(ImageOps.autocontrast(contrast_enhanced))
    except Exception:
        pass

    # Variant 3: Sharpened for small text
    try:
        sharpened = gray.filter(ImageFilter.SHARPEN)
        variants.append(ImageOps.autocontrast(sharpened))
    except Exception:
        pass

    # Variant 4: OpenCV adaptive thresholding (best for tables)
    if _HAS_CV2:
        try:
            cv_variants = _opencv_table_variants(gray)
            variants.extend(cv_variants)
        except Exception:
            pass

    # Variant 5: High contrast + brightness (for light backgrounds)
    try:
        bright_contrast = ImageEnhance.Brightness(gray).enhance(1.1)
        bright_contrast = ImageEnhance.Contrast(bright_contrast).enhance(2.8)
        variants.append(ImageOps.autocontrast(bright_contrast))
    except Exception:
        pass

    if aggressive:
        # Variant 6: Inverted (for highlighted rows)
        try:
            inverted = ImageOps.invert(gray)
            variants.append(ImageOps.autocontrast(inverted))
        except Exception:
            pass

        # Variant 7: Edge enhancement
        try:
            edge_enhanced = gray.filter(ImageFilter.EDGE_ENHANCE_MORE)
            variants.append(ImageOps.autocontrast(edge_enhanced))
        except Exception:
            pass

    # Remove duplicates
    return _deduplicate_variants(variants)


def _opencv_table_variants(gray_img: Image.Image) -> list[Image.Image]:
    """Generate OpenCV-based preprocessing variants optimized for table text."""
    if not _HAS_CV2:
        return []

    variants = []
    np_img = np.array(gray_img)
    adaptive_gaussian = None

    # Adaptive threshold - Gaussian (best for varying lighting)
    try:
        adaptive_gaussian = cv2.adaptiveThreshold(
            np_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        variants.append(Image.fromarray(adaptive_gaussian))
    except Exception:
        pass

    # Adaptive threshold - Mean
    try:
        adaptive_mean = cv2.adaptiveThreshold(
            np_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3
        )
        variants.append(Image.fromarray(adaptive_mean))
    except Exception:
        pass

    # Otsu's binarization (good for bimodal histograms)
    try:
        _, otsu = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(Image.fromarray(otsu))
    except Exception:
        pass

    # Denoise + adaptive threshold (reduces noise in scans)
    try:
        denoised = cv2.fastNlMeansDenoising(np_img, h=10)
        denoised_thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        variants.append(Image.fromarray(denoised_thresh))
    except Exception:
        pass

    # Morphological operations to clean up text
    if adaptive_gaussian is not None:
        try:
            kernel = np.ones((2, 2), np.uint8)
            morph = cv2.morphologyEx(adaptive_gaussian, cv2.MORPH_CLOSE, kernel)
            variants.append(Image.fromarray(morph))
        except Exception:
            pass

    return variants


def preprocess_for_invoice_rows(img: Image.Image) -> list[Image.Image]:
    """
    Specialized preprocessing for invoice row detection in tree views.
    Optimized for:
    - Small fonts (dates, amounts, invoice numbers)
    - Light/white backgrounds
    - Tree structure indentation
    """
    variants = []

    try:
        gray = img.convert("L")
    except Exception:
        gray = ImageOps.grayscale(img)

    # For invoice rows, prioritize high-contrast, sharp preprocessing

    # 1. High contrast with sharpening (best for dates and numbers)
    try:
        sharp_contrast = ImageEnhance.Sharpness(gray).enhance(2.0)
        sharp_contrast = ImageEnhance.Contrast(sharp_contrast).enhance(3.0)
        variants.append(ImageOps.autocontrast(sharp_contrast))
    except Exception:
        pass

    # 2. OpenCV adaptive thresholding (critical for table cells)
    if _HAS_CV2:
        try:
            np_img = np.array(gray)

            # Fine-tuned for small text in tables
            adaptive = cv2.adaptiveThreshold(
                np_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2
            )

            # Clean up with morphology
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)

            variants.append(Image.fromarray(cleaned))
        except Exception:
            pass

    # 3. Basic high-contrast
    try:
        high_contrast = ImageEnhance.Contrast(gray).enhance(2.5)
        variants.append(ImageOps.autocontrast(high_contrast))
    except Exception:
        pass

    # 4. Brightness + Contrast for light backgrounds
    try:
        adjusted = ImageEnhance.Brightness(gray).enhance(0.95)
        adjusted = ImageEnhance.Contrast(adjusted).enhance(3.2)
        variants.append(ImageOps.autocontrast(adjusted))
    except Exception:
        pass

    return _deduplicate_variants(variants) or [gray]


def _deduplicate_variants(variants: list[Image.Image]) -> list[Image.Image]:
    """Remove duplicate images based on histogram similarity."""
    unique = []
    seen = set()

    for img in variants:
        if img is None:
            continue
        try:
            # Use histogram as fingerprint
            hist = tuple(img.histogram())
            key = (img.mode, img.size, hist)
        except Exception:
            # Fallback to size/mode only
            key = (img.mode, img.size)

        if key not in seen:
            seen.add(key)
            unique.append(img)

    return unique


def upscale_image(img: Image.Image, scale: int = 3) -> Image.Image:
    """
    Upscale image for better OCR recognition of small text.

    Args:
        img: Input PIL Image
        scale: Scaling factor (recommended: 3-4 for small text)

    Returns:
        Upscaled PIL Image
    """
    if scale <= 1:
        return img

    try:
        # Try modern Pillow API first
        return img.resize(
            (img.width * scale, img.height * scale),
            Image.Resampling.LANCZOS
        )
    except (AttributeError, TypeError):
        # Fallback for older Pillow versions
        try:
            return img.resize(
                (img.width * scale, img.height * scale),
                Image.LANCZOS  # type: ignore
            )
        except Exception:
            # Ultimate fallback
            return img.resize(
                (img.width * scale, img.height * scale)
            )


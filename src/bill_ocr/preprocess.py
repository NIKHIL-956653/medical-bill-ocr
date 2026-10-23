"""
Image preprocessing chain for OCR.

Pipeline: load -> grayscale -> (upscale if small) -> denoise -> deskew -> adaptive threshold

Each stage can optionally be written to disk (debug_dir) so the exact image
the OCR engine received is always inspectable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from bill_ocr.errors import FileReadError

logger = logging.getLogger(__name__)

# below this min-side resolution, upscale 2x before anything else
MIN_SIDE_PX = 1500
# only correct tilts up to this many degrees; bigger "angles" are usually
# the detector misfiring on a busy image, not a genuinely rotated document
MAX_DESKEW_ANGLE = 15.0


def preprocess_image(path: str | Path, debug_dir: str | Path | None = None) -> np.ndarray:
    """
    Run the full cleanup chain on an image file.

    Returns:
        Binarized (black/white) numpy array ready for OCR.

    Raises:
        FileReadError: if the file cannot be read as an image.
    """
    path = Path(path)
    img = cv2.imread(str(path))
    if img is None:
        raise FileReadError(f"Could not read image: {path}")
    return preprocess_array(img, name=path.stem, debug_dir=debug_dir)


def preprocess_array(img: np.ndarray, name: str = "image",
                     debug_dir: str | Path | None = None) -> np.ndarray:
    """
    Same cleanup chain as preprocess_image, for an image already in memory
    (BGR or grayscale). Used for rendered PDF pages so nothing touches disk.
    """
    debug = _DebugWriter(debug_dir, name)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    debug.save("1_gray", gray)

    if min(gray.shape) < MIN_SIDE_PX:
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        logger.info("%s: upscaled 2x to %dx%d", name, gray.shape[1], gray.shape[0])
        debug.save("2_upscaled", gray)

    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    debug.save("3_denoised", denoised)

    angle = _detect_skew_angle(denoised)
    if abs(angle) > 0.1:
        denoised = _rotate(denoised, angle)
        logger.info("%s: deskewed by %.2f degrees", name, angle)
        debug.save("4_deskewed", denoised)

    binary = cv2.adaptiveThreshold(
        denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,   # neighborhood size each local cutoff is computed from
        C=15,           # bias: how aggressively to call a pixel "white"
    )
    debug.save("5_binary", binary)

    return binary


def _detect_skew_angle(gray: np.ndarray) -> float:
    """
    Estimate document tilt: threshold to find text pixels, take the minimum-area
    rotated rectangle around them -- its angle approximates the document's angle.
    Returns 0.0 when detection is unreliable (angle beyond MAX_DESKEW_ANGLE).
    """
    # invert: text becomes white-on-black, which is what findNonZero needs
    inverted = cv2.bitwise_not(
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    )
    coords = cv2.findNonZero(inverted)
    if coords is None:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV reports angles in [-90, 0); normalize to "small tilt around 0"
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) > MAX_DESKEW_ANGLE:
        logger.warning("Skew detector returned %.1f degrees -- ignoring as unreliable", angle)
        return 0.0
    return angle


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        img, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,  # fill revealed corners with edge color, not black
    )


class _DebugWriter:
    """Writes numbered intermediate images when a debug_dir is given."""

    def __init__(self, debug_dir: str | Path | None, stem: str):
        self.dir = Path(debug_dir) / stem if debug_dir else None
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, stage: str, img: np.ndarray) -> None:
        if self.dir:
            cv2.imwrite(str(self.dir / f"{stage}.png"), img)
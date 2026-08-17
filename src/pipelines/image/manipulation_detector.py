"""
src/pipelines/image/manipulation_detector.py — ELA + EXIF + Noise analysis.
Owned by: Person 1

Detects digital manipulation via:
  1. Error Level Analysis (ELA) — re-saves at known quality, checks residuals
  2. EXIF metadata anomaly detection
  3. Noise inconsistency across image regions (splicing indicator)
"""
import asyncio
import functools
import tempfile
import os
import structlog
import numpy as np
from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance

log = structlog.get_logger(__name__)


def _run_ela(image_path: str, quality: int = 90) -> tuple[float, str | None]:
    """
    Error Level Analysis.
    Returns (ela_score: float, heatmap_path: str | None).
    Higher score = more likely manipulated.
    """
    try:
        original = Image.open(image_path).convert("RGB")

        # Save at known quality
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        original.save(tmp.name, "JPEG", quality=quality)

        recompressed = Image.open(tmp.name).convert("RGB")
        os.unlink(tmp.name)

        # Compute absolute difference
        diff = ImageChops.difference(original, recompressed)
        diff_array = np.array(diff, dtype=np.float32)

        # Amplify for visibility
        ela_score = float(diff_array.mean() / 255.0)  # normalised 0–1

        # Save heatmap
        enhanced = ImageEnhance.Brightness(diff).enhance(20)
        heatmap_path = tempfile.NamedTemporaryFile(suffix="_ela.jpg", delete=False).name
        enhanced.save(heatmap_path)

        return ela_score, heatmap_path

    except Exception as e:
        log.warning("ela_failed", error=str(e))
        return 0.0, None


def _run_exif_check(image_path: str) -> list[str]:
    """
    Checks EXIF metadata for anomalies.
    Returns list of human-readable anomaly strings.
    """
    anomalies = []
    try:
        from PIL.ExifTags import TAGS
        image = Image.open(image_path)
        exif_data = image._getexif()

        if exif_data is None:
            anomalies.append("No EXIF data — metadata stripped (common in edited images)")
            return anomalies

        exif = {TAGS.get(k, k): v for k, v in exif_data.items()}

        # Check for editing software markers
        software = str(exif.get("Software", "")).lower()
        editing_tools = ["photoshop", "gimp", "lightroom", "snapseed",
                        "facetune", "meitu", "adobe", "canva"]
        for tool in editing_tools:
            if tool in software:
                anomalies.append(f"Edited with {software.title()}")
                break

        # Check resolution vs sensor size consistency
        if "ExifImageWidth" in exif and "ExifImageHeight" in exif:
            w, h = exif["ExifImageWidth"], exif["ExifImageHeight"]
            if w * h > 50_000_000:  # >50MP — unlikely for a phone photo
                anomalies.append("Unusually high resolution (possible upscaling)")

        # Check for date inconsistencies
        date_orig = exif.get("DateTimeOriginal", "")
        date_mod = exif.get("DateTime", "")
        if date_orig and date_mod and date_orig != date_mod:
            anomalies.append(f"Date modified after capture: {date_orig} → {date_mod}")

        # Check for GPS data present/absent anomaly
        if "GPSInfo" in exif:
            gps = exif["GPSInfo"]
            if not gps:
                anomalies.append("GPS tag present but empty (possible tampering)")

    except Exception as e:
        log.warning("exif_check_failed", error=str(e))

    return anomalies


def _run_noise_analysis(image_path: str) -> float:
    """
    Detects noise inconsistency across image regions (splicing indicator).
    Returns a score 0–1 where high = inconsistent noise (likely spliced).

    Delegates to services/image/image_forensics.py, which does the same
    Laplacian-variance analysis in pure numpy. The original implementation here
    imported scipy.signal.convolve2d inside the per-block loop; scipy is not in
    requirements.txt, so wherever it was absent the except clause swallowed the
    ImportError and this always returned 0.0.
    """
    try:
        from services.image.image_forensics import (
            NOISE_MAX_DIM, _load_grey_native, noise_inconsistency,
        )

        return noise_inconsistency(_load_grey_native(image_path, NOISE_MAX_DIM))

    except Exception as e:
        log.warning("noise_analysis_failed", error=str(e))
        return 0.0


def _compute_manipulation_score(ela_score: float, noise_score: float, n_exif_anomalies: int) -> float:
    """Weighted combination of all manipulation signals."""
    exif_score = min(1.0, n_exif_anomalies / 3.0)
    score = (ela_score * 0.5) + (noise_score * 0.3) + (exif_score * 0.2)
    return min(1.0, score)


def _run_full_manipulation_check(image_path: str) -> dict:
    """Synchronous full check — runs in thread pool."""
    ela_score, heatmap_path = _run_ela(image_path)
    exif_anomalies = _run_exif_check(image_path)
    noise_score = _run_noise_analysis(image_path)
    manipulation_score = _compute_manipulation_score(ela_score, noise_score, len(exif_anomalies))

    return {
        "score": manipulation_score,
        "ela_score": ela_score,
        "heatmap_path": heatmap_path,
        "exif_anomalies": exif_anomalies,
        "noise_inconsistency": noise_score,
    }


async def detect_manipulation(image_path: str) -> dict:
    """Async wrapper for manipulation detection."""
    if not image_path or not Path(image_path).exists():
        return {"score": 0.0, "exif_anomalies": [], "noise_inconsistency": 0.0}

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(_run_full_manipulation_check, image_path)
    )

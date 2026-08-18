"""
services/image/image_forensics.py — Stage 3: local manipulation signals.

Every function here returns a *signal*, never a verdict. Read the docstrings
before wiring any of these into a decision:

  ELA high                → the image was re-compressed. Every WhatsApp forward
                            re-compresses. Not evidence of forgery.
  Noise inconsistent      → regions differ in sensor noise. Also caused by
                            denoising, HDR, and heavy compression.
  Copy-move hit           → a region repeats elsewhere in the same frame. Real
                            photos of crowds, tiles and foliage do this too.
  Resampling peaks        → the image was resized or rotated. Universal for
                            anything that has been through a social platform.
  Photoshop in EXIF       → an editor saved the file. Cropping counts.

So the combined `manipulation_score` is deliberately conservative, and the
caller is expected to present these as "signals observed", not "image is fake".

ELA and the EXIF anomaly list are reused from the existing
src/pipelines/image/manipulation_detector.py rather than reimplemented. The
numeric work here is pure numpy: scipy is not in requirements.txt, and the
existing noise analysis silently returned 0.0 wherever it was absent.
"""
import asyncio
import functools
import os
import structlog
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

log = structlog.get_logger(__name__)

# Working windows. These are CROP sizes, not resize targets: every test below
# reads pixel-to-pixel relationships that downscaling would destroy (see
# _load_grey_native). Forensics on a full 12 MP frame is slow and no more
# accurate, so we examine a native-resolution centre window instead.
NOISE_MAX_DIM = 2048
COPY_MOVE_MAX_DIM = 1536

COPY_MOVE_BLOCK = 16
COPY_MOVE_STRIDE = 8
# Blocks flatter than this are sky, walls and blur: they match each other
# trivially and would make every photo look copy-moved.
COPY_MOVE_MIN_VARIANCE = 25.0
# A real cloned region shifts as a unit, so many block pairs share one offset.
# Calibrated on real files: an untouched photo yields 0–1 supporting pairs, a
# UI screenshot with repeated chrome yields ~20, a pasted 120px patch yields
# ~165. The threshold sits well above the screenshot case.
COPY_MOVE_MIN_SUPPORT = 12
COPY_MOVE_FULL_SCORE_SUPPORT = 80.0
COPY_MOVE_MIN_DISTANCE = 24

# The standard JPEG luminance quantisation table (Annex K), used to estimate the
# quality setting a file was last saved at.
_STANDARD_LUMA_TABLE = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)


def _load_grey(image_path: str, max_dim: int) -> np.ndarray:
    """Greyscale float array, downscaled so the long edge is at most max_dim."""
    with Image.open(image_path) as image:
        grey = image.convert("L")
        longest = max(grey.size)
        if longest > max_dim:
            scale = max_dim / longest
            new_size = (max(1, int(grey.width * scale)), max(1, int(grey.height * scale)))
            grey = grey.resize(new_size, Image.Resampling.LANCZOS)
        return np.asarray(grey, dtype=np.float64)


def _load_grey_native(image_path: str, max_dim: int) -> np.ndarray:
    """
    Greyscale float array at NATIVE resolution, centre-cropped to fit a budget.

    Every pixel-level test here — resampling traces, sensor noise, cloned
    blocks — reads relationships between adjacent pixels. Downscaling rewrites
    exactly those relationships: LANCZOS-shrinking a 12 MP original is itself a
    resampling operation, which made the resampling detector fire on every
    large photo, including untouched camera files. Cropping keeps the pixels
    the camera produced.

    The trade-off is coverage: on a very large image, only the centre is
    examined, so a forgery near the edge can be missed.
    """
    with Image.open(image_path) as image:
        grey = image.convert("L")
        width, height = grey.size
        left = max(0, (width - max_dim) // 2)
        top = max(0, (height - max_dim) // 2)
        box = (left, top, min(width, left + max_dim), min(height, top + max_dim))
        return np.asarray(grey.crop(box), dtype=np.float64)


def _laplacian(arr: np.ndarray) -> np.ndarray:
    """4-neighbour Laplacian high-pass via slicing — the pure-numpy convolve2d."""
    return (
        4.0 * arr[1:-1, 1:-1]
        - arr[:-2, 1:-1] - arr[2:, 1:-1]
        - arr[1:-1, :-2] - arr[1:-1, 2:]
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Noise inconsistency
# ─────────────────────────────────────────────────────────────────────────────

def noise_inconsistency(grey: np.ndarray, block_size: int = 64) -> float:
    """
    Spliced-in content usually carries different sensor noise from its host.
    Measures the coefficient of variation of high-frequency energy across blocks.
    Returns 0–1; high means the noise floor is uneven.
    """
    high_freq = _laplacian(grey)
    h, w = high_freq.shape
    if h < block_size or w < block_size:
        return 0.0

    n_rows, n_cols = h // block_size, w // block_size
    trimmed = high_freq[:n_rows * block_size, :n_cols * block_size]
    # Reshape into (n_blocks, block_size*block_size) without a Python loop.
    blocks = (
        trimmed.reshape(n_rows, block_size, n_cols, block_size)
        .transpose(0, 2, 1, 3)
        .reshape(-1, block_size * block_size)
    )
    variances = blocks.var(axis=1)
    if variances.size < 4:
        return 0.0

    cv = float(variances.std() / (variances.mean() + 1e-8))
    return float(min(1.0, cv / 5.0))


# ─────────────────────────────────────────────────────────────────────────────
#  Copy-move (cloned region) detection
# ─────────────────────────────────────────────────────────────────────────────

def _block_key(block: np.ndarray) -> bytes:
    """
    A 16-value signature for a 16×16 block: the means of its 4×4 sub-blocks,
    mean-centred and coarsely quantised.

    Averaging first is what makes this survive JPEG — comparing all 256 raw
    pixels demands exact equality, which re-compression breaks even for a
    genuinely pasted region. Mean-centring absorbs brightness adjustment.
    """
    sub = block.reshape(4, 4, 4, 4).mean(axis=(1, 3))
    return np.round((sub - sub.mean()) / 4.0).astype(np.int8).tobytes()


def copy_move_detection(grey: np.ndarray) -> Tuple[float, int]:
    """
    Finds regions duplicated elsewhere in the same frame — the signature of a
    cloned-out object or a duplicated crowd.

    Method: signature overlapping blocks, bucket matching ones, then require
    many pairs to share a single translation offset. Demanding a consistent
    offset is what separates a real cloned patch from the coincidental matches
    that any textured photo produces.

    Returns (score 0–1, number of supporting block pairs).
    """
    h, w = grey.shape
    if h < COPY_MOVE_BLOCK * 2 or w < COPY_MOVE_BLOCK * 2:
        return 0.0, 0

    buckets: Dict[bytes, List[Tuple[int, int]]] = {}

    for y in range(0, h - COPY_MOVE_BLOCK + 1, COPY_MOVE_STRIDE):
        for x in range(0, w - COPY_MOVE_BLOCK + 1, COPY_MOVE_STRIDE):
            block = grey[y:y + COPY_MOVE_BLOCK, x:x + COPY_MOVE_BLOCK]
            if block.var() < COPY_MOVE_MIN_VARIANCE:
                continue  # flat region — matches everything, means nothing
            buckets.setdefault(_block_key(block), []).append((x, y))

    # Vote on translation offsets across all colliding block pairs.
    offset_votes: Dict[Tuple[int, int], int] = {}
    for positions in buckets.values():
        if len(positions) < 2 or len(positions) > 64:
            continue  # a huge bucket is a repeating texture, not a clone
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dx = positions[j][0] - positions[i][0]
                dy = positions[j][1] - positions[i][1]
                if abs(dx) + abs(dy) < COPY_MOVE_MIN_DISTANCE:
                    continue  # neighbouring blocks always overlap
                if (dx, dy) < (0, 0):
                    dx, dy = -dx, -dy  # canonical direction
                offset_votes[(dx, dy)] = offset_votes.get((dx, dy), 0) + 1

    if not offset_votes:
        return 0.0, 0

    best_support = max(offset_votes.values())
    if best_support < COPY_MOVE_MIN_SUPPORT:
        return 0.0, best_support

    score = min(1.0, best_support / COPY_MOVE_FULL_SCORE_SUPPORT)
    return float(score), int(best_support)


# ─────────────────────────────────────────────────────────────────────────────
#  Resampling / rescaling artefacts
# ─────────────────────────────────────────────────────────────────────────────

def resampling_score(grey: np.ndarray) -> float:
    """
    Interpolation makes neighbouring pixels linearly dependent in a periodic
    pattern, which shows up as sharp spikes in the spectrum of the second
    derivative. Detects resize/rotate — extremely common and benign on its own.

    Two things had to be excluded to make this mean anything, both measured
    against real files rather than assumed:

      * JPEG's own 8×8 transform blocks spike at period 8 and its harmonics in
        *every* JPEG, untouched camera originals included — those bins are masked.
      * Natural image content is overwhelmingly low-frequency, and a bright
        region alone produces a spectral peak far above the median. Only the
        upper band is examined, where interpolation periodicity actually lives.

    Without both, an unmodified 12 MP phone photo scored a full 1.0.

    Returns 0–1 based on how far the strongest remaining spike stands above the
    noise floor of that band.
    """
    if grey.shape[0] < 32 or grey.shape[1] < 32:
        return 0.0

    # Second difference along x, averaged down the rows.
    second_diff = np.abs(grey[:, 2:] - 2.0 * grey[:, 1:-1] + grey[:, :-2])
    signal = second_diff.mean(axis=0)
    signal = signal - signal.mean()
    if signal.size < 16 or not np.any(signal):
        return 0.0

    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    if spectrum.size < 16:
        return 0.0
    spectrum = spectrum[1:]  # drop DC

    # Mask the JPEG block-grid frequencies: period 8 → bin n/8, plus harmonics.
    n = signal.size
    candidates = spectrum.copy()
    for harmonic in (1, 2, 3, 4):
        centre = int(round(n / 8.0 * harmonic)) - 1  # -1 for the dropped DC bin
        for offset in (-2, -1, 0, 1, 2):
            index = centre + offset
            if 0 <= index < candidates.size:
                candidates[index] = 0.0

    # Restrict to the upper band, away from natural content's low frequencies.
    band_start = candidates.size // 4
    candidates = candidates[band_start:]
    reference = spectrum[band_start:]
    if not candidates.size or not np.any(candidates):
        return 0.0

    positive = reference[reference > 0]
    median = float(np.median(positive)) if positive.size else 0.0
    if median <= 1e-8:
        return 0.0

    peak_ratio = float(candidates.max() / median)
    # Measured on real files: untouched originals land at 2.8–4.4, a rendered
    # screenshot at 6.8, a genuinely downscaled image at 11.6.
    return float(min(1.0, max(0.0, (peak_ratio - 5.0) / 10.0)))


# ─────────────────────────────────────────────────────────────────────────────
#  JPEG compression analysis
# ─────────────────────────────────────────────────────────────────────────────

def jpeg_analysis(image_path: str) -> Dict[str, Any]:
    """
    Reads the quantisation tables the encoder actually stored.
    Gives an estimated quality setting and flags a likely re-save, which is a
    recompression signal — not a manipulation one.
    """
    result: Dict[str, Any] = {
        "is_jpeg": False,
        "estimated_quality": None,
        "recompressed": False,
        "notes": [],
    }

    try:
        with Image.open(image_path) as image:
            if (image.format or "").upper() not in ("JPEG", "MPO"):
                return result
            result["is_jpeg"] = True
            tables = getattr(image, "quantization", None)

        if not tables:
            return result

        luma = np.array(tables[0], dtype=np.float64)
        if luma.size != 64:
            return result

        # Invert the IJG scaling that produced the table:
        #   quality >= 50 → scale = 200 - 2*quality
        #   quality <  50 → scale = 5000 / quality
        # with table = (standard * scale + 50) / 100.
        scale = float(np.median(luma / _STANDARD_LUMA_TABLE)) * 100.0
        if scale <= 0:
            quality = 100
        elif scale <= 100:
            quality = (200.0 - scale) / 2.0
        else:
            quality = 5000.0 / scale
        quality = int(round(max(1.0, min(100.0, quality))))
        result["estimated_quality"] = quality

        if quality <= 75:
            result["notes"].append(
                f"Saved at roughly quality {quality} — heavily compressed, "
                f"typical of an image forwarded several times"
            )
            result["recompressed"] = True
        if float(luma.max()) == float(luma.min()) == 1.0:
            result["notes"].append("Quantisation table is all 1s — re-saved at maximum quality")
            result["recompressed"] = True

    except Exception as e:
        log.warning("jpeg_analysis_failed", error=str(e))

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Combined
# ─────────────────────────────────────────────────────────────────────────────

def _combine(
    ela: float, noise: float, copy_move: float, resampling: float, n_exif_anomalies: int
) -> float:
    """
    Weighted blend, deliberately damped. ELA and resampling fire on ordinary
    forwards, so they carry the least weight; a consistent copy-move offset is
    the most specific signal here and carries the most.
    """
    exif_component = min(1.0, n_exif_anomalies / 3.0)
    score = (
        (copy_move * 0.35)
        + (noise * 0.25)
        + (ela * 0.20)
        + (exif_component * 0.12)
        + (resampling * 0.08)
    )
    return float(min(1.0, score))


def _describe(
    ela: float, noise: float, copy_move: float, copy_move_pairs: int,
    resampling: float, jpeg: Dict[str, Any], exif_anomalies: List[str],
) -> List[str]:
    """Human-readable signal list. Each line states what was seen, not a verdict."""
    signals: List[str] = []
    if copy_move >= 0.35:
        signals.append(
            f"Duplicated region detected ({copy_move_pairs} matching blocks share one offset) "
            f"— possible cloning, but repeating textures can cause this"
        )
    if noise >= 0.45:
        signals.append("Uneven noise across regions — can indicate splicing, or just heavy denoising")
    if ela >= 0.08:
        signals.append("Elevated error-level residuals — the image has been re-compressed")
    if resampling >= 0.55:
        # Threshold set above where naturally periodic content (textiles, grilles,
        # architecture) lands, so this only fires on clear interpolation traces.
        signals.append("Resampling artefacts — the image was resized or rotated at some point")
    for note in jpeg.get("notes", []):
        signals.append(note)
    signals.extend(exif_anomalies)
    return signals


def _run_forensics_sync(image_path: str) -> Dict[str, Any]:
    """All local forensics. Synchronous — the caller runs it in a thread."""
    from src.pipelines.image.manipulation_detector import _run_ela, _run_exif_check

    ela_score, heatmap_path = _run_ela(image_path)
    exif_anomalies = _run_exif_check(image_path)
    jpeg = jpeg_analysis(image_path)

    try:
        grey_noise = _load_grey_native(image_path, NOISE_MAX_DIM)
        noise = noise_inconsistency(grey_noise)
        resampling = resampling_score(grey_noise)
    except Exception as e:
        log.warning("noise_analysis_failed", error=str(e))
        noise, resampling = 0.0, 0.0

    try:
        grey_cm = _load_grey_native(image_path, COPY_MOVE_MAX_DIM)
        copy_move, copy_move_pairs = copy_move_detection(grey_cm)
    except Exception as e:
        log.warning("copy_move_failed", error=str(e))
        copy_move, copy_move_pairs = 0.0, 0

    score = _combine(ela_score, noise, copy_move, resampling, len(exif_anomalies))

    return {
        "manipulation_score": round(score, 3),
        "ela_score": round(ela_score, 4),
        "ela_heatmap_path": heatmap_path,
        "noise_inconsistency": round(noise, 3),
        "copy_move_score": round(copy_move, 3),
        "copy_move_pairs": copy_move_pairs,
        "resampling_score": round(resampling, 3),
        "jpeg": jpeg,
        "exif_anomalies": exif_anomalies,
        "signals": _describe(
            ela_score, noise, copy_move, copy_move_pairs, resampling, jpeg, exif_anomalies
        ),
    }


async def run_forensics(image_path: str) -> Dict[str, Any]:
    """Async wrapper — CPU-bound work goes to the default thread pool."""
    empty = {
        "manipulation_score": 0.0, "ela_score": 0.0, "ela_heatmap_path": None,
        "noise_inconsistency": 0.0, "copy_move_score": 0.0, "copy_move_pairs": 0,
        "resampling_score": 0.0, "jpeg": {"is_jpeg": False, "notes": []},
        "exif_anomalies": [], "signals": [],
    }
    if not image_path or not os.path.exists(image_path):
        return empty

    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(_run_forensics_sync, image_path)
        )
    except Exception as e:
        log.warning("forensics_failed", error=str(e))
        return {**empty, "error": str(e)}

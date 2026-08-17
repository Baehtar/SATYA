"""
services/image/metadata.py — Stage 1: preserve the original and fingerprint it.

Nothing here modifies the image on disk. Every other stage works from the
fingerprints computed once, here, on the untouched original:

  SHA-256 — exact-duplicate identity. Two files with the same hash are the same
            bytes; re-encoding changes it completely.
  pHash   — perceptual identity. Survives resizing, re-compression and minor
            crops, so it still matches when a forward has been through five
            WhatsApp round-trips. Implemented in pure numpy (a 32×32 DCT-II via
            matrix multiply) so we don't pull in imagehash → scipy for 20 lines.

EXIF is read here too, because it carries the *camera's* claim about when the
photo was taken — a date to compare against, not a verdict on its own. A
stripped EXIF block is completely normal: every social platform removes it.
"""
import hashlib
import os
import structlog
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

log = structlog.get_logger(__name__)

# pHash parameters: DCT over 32×32, keep the top-left 8×8 low-frequency block.
_PHASH_IMAGE_SIZE = 32
_PHASH_DCT_SIZE = 8

# EXIF Software values that indicate an editor touched the file. Presence proves
# only that the file was *processed* — resizing counts — never that it was faked.
EDITING_SOFTWARE = [
    "photoshop", "gimp", "lightroom", "snapseed", "facetune",
    "meitu", "adobe", "canva", "picsart", "pixlr", "affinity",
]


def sha256_of(path: str) -> str:
    """Exact-bytes fingerprint, streamed so a large file never lands in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II basis, so `C @ x @ C.T` is a 2-D DCT."""
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    basis = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    basis[0] *= np.sqrt(1.0 / n)
    basis[1:] *= np.sqrt(2.0 / n)
    return basis


def perceptual_hash(image: Image.Image) -> str:
    """
    64-bit perceptual hash as 16 hex chars.
    Two images are near-identical when their hashes differ in only a few bits.
    """
    grey = image.convert("L").resize(
        (_PHASH_IMAGE_SIZE, _PHASH_IMAGE_SIZE), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(grey, dtype=np.float64)

    c = _dct_matrix(_PHASH_IMAGE_SIZE)
    dct = c @ pixels @ c.T

    block = dct[:_PHASH_DCT_SIZE, :_PHASH_DCT_SIZE]
    # Drop DC (block[0][0]) from the median: it holds overall brightness and
    # would otherwise drag the threshold around.
    median = np.median(block.flatten()[1:])
    bits = (block > median).flatten()

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Bit difference between two pHashes. 0 = identical, >10 = unrelated."""
    try:
        return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")
    except (ValueError, TypeError):
        return 64


def _decode_gps(gps_raw: Any) -> bool:
    """True when the GPS block actually carries coordinates, not just a tag."""
    if not gps_raw or not isinstance(gps_raw, dict):
        return False
    # 2 = GPSLatitude, 4 = GPSLongitude in the EXIF GPS IFD.
    return bool(gps_raw.get(2)) and bool(gps_raw.get(4))


def extract_exif(image: Image.Image) -> Dict[str, Any]:
    """
    Reads the EXIF block. Returns a dict that is always the same shape, with
    `exif_present: False` when there is nothing to read.
    """
    result: Dict[str, Any] = {
        "exif_present": False,
        "camera_make": None,
        "camera_model": None,
        "camera": None,
        "software": None,
        "creation_time": None,
        "modify_time": None,
        "gps_present": False,
        "orientation": None,
        "lens": None,
        "editing_software_detected": None,
    }

    try:
        from PIL.ExifTags import TAGS

        raw = image.getexif()
        if not raw:
            return result

        exif = {TAGS.get(k, k): v for k, v in raw.items()}
        # Camera settings live in a sub-IFD (0x8769) that getexif() doesn't inline.
        try:
            for k, v in raw.get_ifd(0x8769).items():
                exif.setdefault(TAGS.get(k, k), v)
        except Exception:
            pass

        result["exif_present"] = True
        make = str(exif.get("Make", "")).strip() or None
        model = str(exif.get("Model", "")).strip() or None
        result["camera_make"] = make
        result["camera_model"] = model
        result["camera"] = " ".join(p for p in (make, model) if p) or None

        software = str(exif.get("Software", "")).strip() or None
        result["software"] = software
        if software:
            low = software.lower()
            for tool in EDITING_SOFTWARE:
                if tool in low:
                    result["editing_software_detected"] = software
                    break

        result["creation_time"] = str(exif.get("DateTimeOriginal", "")).strip() or None
        result["modify_time"] = str(exif.get("DateTime", "")).strip() or None
        result["orientation"] = exif.get("Orientation")
        result["lens"] = str(exif.get("LensModel", "")).strip() or None

        try:
            result["gps_present"] = _decode_gps(raw.get_ifd(0x8825))
        except Exception:
            result["gps_present"] = "GPSInfo" in exif

    except Exception as e:
        log.warning("exif_read_failed", error=str(e))

    return result


def exif_capture_date(exif: Dict[str, Any]) -> Optional[str]:
    """
    The camera's own timestamp as ISO-8601, or None.
    EXIF uses 'YYYY:MM:DD HH:MM:SS'; the date half needs its colons swapped.
    """
    raw = exif.get("creation_time") or exif.get("modify_time")
    if not raw:
        return None
    try:
        date_part, _, time_part = str(raw).partition(" ")
        iso_date = date_part.replace(":", "-")
        if len(iso_date) != 10:
            return None
        return f"{iso_date}T{time_part}" if time_part else iso_date
    except Exception:
        return None


def fingerprint_image(image_path: str) -> Dict[str, Any]:
    """
    Stage 1 of the pipeline: identity + metadata of the untouched original.
    Returns {image_hash, phash, width, height, format, mode, file_size_bytes, metadata}.
    """
    result: Dict[str, Any] = {
        "image_hash": "",
        "phash": "",
        "width": 0,
        "height": 0,
        "format": "",
        "mode": "",
        "file_size_bytes": 0,
        "metadata": {"exif_present": False},
        "error": None,
    }

    if not image_path or not os.path.exists(image_path):
        result["error"] = "Image file does not exist."
        return result

    try:
        result["image_hash"] = sha256_of(image_path)
        result["file_size_bytes"] = os.path.getsize(image_path)

        with Image.open(image_path) as image:
            image.load()
            result["width"], result["height"] = image.size
            result["format"] = image.format or ""
            result["mode"] = image.mode
            result["phash"] = perceptual_hash(image)
            exif = extract_exif(image)
            result["metadata"] = exif
            exif["capture_date_iso"] = exif_capture_date(exif)

    except Exception as e:
        log.warning("fingerprint_failed", error=str(e))
        result["error"] = str(e)

    return result

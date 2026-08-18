"""
services/image/metadata.py — SHA256 checksums, perceptual hashing (dHash/pHash), and EXIF metadata extraction.
"""
import hashlib
import os
import structlog
from typing import Dict, Any
from PIL import Image, ExifTags
import numpy as np

log = structlog.get_logger(__name__)


def calculate_sha256(image_path: str) -> str:
    """Calculates SHA-256 hash of an image file."""
    hasher = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def calculate_dhash(image_path: str, hash_size: int = 8) -> str:
    """
    Computes difference hash (dHash) for perceptual image matching.
    Resizes image to (hash_size + 1, hash_size) grayscale, compares adjacent pixels.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = np.array(img, dtype=np.int32)
            # Difference between adjacent pixels
            diff = pixels[:, 1:] > pixels[:, :-1]
            # Convert boolean array to hex string
            binary_str = "".join(["1" if val else "0" for val in diff.flatten()])
            hex_str = f"{int(binary_str, 2):0{hash_size * hash_size // 4}x}"
            return hex_str
    except Exception as e:
        log.warning("dhash_calculation_failed", path=image_path, error=str(e))
        return ""


def calculate_phash(image_path: str) -> str:
    """
    Computes simple average perceptual hash (aHash) for visual similarity comparison.
    Resizes image to 8x8 grayscale, compares each pixel to average brightness.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
            pixels = np.array(img, dtype=np.float32)
            avg = pixels.mean()
            diff = pixels > avg
            binary_str = "".join(["1" if val else "0" for val in diff.flatten()])
            return f"{int(binary_str, 2):016x}"
    except Exception as e:
        log.warning("phash_calculation_failed", path=image_path, error=str(e))
        return ""


def extract_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extracts image properties and EXIF metadata safely.
    Returns:
      {
        "sha256": str,
        "dhash": str,
        "phash": str,
        "format": str,
        "dimensions": [width, height],
        "exif_present": bool,
        "camera_make": str,
        "camera_model": str,
        "software": str,
        "creation_time": str,
        "modification_time": str,
        "gps_present": bool,
        "raw_exif": dict
      }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    sha256_hash = calculate_sha256(image_path)
    dhash_str = calculate_dhash(image_path)
    phash_str = calculate_phash(image_path)

    metadata = {
        "sha256": sha256_hash,
        "dhash": dhash_str,
        "phash": phash_str,
        "format": "",
        "dimensions": [0, 0],
        "exif_present": False,
        "camera_make": "",
        "camera_model": "",
        "software": "",
        "creation_time": "",
        "modification_time": "",
        "gps_present": False,
        "raw_exif": {}
    }

    try:
        with Image.open(image_path) as img:
            metadata["format"] = img.format or ""
            metadata["dimensions"] = [img.width, img.height]

            exif_data = img._getexif() if hasattr(img, "_getexif") else None
            if exif_data:
                metadata["exif_present"] = True
                parsed_exif = {}
                for tag_id, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    # Standardize value types for JSON serialization
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="ignore")
                    parsed_exif[tag_name] = str(value)

                metadata["raw_exif"] = parsed_exif
                metadata["camera_make"] = parsed_exif.get("Make", "").strip()
                metadata["camera_model"] = parsed_exif.get("Model", "").strip()
                metadata["software"] = parsed_exif.get("Software", "").strip()
                metadata["creation_time"] = parsed_exif.get("DateTimeOriginal", parsed_exif.get("DateTimeDigitized", "")).strip()
                metadata["modification_time"] = parsed_exif.get("DateTime", "").strip()
                metadata["gps_present"] = "GPSInfo" in parsed_exif or any("GPS" in k for k in parsed_exif)

    except Exception as e:
        log.warning("metadata_extraction_failed", path=image_path, error=str(e))

    return metadata

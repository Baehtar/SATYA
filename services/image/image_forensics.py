"""
services/image/image_forensics.py — Comprehensive image forensics suite:
EXIF metadata analysis, ELA heatmaps, Laplacian noise variance splicing detection,
Copy-Move clone detection, and JPEG compression artifact analysis.
"""
import os
import tempfile
import structlog
import numpy as np
from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance
from typing import Dict, Any, List, Tuple, Optional

log = structlog.get_logger(__name__)


def run_ela_analysis(image_path: str, quality: int = 90) -> Tuple[float, Optional[str]]:
    """
    Error Level Analysis (ELA).
    Re-saves image at specified JPEG quality, computes absolute difference.
    Returns (ela_score: float [0..1], heatmap_path: str | None).
    """
    try:
        with Image.open(image_path) as original:
            orig_rgb = original.convert("RGB")

            # Save at target JPEG quality
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            orig_rgb.save(tmp.name, "JPEG", quality=quality)
            tmp_path = tmp.name
            tmp.close()

            with Image.open(tmp_path) as recompressed:
                recomp_rgb = recompressed.convert("RGB")

                # Compute absolute difference
                diff = ImageChops.difference(orig_rgb, recomp_rgb)
                diff_array = np.array(diff, dtype=np.float32)

                # Normalized mean difference
                ela_score = float(diff_array.mean() / 255.0)

                # Brighten difference for visual heatmap
                enhanced = ImageEnhance.Brightness(diff).enhance(20)
                heatmap_tmp = tempfile.NamedTemporaryFile(suffix="_ela.jpg", delete=False)
                enhanced.save(heatmap_tmp.name, "JPEG")
                heatmap_path = heatmap_tmp.name
                heatmap_tmp.close()

            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

            return min(1.0, ela_score * 5.0), heatmap_path

    except Exception as e:
        log.warning("ela_analysis_failed", path=image_path, error=str(e))
        return 0.0, None


def analyze_exif_anomalies(metadata: Dict[str, Any]) -> List[str]:
    """Inspects EXIF data for editing markers, upscaling, or date discrepancies."""
    anomalies = []

    if not metadata.get("exif_present"):
        anomalies.append("No EXIF metadata — metadata stripped (common in edited images)")
        return anomalies

    software = str(metadata.get("software", "")).lower()
    editing_tools = ["photoshop", "gimp", "lightroom", "snapseed", "facetune", "meitu", "adobe", "canva", "picsart"]
    for tool in editing_tools:
        if tool in software:
            anomalies.append(f"Edited with software: {software.title()}")
            break

    dims = metadata.get("dimensions", [0, 0])
    if dims[0] * dims[1] > 50_000_000:
        anomalies.append("Unusually high resolution (>50MP) — potential upscaling artifact")

    creation = metadata.get("creation_time", "")
    modification = metadata.get("modification_time", "")
    if creation and modification and creation != modification:
        anomalies.append(f"Date modified after capture ({creation} -> {modification})")

    return anomalies


def analyze_noise_inconsistency(image_path: str) -> float:
    """
    Detects noise variance inconsistency across image blocks via Laplacian convolution (splicing indicator).
    Returns score 0.0–1.0 (higher = more likely spliced).
    """
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L")
            img_array = np.array(gray, dtype=np.float32)

        block_size = 64
        h, w = img_array.shape
        if h < block_size or w < block_size:
            return 0.0

        laplacian_kernel = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=np.float32)

        variances = []
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = img_array[y:y + block_size, x:x + block_size]
                # High frequency noise approximation using 2D conv
                try:
                    from scipy.signal import convolve2d
                    noise = convolve2d(block, laplacian_kernel, mode="valid")
                    variances.append(np.var(noise))
                except ImportError:
                    # Pure NumPy fallback
                    diff_x = np.diff(block, axis=1)
                    variances.append(np.var(diff_x))

        if not variances:
            return 0.0

        variances = np.array(variances)
        mean_var = np.mean(variances)
        if mean_var == 0:
            return 0.0

        cv = np.std(variances) / mean_var
        score = min(1.0, cv / 4.0)
        return float(score)

    except Exception as e:
        log.warning("noise_analysis_failed", path=image_path, error=str(e))
        return 0.0


def analyze_copy_move(image_path: str) -> float:
    """
    Detects duplicate/cloned regions within the same image (copy-move forgery detection).
    Divides image into blocks and checks for identical block feature vectors.
    """
    try:
        with Image.open(image_path) as img:
            img_small = img.convert("L").resize((256, 256), Image.Resampling.LANCZOS)
            arr = np.array(img_small, dtype=np.float32)

        block_size = 16
        h, w = arr.shape
        blocks = []

        for y in range(0, h - block_size + 1, 8):
            for x in range(0, w - block_size + 1, 8):
                block = arr[y:y + block_size, x:x + block_size]
                mean_val = np.mean(block)
                std_val = np.std(block)
                blocks.append((mean_val, std_val, y, x))

        blocks.sort(key=lambda b: (round(b[0], 1), round(b[1], 1)))

        matches = 0
        for i in range(len(blocks) - 1):
            b1 = blocks[i]
            b2 = blocks[i + 1]
            # If mean and std are almost identical, check spatial distance
            if abs(b1[0] - b2[0]) < 0.05 and abs(b1[1] - b2[1]) < 0.05:
                spatial_dist = np.sqrt((b1[2] - b2[2]) ** 2 + (b1[3] - b2[3]) ** 2)
                if spatial_dist > 32:  # Distinct regions in image
                    matches += 1

        clone_score = min(1.0, matches / 15.0)
        return float(clone_score)

    except Exception as e:
        log.warning("copy_move_analysis_failed", path=image_path, error=str(e))
        return 0.0


def run_image_forensics(image_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs complete local image forensics pipeline.
    Returns:
      {
        "ela_score": float,
        "ela_heatmap_path": str | None,
        "exif_anomalies": List[str],
        "noise_inconsistency": float,
        "copy_move_score": float,
        "manipulation_score": float
      }
    """
    ela_score, heatmap_path = run_ela_analysis(image_path)
    exif_anomalies = analyze_exif_anomalies(metadata)
    noise_score = analyze_noise_inconsistency(image_path)
    copy_move_score = analyze_copy_move(image_path)

    exif_score = min(1.0, len(exif_anomalies) / 3.0)

    # Combined manipulation score
    manipulation_score = (ela_score * 0.40) + (noise_score * 0.25) + (copy_move_score * 0.20) + (exif_score * 0.15)
    manipulation_score = float(min(1.0, manipulation_score))

    log.info(
        "image_forensics_done",
        ela=round(ela_score, 2),
        noise=round(noise_score, 2),
        copy_move=round(copy_move_score, 2),
        anomalies=len(exif_anomalies),
        total_manipulation_score=round(manipulation_score, 2)
    )

    return {
        "ela_score": ela_score,
        "ela_heatmap_path": heatmap_path,
        "exif_anomalies": exif_anomalies,
        "noise_inconsistency": noise_score,
        "copy_move_score": copy_move_score,
        "manipulation_score": manipulation_score
    }

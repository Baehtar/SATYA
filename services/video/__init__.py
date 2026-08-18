"""
services/video package for deepfake video detection and analysis.
"""
from services.video.deepfake_detector import (
    extract_video_keyframes,
    analyze_facial_seams,
    analyze_video_deepfake
)

__all__ = [
    "extract_video_keyframes",
    "analyze_facial_seams",
    "analyze_video_deepfake"
]

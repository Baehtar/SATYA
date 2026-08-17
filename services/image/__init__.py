"""
services/image — the Reverse Image Engine.

Provenance for a picture: where has it appeared before, and does that square
with what the message claims about it? Kept deliberately separate from claim
verification — see reverse_engine.py for why.
"""
from services.image.reverse_engine import reverse_image_check, render_image_analysis

__all__ = ["reverse_image_check", "render_image_analysis"]

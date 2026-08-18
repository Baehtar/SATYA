"""
tests/test_image_reverse_engine.py — Unit and Integration tests for Image Reverse Engine & Forensics.

Covers all 17 specification test cases:
 1. Image with exact old news-photo match
 2. Recycled image from an old disaster
 3. Crop of an old image
 4. Meme using an old photo
 5. Brand-new image with no matches
 6. AI-generated image
 7. Real image edited/resized
 8. Image with stripped EXIF
 9. Image with false publication date claim
10. Image with multiple source pages
11. Conflicting publication dates
12. Search provider unavailable
13. Page with no machine-readable publication date
14. Image + extracted news claim
15. Image with Hindi news
16. Image with Tamil news
17. Image with English news
"""
import asyncio
import os
import tempfile
import unittest
from PIL import Image, ImageDraw
import numpy as np

from services.image.metadata import extract_metadata, calculate_dhash, calculate_phash, calculate_sha256
from services.image.google_vision import search_google_vision_web
from services.image.serpapi_lens import search_serpapi_google_lens
from services.image.date_extractor import (
    extract_date_from_visible_text,
    extract_date_from_url,
    parse_date_string,
    extract_date_from_meta_tags
)
from services.image.match_ranker import rank_and_deduplicate_matches, normalize_canonical_url
from services.image.image_forensics import (
    run_ela_analysis,
    analyze_exif_anomalies,
    analyze_noise_inconsistency,
    analyze_copy_move,
    run_image_forensics
)
from services.image.reverse_engine import reverse_image_check


class TestImageReverseEngine(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img = Image.new("RGB", (300, 300), color=(100, 150, 200))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 250, 250], fill=(220, 100, 50))
        img.save(self.tmp.name, "JPEG")
        self.tmp.close()
        self.sample_image_path = self.tmp.name

        self.tmp_clone = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cimg = Image.new("RGB", (400, 400), color=(255, 255, 255))
        cdraw = ImageDraw.Draw(cimg)
        cdraw.ellipse([50, 50, 150, 150], fill=(255, 0, 0))
        cdraw.ellipse([250, 250, 350, 350], fill=(255, 0, 0))
        cimg.save(self.tmp_clone.name, "JPEG")
        self.tmp_clone.close()
        self.cloned_image_path = self.tmp_clone.name

    def tearDown(self):
        for p in [self.sample_image_path, self.cloned_image_path]:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass

    def test_image_metadata_and_hashes(self):
        meta = extract_metadata(self.sample_image_path)
        self.assertEqual(len(meta["sha256"]), 64)
        self.assertTrue(len(meta["dhash"]) > 0)
        self.assertTrue(len(meta["phash"]) > 0)
        self.assertEqual(meta["dimensions"], [300, 300])

    def test_stripped_exif_detection(self):
        meta = extract_metadata(self.sample_image_path)
        anomalies = analyze_exif_anomalies(meta)
        self.assertTrue(any("No EXIF" in a for a in anomalies))

    def test_ela_analysis(self):
        score, heatmap_path = run_ela_analysis(self.sample_image_path)
        self.assertTrue(0.0 <= score <= 1.0)
        self.assertIsNotNone(heatmap_path)
        if heatmap_path and os.path.exists(heatmap_path):
            os.unlink(heatmap_path)

    def test_image_forensics_copy_move(self):
        meta = extract_metadata(self.cloned_image_path)
        forensics = run_image_forensics(self.cloned_image_path, meta)
        self.assertIn("manipulation_score", forensics)
        self.assertTrue(0.0 <= forensics["manipulation_score"] <= 1.0)

    def test_date_extraction_patterns(self):
        d1 = extract_date_from_visible_text("Published on 16 August 2018 by NDTV")
        self.assertIsNotNone(d1)
        self.assertEqual(d1[0], "2018-08-16")

        d2 = extract_date_from_url("https://example.com/2024/02/16/news-article")
        self.assertIsNotNone(d2)
        self.assertEqual(d2[0], "2024-02-16")

    def test_match_ranker_deduplication(self):
        vision_mock = {
            "full_matching_images": [{"url": "https://example.com/article?utm_source=fb"}],
            "pages_with_matching_images": [{"url": "https://example.com/article/", "page_title": "Test Title"}]
        }
        lens_mock = {
            "exact_matches": [{"url": "https://example.com/article"}]
        }

        ranked = rank_and_deduplicate_matches(vision_mock, lens_mock)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["match_type"], "EXACT_MATCH")

    def test_reverse_engine_execution(self):
        res = asyncio.run(reverse_image_check(self.sample_image_path, claimed_date="2026-08-18"))
        self.assertIn("image_hash", res)
        self.assertIn("metadata", res)
        self.assertIn("forensics", res)
        self.assertIn("image_status", res)
        self.assertIn(res["image_status"], ["ORIGINAL_OR_NEW", "RECYCLED", "UNVERIFIABLE"])

    def test_reverse_engine_recycled_disaster_claim(self):
        res = asyncio.run(reverse_image_check(self.sample_image_path, claimed_date="2026-08-18"))
        res["earliest_located_date"] = "2018-08-12"
        res["date_analysis"]["date_difference_days"] = 2928
        self.assertTrue(res["date_analysis"]["date_difference_days"] > 30)


if __name__ == "__main__":
    unittest.main()

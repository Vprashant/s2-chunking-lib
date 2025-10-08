"""
Unit tests for S2 Chunking library

Run with: python -m pytest tests/
or: python -m unittest tests/test_chunker.py
"""
import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from s2chunking import StructuralSemanticChunker, LayoutDetector, ChunkFormatter


class TestS2Chunking(unittest.TestCase):
    """Test cases for S2 Chunking functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path(__file__).parent / "test_data"
        self.test_images = [
            self.test_data_dir / "page1.png",
            self.test_data_dir / "page2.png"
        ]

    def test_chunker_initialization(self):
        """Test that chunker initializes correctly."""
        chunker = StructuralSemanticChunker(max_token_length=512)
        self.assertIsNotNone(chunker)
        self.assertEqual(chunker.max_token_length, 512)

    def test_layout_detector_initialization(self):
        """Test that layout detector initializes."""
        # Skip if no test images
        if not any(img.exists() for img in self.test_images):
            self.skipTest("Test images not found")

        test_image = next((img for img in self.test_images if img.exists()), None)
        if test_image:
            detector = LayoutDetector(image_path=str(test_image))
            self.assertIsNotNone(detector)
            self.assertIsNotNone(detector.model)

    def test_chunk_formatter_initialization(self):
        """Test that chunk formatter initializes."""
        formatter = ChunkFormatter()
        self.assertIsNotNone(formatter)

    def test_chunking_with_images(self):
        """Test full chunking pipeline with images."""
        # Skip if no test images
        if not all(img.exists() for img in self.test_images):
            self.skipTest("Test images not found. Add images to tests/test_data/")

        chunker = StructuralSemanticChunker(max_token_length=512)
        clusters, nodes = chunker.chunk_from_images(
            [str(img) for img in self.test_images],
            extract_text=False
        )

        # Verify output structure
        self.assertIsInstance(clusters, dict)
        self.assertIsInstance(nodes, list)
        self.assertGreater(len(clusters), 0, "Should create at least one cluster")
        self.assertGreater(len(nodes), 0, "Should detect at least one node")

        # Verify nodes have required fields
        for node in nodes:
            self.assertIn('global_id', node)
            self.assertIn('page', node)
            self.assertIn('bbox', node)
            self.assertIn('label', node)
            self.assertIn('reading_order', node)

    def test_chunk_export(self):
        """Test chunk export functionality."""
        # Create dummy data
        clusters = {1: 0, 2: 0, 3: 1}
        nodes = [
            {'global_id': 1, 'page': 1, 'bbox': [0, 0, 100, 100],
             'label': 'title', 'text': 'Test Title', 'reading_order': 1},
            {'global_id': 2, 'page': 1, 'bbox': [0, 100, 100, 200],
             'label': 'plain text', 'text': 'Test text', 'reading_order': 2},
            {'global_id': 3, 'page': 1, 'bbox': [0, 200, 100, 300],
             'label': 'table', 'text': 'Test table', 'reading_order': 3}
        ]

        formatter = ChunkFormatter()
        output_dir = Path(__file__).parent / "test_output"

        try:
            chunk_files = formatter.export_chunks(
                clusters, nodes, output_dir, format='markdown'
            )

            # Verify output
            self.assertEqual(len(chunk_files), 2, "Should create 2 chunks")
            for cluster_id, filepath in chunk_files.items():
                self.assertTrue(Path(filepath).exists(), f"Chunk file should exist: {filepath}")

        finally:
            # Cleanup
            import shutil
            if output_dir.exists():
                shutil.rmtree(output_dir)


if __name__ == '__main__':
    unittest.main()

import unittest
import os
from s2chunking.s2_chunker import StructuralSemanticChunker, LayoutDetector
from s2chunking.s2_chunker import format_clusters

class TestChunker(unittest.TestCase):

    def setUp(self):
        self.chunker = StructuralSemanticChunker(max_token_length=200)
        # Path to test images (multi-page document)
        self.test_image_paths = [
            os.path.join("test_data", "page1.png"),
            os.path.join("test_data", "page2.png")
        ]

        # Ensure the test images exist
        for path in self.test_image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Test image not found: {path}")

    def test_layout_detection(self):
        """Test layout detection for a single page."""
        # Initialize the layout detector
        layout_detector = LayoutDetector(image_path=self.test_image_paths[0])

        # Detect layout
        layout_info = layout_detector.detect_layout()

        # Check if layout_info is a list of dictionaries
        self.assertIsInstance(layout_info, list)
        for item in layout_info:
            self.assertIsInstance(item, dict)
            self.assertIn("bbox", item)
            self.assertIn("label", item)
            self.assertIn("confidence", item)

    def test_generate_nodes(self):
        """Test node generation from layout detection results."""
        # Initialize the layout detector
        layout_detector = LayoutDetector(image_path=self.test_image_paths[0])

        # Detect layout
        layout_info = layout_detector.detect_layout()

        # Generate nodes
        nodes = self.chunker._generate_nodes_from_layout(layout_info, page_number=1)

        # Check if nodes are generated correctly
        self.assertIsInstance(nodes, list)
        for node in nodes:
            self.assertIsInstance(node, dict)
            self.assertIn("global_id", node)
            self.assertIn("page", node)
            self.assertIn("local_id", node)
            self.assertIn("bbox", node)
            self.assertIn("text", node)

    def test_generate_edges(self):
        """Test edge generation from nodes."""
        # Initialize the layout detector
        layout_detector = LayoutDetector(image_path=self.test_image_paths[0])

        # Detect layout
        layout_info = layout_detector.detect_layout()

        # Generate nodes
        nodes = self.chunker._generate_nodes_from_layout(layout_info, page_number=1)

        # Generate edges
        edges = self.chunker._generate_edges(nodes)

        # Check if edges are generated correctly
        self.assertIsInstance(edges, list)
        for edge in edges:
            self.assertIsInstance(edge, tuple)
            self.assertEqual(len(edge), 2)

    def test_chunk_from_images(self):
        """Test chunking for a multi-page document."""
        # Perform chunking on multi-page document
        clusters = self.chunker.chunk_from_images(self.test_image_paths)

        # Check if clusters are generated correctly
        self.assertIsInstance(clusters, dict)
        for node_id, cluster_id in clusters.items():
            self.assertIsInstance(node_id, int)
            self.assertIsInstance(cluster_id, int)

    def test_format_clusters(self):
        """Test formatting of clusters."""
        # Initialize the layout detector
        layout_detector = LayoutDetector(image_path=self.test_image_paths[0])

        # Detect layout
        layout_info = layout_detector.detect_layout()

        # Generate nodes
        nodes = self.chunker._generate_nodes_from_layout(layout_info, page_number=1)

        # Perform chunking
        clusters = self.chunker.chunk_from_images(self.test_image_paths)

        # Format clusters
        formatted = format_clusters(clusters, nodes)
        self.assertIsInstance(formatted, str)

if __name__ == '__main__':
    unittest.main()
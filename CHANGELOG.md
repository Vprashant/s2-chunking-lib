# Changelog

## [1.0.0] - 2025-01-08

### Added
- Complete S2 Chunking implementation based on arXiv:2501.05485
- Layout detection using doclayout_yolo with layout_detect.pt model
- Reading order detection with column awareness
- Spatial + Semantic clustering using spectral clustering
- Complete chunk generation with markdown formatting
- OCR support via EasyOCR and Pytesseract
- Multi-page document support
- Visualization with annotated images and cropped regions
- Comprehensive API and documentation

### Features
- **StructuralSemanticChunker**: Main chunking class
- **LayoutDetector**: YOLO-based layout detection (10 categories)
- **BBoxOrderer**: Column-aware reading order detection
- **ChunkFormatter**: Complete chunk generation with metadata
- **format_clusters**: Human-readable clustering output

### Models
- Integrated layout_detect.pt YOLO model for document layout
- Support for custom YOLO models
- BERT-based semantic embeddings (sentence-transformers)

### Documentation
- Complete README with examples
- API reference documentation
- Basic usage example
- Advanced example with visualization

### Output
- Markdown formatted chunks with metadata
- Annotated images showing detected regions
- Cropped region images for verification
- Summary statistics and metadata

---

## Implementation Details

### Layout Categories Detected
1. Title (0)
2. Plain Text (1)
3. Abandon (2) - filtered
4. Figure (3)
5. Figure Caption (4)
6. Table (5)
7. Table Caption (6)
8. Table Footnote (7)
9. Isolated Formula (8)
10. Formula Caption (9)

### Clustering Algorithm
- Graph-based approach with weighted edges
- Spatial weights: Based on bbox centroid distances
- Semantic weights: Cosine similarity of BERT embeddings
- Spectral clustering for graph partitioning
- Token length constraint enforcement

### Reading Order
- Column detection using K-means or gap-based clustering
- Left-to-right, top-to-bottom ordering
- Respects multi-column layouts
- Maintains document flow

---

## Dependencies

### Core
- pydantic>=1.10.0
- networkx>=3.0
- numpy>=1.23.0
- scikit-learn>=1.1.0
- torch>=2.0.0
- transformers>=4.25.0
- opencv-python>=4.6.0
- doclayout-yolo

### Optional
- easyocr (for OCR)
- pytesseract (for OCR)

---

## Usage

### Basic
```python
from s2chunking import StructuralSemanticChunker, ChunkFormatter

chunker = StructuralSemanticChunker()
clusters, nodes = chunker.chunk_from_images(["page1.jpg"])

formatter = ChunkFormatter()
formatter.export_chunks(clusters, nodes, "chunks/")
```

### Command Line
```bash
python examples/basic_usage.py page1.jpg page2.jpg
```

---

## Performance

### Tested On
- CPU: Apple M-series
- GPU: CUDA-capable devices
- Image sizes: 1024x768 to 1024x1024
- Document types: Research papers, reports, multi-column layouts

### Benchmarks
- Layout detection: ~1-2 seconds per page (CPU)
- Edge generation: ~0.1 seconds for 24 nodes
- Clustering: ~0.05 seconds
- Total: ~2-3 seconds per page (without OCR)

---

## Known Limitations

1. Requires pre-trained YOLO model (layout_detect.pt)
2. OCR quality depends on image quality
3. Best for structured documents (papers, reports)
4. May struggle with very complex layouts

---

## Future Improvements

- [ ] Support for PDF input directly
- [ ] Enhanced table structure recognition
- [ ] Formula recognition improvements
- [ ] Better handling of irregular layouts
- [ ] Parallel processing for multi-page documents
- [ ] Cloud deployment options

---

## Credits

- Paper: "S2 Chunking: A Hybrid Framework for Document Segmentation Through Integrated Spatial and Semantic Analysis"
- Author: Prashant Verma
- arXiv: 2501.05485
- Year: 2025

---

## License

MIT License - see LICENSE file for details

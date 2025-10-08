# S2 Chunking: Structural-Semantic Document Chunking

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2501.05485-b31b1b)](https://arxiv.org/abs/2501.05485)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**S2 Chunking** is a Python library for intelligent document chunking that combines **structural layout analysis** with **semantic understanding**. It's designed for processing complex documents (research papers, reports, multi-column layouts) to create meaningful chunks for RAG and LLM applications.

## Key Features

- **Layout-Aware**: Detects document structure using YOLO (titles, text, figures, tables, captions)
- **Reading Order**: Automatically determines correct reading flow (columns, top-to-bottom)
- **Semantic Clustering**: Groups related content using BERT embeddings
- **Spatial Analysis**: Considers physical proximity of document elements
- **Complete Chunks**: Generates markdown-formatted chunks with full text content
- **Multi-Page Support**: Handles documents with multiple pages seamlessly

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/s2-chunking-lib.git
cd s2-chunking-lib

# Install dependencies
pip install -r requirements.txt

# Install doclayout-yolo for layout detection
pip install doclayout-yolo

# Optional: Install OCR support
pip install easyocr pytesseract
```

## Quick Start

```python
from s2chunking import StructuralSemanticChunker, ChunkFormatter

# Initialize chunker
chunker = StructuralSemanticChunker(max_token_length=512)

# Process document images
clusters, nodes = chunker.chunk_from_images(["page1.jpg", "page2.jpg"])

# Generate complete markdown chunks
formatter = ChunkFormatter()
chunk_files = formatter.export_chunks(clusters, nodes, "output/chunks")

print(f"Generated {len(chunk_files)} chunks")
```

## How It Works

### 1. Layout Detection
Uses a fine-tuned YOLO model (`layout_detect.pt`) to detect document regions:
- Titles and headers
- Plain text paragraphs
- Figures and figure captions
- Tables and table captions
- Formulas

### 2. Reading Order Detection
Implements column-aware ordering:
- Detects multi-column layouts
- Orders regions left-to-right, top-to-bottom
- Respects document flow

### 3. Graph-Based Clustering
Creates a weighted graph where:
- **Nodes** = detected regions
- **Edges** = spatial + semantic relationships
- **Weights** = combined proximity and similarity scores

Uses spectral clustering to group related regions.

### 4. Chunk Generation
Produces complete, self-contained markdown files:
```markdown
<!--
Cluster: 0
Nodes: 9
Pages: [1, 2]
Reading Order: 1-9
Categories: {'title': 3, 'plain text': 4, 'table_caption': 2}
-->

# Chunk 0

## Introduction

[Full text content here...]

## Formatting Guidelines

[Full text content here...]
```

## Usage Examples

### Basic Usage

```python
from s2chunking import StructuralSemanticChunker

chunker = StructuralSemanticChunker(max_token_length=512)
clusters, nodes = chunker.chunk_from_images(["doc.jpg"])
```

### With OCR Text Extraction

```python
# Extract actual text from images using EasyOCR
clusters, nodes = chunker.chunk_from_images(
    ["page1.jpg", "page2.jpg"],
    extract_text=True  # Enable OCR
)
```

### Custom Configuration

```python
chunker = StructuralSemanticChunker(
    max_token_length=300,  # Smaller chunks
)

# Use custom model path
from s2chunking import LayoutDetector
detector = LayoutDetector(
    image_path="page.jpg",
    model_path="path/to/custom_model.pt"
)
```

### Generate Chunks for RAG

```python
from s2chunking import ChunkFormatter

formatter = ChunkFormatter()

# Export as markdown
chunk_files = formatter.export_chunks(
    clusters, nodes,
    output_dir="chunks",
    format='markdown'
)

# Now chunks are ready for embedding and indexing
for cluster_id, filepath in chunk_files.items():
    # Load and embed each chunk
    with open(filepath, 'r') as f:
        chunk_text = f.read()
        # Your embedding logic here
```

## Command Line Usage

```bash
# Basic usage
python examples/example.py page1.jpg page2.jpg

# Full example with visualization
python examples/example.py
```

## Project Structure

```
s2-chunking-lib/
├── src/s2chunking/
│   ├── __init__.py
│   ├── s2_chunker.py          # Main chunking logic
│   ├── layout_deter.py        # YOLO layout detection
│   ├── bbox_order.py          # Reading order detection
│   └── chunk_formatter.py     # Chunk generation
├── models/
│   └── layout_detect.pt        # Pre-trained YOLO model
├── examples/
│   ├── example.py         # Simple example
│   ├── example.py        # Complete example with visualization
│   └── image1.jpg, image2.jpg # Sample images
├── requirements.txt
└── README.md
```

## Model

The library uses `layout_detect.pt`, a YOLO-based model trained for document layout detection. The model detects:
- Title (0)
- Plain Text (1)
- Abandon (2) - filtered out
- Figure (3)
- Figure Caption (4)
- Table (5)
- Table Caption (6)
- Table Footnote (7)
- Isolated Formula (8)
- Formula Caption (9)

To use your own model, place it in the `models/` folder or specify the path:
```python
detector = LayoutDetector(image_path="...", model_path="your_model.pt")
```

## Requirements

```
pydantic>=1.10.0
networkx>=3.0
numpy>=1.23.0
scikit-learn>=1.1.0
torch>=2.0.0
transformers>=4.25.0
opencv-python>=4.6.0
doclayout-yolo
```

Optional for OCR:
```
easyocr
pytesseract
```

## Output Structure

```
output/
├── chunks/
│   ├── chunk_000.md    # Markdown formatted chunks
│   ├── chunk_001.md
│   └── ...
├── image1/
│   ├── image1_annotated.jpg  # Visualizations
│   └── cropped regions...
└── chunking_results.txt      # Summary metadata
```

## Citation

If you use this library in your research, please cite:

```bibtex
@article{s2chunking2025,
  title={S2 Chunking: A Hybrid Framework for Document Segmentation Through Integrated Spatial and Semantic Analysis},
  author={Prashant Verma},
  journal={arXiv preprint arXiv:2501.05485},
  year={2025}
}
```

## Paper Reference

This implementation is based on the paper:
**"S2 Chunking: A Hybrid Framework for Document Segmentation Through Integrated Spatial and Semantic Analysis"**

📄 [Read on arXiv](https://arxiv.org/abs/2501.05485)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Paper: [arXiv:2501.05485](https://arxiv.org/abs/2501.05485)
- YOLO: [Ultralytics](https://github.com/ultralytics/ultralytics)
- Transformers: [Hugging Face](https://huggingface.co/)
- Layout Detection: [doclayout-yolo](https://github.com/opendatalab/doclayout-yolo)

## Contact

For questions or feedback:
- Email: prashant27050@gmail.com
- Issues: [GitHub Issues](https://github.com/yourusername/s2-chunking-lib/issues)

---

Made with ❤️ for better document understanding

# S2 Chunking API Reference

## Quick Reference

```python
from s2chunking import (
    StructuralSemanticChunker,  # Main chunker
    ChunkFormatter,              # Format output chunks
    LayoutDetector,              # Layout detection
    BBoxOrderer                  # Reading order
)
```

## StructuralSemanticChunker

Main class for S2 chunking.

### Initialization

```python
chunker = StructuralSemanticChunker(max_token_length=512)
```

### Methods

#### `chunk_from_images(image_paths, extract_text=True)`

Process document images and create chunks.

**Returns:** `(clusters, nodes)` tuple

**Example:**
```python
clusters, nodes = chunker.chunk_from_images(["page1.jpg", "page2.jpg"])
```

---

## ChunkFormatter

Format chunks for output.

### Methods

#### `export_chunks(clusters, nodes, output_dir, format='markdown')`

Export chunks to files.

**Returns:** Dict mapping cluster_id to filepath

**Example:**
```python
formatter = ChunkFormatter()
files = formatter.export_chunks(clusters, nodes, "chunks/")
```

---

## LayoutDetector

Detect document layout regions.

### Initialization

```python
detector = LayoutDetector(
    image_path="page.jpg",
    model_path="models/layout_detect.pt"  # Optional
)
```

### Methods

#### `detect_layout(extract_text=True)`

Detect layout regions.

**Returns:** List of region dictionaries

---

## Node Structure

```python
{
    "global_id": 1,
    "page": 1,
    "bbox": [x1, y1, x2, y2],
    "label": "title",
    "text": "Document Title",
    "confidence": 0.95,
    "reading_order": 1
}
```

---

For complete documentation, see [README.md](../README.md)


## S2 Chunking: A Hybrid Framework for Document Segmentation Through Integrated Spatial and Semantic Analysis
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![arXiv Paper](https://img.shields.io/badge/arXiv-2501.05485-b31b1b)
![YOLO](https://img.shields.io/badge/YOLO-v8-orange)
![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

**S2 Chunking Lib** is a Python library designed for **structural-semantic chunking** of documents. It leverages advanced techniques to break down documents into meaningful chunks based on both **spatial layout** and **semantic content**. This library is particularly useful for processing multi-page documents, such as research papers, reports, and books, where understanding the structure and semantics is crucial.

Inspired by the research in [arXiv:2501.05485](https://arxiv.org/pdf/2501.05485), this library implements a hybrid approach that combines **layout detection** (e.g., bounding boxes, regions) and **semantic analysis** (e.g., text embeddings) to create coherent and contextually meaningful chunks.

---

## Features

- **Layout Detection**: Automatically detects layout elements (e.g., text blocks, figures, tables) using a pre-trained YOLO model.
- **Semantic Analysis**: Uses transformer-based embeddings to understand the semantic relationships between text elements.
- **Multi-Page Support**: Handles multi-page documents seamlessly, preserving the context across pages.
- **Customizable Chunking**: Allows users to define maximum token lengths and thresholds for spatial and semantic relationships.
- **Graph-Based Clustering**: Uses graph theory to cluster related elements into meaningful chunks.
- **Easy Integration**: Provides a simple API for chunking documents and formatting the results.

---

## Installation

You can install the library using `pip`. Ensure you have Python 3.7 or higher installed.

```bash
pip install s2chunking
```

---

## Usage

### Basic Example

Here’s how to use the `StructuralSemanticChunker` to process a multi-page document:

```python
from s2chunking import StructuralSemanticChunker

# Initialize the chunker
chunker = StructuralSemanticChunker(max_token_length=200)

# List of image paths
image_paths = ["path/to/page1.png", "path/to/page2.png"]

# Perform chunking on the multi-page document
clusters = chunker.chunk_from_images(image_paths)

# Print the formatted clusters
formatted_clusters = chunker.format_clusters(clusters)
print(formatted_clusters)
```

### Output Example

The output will be a formatted string showing the clusters of related elements:

```
Cluster 0:
   - (page=1 local=1) Title of the document...
   - (page=1 local=2) Introduction paragraph...
Cluster 1:
   - (page=2 local=1) Figure 1: Description...
   - (page=2 local=2) Table 1: Summary of results...
```

---

## Advanced Usage

### Customizing Layout Detection

You can customize the layout detection by specifying a different YOLO model or thresholds:

```python
from s2chunking import LayoutDetector

# Initialize the layout detector with a custom model
layout_detector = LayoutDetector(
    model_name="custom_model.pt",
    repo_id="your-repo-id",
    weights_folder="weight",
    device="cuda"  # Use GPU if available
)

# Detect layout for a single page
layout_info = layout_detector.detect_layout("path/to/page1.png")
print(layout_info)
```

### Customizing Chunking Parameters

You can adjust the chunking parameters, such as the maximum token length and semantic thresholds:

```python
# Initialize the chunker with custom parameters
chunker = StructuralSemanticChunker(
    max_token_length=300,  # Maximum tokens per chunk
    spatial_threshold=0.5,  # Spatial relationship threshold
    semantic_threshold=0.8  # Semantic relationship threshold
)
```

---

## Documentation

For detailed API documentation, refer to the [docs/api.md](docs/api.md) file.

---

## Contributing

We welcome contributions! If you’d like to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this library in your research, please consider citing the following paper:

```bibtex
@article{arxiv2501.05485,
  title={S2 Chunking: A Hybrid Framework for Document Segmentation Through Integrated Spatial and Semantic Analysis},
  author={Prashant Verma},
  journal={arXiv preprint arXiv:2501.05485},
  year={2025}
}
```

---

## Acknowledgments

- This library is inspired by the research presented in [arXiv:2501.05485](https://arxiv.org/pdf/2501.05485).
- Thanks to the open-source community for providing tools like [YOLO](https://github.com/ultralytics/ultralytics) and [Hugging Face Transformers](https://huggingface.co/).

---

## Contact

For questions or feedback, please contact [Prashant Verma](mailto:prashant27050@gmail.com).


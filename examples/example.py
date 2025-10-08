#!/usr/bin/env python3
"""
S2 Chunking - Complete Example

Demonstrates S2 Chunking with optional visualization.

Usage:
    # Basic usage (just chunks)
    python example.py image1.jpg [image2.jpg ...]

    # With visualization (cropped regions + annotated images)
    python example.py --visualize image1.jpg [image2.jpg ...]
"""
import sys
import argparse
from pathlib import Path
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from s2chunking import StructuralSemanticChunker, ChunkFormatter


def save_visualizations(image_paths, nodes, clusters, output_dir):
    """Save cropped regions and annotated images for visualization."""
    print("\nSaving visualizations...")
    output_dir = Path(output_dir)

    for page_number, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(image_path)
        if image is None:
            continue

        image_name = Path(image_path).stem
        page_dir = output_dir / image_name
        page_dir.mkdir(parents=True, exist_ok=True)

        # Create annotated image
        annotated = image.copy()
        page_nodes = [n for n in nodes if n['page'] == page_number]

        for node in page_nodes:
            x1, y1, x2, y2 = map(int, node['bbox'])
            cluster_id = clusters.get(node['global_id'], -1)
            label = node.get('label', 'unknown')

            # Draw bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"C{cluster_id}:{label}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save cropped region
            cropped = image[y1:y2, x1:x2]
            crop_file = page_dir / f"region_{node['local_id']}_cluster{cluster_id}_{label}.jpg"
            cv2.imwrite(str(crop_file), cropped)

        # Save annotated image
        annotated_file = page_dir / f"{image_name}_annotated.jpg"
        cv2.imwrite(str(annotated_file), annotated)
        print(f"  ✓ Saved {len(page_nodes)} regions for {image_name}")


def main():
    parser = argparse.ArgumentParser(description='S2 Chunking Example')
    parser.add_argument('images', nargs='+', help='Image file paths')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Save cropped regions and annotated images')
    parser.add_argument('--output', '-o', default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Max tokens per chunk (default: 512)')

    args = parser.parse_args()

    # Verify images exist
    for img_path in args.images:
        if not Path(img_path).exists():
            print(f"Error: Image not found: {img_path}")
            sys.exit(1)

    print(f"S2 Chunking - Processing {len(args.images)} image(s)")
    print("=" * 70)

    # Initialize chunker
    print("\nInitializing chunker...")
    chunker = StructuralSemanticChunker(max_token_length=args.max_tokens)

    # Process images
    print("Processing images...")
    clusters, nodes = chunker.chunk_from_images(args.images, extract_text=True)

    if not clusters:
        print("✗ No clusters created")
        return 1

    num_clusters = len(set(clusters.values()))
    num_nodes = len(clusters)
    print(f"\n✓ Created {num_clusters} cluster(s) from {num_nodes} node(s)")

    # Generate chunks
    print("\nGenerating chunks...")
    output_dir = Path(args.output)
    chunks_dir = output_dir / "chunks"

    formatter = ChunkFormatter()
    chunk_files = formatter.export_chunks(clusters, nodes, chunks_dir, format='markdown')

    print(f"\n✓ Generated {len(chunk_files)} chunk file(s) in {chunks_dir}")

    # Save visualizations if requested
    if args.visualize:
        viz_dir = output_dir / "visualizations"
        save_visualizations(args.images, nodes, clusters, viz_dir)
        print(f"\n✓ Saved visualizations to {viz_dir}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Chunks: {len(chunk_files)}")
    for cluster_id, filepath in sorted(chunk_files.items()):
        size = Path(filepath).stat().st_size
        print(f"  - Chunk {cluster_id}: {Path(filepath).name} ({size} bytes)")

    print(f"\n✓ Complete! Output saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

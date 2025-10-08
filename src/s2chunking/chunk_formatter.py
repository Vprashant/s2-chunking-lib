"""
Chunk formatter for generating complete text chunks from clustered nodes.
Produces markdown-formatted chunks ready for RAG/LLM applications.
"""
from typing import List, Dict
from pathlib import Path


class ChunkFormatter:
    """Format clustered nodes into complete text chunks."""

    def __init__(self, max_chars_preview: int = 500):
        self.max_chars_preview = max_chars_preview

    def format_chunk_text(self, nodes: List[Dict], cluster_id: int) -> str:
        """
        Generate complete formatted text for a chunk.

        Args:
            nodes: List of nodes in reading order
            cluster_id: ID of the cluster

        Returns:
            Formatted markdown text for the chunk
        """
        lines = []

        # Add chunk header
        lines.append(f"# Chunk {cluster_id}\n")

        # Sort by reading order
        nodes_sorted = sorted(nodes, key=lambda n: n.get('reading_order', 999))

        # Group consecutive nodes of same category
        current_category = None
        category_content = []

        for node in nodes_sorted:
            category = node.get('label', 'unknown')
            text = node.get('text', '')

            # Handle different categories with proper formatting
            if category == 'title':
                if category_content:
                    lines.append('\n'.join(category_content))
                    category_content = []
                lines.append(f"\n## {text}\n")
                current_category = None

            elif category == 'plain text':
                # Accumulate plain text
                if current_category == 'plain text':
                    category_content.append(text)
                else:
                    if category_content:
                        lines.append('\n'.join(category_content))
                        category_content = []
                    category_content.append(text)
                    current_category = 'plain text'

            elif category == 'figure':
                if category_content:
                    lines.append('\n'.join(category_content))
                    category_content = []
                lines.append(f"\n[Figure: {text[:100]}...]\n")
                current_category = None

            elif category == 'figure_caption':
                if category_content:
                    lines.append('\n'.join(category_content))
                    category_content = []
                lines.append(f"\n*Figure Caption: {text}*\n")
                current_category = None

            elif category == 'table':
                if category_content:
                    lines.append('\n'.join(category_content))
                    category_content = []
                lines.append(f"\n```\nTable:\n{text}\n```\n")
                current_category = None

            elif category == 'table_caption':
                if category_content:
                    lines.append('\n'.join(category_content))
                    category_content = []
                lines.append(f"\n**Table Caption: {text}**\n")
                current_category = None

            elif category == 'table_footnote':
                if category_content:
                    lines.append('\n'.join(category_content))
                    category_content = []
                lines.append(f"\n_Table Note: {text}_\n")
                current_category = None

            elif category == 'isolate_formula':
                if category_content:
                    lines.append('\n'.join(category_content))
                    category_content = []
                lines.append(f"\n$$\n{text}\n$$\n")
                current_category = None

            elif category == 'formula_caption':
                if category_content:
                    lines.append('\n'.join(category_content))
                    category_content = []
                lines.append(f"\n*Formula: {text}*\n")
                current_category = None

            else:
                # Unknown category
                if category_content:
                    lines.append('\n'.join(category_content))
                    category_content = []
                lines.append(f"\n{text}\n")
                current_category = None

        # Add remaining content
        if category_content:
            lines.append('\n'.join(category_content))

        return '\n'.join(lines)

    def generate_chunk_summary(self, nodes: List[Dict], cluster_id: int) -> Dict:
        """
        Generate metadata summary for a chunk.

        Args:
            nodes: List of nodes in the chunk
            cluster_id: Cluster ID

        Returns:
            Dictionary with chunk metadata
        """
        nodes_sorted = sorted(nodes, key=lambda n: n.get('reading_order', 999))

        # Count categories
        categories = {}
        for node in nodes_sorted:
            cat = node.get('label', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        # Get text statistics
        total_chars = sum(len(node.get('text', '')) for node in nodes_sorted)

        # Get page span
        pages = sorted(set(node.get('page', 0) for node in nodes_sorted))

        return {
            'cluster_id': cluster_id,
            'num_nodes': len(nodes_sorted),
            'categories': categories,
            'total_chars': total_chars,
            'pages': pages,
            'first_node_order': nodes_sorted[0].get('reading_order', 0) if nodes_sorted else 0,
            'last_node_order': nodes_sorted[-1].get('reading_order', 0) if nodes_sorted else 0
        }

    def export_chunks(self, clusters: Dict[int, int], nodes: List[Dict],
                     output_dir: Path, format: str = 'markdown') -> Dict[int, str]:
        """
        Export all chunks to files.

        Args:
            clusters: Dict mapping node_id to cluster_id
            nodes: List of all nodes with metadata
            output_dir: Directory to save chunk files
            format: Output format ('markdown', 'text', or 'json')

        Returns:
            Dict mapping cluster_id to output filepath
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group nodes by cluster
        cluster_groups = {}
        for node_id, cluster_id in clusters.items():
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            node = next((n for n in nodes if n['global_id'] == node_id), None)
            if node:
                cluster_groups[cluster_id].append(node)

        chunk_files = {}

        for cluster_id in sorted(cluster_groups.keys()):
            nodes_in_cluster = cluster_groups[cluster_id]

            # Generate chunk text
            chunk_text = self.format_chunk_text(nodes_in_cluster, cluster_id)

            # Generate metadata
            metadata = self.generate_chunk_summary(nodes_in_cluster, cluster_id)

            # Save to file
            if format == 'markdown':
                filename = f"chunk_{cluster_id:03d}.md"
                filepath = output_dir / filename

                # Add metadata header
                header = f"<!--\n"
                header += f"Cluster: {cluster_id}\n"
                header += f"Nodes: {metadata['num_nodes']}\n"
                header += f"Pages: {metadata['pages']}\n"
                header += f"Reading Order: {metadata['first_node_order']}-{metadata['last_node_order']}\n"
                header += f"Categories: {metadata['categories']}\n"
                header += f"-->\n\n"

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(header + chunk_text)

            elif format == 'text':
                filename = f"chunk_{cluster_id:03d}.txt"
                filepath = output_dir / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(chunk_text)

            chunk_files[cluster_id] = str(filepath)
            print(f"  ✓ Saved chunk {cluster_id} to {filename} ({metadata['total_chars']} chars)")

        return chunk_files

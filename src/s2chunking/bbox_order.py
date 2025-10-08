"""
BBox ordering utilities for proper reading order detection.
Based on column detection and reading order logic.
"""
import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans


class BBoxOrderer:
    """Order bounding boxes based on document layout (columns, reading order)."""

    def __init__(self):
        self.x_gap_threshold = 60  # Pixels between columns

    def compute_x_center(self, bbox: List[float]) -> float:
        """Calculate center X coordinate of bbox."""
        return (bbox[0] + bbox[2]) / 2

    def compute_y_center(self, bbox: List[float]) -> float:
        """Calculate center Y coordinate of bbox."""
        return (bbox[1] + bbox[3]) / 2

    def cluster_columns_gap(self, bboxes: List[List[float]]) -> List[List[int]]:
        """
        Cluster bboxes into columns based on x-center gaps.
        Returns list of column indices.
        """
        if len(bboxes) == 0:
            return []

        # Sort by x-center
        indexed_bboxes = [(i, bbox) for i, bbox in enumerate(bboxes)]
        sorted_indexed = sorted(indexed_bboxes, key=lambda x: self.compute_x_center(x[1]))

        columns = []
        current_column = [sorted_indexed[0][0]]

        for i in range(1, len(sorted_indexed)):
            idx, bbox = sorted_indexed[i]
            prev_idx, prev_bbox = sorted_indexed[i-1]

            prev_center = self.compute_x_center(prev_bbox)
            curr_center = self.compute_x_center(bbox)

            if (curr_center - prev_center) > self.x_gap_threshold:
                # Start new column
                columns.append(current_column)
                current_column = [idx]
            else:
                current_column.append(idx)

        columns.append(current_column)
        return columns

    def order_nodes_by_layout(self, nodes: List[Dict]) -> List[Dict]:
        """
        Order nodes based on document layout (columns + reading order).

        Args:
            nodes: List of node dicts with 'bbox', 'global_id', etc.

        Returns:
            Ordered list of nodes in reading order
        """
        if len(nodes) == 0:
            return []

        # Extract bboxes
        bboxes = [node['bbox'] for node in nodes]

        # Detect columns
        column_indices = self.cluster_columns_gap(bboxes)

        # Sort columns left-to-right by their average x-position
        column_avg_x = []
        for col_idx_list in column_indices:
            avg_x = np.mean([self.compute_x_center(bboxes[i]) for i in col_idx_list])
            column_avg_x.append((col_idx_list, avg_x))

        column_avg_x.sort(key=lambda x: x[1])

        # Within each column, sort top-to-bottom
        ordered_nodes = []
        for col_idx_list, _ in column_avg_x:
            # Get nodes in this column
            column_nodes = [nodes[i] for i in col_idx_list]

            # Sort by y-position (top to bottom)
            column_nodes_sorted = sorted(
                column_nodes,
                key=lambda n: (n['bbox'][1] + n['bbox'][3]) / 2  # y-center
            )

            ordered_nodes.extend(column_nodes_sorted)

        return ordered_nodes

    def assign_reading_order(self, nodes: List[Dict]) -> List[Dict]:
        """
        Assign reading order IDs to nodes based on layout.
        Modifies nodes in-place to add 'reading_order' field.

        Args:
            nodes: List of node dicts

        Returns:
            Nodes with 'reading_order' field added
        """
        ordered_nodes = self.order_nodes_by_layout(nodes)

        # Assign reading order
        for idx, node in enumerate(ordered_nodes, start=1):
            node['reading_order'] = idx

        return ordered_nodes

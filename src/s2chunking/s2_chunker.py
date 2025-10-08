from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import networkx as nx
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
import torch
from .layout_deter import LayoutDetector
from .bbox_order import BBoxOrderer

class ChunkerOutput(BaseModel):
    chunks: List[float]

    class Config:
        arbitrary_types_allowed = True

class GraphProcessing(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    def create_graph(self, nodes: List[int], edges: List[tuple]) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    def add_weights_to_graph(self, graph: nx.Graph, weights: np.ndarray) -> nx.Graph:
        for i, (u, v) in enumerate(graph.edges()):
            graph[u][v]['weight'] = weights[i]
        return graph

class EdgeWeightsCalculations(BaseModel):
    relation_threshold: float = Field(default=0.4, description="Threshold for defining relations between nodes.")
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModel] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModel.from_pretrained("bert-base-uncased")
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Transformers module not found. Please install it using `pip install transformers`.")
        except Exception as e:
            raise RuntimeError(f"Error initializing tokenizer or model: {e}")

    def spatial_weights_calculation(self, nodes: List[Dict]) -> np.ndarray:
        try:
            print("Calculating spatial weights...")
            num_nodes = len(nodes)
            spatial_weights = np.zeros((num_nodes, num_nodes))
            for i in range(num_nodes):
                for j in range(num_nodes):
                    bbox_i = nodes[i]['bbox']
                    bbox_j = nodes[j]['bbox']
                    centroid_i = np.array([(bbox_i[0] + bbox_i[2]) / 2, (bbox_i[1] + bbox_i[3]) / 2])
                    centroid_j = np.array([(bbox_j[0] + bbox_j[2]) / 2, (bbox_j[1] + bbox_j[3]) / 2])
                    distance = np.linalg.norm(centroid_i - centroid_j)
                    spatial_weights[i, j] = 1 / (1 + distance)
            return spatial_weights
        except Exception as e:
            print(f"Error in spatial_weights_calculation: {e}")
            return None

    def semantic_weights_calculation(self, nodes: List[Dict]) -> np.ndarray:
        try:
            if not self.tokenizer or not self.model:
                raise ValueError("Tokenizer or model not initialized.")

            print("Calculating semantic weights...")
            texts = [node['text'] for node in nodes if node.get('text', '').strip()]
            
            if not texts:
                print("No valid texts for semantic weights calculation.")
                return None

            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            semantic_weights = cosine_similarity(embeddings)
            return semantic_weights
        except Exception as e:
            print(f"Error in semantic_weights_calculation: {e}")
            return None

    def combined_weights(self, nodes: List[Dict]) -> np.ndarray:
        try:
            spatial_weights = self.spatial_weights_calculation(nodes)
            semantic_weights = self.semantic_weights_calculation(nodes)
            if spatial_weights is None or semantic_weights is None:
                raise ValueError("Spatial or semantic weight calculation failed.")
            combined_weights = (spatial_weights + semantic_weights) / 2  
            print("Combining weights...")
            return combined_weights
        except Exception as e:
            print(f"Error in combined_weights: {e}")
            return None

class GraphClustering(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    def cluster_graph(self, graph: nx.Graph, weights: np.ndarray, n_clusters: int = 3) -> Dict[int, int]:
        try:
            print("Clustering graph...")
            clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
            labels = clustering.fit_predict(weights)
            return {node: label for node, label in zip(graph.nodes(), labels)}
        except Exception as e:
            print(f"Error in clustering graph: {e}")
            return {}

class StructuralSemanticChunker(BaseModel):
    edge_weights: Optional[EdgeWeightsCalculations] = Field(default_factory=EdgeWeightsCalculations)
    graph_processing: Optional[GraphProcessing] = Field(default_factory=GraphProcessing)
    graph_clustering: Optional[GraphClustering] = Field(default_factory=GraphClustering)
    max_token_length: int = Field(default=512, description="Maximum token length for each chunk.")
    layout_detector: Optional[LayoutDetector] = None
    bbox_orderer: Optional[BBoxOrderer] = Field(default_factory=BBoxOrderer)

    class Config:
        arbitrary_types_allowed = True

    def chunk_from_images(self, image_paths: List[str], extract_text: bool = True) -> tuple[Dict[int, int], List[Dict]]:
        """
        Perform chunking directly from a list of images (multi-page document).
        Args:
            image_paths (List[str]): List of paths to images (one per page).
            extract_text (bool): If True, extract text using OCR; otherwise use layout labels
        Returns:
            Tuple of (clusters dict, nodes list with reading_order)
        """
        try:
            all_nodes = []
            all_edges = []
            global_id_counter = 1  # Track global ID across all pages

            for page_number, image_path in enumerate(image_paths, start=1):
                self.layout_detector = LayoutDetector(image_path=image_path)
                layout_info = self.layout_detector.detect_layout(extract_text=extract_text)

                nodes = self._generate_nodes_from_layout(layout_info, page_number, global_id_counter)
                all_nodes.extend(nodes)
                global_id_counter += len(nodes)  # Update counter

            # Apply reading order to nodes
            print("Applying reading order to detected regions...")
            all_nodes = self.bbox_orderer.assign_reading_order(all_nodes)

            # Generate edges for ALL nodes (not just within same page)
            all_edges = self._generate_edges(all_nodes)

            clusters = self.chunk(all_nodes, all_edges)
            return clusters, all_nodes  # Return both clusters and enriched nodes
        except Exception as e:
            print(f"Error in chunk_from_images: {e}")
            import traceback
            traceback.print_exc()
            return {}, []

    def _generate_nodes_from_layout(self, layout_info: List[Dict], page_number: int, global_id_start: int) -> List[Dict]:
        """
        Generate nodes from layout detection results for a specific page.
        Args:
            layout_info (List[Dict]): Layout detection results for a single page.
            page_number (int): The page number associated with the layout info.
            global_id_start (int): Starting global ID for nodes on this page.

        Returns:
            List[Dict]: List of nodes with global_id, page, local_id, bbox, and text.
        """
        nodes = []
        for i, item in enumerate(layout_info):
            nodes.append({
                "global_id": global_id_start + i,
                "page": page_number,
                "local_id": i + 1,
                "bbox": item["bbox"],
                "label": item["label"],  # Keep layout label
                "text": item.get("text", item["label"]),  # Use extracted text or fallback to label
                "confidence": item.get("confidence", 1.0)  # Add confidence score
            })
        return nodes

    def _generate_edges(self, nodes: List[Dict]) -> List[tuple]:
        """
        Generate edges based on spatial and semantic relationships.
        Creates edges between nodes that have sufficient spatial proximity
        OR semantic similarity (not requiring both).
        """
        edges = []
        spatial_threshold = 0.3  # Lowered from implicit high value
        semantic_threshold = 0.5  # Lowered from 0.7

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                bbox_i = nodes[i]['bbox']
                bbox_j = nodes[j]['bbox']

                # Calculate spatial proximity
                centroid_i = np.array([(bbox_i[0] + bbox_i[2]) / 2, (bbox_i[1] + bbox_i[3]) / 2])
                centroid_j = np.array([(bbox_j[0] + bbox_j[2]) / 2, (bbox_j[1] + bbox_j[3]) / 2])
                distance = np.linalg.norm(centroid_i - centroid_j)
                spatial_weight = 1 / (1 + distance)

                # Calculate semantic similarity
                text_i = nodes[i]['text']
                text_j = nodes[j]['text']

                semantic_weight = 0
                if text_i.strip() and text_j.strip():
                    try:
                        with torch.no_grad():
                            inputs = self.edge_weights.tokenizer(
                                [text_i, text_j],
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=512
                            )
                            outputs = self.edge_weights.model(**inputs)
                            embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
                            semantic_weight = cosine_similarity(embeddings)[0, 1]
                    except Exception as e:
                        print(f"Warning: Failed to compute semantic similarity: {e}")

                # Create edge if either spatial OR semantic relationship is strong
                # OR if combined weight is sufficient
                combined_weight = (spatial_weight + semantic_weight) / 2

                if (spatial_weight >= spatial_threshold or
                    semantic_weight >= semantic_threshold or
                    combined_weight >= 0.3):
                    edges.append((nodes[i]['global_id'], nodes[j]['global_id']))

        print(f"Generated {len(edges)} edges from {len(nodes)} nodes")
        return edges

    def chunk(self, nodes: List[Dict], edges: List[tuple]) -> Dict[int, int]:
        try:
            print("Starting chunking process...")
            node_ids = [node['global_id'] for node in nodes]
            graph = self.graph_processing.create_graph(node_ids, edges)
            weight_matrix = self.edge_weights.combined_weights(nodes)
            if weight_matrix is None:
                print("Weight calculation failed. Returning empty clusters.")
                return {}

            # Extract edge weights from the weight matrix for the edges in the graph
            edge_weights = []
            node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
            for u, v in graph.edges():
                i, j = node_id_to_index[u], node_id_to_index[v]
                edge_weights.append(weight_matrix[i, j])

            edge_weights = np.array(edge_weights)
            weighted_graph = self.graph_processing.add_weights_to_graph(graph, edge_weights)

            n_clusters = self._calculate_n_clusters(nodes, weight_matrix)
            clusters = self.graph_clustering.cluster_graph(weighted_graph, weight_matrix, n_clusters)
            clusters = self._split_clusters_by_token_length(clusters, nodes)

            print("Chunking completed.")
            return clusters
        except Exception as e:
            print(f"Error in chunking: {e}")
            return {}

    def _calculate_n_clusters(self, nodes: List[Dict], weights: np.ndarray) -> int:
        try:
            total_token_length = 0
            for node in nodes:
                text = node['text']
                total_token_length += len(self.edge_weights.tokenizer.tokenize(text))

            # Use mean of combined weights as a general relationship strength indicator
            avg_weightage = np.mean(weights)

            n_clusters = max(
                1,
                int(
                    (total_token_length / self.max_token_length)
                    * (1 - avg_weightage)
                )
            )
            return n_clusters
        except Exception as e:
            print(f"Error in _calculate_n_clusters: {e}")
            return 1

    def _split_clusters_by_token_length(self, clusters: Dict[int, int], nodes: List[Dict]) -> Dict[int, int]:
        updated_clusters = {}
        cluster_id_counter = 0

        for cluster_id, node_ids in self._group_nodes_by_cluster(clusters).items():
            current_chunk = []
            current_token_length = 0

            for node_id in node_ids:
                node = next((n for n in nodes if n['global_id'] == node_id), None)
                if not node:
                    continue
                node_text = node['text']
                node_token_length = len(self.edge_weights.tokenizer.tokenize(node_text))

                if current_token_length + node_token_length > self.max_token_length:
                    for node_in_chunk in current_chunk:
                        updated_clusters[node_in_chunk] = cluster_id_counter
                    cluster_id_counter += 1
                    current_chunk = []
                    current_token_length = 0

                current_chunk.append(node_id)
                current_token_length += node_token_length

            for node_in_chunk in current_chunk:
                updated_clusters[node_in_chunk] = cluster_id_counter
            cluster_id_counter += 1

        return updated_clusters

    def _group_nodes_by_cluster(self, clusters: Dict[int, int]) -> Dict[int, List[int]]:
        cluster_groups = {}
        for node, cluster_id in clusters.items():
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(node)
        return cluster_groups

def format_clusters(clusters: Dict[int, int], nodes: List[Dict]) -> str:
    cluster_groups = {}
    for node_id, cluster_id in clusters.items():
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        node = next((n for n in nodes if n['global_id'] == node_id), None)
        if node:
            reading_order = node.get('reading_order', '?')
            text_preview = node['text'][:50] + "..." if len(node['text']) > 50 else node['text']
            snippet = f"(order={reading_order} page={node['page']} local={node['local_id']}) {text_preview}"
            cluster_groups[cluster_id].append((reading_order, snippet))

    lines = []
    for cluster_id in sorted(cluster_groups.keys()):
        snippets = cluster_groups[cluster_id]
        # Sort by reading order within cluster
        snippets_sorted = sorted(snippets, key=lambda x: x[0])
        lines.append(f"Cluster {cluster_id}:")
        for _, s in snippets_sorted:
            lines.append(f"   - {s}")

    return "\n".join(lines)
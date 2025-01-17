from typing import List, Dict, Optional, Union
from pathlib import Path
import cv2
import os
import torch
import numpy as np
from typing import List, Dict, Optional, Union

class LayoutDetector:
    def __init__(
        self,
        image_path: str,
        model_name: str = "model_wt.pt",  # change to "model.pt" for the custom model
        repo_id: str = "vprashant/doclayout_detector",  
        weights_folder: str = "weight",  
        local_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.image = image_path
        self.device = device
        self.repo_id = repo_id
        self.weights_folder = weights_folder
        self.model_name = model_name
        self.local_model_path = local_model_path
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model by downloading weights from Hugging Face if not already available."""
        try:
            if self.local_model_path and os.path.exists(self.local_model_path):
                model_path = self.local_model_path
            else:
                # Download weights from Hugging Face
                try:
                    from huggingface_hub import hf_hub_download
                    model_path = hf_hub_download(
                        repo_id=self.repo_id,
                        filename=f"{self.weights_folder}/{self.model_name}",  # Path to weights in the repo
                        cache_dir="models" # Directory to save downloaded weights
                    )
                except ModuleNotFoundError as e:
                    raise RuntimeError("Please install the `pip install huggingface-hub` package to use the LayoutDetector")
                except Exception as e:
                    raise RuntimeError(f"Failed to download model weights: {e}")
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.model.to(self.device)
            except ModuleNotFoundError as e:
                raise RuntimeError("Please install the `pip install ultralytics` package to use the LayoutDetector")
            except Exception as e:
                raise RuntimeError(f"Failed to load YOLO model: {e}")

        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect_layout(self) -> List[Dict]:
        """Detect layout elements in an image using YOLO."""
        try:
            # Load image
            image = cv2.imread(self.imgae_path)
            if image is None:
                raise ValueError(f"Unable to load image from {self.image_path}")
    
            results = self.model(image)
            layout_info = []
            for result in results:
                for box in result.boxes:
                    layout_info.append({
                        "bbox": box.xyxy.tolist()[0],  # [x1, y1, x2, y2]
                        "label": result.names[int(box.cls)],  # Class label
                        "confidence": float(box.conf)  # Confidence score
                    })
            return layout_info
        except Exception as e:
            raise RuntimeError(f"Error during layout detection: {e}")

from typing import List, Dict, Optional, Union
from pathlib import Path
import cv2
import os
import torch
import numpy as np

class LayoutDetector:
    def __init__(
        self,
        image_path: str,
        model_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.image_path = image_path
        self.device = device
        self.model_path = model_path or self._get_default_model_path()
        self.categories = {
            0: 'title', 1: 'plain text', 2: 'abandon',
            3: 'figure', 4: 'figure_caption',
            5: 'table', 6: 'table_caption',
            7: 'table_footnote', 8: 'isolate_formula', 9: 'formula_caption'
        }
        self._load_model()

    def _get_default_model_path(self) -> str:
        """Get the default model path from project models folder."""
        # Try to find layout_detect.pt in models folder
        current_dir = Path(__file__).parent.parent.parent
        model_path = current_dir / "models" / "layout_detect.pt"

        if model_path.exists():
            return str(model_path)

        # Fallback to old model
        return None

    def _load_model(self) -> None:
        """Load doclayout_yolo model."""
        try:
            if not self.model_path or not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. "
                    f"Please place layout_detect.pt in the models/ folder."
                )

            try:
                from doclayout_yolo import YOLOv10
                self.model = YOLOv10(self.model_path)
                print(f"✓ Loaded doclayout_yolo model from {self.model_path}")
            except ImportError:
                print("doclayout_yolo not found, falling back to ultralytics YOLO")
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                self.model.to(self.device)

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def detect_layout(self, extract_text: bool = True) -> List[Dict]:
        """
        Detect layout elements using model (similar to llm_server implementation).

        Args:
            extract_text: If True, extract text from detected regions using OCR

        Returns:
            List of dicts with bbox, label, confidence, category, and text
        """
        try:
            # Load image
            image = cv2.imread(self.image_path)
            if image is None:
                raise ValueError(f"Unable to load image from {self.image_path}")

            # Run detection with model (similar to llm_server)
            det_res = self.model.predict(
                self.image_path,
                imgsz=1024,
                conf=0.15,
                iou=0.5,
                device=self.device
            )

            print(f"Detected {len(det_res[0].boxes)} regions")

            layout_info = []

            # Initialize OCR if needed
            ocr_reader = None
            if extract_text:
                try:
                    import easyocr
                    ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                except ImportError:
                    try:
                        import pytesseract
                        print("Using pytesseract for OCR")
                    except ImportError:
                        print("Warning: No OCR library available. Using labels as text.")
                        extract_text = False

            # Extract bounding boxes
            for i, box in enumerate(det_res[0].boxes.xyxy):
                category_id = int(det_res[0].boxes.cls[i])
                category_name = self.categories.get(category_id, "Unknown")
                bbox = box.tolist()
                confidence = float(det_res[0].boxes.conf[i])

                # Skip abandoned regions
                if category_name == "abandon":
                    continue

                # Extract text from bbox
                x1, y1, x2, y2 = map(int, bbox)
                crop_img = image[y1:y2, x1:x2]

                text = category_name  # Default to category name

                if extract_text and crop_img.size > 0:
                    try:
                        if ocr_reader:
                            ocr_results = ocr_reader.readtext(crop_img, detail=0)
                            if ocr_results:
                                text = " ".join(ocr_results).replace("\n", " ").replace("\t", " ").strip()
                        else:
                            import pytesseract
                            text = pytesseract.image_to_string(crop_img, config='--psm 6').strip()
                            text = text.replace("\n", " ").replace("\t", " ")
                    except Exception as e:
                        print(f"OCR failed for {category_name}: {e}")

                if not text or text.isspace():
                    text = f"<{category_name}>"

                layout_info.append({
                    "bbox": bbox,
                    "label": category_name,
                    "category": category_name,
                    "confidence": confidence,
                    "text": text
                })

            print(f"Extracted {len(layout_info)} valid regions (after filtering abandon)")
            return layout_info

        except Exception as e:
            raise RuntimeError(f"Error during layout detection: {e}")

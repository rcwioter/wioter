from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from ultralytics import YOLO


@dataclass
class Detection:
    label: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


class YoloV8Detector:
    def __init__(self, weights: str, class_names: Sequence[str]) -> None:
        self.model = YOLO(weights)
        self.class_names = list(class_names)

    def predict(self, image_path: Path) -> List[Detection]:
        results = self.model.predict(source=str(image_path), verbose=False)
        detections: List[Detection] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                label = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        label=label,
                        confidence=confidence,
                        x=int(x1),
                        y=int(y1),
                        width=int(x2 - x1),
                        height=int(y2 - y1),
                    )
                )
        return detections

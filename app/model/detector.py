from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Detection:
    label: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


class TomatoLeafDetector:
    """Mock detector placeholder for YOLOv8 + CBAM pipeline.

    This lightweight implementation makes it easy to replace with a real
    inference pipeline later without changing the Flask API contract.
    """

    def __init__(self) -> None:
        self.labels = [
            "late_blight",
            "early_blight",
            "leaf_mold",
            "septoria_leaf_spot",
        ]

    def predict(self, image_width: int, image_height: int) -> List[Detection]:
        """Return deterministic mock detections based on image dimensions."""
        if image_width == 0 or image_height == 0:
            return []

        box_width = max(40, image_width // 5)
        box_height = max(40, image_height // 6)
        x = image_width // 10
        y = image_height // 8

        return [
            Detection(
                label=self.labels[0],
                confidence=0.91,
                x=x,
                y=y,
                width=box_width,
                height=box_height,
            ),
            Detection(
                label=self.labels[2],
                confidence=0.86,
                x=min(image_width - box_width - 5, x + box_width + 30),
                y=min(image_height - box_height - 5, y + box_height + 20),
                width=box_width,
                height=box_height,
            ),
        ]

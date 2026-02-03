from __future__ import annotations

import base64
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Dict

from flask import Flask, jsonify, render_template, request
from PIL import Image

from model.detector import TomatoLeafDetector

app = Flask(__name__)


class DetectorService:
    def __init__(self) -> None:
        if os.environ.get("USE_YOLO") == "1":
            from model.yolo_detector import YoloV8Detector

            weights = os.environ.get("YOLO_WEIGHTS", "runs/detect/train/weights/best.pt")
            class_names = os.environ.get(
                "YOLO_CLASSES",
                "early_blight,late_blight,leaf_mold,septoria_leaf_spot",
            ).split(",")
            self.detector = YoloV8Detector(weights=weights, class_names=class_names)
            self.use_file_path = True
        else:
            self.detector = TomatoLeafDetector()
            self.use_file_path = False

    def predict(self, image: Image.Image) -> Dict[str, object]:
        width, height = image.size
        if self.use_file_path:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name)
                detections = self.detector.predict(Path(tmp.name))
            Path(tmp.name).unlink(missing_ok=True)
        else:
            detections = self.detector.predict(width, height)

        return {
            "width": width,
            "height": height,
            "detections": [
                {
                    "label": det.label,
                    "confidence": det.confidence,
                    "x": det.x,
                    "y": det.y,
                    "width": det.width,
                    "height": det.height,
                }
                for det in detections
            ],
            "summary": summarize(detections),
        }


def summarize(detections) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return counts


detector_service = DetectorService()


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/detect")
def detect() -> Dict[str, object]:
    if "image" not in request.files:
        return jsonify({"error": "请上传图片文件。"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "未检测到文件名。"}), 400

    image = Image.open(image_file.stream).convert("RGB")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    result = detector_service.predict(image)
    result["image"] = f"data:image/png;base64,{encoded}"
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

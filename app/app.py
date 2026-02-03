from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict

from flask import Flask, jsonify, render_template, request
from PIL import Image

from model.detector import TomatoLeafDetector

app = Flask(__name__)

detector = TomatoLeafDetector()


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
    width, height = image.size

    detections = detector.predict(width, height)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    result = {
        "image": f"data:image/png;base64,{encoded}",
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
    return jsonify(result)


def summarize(detections) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return counts


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

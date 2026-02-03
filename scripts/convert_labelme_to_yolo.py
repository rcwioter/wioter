"""Convert LabelMe JSON annotations to YOLO detection format.

Usage:
  python scripts/convert_labelme_to_yolo.py \
    --input data/raw/labelme \
    --output data/processed/plantvillage_det \
    --classes configs/classes.txt
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LabelMe to YOLO")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    return parser.parse_args()


def load_classes(classes_path: Path) -> Dict[str, int]:
    classes = [line.strip() for line in classes_path.read_text().splitlines() if line.strip()]
    return {name: idx for idx, name in enumerate(classes)}


def convert_box(points: List[List[float]], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height


def main() -> None:
    args = parse_args()
    class_map = load_classes(args.classes)

    image_out = args.output / "images"
    label_out = args.output / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    json_files = list(args.input.glob("*.json"))
    if not json_files:
        raise SystemExit("No LabelMe JSON files found.")

    for json_file in json_files:
        data = json.loads(json_file.read_text())
        img_path = Path(data["imagePath"]) if "imagePath" in data else None
        if img_path is None:
            continue
        src_image = (json_file.parent / img_path).resolve()
        if not src_image.exists():
            continue

        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        if not img_w or not img_h:
            continue

        labels: List[str] = []
        for shape in data.get("shapes", []):
            label = shape.get("label")
            points = shape.get("points", [])
            if label not in class_map or len(points) < 2:
                continue
            x_center, y_center, width, height = convert_box(points, img_w, img_h)
            labels.append(f"{class_map[label]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if not labels:
            continue

        target_image = image_out / src_image.name
        target_label = label_out / f"{src_image.stem}.txt"
        shutil.copy2(src_image, target_image)
        target_label.write_text("\n".join(labels))

    print(f"Converted annotations to {args.output}")


if __name__ == "__main__":
    main()

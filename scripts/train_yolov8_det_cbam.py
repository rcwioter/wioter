"""Train YOLOv8 with CBAM blocks using a custom model definition."""
from __future__ import annotations

import argparse

from ultralytics import YOLO
import ultralytics.nn.modules as modules

from models.cbam import CBAM


modules.CBAM = CBAM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 with CBAM")
    parser.add_argument("--data", default="configs/plantvillage_detect.yaml")
    parser.add_argument("--model", default="configs/yolov8s_cbam.yaml")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()

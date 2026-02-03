#!/usr/bin/env bash
set -euo pipefail

# Example: YOLOv8 classification training using PlantVillage
# Ensure ultralytics is installed: pip install ultralytics

yolo classify train \
  data=data/processed/plantvillage_cls \
  model=yolov8s-cls.pt \
  epochs=80 \
  imgsz=224 \
  batch=64 \
  device=0

#!/usr/bin/env bash
set -euo pipefail

# Example: YOLOv8 detection training
# Ensure ultralytics is installed: pip install ultralytics

yolo detect train \
  data=configs/plantvillage_detect.yaml \
  model=yolov8s.pt \
  epochs=120 \
  imgsz=640 \
  batch=16 \
  device=0

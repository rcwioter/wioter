"""Prepare PlantVillage dataset for YOLOv8 classification.

Expected input structure:
  data/raw/PlantVillage/
    class_a/
      img1.jpg
      img2.jpg
    class_b/
      img3.jpg

This script splits the dataset into train/val/test folders in
  data/processed/plantvillage_cls/images/{train,val,test}/<class_name>/
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PlantVillage dataset")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/PlantVillage"),
        help="PlantVillage root folder with class subfolders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/plantvillage_cls"),
        help="Output folder for YOLOv8 classification format",
    )
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def list_images(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]


def split_items(items: List[Path], ratios: Tuple[float, float, float]) -> Dict[str, List[Path]]:
    train_ratio, val_ratio, _ = ratios
    random.shuffle(items)
    total = len(items)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def copy_split(split: Dict[str, List[Path]], class_name: str, output_root: Path) -> None:
    for split_name, items in split.items():
        target_dir = output_root / "images" / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            shutil.copy2(item, target_dir / item.name)


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise SystemExit(f"Source path not found: {args.source}")

    ratios = (args.train, args.val, 1.0 - args.train - args.val)
    if ratios[2] <= 0:
        raise SystemExit("Train + val ratio must be less than 1.0")

    random.seed(args.seed)
    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    class_dirs = [p for p in args.source.iterdir() if p.is_dir()]
    if not class_dirs:
        raise SystemExit("No class folders found under source path.")

    for class_dir in class_dirs:
        images = list_images(class_dir)
        if not images:
            continue
        split = split_items(images, ratios)
        copy_split(split, class_dir.name, output_root)

    print(f"Prepared dataset at {output_root}")


if __name__ == "__main__":
    main()

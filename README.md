# 基于 YOLOv8 的复杂环境番茄叶片病害目标检测系统

## 项目概述
本项目以 YOLOv8 为核心构建作物叶片病害目标检测系统，面向光照变化、叶片遮挡与背景复杂等复杂环境场景，通过模型结构优化与数据增强提升检测精度与鲁棒性，并实现检测结果的前后端可视化展示。

## 技术路线
1. **数据采集与预处理**
   - 采集覆盖光照变化、叶片遮挡及背景复杂场景的番茄叶片病害图像。
   - 对数据进行标注、清洗与统一格式化，构建标准化训练数据集。
2. **模型优化**
   - 在 YOLOv8 特征融合层引入 **CBAM 注意力模块**：
     - **通道注意力**突出关键特征通道。
     - **空间注意力**强化病斑所在区域。
   - 结合多尺度训练与针对复杂环境的数据增强（光照增强、遮挡增强、背景增强），提升小病斑与稀疏病斑检测能力。
3. **训练与验证**
   - 对优化后的模型进行训练与验证。
   - 评估不同病害类别与复杂场景下的识别性能与检测稳定性。
4. **系统部署与可视化**
   - 将训练完成的模型部署到基于 **Flask** 的后端服务。
   - 通过 HTML 前端页面实现检测结果的实时可视化展示与病害统计分析。

## 系统设计亮点
- **高精度与复杂环境鲁棒性**：CBAM + 多尺度训练 + 场景增强。
- **小目标检测能力增强**：针对小病斑和稀疏病斑的特征强化。
- **端到端展示**：后端检测 + 前端实时可视化与统计分析。

## 模块划分
- **数据层**：图像采集、标注、预处理与增强。
- **模型层**：YOLOv8 主干 + CBAM 注意力增强。
- **训练层**：训练、验证、指标评估。
- **部署层**：Flask 推理服务 + HTML 可视化界面。

## 环境准备
### 基础运行（前后端可视化演示）
```bash
pip install -r requirements.txt
```

### 训练与真实推理（本地 GPU）
```bash
pip install -r requirements-ml.txt
```

> `requirements-ml.txt` 依赖 PyTorch + Ultralytics，仅在训练或真实推理时需要。

## PlantVillage 数据集落地流程
PlantVillage 是公开的番茄叶片病害图像数据集，适合做基础训练与病害类别定义。由于该数据集主要是分类图片，建议：
- **先用分类任务训练特征（YOLOv8-CLS）**
- **再对部分图片进行框标注，转为检测数据集**

### 1. 准备数据集目录
将 PlantVillage 解压到：
```
data/raw/PlantVillage/
  class_a/
  class_b/
```

### 2. 生成 YOLOv8 分类训练集
```bash
python scripts/prepare_plantvillage.py --source data/raw/PlantVillage --output data/processed/plantvillage_cls
```

### 3. 分类训练（本地 GPU）
```bash
bash scripts/train_yolov8_cls.sh
```

### 4. 标注检测数据（LabelMe 推荐）
- 使用 LabelMe 对病斑区域进行框标注，输出 JSON 文件
- 将 JSON 与图片放入 `data/raw/labelme/`
- 维护类别文件 `configs/classes.txt`

```bash
python scripts/convert_labelme_to_yolo.py \
  --input data/raw/labelme \
  --output data/processed/plantvillage_det \
  --classes configs/classes.txt
```

### 5. 检测训练（本地 GPU）
```bash
bash scripts/train_yolov8_det.sh
```

## CBAM 优化训练（检测）
本项目提供了可直接插入 CBAM 的 YOLOv8 检测模型配置与训练脚本：

```bash
python scripts/train_yolov8_det_cbam.py \
  --data configs/plantvillage_detect.yaml \
  --model configs/yolov8s_cbam.yaml \
  --epochs 120 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

> 训练脚本会自动注册 `CBAM` 模块（见 `models/cbam.py`），无需修改 Ultralytics 源码。

## 使用说明
### 1. 启动后端服务
```bash
python app/app.py
```

### 2. 访问前端页面
浏览器打开 `http://localhost:5000`，上传番茄叶片图像即可查看检测结果与统计。

> 当前版本内置轻量级演示检测器，便于后续无缝替换为真实 YOLOv8 推理流程。

## 切换到真实 YOLOv8 推理
训练完成后，使用以下环境变量替换检测器：
```bash
export USE_YOLO=1
export YOLO_WEIGHTS=runs/detect/train/weights/best.pt
export YOLO_CLASSES=early_blight,late_blight,leaf_mold,septoria_leaf_spot
python app/app.py
```

## 预期效果
- 在复杂环境下保持稳定检测性能。
- 提升小病斑识别准确率与召回率。
- 可视化界面可实时展示检测结果与病害统计。

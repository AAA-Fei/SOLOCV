# OCR 识别推理 (OCR Inference)

👉 [回主页文档](../../README.md)

本目录存放 OCR 识别的推理代码，支持多种后端推理框架。返回识别结果 `(box, text, score)`，其中 `box` 为文本框坐标 `(x1, y1, x2, y2)`（或多边形点集），`text` 为识别出的文本内容，`score` 为置信度。

## 推理后端总览

| 后端 | 状态 | 说明 |
| :--- | :---: | :--- |
| ONNXRuntime | ✅ 已支持 | 详见下方章节 |
| TensorRT | 🚧 计划中 | NVIDIA GPU 高性能 FP16/INT8 推理 |
| 昇腾 Ascend (ACL/CANN) | 🚧 计划中 | 华为昇腾 NPU 推理 (`.om` 模型) |
| RKNN | 🚧 计划中 | 瑞芯微 NPU 边缘端推理 (`.rknn` 模型) |
| OpenVINO | 🚧 计划中 | Intel CPU/iGPU 推理 |
| NCNN | ❌ 不支持 | 移动端 ARM 推理 |

> 状态说明：✅ 已支持 ｜ 🚧 计划中 ｜ ❌ 不支持

---

## ONNXRuntime

| 文件 | 模型文件 | 输入尺寸 | 支持设备 | 模型出处 | 说明 |
| :--- | :--- | :---: | :---: | :--- | :--- |
| [plate-ocr-onnxruntime.py](./plate-ocr-onnxruntime.py) | `models/ocr/plate-det-onnxruntime.onnx`、`models/ocr/plate-rec-onnxruntime.onnx` | 检测 640×640、识别 48×168 | CPU/GPU | [Chinese_license_plate](https://github.com/we0091234/Chinese_license_plate_detection_recognition) | 车牌检测 + 文字识别（含车牌颜色），支持单层/双层车牌 |

---

## TensorRT

| 文件 | 模型文件 | 精度 | 支持设备 | 说明 |
| :--- | :--- | :--- | :---: | :--- |
| - | - | - | - | 待添加 |

---

## 昇腾 Ascend (ACL/CANN)

| 文件 | 模型文件 | 精度 | 支持设备 | 说明 |
| :--- | :--- | :--- | :---: | :--- |
| - | - | - | - | 待添加 |

---

## RKNN

| 文件 | 模型文件 | 精度 | 支持设备 | 说明 |
| :--- | :--- | :--- | :---: | :--- |
| - | - | - | - | 待添加 |

---

## OpenVINO

| 文件 | 模型文件 | 精度 | 支持设备 | 说明 |
| :--- | :--- | :--- | :---: | :--- |
| - | - | - | - | 待添加 |

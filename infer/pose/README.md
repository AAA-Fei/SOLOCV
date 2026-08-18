# 姿态估计推理 (Pose Estimation Inference)

👉 [回主页文档](../../README.md)

本目录存放姿态估计 (Pose Estimation) 的推理代码，支持多种后端推理框架。返回人体关键点坐标 `(x, y, conf)` 和 Bounding Box 信息。

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

| 文件 | 模型文件                                         | 输入尺寸 | 支持设备 | 说明 |
| :--- |:---------------------------------------------| :---: | :---: | :--- |
| [yolo-pose-onnxruntime.py](./yolo-pose-onnxruntime.py) | `models/pose/yolo-pose-onnxruntime.onnx`     | 640×640 | CPU/GPU | 标准导出，后处理灵活可控（需手动实现 NMS） |
| [yolo-pose-onnxruntime-nms.py](./yolo-pose-onnxruntime-nms.py) | `models/pose/yolo-pose-onnxruntime-nms.onnx` | 640×640 | CPU/GPU | 部署更轻量，已内置 NMS，无需再做非极大值抑制 |

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

---

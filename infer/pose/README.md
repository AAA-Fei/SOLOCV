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

| 文件 | 模型文件                                         | 输入尺寸 | 支持设备 | 模型出处 | 说明 |
| :--- |:---------------------------------------------| :---: | :---: | :--- | :--- |
| [yolo-pose-onnxruntime.py](./yolo-pose-onnxruntime.py) | `models/pose/yolo-pose-onnxruntime.onnx`     | 640×640 | CPU/GPU | [ultralytics](https://github.com/ultralytics/ultralytics) | 标准导出，后处理灵活可控（需手动实现 NMS） |
| [yolo-pose-onnxruntime-nms.py](./yolo-pose-onnxruntime-nms.py) | `models/pose/yolo-pose-onnxruntime-nms.onnx` | 640×640 | CPU/GPU | [ultralytics](https://github.com/ultralytics/ultralytics) | 部署更轻量，已内置 NMS，无需再做非极大值抑制 |
| [rtmpose-pose-onnxruntime.py](./rtmpose-pose-onnxruntime.py) | `models/pose/rtmpose-pose-onnxruntime.onnx` | 256×192 | CPU/GPU | [RTMPose](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose) | RTMPose，HALPE26 拓扑（26 关键点），不返回检测框，需外部提供人体 bbox |
| [hrnet-pose-onnxruntime.py](./hrnet-pose-onnxruntime.py) | `models/pose/hrnet-pose-onnxruntime.onnx` | 256×192 | CPU/GPU | [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) | HRNet（top-down），COCO 17 关键点，不返回检测框，需外部提供人体 bbox 或默认整图单人 |
| [vitpose-pose-onnxruntime.py](./vitpose-pose-onnxruntime.py) | `models/pose/vitpose-pose-onnxruntime.onnx` | 256×192 | CPU/GPU | [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) | ViTPose（top-down），COCO 17 关键点，UDP 仿射对齐 + DARK 解码，默认整图单人推理 |

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

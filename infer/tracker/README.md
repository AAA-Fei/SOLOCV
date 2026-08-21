# 目标跟踪推理 (Tracking Inference)

👉 [回主页文档](../../README.md)

本目录存放多目标跟踪 (Multiple Object Tracking) 的推理代码，用于在视频流中持续跟踪目标。返回检测框 `(x1, y1, x2, y2, track_id)`，支持在线/离线追踪算法。

==========================追踪的示例视频同样放到了modelscope的根目录下=============================

## 推理后端总览

| 后端 | 状态 | 说明 |
| :--- | :---: | :--- |
| ONNXRuntime | ✅ 已支持 | 详见下方章节 |
| TensorRT | 🚧 计划中 | NVIDIA GPU 高性能 FP16/INT8 推理 |
| 昇腾 Ascend (ACL/CANN) | 🚧 计划中 | 华为昇腾 NPU 推理 (`.om` 模型) |
| RKNN | 🚧 计划中 | 瑞芯微 NPU 边缘端推理 (`.rknn` 模型) |
| OpenVINO | 🚧 计划中 | Intel CPU/iGPU 推理 |
| NCNN | ❌ 不支持 | 移动端 ARM 推理 |

> 状态说明：⚠️ 待实现 ｜ 🚧 计划中 ｜ ❌ 不支持

---

## ONNXRuntime

| 文件 | 模型文件 | 输入尺寸 | 支持设备 | 模型出处 | 说明 |
| :--- | :--- | :---: | :---: | :--- | :--- |
| [deepsort-tracker-onnxruntime.py](./deepsort-tracker-onnxruntime.py) | `models/tracker/deepsort-tracker-onnxruntime.onnx` | 128×64 | CPU/GPU | [Yolov5-Deepsort](https://github.com/Sharpiless/Yolov5-Deepsort) | DeepSort（ONNX ReID 特征提取 + 卡尔曼级联匹配），对接 `infer/det/` 任意检测器，默认仅跟踪 person，输出 `(x1,y1,x2,y2,cls_id,track_id)` |

---

## TensorRT

| 文件 | 模型文件 | 精度 | 支持设备 | 说明 |
| :--- | :--- | :--- | :---: | :--- |
| - | - | - | - | - | 待添加 |

---

## 昇腾 Ascend (ACL/CANN)

| 文件 | 模型文件 | 精度 | 支持设备 | 说明 |
| :--- | :--- | :--- | :---: | :--- |
| - | - | - | - | - | 待添加 |

---

## RKNN

| 文件 | 模型文件 | 精度 | 支持设备 | 说明 |
| :--- | :--- | :--- | :---: | :--- |
| - | - | - | - | - | 待添加 |

---

## OpenVINO

| 文件 | 模型文件 | 精度 | 支持设备 | 说明 |
| :--- | :--- | :--- | :---: | :--- |
| - | - | - | - | - | 待添加 |

---
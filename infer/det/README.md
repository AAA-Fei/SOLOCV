# 目标检测推理 (Detection Inference)

👉 [回主页文档](../../README.md)

本目录存放目标检测的推理代码，支持多种后端推理框架。返回原图坐标 `(x1, y1, x2, y2, class_id, score)`。

## 推理后端总览

| 后端 | 状态 | 说明 |
| :--- | :---: | :--- |
| ONNXRuntime | ✅ 已支持 | 详见下方章节 |
| TensorRT | 🚧 计划中 | NVIDIA GPU 高性能 FP16/INT8 推理 |
| 昇腾 Ascend (ACL/CANN) | 🚧 计划中 | 华为昇腾 NPU 推理 (`.om` 模型) |
| RKNN (RK3588/RK3576) | 🚧 计划中 | 瑞芯微 NPU 边缘端推理 (`.rknn` 模型) |
| OpenVINO | 🚧 计划中 | Intel CPU/iGPU 推理 |
| NCNN | 🚧 计划中 | 移动端 ARM 推理 |

> 状态说明：✅ 已支持 ｜ 🚧 计划中 ｜ ❌ 不支持

---

## ONNXRuntime

| 文件 | 模型文件 | 输出形状 | NMS 位置 | 说明 |
| :--- | :--- | :--- | :---: | :--- |
| [yolo-onnxruntime.py](./yolo-onnxruntime.py) | `models/det/yolo.onnx` | `[1, 84, 8400]` | Python 端 (cv2.dnn.NMSBoxes) | 标准导出，后处理灵活可控 |
| [yolo-onnxruntime-nms.py](./yolo-onnxruntime-nms.py) | `models/det/yolo_nms.onnx` | `[1, 300, 6]` | 模型端 (导出时含 NMS 节点) | 部署更轻量，无需再做 NMS |


---

## TensorRT

| 文件 | 模型文件 | 精度 | NMS 位置 | 说明 |
| :--- | :--- | :--- | :---: | :--- |
| - | - | - | - | 待添加 |

---

## 昇腾 Ascend (ACL/CANN)

| 文件 | 模型文件 | 精度 | NMS 位置 | 说明 |
| :--- | :--- | :--- | :---: | :--- |
| - | - | - | - | 待添加 |

---

## RKNN (RK3588/RK3576)

| 文件 | 模型文件 | 精度 | NMS 位置 | 说明 |
| :--- | :--- | :--- | :---: | :--- |
| - | - | - | - | 待添加 |

---

## OpenVINO

| 文件 | 模型文件 | 精度 | NMS 位置 | 说明 |
| :--- | :--- | :--- | :---: | :--- |
| - | - | - | - | 待添加 |

---

## NCNN

| 文件 | 模型文件 | 精度 | NMS 位置 | 说明 |
| :--- | :--- | :--- | :---: | :--- |
| - | - | - | - | 待添加 |

---

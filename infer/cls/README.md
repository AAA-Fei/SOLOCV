# 图像分类推理 (Classification Inference)

👉 [回主页文档](../../README.md)

本目录存放图像分类的推理代码，支持多种后端推理框架。返回 Top-k 预测结果 `(class_id, class_name, confidence)`。

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
| [ShuffleNetV2-cls-onnxruntime.py](./ShuffleNetV2-cls-onnxruntime.py) | `models/cls/ShuffleNetV2-cls-onnxruntime.onnx` | 224×224 | CPU/GPU | [ShuffleNetV2](https://github.com/Randl/ShuffleNetV2-pytorch) | ShuffleNetV2（ImageNet 1000 类），softmax 取 Top-k，官方 256-resize + 中心裁剪预处理 |

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
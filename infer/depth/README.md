# 深度估计推理 (Depth Estimation Inference)

👉 [回主页文档](../../README.md)

本目录存放单目深度估计 (Monocular Depth Estimation) 的推理代码，支持多种后端推理框架。返回深度图 `(depth_map)`，shape = `(img_h, img_w)`，值越大表示距离越近。

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
| [lite_momo-depth-onnxruntime.py](./lite_momo-depth-onnxruntime.py) | `models/depth/lite_momo-depth-onnxruntime.onnx` | 640×192 | CPU/GPU | [Lite-Mono](https://github.com/noahzn/Lite-Mono) | 单目深度估计，相对深度，JET 颜色映射可视化 |

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
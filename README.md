# 🎯 SOLOCV

觉得有用就给个 Star ⭐ 吧！

> 独立、轻量、即拿即用的 CV 模型推理文件集 —— 每个模型一个文件，拿来就能跑。

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 📖 这是什么？

**SOLOCV** 不是一个框架，而是一个**CV 模型推理文件的集合**。

> 每个模型单独一个 `.py` 文件，下载下来就能用，不需要学习任何框架 API。

你只需要：
1. 找到你需要的模型文件（如 `yolo.py`）
2. 下载它
3. 运行它

就这么简单。

---

## ✨ 核心理念

| 传统框架 | SOLOCV |
|:---|:---|
| 需要安装整个框架 | 只下载你需要的那个文件 |
| 需要学习复杂的 API | 打开文件看一眼就懂 |
| 改一个地方可能影响全局 | 模型之间完全隔离 |
| 添加新模型需要理解整个架构 | 复制一个文件，改几行代码即可 |
| 新手劝退 | 零门槛 |

**SOLOCV 的目标：让你 10 秒内从下载到跑通第一个模型。**

---


> 每个文件都是**完全独立**的，你可以只下载 `yolo.py` 和对应的模型权重，其他什么都不需要。

---

## 🚀 怎么用？

### 方式一：直接运行命令行

```bash
# 下载 yolo.py 和模型权重后，直接跑
python yolo.py --model yolov8n.onnx --image test.jpg


📄 许可证
采用 Apache License 2.0，商业友好，可自由使用和修改。


🙏 致谢
ONNX Runtime

所有开源模型和数据集

每一位贡献者 ❤️

📬 参与贡献
提 Issue 报告问题或建议
提 PR 添加新的模型文件

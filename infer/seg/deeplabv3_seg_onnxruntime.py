import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple


def voc_cmap(N: int = 256, normalized: bool = False) -> np.ndarray:
    """PASCAL VOC 颜色映射（官方 datasets/voc.py voc_cmap），用于解码语义掩码为 RGB 图"""
    def bitget(byteval: int, idx: int) -> bool:
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    return cmap


class DeepLabV3_SEG:
    # VOC 语义分割 21 类（官方 --dataset voc，num_classes=21）
    classes = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle',
               6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'dining table',
               12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'potted plant',
               17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tv/monitor'}
    # 官方预处理：ImageNet 归一化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def __init__(self, onnx_model: str, draw_boxes: bool = False):
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(onnx_model, providers=providers or available)
        self.model_inputs = self.session.get_inputs()
        # 全卷积网络，导出为动态输入，直接使用原图尺寸即可
        self.input_height = self.input_width = 0

        self.draw_boxes = draw_boxes  # 控制是否叠加绘制掩码
        self.cmap = voc_cmap()

    def draw_detections(self, img: np.ndarray, mask: np.ndarray,
                        colorized: np.ndarray) -> None:
        """将 VOC 颜色映射的掩码半透明叠加到原图（官方 decode_target + 可视化）"""
        if mask.max() <= 0:
            return
        img[:] = (img.astype(np.float32) * 0.5 + colorized.astype(np.float32) * 0.5).astype(np.uint8)

    # ---------- 推理 ----------

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        官方预处理（predict.py）：
        T.ToTensor()（RGB /255）+ T.Normalize(ImageNet mean/std)。
        官方默认 crop_val=False，不做 resize，直接使用原图尺寸。
        注：DeepLabV3 输入尺寸最好为 output_stride(=16) 的整数倍，非对齐尺寸也能运行。
        返回 [1, 3, H, W] 输入张量。
        """
        self.img_height, self.img_width = img.shape[:2]
        input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype(np.float32) / 255.0
        input_image = (input_image - self.mean) / self.std
        return input_image

    def postprocess(self, input_image: np.ndarray,
                    outputs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        官方后处理（predict.py）：
        pred = model(img).max(1)[1]（按类别通道 argmax），
        decode_target(pred) = cmap[mask] 得到彩色分割图。

        返回 (绘制后的图像, mask)
        mask 为 uint8 数组，shape = (img_h, img_w)，值为类别 id（0 为背景）。
        """
        output = np.squeeze(outputs[0])  # [C, H, W]
        mask = np.argmax(output, axis=0).astype(np.uint8)  # [H, W]
        colorized = self.cmap[mask]  # [H, W, 3] RGB

        if self.draw_boxes:
            self.draw_detections(input_image, mask, colorized)

        return input_image, mask

    def run(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行推理，返回 (绘制后的图像, 分割掩码)
        """
        img_copy = img.copy()
        input_image = self.preprocess(img)
        outputs = self.session.run(None, {self.model_inputs[0].name: input_image})
        return self.postprocess(img_copy, outputs)


if __name__ == "__main__":
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent
    target_file = script_dir.parent.parent
    model = target_file / Path("models/seg/deeplabv3-seg-onnxruntime.onnx")
    img = target_file / Path("assets/airplane.png")

    # 实例化时设置 draw_boxes=True 可叠加绘制掩码
    segmentor = DeepLabV3_SEG(model, draw_boxes=True)
    output_image, mask = segmentor.run(cv2.imread(str(img)))
    classes = [segmentor.classes.get(int(c), int(c)) for c in np.unique(mask)]
    print(f"掩码尺寸: {mask.shape[1]}x{mask.shape[0]} 含类别: {classes}")

    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
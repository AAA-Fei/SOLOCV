import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple


class LiteMono_DEPTH:
    # 官方 demo 采用的深度图颜色映射
    COLOR_MAP = cv2.COLORMAP_JET

    def __init__(self, onnx_model: str, draw_boxes: bool = False):
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(onnx_model, providers=providers or available)
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        # 仅控制是否生成深度可视化图（深度估计不返回检测框）
        self.draw_boxes = draw_boxes

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        官方预处理（demo_Lite-Mono_onnx.py run_inference）：
        resize 到模型输入尺寸 -> BGR->RGB -> HWC->CHW -> float32 /255
        返回 [1, 3, H, W] 输入张量。
        """
        input_image = cv2.resize(img, (self.input_width, self.input_height))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype(np.float32) / 255.0
        return input_image

    def postprocess(self, outputs: List[np.ndarray],
                    img_shape: Tuple[int, int]) -> np.ndarray:
        """
        官方后处理（demo_Lite-Mono_onnx.py run_inference）：
        squeeze -> *255.0 -> uint8，再缩放回原图尺寸。
        返回深度图 depth_map，shape = (img_h, img_w)，值域 [0, 255]。
        """
        img_h, img_w = img_shape
        depth_map = np.squeeze(outputs[0]).astype(np.float32) * 255.0
        depth_map = np.asarray(depth_map, dtype=np.uint8)
        if (img_w, img_h) != (self.input_width, self.input_height):
            depth_map = cv2.resize(depth_map, (img_w, img_h))
        return depth_map

    def draw_detections(self, img: np.ndarray,
                        depth_map: np.ndarray) -> np.ndarray:
        """对深度图应用 JET 颜色映射并缩放回原图尺寸（参考官方 draw_debug）"""
        depth_image = cv2.applyColorMap(depth_map, self.COLOR_MAP)
        depth_image = cv2.resize(depth_image, (self.img_width, self.img_height))
        return depth_image

    def run(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行推理，返回 (输出图像, 深度图)

        - output_image: 深度可视化图（JET 颜色映射，与原图同尺寸）；
          draw_boxes=False 时返回原图副本。
        - depth_map: 深度图，shape = (img_h, img_w)，值域 [0, 255]，
          值越大表示越近（Lite-Mono 输出相对深度）。
        """
        img_copy = img.copy()
        self.img_height, self.img_width = img.shape[:2]

        input_image = self.preprocess(img)
        outputs = self.session.run(None, {self.model_inputs[0].name: input_image})
        depth_map = self.postprocess(outputs, img.shape[:2])

        if self.draw_boxes:
            output_image = self.draw_detections(img_copy, depth_map)
        else:
            output_image = img_copy
        return output_image, depth_map


if __name__ == "__main__":
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent
    target_file = script_dir.parent.parent
    model = target_file / Path("models/depth/lite_momo-depth-onnxruntime.onnx")
    img = target_file / Path("assets/street.jpg")

    # 实例化时设置 draw_boxes=True 可生成深度可视化图
    depth_estimator = LiteMono_DEPTH(model, draw_boxes=True)
    src_img = cv2.imread(str(img))
    output_image, depth_map = depth_estimator.run(src_img)
    print(f"深度图尺寸: {depth_map.shape[1]}x{depth_map.shape[0]} "
          f"值域: [{int(depth_map.min())}, {int(depth_map.max())}]")

    cv2.imshow("Input", src_img)
    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
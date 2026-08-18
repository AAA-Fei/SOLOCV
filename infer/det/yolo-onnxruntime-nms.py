import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Dict, Any

class YOLO:
    def __init__(self, onnx_model: str, confidence_thres: float, iou_thres: float,
                 draw_boxes: bool = False):  # 新增绘制开关
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(onnx_model, providers=providers or available)
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.draw_boxes = draw_boxes  # 控制是否绘制

        # 类别字典保持不变
        self.classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                        6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
                        11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
                        16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
                        22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
                        27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
                        32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
                        36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
                        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
                        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
                        51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
                        57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
                        62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
                        68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
                        73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                        78: 'hair drier', 79: 'toothbrush'}
        # 固定颜色，避免随机变化（也可保留随机，这里改为固定明亮颜色）
        np.random.seed(21)
        self.color_palette = np.random.uniform(100, 255, size=(len(self.classes), 3)).astype(int)

    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, Tuple[int, int]]:
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, (top, left)

    def draw_detections(self, img: np.ndarray, box: List[float], score: float, class_id: int) -> None:
        x1, y1, w, h = box
        color = self.color_palette[class_id].tolist()
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 修正文字背景框的位置，使其恰好包裹文字
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + h + 10
        cv2.rectangle(img, (label_x, label_y - label_height - baseline),
                      (label_x + label_width, label_y + baseline), color, cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        self.img_height, self.img_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, pad = self.letterbox(img, (self.input_width, self.input_height))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = image_data[None].astype(np.float32)
        return image_data, pad

    def postprocess(self, input_image: np.ndarray, outputs: List[np.ndarray],
                    pad: Tuple[int, int]) -> Tuple[np.ndarray, List[Tuple[float, float, float, float, int, float]]]:
        """
        适配输出 [1, 300, 6]，坐标为输入尺寸（640×640）下的像素值，格式为 [x1, y1, x2, y2, score, class_id]
        """
        dets = outputs[0][0]  # (300, 6)
        gain = min(self.input_height / self.img_height, self.input_width / self.img_width)

        detections = []
        for row in dets:
            score = row[4]
            if score < self.confidence_thres:
                continue
            class_id = int(row[5])

            # 坐标格式为 [x1, y1, x2, y2]
            x1_in, y1_in, x2_in, y2_in = row[0], row[1], row[2], row[3]

            # ---------- 逆 letterbox 变换 ----------
            # 减去填充偏移
            x1_unpad = x1_in - pad[1]  # left
            y1_unpad = y1_in - pad[0]  # top
            x2_unpad = x2_in - pad[1]
            y2_unpad = y2_in - pad[0]

            # 除以缩放比例得到原图坐标
            x1 = x1_unpad / gain
            y1 = y1_unpad / gain
            x2 = x2_unpad / gain
            y2 = y2_unpad / gain

            # 边界裁剪（防止四舍五入误差）
            x1 = np.clip(x1, 0, self.img_width)
            y1 = np.clip(y1, 0, self.img_height)
            x2 = np.clip(x2, 0, self.img_width)
            y2 = np.clip(y2, 0, self.img_height)

            # 确保坐标顺序
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1

            # 保存结果 (x1, y1, x2, y2, class_id, score)
            detections.append((int(x1), int(y1), int(x2), int(y2), class_id, float(score)))

            if self.draw_boxes:
                box_xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                self.draw_detections(input_image, box_xywh, float(score), class_id)

        return input_image, detections

    def run(self, img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[float, float, float, float, int, float]]]:
        """
        执行推理，返回 (绘制后的图像, 检测结果)
        """
        img_copy = img.copy()
        img_data, pad = self.preprocess(img)
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
        return self.postprocess(img_copy, outputs, pad)


if __name__ == "__main__":
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent
    target_file = script_dir.parent.parent
    model = target_file / Path("models/det/yolo_nms.onnx")
    img = target_file / Path("assets/bus.jpg")
    conf_thres = 0.5
    iou_thres = 0.5

    # 实例化时设置 draw_boxes=True 可绘制
    detection = YOLO(model, conf_thres, iou_thres, draw_boxes=True)
    output_image, detections = detection.run(cv2.imread(str(img)))
    print(f"检测到 {len(detections)} 个目标：")
    for det in detections:
        print(f"  {det}")

    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
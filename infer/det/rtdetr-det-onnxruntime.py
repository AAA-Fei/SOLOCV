import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple


class RTDETR:
    def __init__(self, onnx_model: str, score_thr: float = 0.5,
                 iou_thr: float = 0.55, draw_boxes: bool = False):
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(onnx_model, providers=providers or available)
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.model_input_size = (self.input_height, self.input_width)  # (h, w)

        self.score_thr = score_thr
        self.iou_thr = iou_thr  # 官方 postprocess() 的 IoU 阈值，设为 0 可关闭
        self.draw_boxes = draw_boxes  # 控制是否绘制

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
        # 固定颜色，避免随机变化
        np.random.seed(21)
        self.color_palette = np.random.uniform(100, 255, size=(len(self.classes), 3)).astype(int)

    def draw_detections(self, img: np.ndarray, box: List[float], score: float, class_id: int) -> None:
        x1, y1, w, h = box
        color = self.color_palette[class_id % len(self.classes)].tolist()
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        label = f"{self.classes.get(class_id, class_id)}: {score:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 修正文字背景框的位置，使其恰好包裹文字
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + h + 10
        cv2.rectangle(img, (label_x, label_y - label_height - baseline),
                      (label_x + label_width, label_y + baseline), color, cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # ---------- IoU 框合并（官方 infer.py 的 postprocess()） ----------

    @staticmethod
    def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个 xyxy 框的 IoU。"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area != 0 else 0

    def merge_boxes(self, detections: List[Tuple], iou_thr: float) -> List[Tuple]:
        """
        官方 postprocess()：将同类别且 IoU >= iou_thr 的框合并为一个包围盒，
        分数取合并组内最大值。iou_thr <= 0 时直接返回原始结果。
        """
        if iou_thr <= 0:
            return detections

        merged = []
        used_indices = set()
        for i in range(len(detections)):
            if i in used_indices:
                continue
            x1, y1, x2, y2, cls_id, score = detections[i]
            boxes_to_merge = [(x1, y1, x2, y2)]
            scores_to_merge = [score]
            used_indices.add(i)
            for j in range(i + 1, len(detections)):
                if j in used_indices:
                    continue
                bx1, by1, bx2, by2, bcls, bscore = detections[j]
                if bcls != cls_id:
                    continue
                if self.calculate_iou((x1, y1, x2, y2), (bx1, by1, bx2, by2)) >= iou_thr:
                    boxes_to_merge.append((bx1, by1, bx2, by2))
                    scores_to_merge.append(bscore)
                    used_indices.add(j)
            xs = np.concatenate([[b[0], b[2]] for b in boxes_to_merge])
            ys = np.concatenate([[b[1], b[3]] for b in boxes_to_merge])
            merged_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
            merged_score = max(scores_to_merge)
            merged.append((int(merged_box[0]), int(merged_box[1]),
                           int(merged_box[2]), int(merged_box[3]),
                           cls_id, float(merged_score)))
        return merged

    # ---------- 推理 ----------

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        官方预处理（infer.py）：
        T.Resize((640, 640)) + T.ToTensor(/255)，直接用 cv2 实现。
        BGR->RGB -> resize 到模型输入尺寸 -> HWC->CHW -> float32 /255
        返回 [1, 3, H, W] 输入张量。
        """
        self.img_height, self.img_width = img.shape[:2]
        input_image = cv2.resize(img, (self.input_width, self.input_height))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype(np.float32) / 255.0
        return input_image

    def postprocess(self, input_image: np.ndarray,
                    outputs: List[np.ndarray]) -> Tuple[np.ndarray, List[Tuple]]:
        """
        官方后处理（infer.py）：
        模型已内置解码（topk 300），输出 labels/boxes/scores，
        boxes 为 xyxy 且已由 orig_target_sizes 缩放回原图坐标，无需额外解码。
        按 score_thr 过滤后再执行官方 postprocess() 的同类别 IoU 框合并。

        返回 (绘制后的图像, 检测结果列表)
        检测结果列表元素: (x1, y1, x2, y2, class_id, score)
        坐标均为原图上的整数坐标（左上角、右下角）
        """
        labels = np.squeeze(outputs[0])
        boxes = np.squeeze(outputs[1])
        scores = np.squeeze(outputs[2])
        if labels.ndim == 0:
            labels, boxes, scores = labels[None], boxes[None, None], scores[None]

        detections = []
        for label, box, score in zip(labels, boxes, scores):
            if score < self.score_thr:
                continue
            class_id = int(label)
            x1, y1, x2, y2 = box
            detections.append((int(x1), int(y1), int(x2), int(y2),
                               class_id, float(score)))

        detections = self.merge_boxes(detections, self.iou_thr)

        if self.draw_boxes:
            for x1, y1, x2, y2, class_id, score in detections:
                box_xywh = [x1, y1, x2 - x1, y2 - y1]
                self.draw_detections(input_image, box_xywh, score, class_id)

        return input_image, detections

    def run(self, img: np.ndarray) -> Tuple[np.ndarray, List[Tuple]]:
        """
        执行推理，返回 (绘制后的图像, 检测结果)
        """
        img_copy = img.copy()
        input_image = self.preprocess(img)
        # 官方传入原始图像尺寸 [w, h]，模型据此将归一化 bbox 缩放到原图坐标
        orig_target_sizes = np.array(
            [[self.img_width, self.img_height]], dtype=np.int64)
        outputs = self.session.run(
            None, {'images': input_image, 'orig_target_sizes': orig_target_sizes})
        return self.postprocess(img_copy, outputs)


if __name__ == "__main__":
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent
    target_file = script_dir.parent.parent
    model = target_file / Path("models/det/rtdetr-det-onnxruntime.onnx")
    img = target_file / Path("assets/bus.jpg")
    score_thr = 0.5

    detection = RTDETR(model, score_thr, draw_boxes=True)
    output_image, detections = detection.run(cv2.imread(str(img)))
    print(f"检测到 {len(detections)} 个目标：")
    for det in detections:
        print(f"  {det}")

    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
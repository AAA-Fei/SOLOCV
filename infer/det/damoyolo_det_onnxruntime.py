import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple


class DAMOYOLO:
    def __init__(self, onnx_model: str, conf_thre: float = 0.5,
                 nms_thre: float = 0.7, draw_boxes: bool = False):
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(onnx_model, providers=providers or available)
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.model_input_size = (self.input_height, self.input_width)  # (h, w)

        # 与官方 ZeroHead 配置一致：nms_conf_thre=0.05, nms_iou_thre=0.7
        self.conf_thre = conf_thre
        self.nms_thre = nms_thre
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

    # ---------- NMS（纯 NumPy 实现，与官方 postprocess 的 batched_nms 等价） ----------

    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, nms_thr: float) -> List[int]:
        """单类别 NMS。"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]
        return keep

    @staticmethod
    def multiclass_nms(boxes: np.ndarray, scores: np.ndarray,
                       conf_thr: float, nms_thr: float, max_num: int = 500) -> np.ndarray:
        """类别感知多类别 NMS，等价于官方 torchvision batched_nms。

        Returns:
            dets (np.ndarray): (M, 6)，格式 (x1, y1, x2, y2, score, cls_ind)。
        """
        final_dets = []
        num_classes = scores.shape[1]
        all_indices = np.arange(boxes.shape[0])
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_mask = cls_scores > conf_thr
            if valid_mask.sum() == 0:
                continue
            keep = DAMOYOLO.nms(boxes[valid_mask], cls_scores[valid_mask], nms_thr)
            for k in keep:
                final_dets.append([*boxes[valid_mask][k], cls_scores[valid_mask][k], cls_ind])
        if len(final_dets) == 0:
            return np.zeros((0, 6))
        dets = np.array(final_dets)
        if max_num > 0 and len(dets) > max_num:
            dets = dets[dets[:, 4].argsort()[::-1][:max_num]]
        return dets

    # ---------- 推理 ----------

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        官方预处理（demo.py preprocess + damo/utils/demo_utils.transform_img）：
        默认测试配置 image_mean/std=[0,0,0]/[1,1,1]、keep_ratio=False、flip_prob=0。
        即直接 resize 到模型输入尺寸 -> BGR->CHW -> float32（无归一化、无 /255）。
        返回 [1, 3, H, W] 输入张量。
        """
        self.img_height, self.img_width = img.shape[:2]
        input_image = cv2.resize(img, (self.input_width, self.input_height),
                                 interpolation=cv2.INTER_LINEAR)
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype(np.float32)
        return input_image

    def postprocess(self, input_image: np.ndarray,
                    outputs: List[np.ndarray]) -> Tuple[np.ndarray, List[Tuple]]:
        """
        官方后处理（demo.py postprocess + damo/utils/boxes.postprocess）：
        模型输出两个 head：scores [1, 8400, nc] 与 boxes [1, 8400, 4]（已解码的 xyxy，输入坐标系）。
        按 conf_thre 过滤 + 类别感知 NMS（top 500），再缩放到原图尺寸。

        返回 (绘制后的图像, 检测结果列表)
        检测结果列表元素: (x1, y1, x2, y2, class_id, score)
        坐标均为原图上的整数坐标（左上角、右下角）
        """
        scores = np.squeeze(outputs[0])  # [8400, nc]
        boxes = np.squeeze(outputs[1])   # [8400, 4] xyxy

        dets = self.multiclass_nms(boxes, scores, self.conf_thre, self.nms_thre)
        ratio_w = self.img_width / self.input_width
        ratio_h = self.img_height / self.input_height

        detections = []
        for x1, y1, x2, y2, score, cls_ind in dets:
            x1 = int(np.clip(x1 * ratio_w, 0, self.img_width))
            y1 = int(np.clip(y1 * ratio_h, 0, self.img_height))
            x2 = int(np.clip(x2 * ratio_w, 0, self.img_width))
            y2 = int(np.clip(y2 * ratio_h, 0, self.img_height))
            detections.append((x1, y1, x2, y2, int(cls_ind), float(score)))

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
        outputs = self.session.run(None, {self.model_inputs[0].name: input_image})
        return self.postprocess(img_copy, outputs)


if __name__ == "__main__":
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent
    target_file = script_dir.parent.parent
    model = target_file / Path("models/det/damoyolo-det-onnxruntime.onnx")
    img = target_file / Path("assets/bus.jpg")

    detection = DAMOYOLO(model, conf_thre=0.5, nms_thre=0.7, draw_boxes=True)
    output_image, detections = detection.run(cv2.imread(str(img)))
    print(f"检测到 {len(detections)} 个目标：")
    for det in detections:
        print(f"  {det}")

    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple


class YOLOX:
    def __init__(self, onnx_model: str, conf_thres: float = 0.5, nms_thres: float = 0.5,
                 draw_boxes: bool = False):
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(onnx_model, providers=providers or available)
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.model_input_size = (self.input_height, self.input_width)  # (h, w)

        self.conf_thres = conf_thres  # 最终置信度阈值：obj_conf * class_conf
        self.nms_thres = nms_thres    # NMS IoU 阈值
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

    # ---------- NMS（纯 NumPy 实现，与官方 demo_utils.nms 一致） ----------

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

    # ---------- 推理 ----------

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        官方预处理（demo/ONNXRuntime/onnx_inference.py 中的 preproc）：
        等比例缩放 + 右下角以 114 填充到模型输入尺寸 -> BGR->CHW -> float32。
        注意：YOLOX ONNX 模型输入为 [0, 255] 原始像素值（无需除以 255）。
        返回 [1, 3, H, W] 输入张量，并记录缩放比例 ratio。
        """
        self.img_height, self.img_width = img.shape[:2]
        padded_img = np.ones((self.input_height, self.input_width, 3), dtype=np.uint8) * 114
        self.ratio = min(self.input_height / img.shape[0], self.input_width / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * self.ratio), int(img.shape[0] * self.ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * self.ratio), : int(img.shape[1] * self.ratio)] = resized_img

        input_image = padded_img.transpose(2, 0, 1)
        input_image = np.ascontiguousarray(input_image, dtype=np.float32)
        input_image = np.expand_dims(input_image, axis=0)
        return input_image

    def demo_postprocess(self, outputs: np.ndarray) -> np.ndarray:
        """
        官方解码（demo/ONNXRuntime/onnx_inference.py 中的 demo_postprocess）：
        模型输出 [1, 8400, 85] 为每个 anchor 的原始回归值
        [cx, cy, w, h, obj_conf, cls_0..79]（obj/cls 已 sigmoid）。
        这里做 grid + strides 解码，把 cxcywh 还原到输入坐标系（640×640 像素空间）。
        """
        strides = [8, 16, 32]
        grids = []
        expanded_strides = []
        for stride in strides:
            hsize = self.input_height // stride
            wsize = self.input_width // stride
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1).astype(np.float32)
        expanded_strides = np.concatenate(expanded_strides, 1).astype(np.float32)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        return outputs

    def postprocess(self, input_image: np.ndarray,
                    output: np.ndarray) -> Tuple[np.ndarray, List[Tuple]]:
        """
        官方后处理（tools/demo.py 的 postprocess + class_agnostic=True）：
        解码后 [8400, 85] -> 取每 anchor 类别最大置信度；
        以 obj_conf * class_conf >= conf_thres 过滤；
        输出 (x1, y1, x2, y2, obj_conf, class_conf, class_pred)，再做类别无关 NMS；
        最后按 ratio 缩放回原图坐标。

        返回 (绘制后的图像, 检测结果列表)
        检测结果列表元素: (x1, y1, x2, y2, class_id, score)
        坐标均为原图上的整数坐标（左上角、右下角）
        """
        pred = np.squeeze(output)
        pred = self.demo_postprocess(pred)

        # cxcywh -> xyxy
        box_corner = pred.copy()
        box_corner[:, 0] = pred[:, 0] - pred[:, 2] / 2
        box_corner[:, 1] = pred[:, 1] - pred[:, 3] / 2
        box_corner[:, 2] = pred[:, 0] + pred[:, 2] / 2
        box_corner[:, 3] = pred[:, 1] + pred[:, 3] / 2
        pred[:, :4] = box_corner[:, :4]

        num_classes = pred.shape[1] - 5
        class_conf = pred[:, 5: 5 + num_classes].max(axis=1)
        class_pred = pred[:, 5: 5 + num_classes].argmax(axis=1)

        # (obj_conf * class_conf >= conf_thres) 过滤
        conf_mask = pred[:, 4] * class_conf >= self.conf_thres
        detections = np.column_stack([pred[:, :5], class_conf, class_pred.astype(np.float32)])
        detections = detections[conf_mask]

        final_dets = []
        if detections.shape[0] > 0:
            # 类别无关 NMS，score = obj_conf * class_conf
            keep = self.nms(detections[:, :4], detections[:, 4] * detections[:, 5], self.nms_thres)
            final_dets = detections[keep]

        detections_out = []
        for x1, y1, x2, y2, obj_conf, class_conf, class_id in final_dets:
            score = float(obj_conf * class_conf)
            x1 = int(np.clip(x1 / self.ratio, 0, self.img_width))
            y1 = int(np.clip(y1 / self.ratio, 0, self.img_height))
            x2 = int(np.clip(x2 / self.ratio, 0, self.img_width))
            y2 = int(np.clip(y2 / self.ratio, 0, self.img_height))
            detections_out.append((x1, y1, x2, y2, int(class_id), score))
            if self.draw_boxes:
                box_xywh = [x1, y1, x2 - x1, y2 - y1]
                self.draw_detections(input_image, box_xywh, score, int(class_id))

        return input_image, detections_out

    def run(self, img: np.ndarray) -> Tuple[np.ndarray, List[Tuple]]:
        """
        执行推理，返回 (绘制后的图像, 检测结果)
        """
        img_copy = img.copy()
        input_image = self.preprocess(img)
        outputs = self.session.run(None, {self.model_inputs[0].name: input_image})
        return self.postprocess(img_copy, outputs[0])


if __name__ == "__main__":
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent
    target_file = script_dir.parent.parent
    model = target_file / Path("models/det/yolox-det-onnxruntime.onnx")
    img = target_file / Path("assets/bus.jpg")
    conf_thres = 0.3
    nms_thres = 0.45

    # 实例化时设置 draw_boxes=True 可绘制
    detection = YOLOX(model, conf_thres, nms_thres, draw_boxes=True)
    output_image, detections = detection.run(cv2.imread(str(img)))
    print(f"检测到 {len(detections)} 个目标：")
    for det in detections:
        print(f"  {det}")

    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple


class RTMDET:
    def __init__(self, onnx_model: str, score_thr: float = 0.3, nms_thr: float = 0.45,
                 mean: Tuple[float, float, float] = (103.5300, 116.2800, 123.6750),
                 std: Tuple[float, float, float] = (57.3750, 57.1200, 58.3950),
                 draw_boxes: bool = False):
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(onnx_model, providers=providers or available)
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.model_input_size = (self.input_height, self.input_width)  # (h, w)

        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
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

    # ---------- NMS（纯 NumPy 实现，与 rtmlib 一致） ----------

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
                       nms_thr: float, score_thr: float):
        """类别感知（class-aware）多类别 NMS。

        Args:
            boxes (np.ndarray): (N, 4)，xyxy 格式。
            scores (np.ndarray): (N, num_classes) 每类别分数。

        Returns:
            tuple:
            - dets (np.ndarray | None): (M, 6)，格式 (x1, y1, x2, y2, score, cls_ind)。
            - keep (np.ndarray | None): 每个检测在原始 boxes 中的下标。
        """
        final_dets = []
        final_keep = []
        num_classes = scores.shape[1]
        all_indices = np.arange(boxes.shape[0])
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                continue
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            valid_indices = all_indices[valid_score_mask]
            keep = RTMDET.nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                final_dets.append(dets)
                final_keep.append(valid_indices[keep])
        if len(final_dets) == 0:
            return None, None
        return np.concatenate(final_dets, 0), np.concatenate(final_keep, 0)

    # ---------- 推理 ----------

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """RTMDet 预处理：等比缩放并填充到模型输入尺寸（左上角对齐，值 114），再归一化。

        Returns:
            tuple:
            - image_data (np.ndarray): 模型输入。
            - ratio (float): 图像缩放比例。
        """
        self.img_height, self.img_width = img.shape[:2]
        if img.shape[:2] == self.model_input_size:
            padded_img = img.copy()
            ratio = 1.
        else:
            padded_img = np.ones((*self.model_input_size, 3), dtype=np.uint8) * 114
            ratio = min(self.model_input_size[0] / img.shape[0],
                        self.model_input_size[1] / img.shape[1])
            resized_img = cv2.resize(
                img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
                interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
            padded_img[:padded_shape[0], :padded_shape[1]] = resized_img

        # RTMDet 的 mean/std 为 BGR 顺序（对应 ImageNet RGB 均值反转），无需转 RGB
        image_data = (padded_img.astype(np.float32) - self.mean) / self.std
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = image_data[None].astype(np.float32)
        return image_data, ratio

    def postprocess(self, input_image: np.ndarray, outputs: List[np.ndarray],
                    ratio: float) -> Tuple[np.ndarray, List[Tuple]]:
        """
        返回 (绘制后的图像, 检测结果列表)
        检测结果列表元素: (x1, y1, x2, y2, class_id, score)
        坐标均为原图上的整数坐标（左上角、右下角）

        支持两种导出格式：
        - 无内置 NMS：[1, N, 4+nc]，解码 + 类别感知 NMS 后处理
        - 内置 NMS：dets [1, M, 5] (xyxy+score) 与可选 labels [1, M]
        """
        output = np.squeeze(outputs[0])
        if output.ndim == 1:
            output = output[None, :]

        detections = []
        if output.shape[-1] == 4 or output.shape[-1] > 5:
            # ---------- 无内置 NMS：解码预测 ----------
            strides = [8, 16, 32]
            grids, expanded_strides = [], []
            for stride in strides:
                hsize = self.model_input_size[0] // stride
                wsize = self.model_input_size[1] // stride
                xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
                grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
                grids.append(grid)
                expanded_strides.append(np.full((*grid.shape[:2], 1), stride))

            grids = np.concatenate(grids, 1)
            expanded_strides = np.concatenate(expanded_strides, 1)
            output[..., :2] = (output[..., :2] + grids) * expanded_strides
            output[..., 2:4] = np.exp(output[..., 2:4]) * expanded_strides

            predictions = output
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            boxes_xyxy /= ratio
            dets, _ = self.multiclass_nms(boxes_xyxy, scores, self.nms_thr, self.score_thr)

            if dets is not None:
                for x1, y1, x2, y2, score, cls_ind in dets:
                    class_id = int(cls_ind)
                    detections.append((int(x1), int(y1), int(x2), int(y2),
                                       class_id, float(score)))

        elif output.shape[-1] == 5:
            # ---------- 内置 NMS：直接取检测结果 ----------
            final_boxes = output[:, :4] / ratio
            final_scores = output[:, 4]
            labels = None
            if len(outputs) > 1:
                labels = np.squeeze(outputs[1])

            isscore = final_scores > self.score_thr
            for i in np.where(isscore)[0]:
                class_id = int(labels[i]) if labels is not None else 0
                x1, y1, x2, y2 = final_boxes[i]
                detections.append((int(x1), int(y1), int(x2), int(y2),
                                   class_id, float(final_scores[i])))

        else:
            raise ValueError(
                f'Unexpected RTMDet output shape {outputs[0].shape}: last '
                'dimension must be 4/5+ (无内置 NMS) or 5 (内置 NMS).')

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
        img_data, ratio = self.preprocess(img)
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
        return self.postprocess(img_copy, outputs, ratio)


if __name__ == "__main__":
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent
    target_file = script_dir.parent.parent
    model = target_file / Path("models/det/rtmdet-det-onnxruntime.onnx")
    img = target_file / Path("assets/bus.jpg")
    score_thr = 0.5
    nms_thr = 0.5

    detection = RTMDET(model, score_thr, nms_thr, draw_boxes=True)
    output_image, detections = detection.run(cv2.imread(str(img)))
    print(f"检测到 {len(detections)} 个目标：")
    for det in detections:
        print(f"  {det}")

    cv2.imshow("Output", output_image)
    cv2.waitKey(0)

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple


class YOLO_SEG:
    def __init__(self, onnx_model: str, confidence_thres: float, iou_thres: float,
                 draw_boxes: bool = False):
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(onnx_model, providers=providers or available)
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.draw_boxes = draw_boxes

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

    def draw_detections(self, img: np.ndarray, box: List[float], score: float,
                        class_id: int, mask: np.ndarray = None) -> None:
        x1, y1, w, h = box
        color = self.color_palette[class_id].tolist()
        # 绘制分割掩码（半透明叠加）
        if mask is not None:
            color_arr = np.array(color, dtype=np.float32)
            img[mask] = (img[mask].astype(np.float32) * 0.5 + color_arr * 0.5).astype(np.uint8)
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

    def process_mask(self, protos: np.ndarray, masks_in: np.ndarray,
                     bboxes: List[Tuple[int, int, int, int]], pad: Tuple[int, int],
                     gain: float) -> List[np.ndarray]:

        c, mh, mw = protos.shape  # CHW
        masks = (masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)  # (N, mh, mw)

        # 掩码空间相对输入空间的缩放（通常 160/640 = 0.25）
        mask_scale = mh / self.input_height
        top_mask = int(pad[0] * mask_scale)
        left_mask = int(pad[1] * mask_scale)
        new_unpad_h = round(self.img_height * gain)
        new_unpad_w = round(self.img_width * gain)
        h_mask = int(new_unpad_h * mask_scale)
        w_mask = int(new_unpad_w * mask_scale)

        result_masks = []
        for i, bbox in enumerate(bboxes):
            mask = masks[i]
            # 裁剪掉 padding 区域，只保留原图对应部分
            mask_unpadded = mask[top_mask:top_mask + h_mask, left_mask:left_mask + w_mask]
            # 缩放到原图尺寸
            if mask_unpadded.size > 0:
                mask_resized = cv2.resize(mask_unpadded, (self.img_width, self.img_height),
                                          interpolation=cv2.INTER_LINEAR)
            else:
                mask_resized = np.zeros((self.img_height, self.img_width), dtype=np.float32)
            # 裁剪到检测框范围（框外置零）
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, min(self.img_width, x1))
            y1 = max(0, min(self.img_height, y1))
            x2 = max(0, min(self.img_width, x2))
            y2 = max(0, min(self.img_height, y2))
            mask_cropped = np.zeros_like(mask_resized)
            mask_cropped[y1:y2, x1:x2] = mask_resized[y1:y2, x1:x2]
            # 阈值化得到二值掩码
            result_masks.append(mask_cropped > 0.0)
        return result_masks

    def postprocess(self, input_image: np.ndarray, output: List[np.ndarray],
                    pad: Tuple[int, int]) -> Tuple[np.ndarray, List[Tuple]]:
        """
        返回 (绘制后的图像, 检测结果列表)
        检测结果列表元素: (x1, y1, x2, y2, class_id, score, mask)
        坐标均为原图上的整数坐标（左上角、右下角）
        mask 为布尔数组，shape = (img_h, img_w)
        """
        preds = np.transpose(np.squeeze(output[0]))  # (num_anchors, 4+nc+nm)
        protos = np.squeeze(output[1])  # (mask_dim, mask_h, mask_w)
        rows = preds.shape[0]
        nc = len(self.classes)

        boxes = []
        scores = []
        class_ids = []
        mask_coefs_list = []
        gain = min(self.input_height / self.img_height, self.input_width / self.img_width)

        # 先去除 padding (bbox cx, cy)
        preds[:, 0] -= pad[1]  # x
        preds[:, 1] -= pad[0]  # y

        for i in range(rows):
            classes_scores = preds[i][4:4 + nc]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                cx, cy, w, h = preds[i][0], preds[i][1], preds[i][2], preds[i][3]
                left = int((cx - w / 2) / gain)
                top = int((cy - h / 2) / gain)
                width = int(w / gain)
                height = int(h / gain)
                mask_coefs = preds[i][4 + nc:]  # (mask_dim,)
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
                mask_coefs_list.append(mask_coefs)

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        detections = []
        if len(indices) > 0:
            surviving_boxes = []
            surviving_coefs = []
            surviving_scores = []
            surviving_class_ids = []
            for i in np.array(indices).flatten():
                idx = int(i)
                surviving_boxes.append(boxes[idx])
                surviving_coefs.append(mask_coefs_list[idx])
                surviving_scores.append(scores[idx])
                surviving_class_ids.append(class_ids[idx])

            # 转为 (x1, y1, x2, y2) 以裁剪掩码
            bboxes_xyxy = []
            for box in surviving_boxes:
                x1, y1, w, h = box
                bboxes_xyxy.append((x1, y1, x1 + w, y1 + h))

            # 批量生成掩码
            masks = []
            if surviving_coefs:
                masks_in_arr = np.stack(surviving_coefs)
                masks = self.process_mask(protos, masks_in_arr, bboxes_xyxy, pad, gain)

            for i, (box, score, class_id) in enumerate(zip(surviving_boxes, surviving_scores, surviving_class_ids)):
                x1, y1, w, h = box
                x2, y2 = x1 + w, y1 + h
                mask = masks[i] if i < len(masks) else None
                detections.append((x1, y1, x2, y2, class_id, score, mask))
                if self.draw_boxes:
                    self.draw_detections(input_image, box, score, class_id, mask)

        return input_image, detections

    def run(self, img: np.ndarray) -> Tuple[np.ndarray, List[Tuple]]:
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
    model = target_file / Path("models/seg/yolo-seg-onnxruntime.onnx")
    img = target_file / Path("assets/bus.jpg")
    conf_thres = 0.5
    iou_thres = 0.5

    # 实例化时设置 draw_boxes=True 可绘制
    detection = YOLO_SEG(model, conf_thres, iou_thres, draw_boxes=True)
    output_image, detections = detection.run(cv2.imread(str(img)))
    print(f"检测到 {len(detections)} 个目标：")
    for det in detections:
        print(f"  {det[:6]}")  # mask 数组过大，只打印前 6 个字段

    cv2.imshow("Output", output_image)
    cv2.waitKey(0)

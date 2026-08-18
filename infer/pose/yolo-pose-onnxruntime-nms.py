import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple


class YOLO_POSE:
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2),
        (1, 3), (2, 4),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 6), (5, 11), (6, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]

    KEYPOINT_COLORS = [
        (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
        (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
        (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
        (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
        (255, 0, 170)
    ]

    LIMB_COLORS = [
        (255, 128, 0), (255, 128, 0), (255, 200, 0), (255, 200, 0),
        (0, 255, 0), (0, 255, 150), (0, 255, 0), (0, 255, 150),
        (0, 150, 255), (0, 100, 255), (0, 100, 255),
        (130, 0, 255), (230, 0, 255), (130, 0, 255), (230, 0, 255)
    ]

    def __init__(self, onnx_model: str, confidence_thres: float, iou_thres: float,
                 kpt_conf_thres: float, draw_boxes: bool = False):
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(onnx_model, providers=providers or available)
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.kpt_conf_thres = kpt_conf_thres
        self.draw_boxes = draw_boxes  # 控制是否绘制

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
                        keypoints: List[Tuple[int, int, float]]) -> None:
        x1, y1, w, h = box
        color = (0, 255, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        label = f"Person: {score:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 修正文字背景框的位置，使其恰好包裹文字
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + h + 10
        cv2.rectangle(img, (label_x, label_y - label_height - baseline),
                      (label_x + label_width, label_y + baseline), color, cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw skeleton
        for limb_idx, (start_idx, end_idx) in enumerate(self.SKELETON_CONNECTIONS):
            sp = keypoints[start_idx]
            ep = keypoints[end_idx]
            if sp[2] > self.kpt_conf_thres and ep[2] > self.kpt_conf_thres:
                cv2.line(img, (sp[0], sp[1]), (ep[0], ep[1]),
                         self.LIMB_COLORS[limb_idx], 2, cv2.LINE_AA)

        # Draw keypoints
        for kpt_idx, (kx, ky, conf) in enumerate(keypoints):
            if conf > self.kpt_conf_thres:
                cv2.circle(img, (kx, ky), 4, self.KEYPOINT_COLORS[kpt_idx], -1)

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        self.img_height, self.img_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, pad = self.letterbox(img, (self.input_width, self.input_height))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = image_data[None].astype(np.float32)
        return image_data, pad

    def postprocess(self, input_image: np.ndarray, outputs: List[np.ndarray],
                    pad: Tuple[int, int]) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, float, List]]]:
        """
        适配 NMS 模型输出 [1, 300, 57]
        每行格式: [x1, y1, x2, y2, score, class_id, kpt0_x, kpt0_y, kpt0_conf, ..., kpt16_x, kpt16_y, kpt16_conf]
        坐标为输入尺寸（640×640）下的像素值，模型已内置 sigmoid 和 NMS
        """
        dets = outputs[0][0]  # (300, 57)
        gain = min(self.input_height / self.img_height, self.input_width / self.img_width)

        detections = []
        for row in dets:
            score = row[4]
            if score < self.confidence_thres:
                continue

            # bbox 坐标 [x1, y1, x2, y2] 已在 640x640 像素空间
            x1_in, y1_in, x2_in, y2_in = row[0], row[1], row[2], row[3]

            # ---------- 逆 letterbox 变换 ----------
            x1_unpad = x1_in - pad[1]
            y1_unpad = y1_in - pad[0]
            x2_unpad = x2_in - pad[1]
            y2_unpad = y2_in - pad[0]

            x1 = x1_unpad / gain
            y1 = y1_unpad / gain
            x2 = x2_unpad / gain
            y2 = y2_unpad / gain

            # 边界裁剪
            x1 = np.clip(x1, 0, self.img_width)
            y1 = np.clip(y1, 0, self.img_height)
            x2 = np.clip(x2, 0, self.img_width)
            y2 = np.clip(y2, 0, self.img_height)

            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1

            # ---------- 关键点坐标（同样逆 letterbox） ----------
            keypoints = []
            for kpt_idx in range(17):
                base = 6 + kpt_idx * 3
                kx = int((row[base] - pad[1]) / gain)
                ky = int((row[base + 1] - pad[0]) / gain)
                kconf = row[base + 2]
                keypoints.append((kx, ky, kconf))

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            detections.append((x1, y1, x2, y2, float(score), keypoints))

            if self.draw_boxes:
                box_xywh = [x1, y1, x2 - x1, y2 - y1]
                self.draw_detections(input_image, box_xywh, float(score), keypoints)

        return input_image, detections

    def run(self, img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, float, List]]]:
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
    model = target_file / Path("models/pose/yolo-pose-onnxruntime-nms.onnx")
    img = target_file / Path("assets/bus.jpg")
    conf_thres = 0.5
    iou_thres = 0.5
    kpt_conf_thres = 0.5

    # 实例化时设置 draw_boxes=True 可绘制
    detection = YOLO_POSE(model, conf_thres, iou_thres, kpt_conf_thres, draw_boxes=True)
    output_image, detections = detection.run(cv2.imread(str(img)))
    print(f"检测到 {len(detections)} 个人：")
    for i, det in enumerate(detections):
        x1, y1, x2, y2, score, keypoints = det
        print(f"  person {i}: bbox=({x1}, {y1}, {x2}, {y2}) score={score:.2f}")
        for kpt_name, (kx, ky, kconf) in zip(detection.KEYPOINT_NAMES, keypoints):
            print(f"    {kpt_name}: ({kx}, {ky}, {kconf:.2f})")

    cv2.imshow("Output", output_image)
    cv2.waitKey(0)

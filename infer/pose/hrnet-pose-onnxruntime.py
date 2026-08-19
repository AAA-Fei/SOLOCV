import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple


class HRNet_POSE:
    # COCO 17 关键点（HRNet 官方拓扑）
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

    # ImageNet 归一化
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, onnx_model: str, kpt_conf_thres: float,
                 search_region_ratio: float = 0.1,
                 draw_boxes: bool = False):
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(onnx_model, providers=providers or available)
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.model_input_size = (self.input_width, self.input_height)  # (w, h)

        # 由输出通道数推断关键点数量（HRNet COCO 为 17）
        self.num_joints = 17
        out_shape = self.session.get_outputs()[0].shape
        if len(out_shape) >= 2 and isinstance(out_shape[1], int) and out_shape[1] > 0:
            self.num_joints = out_shape[1]

        self.kpt_conf_thres = kpt_conf_thres
        # 多目标时，人体 bbox 外扩比例，扩大搜索区域
        self.search_region_ratio = search_region_ratio
        # 仅控制是否绘制骨架/关键点（HRNet 不返回检测框）
        self.draw_boxes = draw_boxes

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        单目标预处理（参考官方 top-down 流程）：
        BGR -> RGB -> resize 到模型输入尺寸 -> /255 -> ImageNet 归一化
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.model_input_size)

        mean = np.array(self.IMAGENET_MEAN, dtype=np.float32)[None, :, None, None]
        std = np.array(self.IMAGENET_STD, dtype=np.float32)[None, :, None, None]
        image_data = img_resized.astype(np.float32) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))[None]
        image_data = (image_data - mean) / std
        return image_data.astype(np.float32)

    def postprocess(self, outputs: List[np.ndarray],
                    img_shape: Tuple[int, int]) -> Tuple[List[Tuple[int, int, float]], float]:
        """
        热图峰值解码：每个关键点通道取最大值位置，并映射回图像坐标。

        outputs[0]: (1, C, map_h, map_w) 热图
        img_shape: (h, w) 输入图像尺寸
        返回 (keypoints, score)
        keypoints: C 个 (x, y, conf)，坐标为图像整数坐标
        score: 该人整体置信度（全部关键点分值均值）
        """
        img_h, img_w = img_shape
        heatmaps = outputs[0][0]  # [C, map_h, map_w]
        C, map_h, map_w = heatmaps.shape

        heatmap_flat = heatmaps.reshape(C, -1)
        max_vals = heatmap_flat.max(axis=1)
        max_idx = heatmap_flat.argmax(axis=1)
        peaks_y, peaks_x = np.divmod(max_idx, map_w)

        # 将峰值坐标线性缩放回原图（参考官方实现 peaks[:, ::-1] * 比例）
        xs = peaks_x.astype(np.float32) * img_w / map_w
        ys = peaks_y.astype(np.float32) * img_h / map_h

        keypoints = [(int(x), int(y), float(v))
                     for x, y, v in zip(xs, ys, max_vals)]
        return keypoints, float(np.mean(max_vals))

    def draw_detections(self, img: np.ndarray,
                        keypoints: List[Tuple[int, int, float]]) -> None:
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

    def run(self, img: np.ndarray,
            bboxes: List[List[float]] = None) -> Tuple[np.ndarray, List[Tuple]]:
        """
        执行推理，返回 (绘制后的图像, 检测结果列表)

        HRNet 为 top-down 单目标模型，不返回检测框。
        未提供 bbox 时默认以整张图片作为单人区域；
        提供人体 bbox（xyxy）时按 bbox 裁剪并外扩搜索区域，逐人推理后映射回原图。
        检测结果元素: (keypoints, score)
        keypoints: 17 个 (x, y, conf)，坐标为原图整数坐标
        score: 该人整体置信度
        """
        img_copy = img.copy()
        self.img_height, self.img_width = img.shape[:2]
        if bboxes is None or len(bboxes) == 0:
            bboxes = [[0, 0, self.img_width, self.img_height]]

        detections = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            box_w, box_h = x2 - x1, y2 - y1

            # Enlarge search region（参考官方实现）
            x1 = max(int(x1 - box_w * self.search_region_ratio), 0)
            x2 = min(int(x2 + box_w * self.search_region_ratio), self.img_width)
            y1 = max(int(y1 - box_h * self.search_region_ratio), 0)
            y2 = min(int(y2 + box_h * self.search_region_ratio), self.img_height)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img_copy[y1:y2, x1:x2]
            img_data = self.preprocess(crop)
            outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
            keypoints, score = self.postprocess(outputs, crop.shape[:2])

            # Fix the pose to the original image（参考官方实现）
            keypoints = [(kx + x1, ky + y1, conf) for kx, ky, conf in keypoints]

            detections.append((keypoints, float(score)))
            if self.draw_boxes:
                self.draw_detections(img_copy, keypoints)

        return img_copy, detections


if __name__ == "__main__":
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent
    target_file = script_dir.parent.parent
    model = target_file / Path("models/pose/hrnet-pose-onnxruntime.onnx")
    img_path = target_file / Path("assets/human.jpg")
    kpt_conf_thres = 0.5

    # HRNet 为单目标模型，默认以整张图片作为单人区域
    src_img = cv2.imread(str(img_path))
    h, w = src_img.shape[:2]

    pose = HRNet_POSE(model, kpt_conf_thres, draw_boxes=True)
    output_image, detections = pose.run(src_img, bboxes=[[0, 0, w, h]])
    print(f"检测到 {len(detections)} 个人：")
    for i, (keypoints, score) in enumerate(detections):
        print(f"  person {i}: score={score:.3f}")
        for kpt_name, (kx, ky, kconf) in zip(pose.KEYPOINT_NAMES, keypoints):
            print(f"    {kpt_name}: ({kx}, {ky}, {kconf:.3f})")

    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
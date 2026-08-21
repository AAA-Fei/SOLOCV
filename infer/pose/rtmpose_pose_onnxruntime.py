import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple


class RTMPOSE:
    # HALPE26 拓扑（body + head/neck/hip + toes/heels），共 26 个关键点
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
        "head", "neck", "hip",
        "left_big_toe", "right_big_toe", "left_small_toe",
        "right_small_toe", "left_heel", "right_heel"
    ]

    SKELETON_CONNECTIONS = [
        (15, 13), (13, 11), (11, 19),
        (16, 14), (14, 12), (12, 19),
        (17, 18), (18, 19),
        (18, 5), (5, 7), (7, 9),
        (18, 6), (6, 8), (8, 10),
        (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
        (15, 20), (15, 22), (15, 24),
        (16, 21), (16, 23), (16, 25)
    ]

    # BGR 颜色（由 halpe26 定义的 RGB 颜色反转得到）
    # 头部/面部为蓝色，左半身为绿色，右半身为橙色
    KEYPOINT_COLORS = [
        (255, 153, 51), (255, 153, 51), (255, 153, 51), (255, 153, 51), (255, 153, 51),
        (0, 255, 0), (0, 128, 255), (0, 255, 0), (0, 128, 255), (0, 255, 0), (0, 128, 255),
        (0, 255, 0), (0, 128, 255), (0, 255, 0), (0, 128, 255), (0, 255, 0), (0, 128, 255),
        (0, 128, 255), (0, 128, 255), (0, 128, 255),
        (0, 128, 255), (0, 128, 255), (0, 128, 255), (0, 128, 255), (0, 128, 255), (0, 128, 255)
    ]

    LIMB_COLORS = [
        (0, 255, 0), (0, 255, 0), (0, 255, 0),
        (0, 128, 255), (0, 128, 255), (0, 128, 255),
        (255, 153, 51), (255, 153, 51),
        (0, 255, 0), (0, 255, 0), (0, 255, 0),
        (0, 128, 255), (0, 128, 255), (0, 128, 255),
        (255, 153, 51), (255, 153, 51), (255, 153, 51),
        (255, 153, 51), (255, 153, 51), (255, 153, 51), (255, 153, 51),
        (0, 255, 0), (0, 255, 0), (0, 255, 0),
        (0, 128, 255), (0, 128, 255), (0, 128, 255)
    ]

    def __init__(self, onnx_model: str, kpt_conf_thres: float,
                 mean: Tuple[float, float, float] = (123.675, 116.28, 103.53),
                 std: Tuple[float, float, float] = (58.395, 57.12, 57.375),
                 draw_boxes: bool = False):
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(onnx_model, providers=providers or available)
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.model_input_size = (self.input_width, self.input_height)  # (w, h)

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.kpt_conf_thres = kpt_conf_thres
        # RTMPose 不返回检测框，此开关控制是否绘制骨架
        self.draw_boxes = draw_boxes

    # ---------- 仿射变换辅助函数（top-down 预处理） ----------

    @staticmethod
    def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        return np.array([[cs, -sn], [sn, cs]]) @ pt

    @staticmethod
    def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        direction = a - b
        return b + np.r_[-direction[1], direction[0]]

    @staticmethod
    def bbox_xyxy2cs(bbox: np.ndarray,
                     padding: float = 1.25) -> Tuple[np.ndarray, np.ndarray]:
        """将 xyxy 检测框转换为 (center, scale)。scale 由 bbox 宽高乘以 padding 得到。"""
        bbox = np.array(bbox)
        dim = bbox.ndim
        if dim == 1:
            bbox = bbox[None, :]
        x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
        center = np.hstack([x1 + x2, y1 + y2]) * 0.5
        scale = np.hstack([x2 - x1, y2 - y1]) * padding
        if dim == 1:
            center = center[0]
            scale = scale[0]
        return center, scale

    @staticmethod
    def get_warp_matrix(center: np.ndarray, scale: np.ndarray, rot: float,
                        output_size: Tuple[int, int],
                        shift: Tuple[float, float] = (0., 0.),
                        inv: bool = False) -> np.ndarray:
        """计算将 bbox 区域仿射到输出尺寸的 2x3 变换矩阵。"""
        shift = np.array(shift)
        src_w = scale[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.deg2rad(rot)
        src_dir = RTMPOSE._rotate_point(np.array([0., src_w * -0.5]), rot_rad)
        dst_dir = np.array([0., dst_w * -0.5])

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale * shift
        src[1, :] = center + src_dir + scale * shift
        src[2, :] = RTMPOSE._get_3rd_point(src[0, :], src[1, :])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst[2, :] = RTMPOSE._get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return warp_mat

    def top_down_affine(self, center: np.ndarray, scale: np.ndarray,
                        img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """将 bbox 区域仿射变换为模型输入尺寸，返回 (仿射后图像, 修正宽高比后的 scale)。"""
        w, h = self.model_input_size
        warp_size = (int(w), int(h))
        aspect_ratio = w / h
        b_w, b_h = np.hsplit(scale, [1])
        scale = np.where(b_w > b_h * aspect_ratio,
                         np.hstack([b_w, b_w / aspect_ratio]),
                         np.hstack([b_h * aspect_ratio, b_h]))
        warp_mat = self.get_warp_matrix(center, scale, 0, output_size=(w, h))
        img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
        return img, scale

    # ---------- 推理 ----------

    def preprocess(self, img: np.ndarray, bbox: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """bbox: [x1, y1, x2, y2]（原图坐标）。返回 (模型输入, center, scale)。"""
        center, scale = self.bbox_xyxy2cs(bbox, padding=1.25)
        img, scale = self.top_down_affine(center, scale, img)
        image_data = (img.astype(np.float32) - self.mean) / self.std
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = image_data[None].astype(np.float32)
        return image_data, center, scale

    @staticmethod
    def get_simcc_maximum(simcc_x: np.ndarray,
                          simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """取 SimCC 两条轴上最大响应的位置与分值，合并置信度为两轴均分。"""
        N, K, _ = simcc_x.shape
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)
        x_locs = np.argmax(simcc_x, axis=1)
        y_locs = np.argmax(simcc_y, axis=1)
        locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
        max_val_x = np.amax(simcc_x, axis=1)
        max_val_y = np.amax(simcc_y, axis=1)
        vals = 0.5 * (max_val_x + max_val_y)
        locs[vals <= 0.] = -1
        return locs.reshape(N, K, 2), vals.reshape(N, K)

    def postprocess(self, outputs: List[np.ndarray], center: np.ndarray,
                    scale: np.ndarray) -> Tuple[List[Tuple[int, int, float]], float]:
        """
        解码 SimCC 输出并映射回原图坐标。

        RTMPose 输出两个向量:
        output0 (simcc_x): (1, K, Wx)
        output1 (simcc_y): (1, K, Wy)

        返回 (keypoints, score)
        keypoints: 26 个 (x, y, conf)，坐标为原图整数坐标
        score: 该人整体置信度（全部关键点分值均值）
        """
        simcc_x, simcc_y = outputs[0], outputs[1]
        locs, scores = self.get_simcc_maximum(simcc_x, simcc_y)
        # simcc_split_ratio = 2.0：热图坐标还原为模型输入尺寸坐标
        keypoints = locs / 2.0
        # 由模型输入尺寸坐标映射回原图坐标
        keypoints = keypoints / np.array(self.model_input_size, dtype=np.float32) * scale
        keypoints = keypoints + center - scale / 2

        keypoints = keypoints[0]  # (K, 2)
        scores = scores[0]  # (K,)

        keypoints_list = [(int(kx), int(ky), float(conf))
                          for (kx, ky), conf in zip(keypoints, scores)]
        return keypoints_list, float(scores.mean())

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

    def run(self, img: np.ndarray, bboxes: List[List[float]] = None) -> Tuple[np.ndarray, List[Tuple]]:
        """
        执行推理，返回 (绘制后的图像, 检测结果列表)

        RTMPose 不返回检测框，需要外部提供人体 bbox（xyxy）。
        未提供时默认使用整张图片作为单人区域。
        检测结果元素: (keypoints, score)
        keypoints: 26 个 (x, y, conf)，坐标为原图整数坐标
        score: 该人整体置信度
        """
        img_copy = img.copy()
        if bboxes is None or len(bboxes) == 0:
            bboxes = [[0, 0, img.shape[1], img.shape[0]]]

        detections = []
        for bbox in bboxes:
            img_data, center, scale = self.preprocess(img_copy, bbox)
            outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
            keypoints, score = self.postprocess(outputs, center, scale)
            detections.append((keypoints, score))
            if self.draw_boxes:
                self.draw_detections(img_copy, keypoints)

        return img_copy, detections


if __name__ == "__main__":
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent
    target_file = script_dir.parent.parent
    model = target_file / Path("models/pose/rtmpose-pose-onnxruntime.onnx")
    img_path = target_file / Path("assets/human.jpg")
    kpt_conf_thres = 0.5

    # RTMPose 不返回检测框，需提供人体 bbox（xyxy），默认整张图片作为单人区域
    src_img = cv2.imread(str(img_path))
    h, w = src_img.shape[:2]

    pose = RTMPOSE(model, kpt_conf_thres, draw_boxes=True)
    output_image, detections = pose.run(src_img, bboxes=[[0, 0, w, h]])
    print(f"检测到 {len(detections)} 个人：")
    for i, (keypoints, score) in enumerate(detections):
        print(f"  person {i}: score={score:.3f}")
        for kpt_name, (kx, ky, kconf) in zip(pose.KEYPOINT_NAMES, keypoints):
            print(f"    {kpt_name}: ({kx}, {ky}, {kconf:.3f})")

    cv2.imshow("Output", output_image)
    cv2.waitKey(0)

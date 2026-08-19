import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple


class VITPOSE_POSE:
    # COCO 17 关键点（ViTPose 官方拓扑）
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

    # ImageNet 归一化（官方 NormalizeTensor，输入为 RGB）
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # COCO 左右对称关键点对（官方 flip_pairs）
    FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

    # DARK 调制核大小（官方 modulate_kernel=11）
    MODULATE_KERNEL = 11

    def __init__(self, onnx_model: str, kpt_conf_thres: float = 0.5,
                 draw_boxes: bool = False, flip_test: bool = True):
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(onnx_model, providers=providers or available)
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.model_input_size = (self.input_width, self.input_height)  # (w, h)

        # 由输出通道数推断关键点数量（ViTPose COCO 为 17）
        self.num_joints = 17
        out_shape = self.session.get_outputs()[0].shape
        if len(out_shape) >= 2 and isinstance(out_shape[1], int) and out_shape[1] > 0:
            self.num_joints = out_shape[1]

        self.kpt_conf_thres = kpt_conf_thres
        # 仅控制是否绘制骨架/关键点（ViTPose 不返回检测框）
        self.draw_boxes = draw_boxes
        # 官方 test_cfg: flip_test=True，翻转两次推理取平均
        self.flip_test = flip_test

        self.mean = np.array(self.IMAGENET_MEAN, dtype=np.float32)[None, :, None, None]
        self.std = np.array(self.IMAGENET_STD, dtype=np.float32)[None, :, None, None]

    @staticmethod
    def _xyxy2xywh(bbox_xyxy):
        """xyxy -> xywh（参考官方 inference.py）"""
        return [bbox_xyxy[0], bbox_xyxy[1],
                bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1]]

    def _box2cs(self, bbox_xywh):
        """bbox -> (center, scale)（参考官方 inference.py _box2cs）"""
        x, y, w, h = bbox_xywh[:4]
        aspect_ratio = self.input_width / self.input_height
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        # pixel_std=200.0，官方再乘 1.25
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        scale = scale * 1.25
        return center, scale

    def _get_udp_warp_matrix(self, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """UDP 仿射变换矩阵（参考官方 TopDownAffine(use_udp=True) + get_warp_matrix）"""
        theta = np.deg2rad(0.0)
        size_input = center * 2.0
        size_dst = np.array(self.model_input_size, dtype=np.float32) - 1.0
        size_target = scale * 200.0
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        matrix = np.zeros((2, 3), dtype=np.float32)
        scale_x = size_dst[0] / size_target[0]
        scale_y = size_dst[1] / size_target[1]
        matrix[0, 0] = cos_t * scale_x
        matrix[0, 1] = -sin_t * scale_x
        matrix[0, 2] = scale_x * (-0.5 * size_input[0] * cos_t +
                                  0.5 * size_input[1] * sin_t +
                                  0.5 * size_target[0])
        matrix[1, 0] = sin_t * scale_y
        matrix[1, 1] = cos_t * scale_y
        matrix[1, 2] = scale_y * (-0.5 * size_input[0] * sin_t -
                                  0.5 * size_input[1] * cos_t +
                                  0.5 * size_target[1])
        return matrix

    def preprocess(self, img_rgb: np.ndarray,
                   bboxes: List[List[float]]) -> np.ndarray:
        """
        官方 top-down 流程：_box2cs -> UDP warpAffine -> ToTensor(/255) -> NormalizeTensor
        输入为 RGB 图像，bbox 为 xyxy，逐人仿射裁剪并拼成 batch。
        返回 [N, 3, H, W] 的归一化张量。
        """
        crops = []
        self._centers = []
        self._scales = []
        for bbox in bboxes:
            center, scale = self._box2cs(self._xyxy2xywh(bbox))
            self._centers.append(center)
            self._scales.append(scale)
            trans = self._get_udp_warp_matrix(center, scale)
            crop = cv2.warpAffine(img_rgb, trans, self.model_input_size,
                                  flags=cv2.INTER_LINEAR)
            crops.append(crop)

        image_data = np.stack(crops).astype(np.float32) / 255.0  # [N, H, W, C]
        image_data = np.transpose(image_data, (0, 3, 1, 2))      # [N, C, H, W]
        image_data = (image_data - self.mean) / self.std
        return image_data.astype(np.float32)

    @staticmethod
    def _flip_back(heatmaps: np.ndarray) -> np.ndarray:
        """翻转热图还原：先交换左右关键点通道，再水平翻转（参考官方 flip_back）"""
        out = heatmaps.copy()
        for left, right in [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]:
            out[:, [left, right], :, :] = out[:, [right, left], :, :]
        return out[:, :, :, ::-1]

    @staticmethod
    def _get_max_preds(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """热图最大响应位置（参考官方 top_down_eval.py）"""
        N, K, _, W = heatmaps.shape
        heatmaps_reshaped = heatmaps.reshape((N, K, -1))
        idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
        maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
        preds[:, :, 0] = preds[:, :, 0] % W
        preds[:, :, 1] = preds[:, :, 1] // W

        preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
        return preds, maxvals

    @staticmethod
    def _post_dark_udp(coords: np.ndarray, batch_heatmaps: np.ndarray,
                       kernel: int = 11) -> np.ndarray:
        """DARK UDP 后处理（参考官方 post_dark_udp）"""
        B, K, H, W = batch_heatmaps.shape
        N = coords.shape[0]

        for heatmaps in batch_heatmaps:
            for heatmap in heatmaps:
                cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
        np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
        np.log(batch_heatmaps, batch_heatmaps)

        batch_heatmaps_pad = np.pad(
            batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)),
            mode='edge').flatten()

        index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = batch_heatmaps_pad[index]
        ix1 = batch_heatmaps_pad[index + 1]
        iy1 = batch_heatmaps_pad[index + W + 2]
        ix1y1 = batch_heatmaps_pad[index + W + 3]
        ix1_y1_ = batch_heatmaps_pad[index - W - 3]
        ix1_ = batch_heatmaps_pad[index - 1]
        iy1_ = batch_heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(N, K, 2, 1)
        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(N, K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
        return coords

    def _transform_preds(self, coords: np.ndarray, center: np.ndarray,
                         scale: np.ndarray, output_size: Tuple[int, int],
                         use_udp: bool = True) -> np.ndarray:
        """预测坐标映射回原图（参考官方 transform_preds）"""
        scale = scale * 200.0
        if use_udp:
            scale_x = scale[0] / (output_size[0] - 1.0)
            scale_y = scale[1] / (output_size[1] - 1.0)
        else:
            scale_x = scale[0] / output_size[0]
            scale_y = scale[1] / output_size[1]

        target_coords = np.ones_like(coords)
        target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
        target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5
        return target_coords

    def postprocess(self, outputs: List[np.ndarray],
                    centers: List[np.ndarray], scales: List[np.ndarray],
                    img_shape: Tuple[int, int]) -> Tuple[List[Tuple[int, int, float]], float]:
        """
        UDP 热图解码：_get_max_preds -> post_dark_udp -> transform_preds(use_udp=True)

        outputs[0]: (N, C, map_h, map_w) 热图
        返回 (keypoints, score)
        keypoints: C 个 (x, y, conf)，坐标为图像整数坐标
        score: 该人整体置信度（全部关键点分值均值）
        """
        img_h, img_w = img_shape
        heatmaps = outputs[0].copy()
        N = heatmaps.shape[0]
        heatmap_size = (heatmaps.shape[3], heatmaps.shape[2])  # (w, h)

        preds, maxvals = self._get_max_preds(heatmaps)
        preds = self._post_dark_udp(preds, heatmaps, kernel=self.MODULATE_KERNEL)
        for i in range(N):
            preds[i] = self._transform_preds(preds[i], centers[i], scales[i],
                                             heatmap_size, use_udp=True)

        keypoints = []
        for i in range(N):
            kpts = []
            for j in range(self.num_joints):
                x, y = preds[i][j]
                conf = float(maxvals[i][j][0])
                kpts.append((int(x), int(y), conf))
            keypoints.append(kpts)
        score = float(np.mean(maxvals))
        return keypoints, score

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

        ViTPose 为 top-down 模型，不返回检测框。
        未提供 bbox 时默认以整张图片作为单人区域；
        提供人体 bbox（xyxy）时逐人仿射对齐后推理，坐标映射回原图。
        检测结果元素: (keypoints, score)
        keypoints: 17 个 (x, y, conf)，坐标为原图整数坐标
        score: 该人整体置信度
        """
        img_copy = img.copy()
        self.img_height, self.img_width = img.shape[:2]
        if bboxes is None or len(bboxes) == 0:
            bboxes = [[0, 0, self.img_width, self.img_height]]

        # 官方 LoadImageFromFile channel_order='rgb'，输入为 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_data = self.preprocess(img_rgb, bboxes)

        # 官方 test_cfg: flip_test=True，翻转输入再推理一次取平均
        if self.flip_test:
            img_data_flipped = img_data[:, :, :, ::-1]
            outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
            outputs_flipped = self.session.run(None, {self.model_inputs[0].name: img_data_flipped})
            heatmaps = outputs[0]
            heatmaps_flipped = self._flip_back(outputs_flipped[0])
            heatmaps = 0.5 * (heatmaps + heatmaps_flipped)
            outputs = [heatmaps]
        else:
            outputs = self.session.run(None, {self.model_inputs[0].name: img_data})

        keypoints_list, score = self.postprocess(
            outputs, self._centers, self._scales, (self.img_height, self.img_width))

        detections = []
        for keypoints in keypoints_list:
            detections.append((keypoints, float(score)))
            if self.draw_boxes:
                self.draw_detections(img_copy, keypoints)

        return img_copy, detections


if __name__ == "__main__":
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent
    target_file = script_dir.parent.parent
    model = target_file / Path("models/pose/vitpose-pose-onnxruntime.onnx")
    img_path = target_file / Path("assets/human.jpg")
    kpt_conf_thres = 0.5

    # ViTPose 为单目标模型，默认以整张图片作为单人区域
    src_img = cv2.imread(str(img_path))
    h, w = src_img.shape[:2]

    pose = VITPOSE_POSE(model, kpt_conf_thres, draw_boxes=True)
    output_image, detections = pose.run(src_img, bboxes=[[0, 0, w, h]])
    print(f"检测到 {len(detections)} 个人：")
    for i, (keypoints, score) in enumerate(detections):
        print(f"  person {i}: score={score:.3f}")
        for kpt_name, (kx, ky, kconf) in zip(pose.KEYPOINT_NAMES, keypoints):
            print(f"    {kpt_name}: ({kx}, {ky}, {kconf:.3f})")

    cv2.imshow("Output", output_image)
    cv2.waitKey(0)

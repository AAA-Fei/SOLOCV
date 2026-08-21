import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Tuple, Optional


# ====================================================================================
# DeepSort 追踪核心（移植自官方实现 https://github.com/Sharpiless/Yolov5-Deepsort）
# 包含：Kalman 滤波、Detection、Track、最近邻度量、级联匹配、IoU 匹配、Tracker。
# 改动点：
#   1) 移除 torch 依赖，特征提取改为 ONNXRuntime（见下方 DEEPSORT._get_features）。
#   2) numpy 化：bbox/conf/clss 均为 ndarray；类标签用 int（COCO class_id）而非字符串。
#   3) 去掉未使用的 NMS 预处理（官方 update() 同样未调用）。
# ====================================================================================


# ---------- 卡尔曼滤波 (sort/kalman_filter.py) ----------

chi2inv95 = {
    1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070,
    6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919,
}


class KalmanFilter(object):
    """
    用于图像空间边界框追踪的卡尔曼滤波。
    状态空间 (x, y, a, h, vx, vy, va, vh)：中心 (x,y)、宽高比 a、高 h 及其速度。
    """

    def __init__(self):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        # 等价于官方 cho_factor/cho_solve，用 numpy.linalg.solve 求 K = cov @ H^T @ S^{-1}
        kalman_gain = np.linalg.solve(
            projected_cov, np.dot(covariance, self._update_mat.T).T).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        # 等价于官方 solve_triangular：z = L^{-1} d^T
        z = np.linalg.solve(cholesky_factor, d.T)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha


# ---------- 检测 (sort/detection.py) ----------

class Detection(object):
    def __init__(self, tlwh, cls_, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        self.cls_ = cls_
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


# ---------- 轨迹状态与 Track (sort/track.py) ----------

class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track(object):
    def __init__(self, mean, cls_, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.cls_ = cls_
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)
        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        self.cls_ = detection.cls_
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted


# ---------- 最近邻距离度量 (sort/nn_matching.py) ----------

def _pdist(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_cosine_distance(x, y):
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":
            self._metric = lambda x, y: np.maximum(0.0, _pdist(x, y).min(axis=0))
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix


# ---------- 线性分配 / 级联匹配 (sort/linear_assignment.py) ----------

INFTY_COST = 1e+5


def linear_sum_assignment(cost_matrix: np.ndarray):
    """线性分配（匈牙利算法，纯 numpy）。

    行为与 scipy.optimize.linear_sum_assignment 一致：对成本矩阵求最小总成本
    的最大基数匹配，返回 (行索引, 列索引)。支持矩形矩阵。
    实现：1-indexed 交替路径增广（cp-algorithms），矩形矩阵以大常数填充为方阵。
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost_matrix 必须为二维数组")
    n_rows, n_cols = cost.shape
    if cost.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    n = max(n_rows, n_cols)
    big = float(cost.max()) + 1.0
    C = np.full((n, n), big, dtype=np.float64)
    C[:n_rows, :n_cols] = cost

    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=np.int64)   # p[j] = 分配到列 j 的行(1-indexed), 0=未分配
    way = np.zeros(n + 1, dtype=np.int64)
    INF = float('inf')

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = C[i0 - 1] - u[i0] - v[1:n + 1]          # 列 1..n 的 reduced cost
            notused = ~used[1:n + 1]
            improve = notused & (cur < minv[1:n + 1])
            way[1:n + 1] = np.where(improve, j0, way[1:n + 1])
            minv[1:n + 1] = np.where(improve, cur, minv[1:n + 1])
            cand = np.where(notused, minv[1:n + 1], INF)
            j1 = int(np.argmin(cand)) + 1
            delta = minv[j1]
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:                                       # 增广路径回溯
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    rows, cols = [], []
    for j in range(1, n + 1):
        i = p[j]
        if i != 0:
            r, c = i - 1, j - 1
            if r < n_rows and c < n_cols:
                rows.append(r)
                cols.append(c)
    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)


def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break
        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        if len(track_indices_l) == 0:
            continue
        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix


# ---------- IoU 匹配 (sort/iou_matching.py) ----------

def iou(bbox, candidates):
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]
    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)
    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = INFTY_COST
            continue
        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix


# ---------- 追踪器 (sort/tracker.py) ----------

class Tracker(object):
    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            return cost_matrix

        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        matches_a, unmatched_tracks_a, unmatched_detections = \
            matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            min_cost_matching(
                iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, detection.cls_, covariance, self._next_id, self.n_init,
            self.max_age, detection.feature))
        self._next_id += 1


# ====================================================================================
# DEEPSORT：ONNXRuntime 特征提取 + 追踪整合
# 与 infer/det/ 下任意检测器对接（检测器需有 run(img)->(img,[...])，
# 检测元素为 (x1,y1,x2,y2,class_id,score)）。默认只跟踪 person (COCO class_id=0)。
# 参数默认值取自官方 deep_sort/configs/deep_sort.yaml。
# ====================================================================================


class DEEPSORT:
    def __init__(self, onnx_model: str,
                 max_dist: float = 0.2,
                 min_confidence: float = 0.3,
                 nms_max_overlap: float = 0.5,
                 max_iou_distance: float = 0.7,
                 max_age: int = 70,
                 n_init: int = 3,
                 nn_budget: int = 100,
                 target_classes: Tuple[int, ...] = (0,)):
        """
        onnx_model: ReID 特征提取 ONNX 模型路径。
        target_classes: 需要跟踪的 COCO 类别 id 元组，默认 (0,) 即 person。
        """
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(onnx_model, providers=providers or available)
        # 实际生效的 EP（CUDA EP 若依赖的 CUDA/cuDNN 版本不匹配会静默回退 CPU）
        active = self.session.get_providers()
        print(f"[DEEPSORT] 推理后端: {active}"
              + ("  ✅ GPU" if "CUDAExecutionProvider" in active
                 else "  ⚠️ CPU only（GPU EP 未加载，多为 CUDA/cuDNN 版本不匹配）"))
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        # ReID 输入 [N, 3, H, W]，官方为 H=128, W=64
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        # cv2.resize 的 dsize 为 (W, H)
        self.reid_size = (self.input_width, self.input_height)
        # ImageNet 归一化（与官方 Extractor 一致，输入保持 BGR 不做通道转换）
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap  # 与官方一致：update 不再调用
        self.target_classes = set(target_classes)

        metric = NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
        self.height = 0
        self.width = 0

    # ---------- 特征提取（ONNX 版官方 Extractor） ----------

    def _preprocess(self, im_crops: List[np.ndarray]) -> np.ndarray:
        """官方预处理：resize 到 (64,128) -> /255 -> CHW -> ImageNet 归一化。"""
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)
        im_batch = np.stack([_resize(im, self.reid_size) for im in im_crops])  # (N,H,W,3)
        im_batch = im_batch.transpose(0, 3, 1, 2)  # (N,3,H,W)
        im_batch = (im_batch - self._mean) / self._std
        return im_batch.astype(np.float32)

    def _get_features(self, bbox_xywh: np.ndarray, ori_img: np.ndarray) -> np.ndarray:
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            im_batch = self._preprocess(im_crops)
            features = self.session.run(None, {self.model_inputs[0].name: im_batch})[0]
            return np.asarray(features)
        return np.array([])

    # ---------- 坐标转换 ----------

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh: np.ndarray) -> np.ndarray:
        bbox_tlwh = bbox_xywh.copy()
        if bbox_tlwh.shape[0] > 0:
            bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
            bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh) -> Tuple[int, int, int, int]:
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    @staticmethod
    def _tlwh_to_xyxy(bbox_tlwh, width: int, height: int) -> Tuple[int, int, int, int]:
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), height - 1)
        return x1, y1, x2, y2

    # ---------- 追踪更新（官方 DeepSort.update） ----------

    def update(self, bbox_xywh: np.ndarray, confidences: np.ndarray,
               clss: np.ndarray, ori_img: np.ndarray):
        """
        bbox_xywh: (N,4) 中心点坐标 + 宽高
        confidences: (N,) 置信度
        clss: (N,) 类别 id
        返回: [(x1,y1,x2,y2,cls_id,track_id), ...]
        """
        self.height, self.width = ori_img.shape[:2]
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], clss[i], conf, features[i])
                      for i, conf in enumerate(confidences) if conf > self.min_confidence]
        self.tracker.predict()
        self.tracker.update(detections)

        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box, self.width, self.height)
            outputs.append((x1, y1, x2, y2, int(track.cls_), int(track.track_id)))
        return outputs

    # ---------- 单帧追踪：检测器 + 过滤 + 更新 + 绘制 ----------

    def track_frame(self, detector, img: np.ndarray,
                    draw: bool = True) -> Tuple[np.ndarray, List[Tuple]]:
        """
        用给定检测器跑一帧：检测 -> 过滤目标类别 -> DeepSort 更新 -> 绘制。
        返回 (绘制后的图像, 追踪结果 [(x1,y1,x2,y2,cls_id,track_id), ...])。
        """
        frame_out, detections = detector.run(img)
        bbox_xywh, confs, clss = [], [], []
        for x1, y1, x2, y2, cls_id, score in detections:
            if int(cls_id) not in self.target_classes:
                continue
            bbox_xywh.append([int((x1 + x2) / 2), int((y1 + y2) / 2), x2 - x1, y2 - y1])
            confs.append(float(score))
            clss.append(int(cls_id))

        bbox_xywh = np.asarray(bbox_xywh, dtype=np.float32).reshape(-1, 4)
        confs = np.asarray(confs, dtype=np.float32).reshape(-1)
        clss = np.asarray(clss, dtype=np.int64).reshape(-1)

        outputs = self.update(bbox_xywh, confs, clss, img)
        if draw:
            self._draw_tracks(frame_out, outputs)
        return frame_out, outputs

    @staticmethod
    def _draw_tracks(image: np.ndarray,
                     tracks: List[Tuple[int, int, int, int, int, int]]) -> None:
        tl = max(round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1, 2)
        for x1, y1, x2, y2, cls_id, track_id in tracks:
            # person 红色，其余绿色（与官方 plot_bboxes 一致）
            color = (0, 0, 255) if cls_id == 0 else (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=tl, lineType=cv2.LINE_AA)
            label = f"{cls_id} ID-{track_id}"
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.rectangle(image, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 3),
                          color, -1, cv2.LINE_AA)
            cv2.putText(image, label, (x1, y1 - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# ====================================================================================
# 检测器对接：下方 __main__ 直接 from-import infer/det/ 下的检测器模块，
# 无需工厂函数。切换检测器只需改 import 的模块名 + 对应模型路径即可。
# ====================================================================================


if __name__ == "__main__":
    import sys

    script_dir = Path(__file__).resolve().parent            # infer/tracker
    project_root = script_dir.parent.parent                  # SOLOCV
    sys.path.insert(0, str(project_root))                    # 使 infer 包可导入
    models_dir = project_root / "models"
    assets_dir = project_root / "assets"

    from infer.det.rtdetrv2_det_onnxruntime import RTDETRv2

    det_model = models_dir / "det" / "rtdetrv2-det-onnxruntime.onnx"
    reid_model = models_dir / "tracker" / "deepsort-tracker-onnxruntime.onnx"
    video_path = assets_dir / "pedestrian.mp4"

    detector = RTDETRv2(str(det_model))
    tracker = DEEPSORT(str(reid_model), target_classes=(0,))  # 只跟踪 person

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        sys.exit(1)

    print(f"[INFO] 检测器: RTDETRv2 | ReID: {reid_model.name} | 视频: {video_path.name}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        result_img, tracks = tracker.track_frame(detector, frame, draw=True)
        cv2.imshow("DeepSort Tracking", result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_idx % 20 == 0:
            print(f"  frame {frame_idx}: {len(tracks)} tracks")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] 完成，共处理 {frame_idx} 帧")

"""
车牌 OCR 推理（检测 + 识别，含车牌颜色）—— 一个任务一个文件。

参照 https://github.com/we0091234/Chinese_license_plate_detection_recognition/blob/main/onnx_infer.py

- 检测模型输出 YOLO 风格 (1, N, 15)：xywh + 目标置信度 + 4 角点(8) + 2 类得分（0=单层/1=双层）
- 识别模型输出两个分支：字符序列（CTC 解码）+ 车牌颜色（黑色/蓝色/绿色/白色/黄色）

使用示例：
    from plate_ocr_onnxruntime import PlateOCR   # 文件名含连字符，需用 importlib 加载
    import cv2
    ocr = PlateOCR()                              # 默认：车牌检测 + 颜色识别
    results = ocr.ocr(cv2.imread("plate.jpg"))
    # [{"rect": [x1,y1,x2,y2], "landmarks": [[x,y]x4], "plate_no": "京A12345",
    #   "plate_color": "蓝色", "roi_height": int, "score": float, "type": "single_layer"/"double_layer"}]

============================= 交叉使用约定 =============================
PlateOCR 不关心具体模型，只要求（新增 OCR 任务模型遵守即可互相组合）：

    1) 检测器：run(img) -> (绘制图, detections)
       detections 元素统一为: (x1, y1, x2, y2, label, score, quad)
       quad 为 (4, 2) 角点数组；纯矩形检测器可传 None。
    2) 识别器（二选一）：
       - 车牌式：recognize(img, quad, label) -> (plate_no, plate_color, roi_height)
       - 通用式：run(crop_img) -> str                // 直接识别矩形裁剪图
====================================================================
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

DEFAULT_DET_MODEL = "models/ocr/plate-det-onnxruntime.onnx"
DEFAULT_REC_MODEL = "models/ocr/plate-rec-onnxruntime.onnx"

PLATE_CHARS = (
    "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
    "学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
)
PLATE_COLORS = ['黑色', '蓝色', '绿色', '白色', '黄色']
MEAN_VALUE = 0.588
STD_VALUE = 0.193


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


class PlateDet:
    """车牌检测模型（plate_detect.onnx）的 ONNXRuntime 推理封装。

    模型输出为 YOLO 风格 (1, N, 15)：
        [:4]  xywh 中心坐标宽高 | [4] 目标置信度 | [5:13] 4 个角点 (8 个值) | [13:15] 2 类得分
    类别：0 = 单层车牌(single_layer)，1 = 双层车牌(double_layer)
    """

    def __init__(self, onnx_model: str, confidence_thres: float = 0.5,
                 iou_thres: float = 0.5, draw_boxes: bool = False):
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

        self.classes = {0: 'single_layer', 1: 'double_layer'}
        self.color_palette = np.array([[64, 128, 255], [0, 200, 255]], dtype=int)

    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)
                  ) -> Tuple[np.ndarray, float, int, int]:
        """等比缩放 + 居中灰边填充，返回 (填充图, 缩放比, 左填充, 上填充)。"""
        height, width, _ = img.shape
        ratio = min(new_shape[0] / height, new_shape[1] / width)
        new_h, new_w = int(height * ratio), int(width * ratio)
        top = int((new_shape[0] - new_h) / 2)
        left = int((new_shape[1] - new_w) / 2)
        bottom = new_shape[0] - new_h - top
        right = new_shape[1] - new_w - left
        resized = cv2.resize(img, (new_w, new_h))
        boxed = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return boxed, ratio, left, top

    def draw_detections(self, img: np.ndarray, box: Tuple[float, float, float, float],
                        score: float, label: int, landmarks: np.ndarray) -> None:
        x1, y1, x2, y2 = box
        color = self.color_palette[label].tolist()
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        label_text = f"{self.classes[label]}: {score:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = int(x1)
        label_y = int(y1) - 10 if int(y1) - 10 > label_height else int(y2) + 10
        cv2.rectangle(img, (label_x, label_y - label_height - baseline),
                      (label_x + label_width, label_y + baseline), color, cv2.FILLED)
        cv2.putText(img, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)

        pts = landmarks.astype(int).reshape(4, 2)
        cv2.polylines(img, [pts], True, color, 1)
        for i, p in enumerate(pts):
            cv2.circle(img, (int(p[0]), int(p[1])), 3, color, -1)

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """预处理：letterbox + BGR->RGB + /255 + HWC->CHW，返回 (输入, 缩放比, 左填充, 上填充)。"""
        self.img_height, self.img_width = img.shape[:2]
        img, ratio, left, top = self.letterbox(img, (self.input_width, self.input_height))
        img = img[:, :, ::-1].transpose(2, 0, 1).copy().astype(np.float32)  # BGR->RGB, HWC->CHW
        img = img / 255.0
        img = img[None].astype(np.float32)
        return img, ratio, left, top

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        xyxy = boxes.copy()
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return xyxy

    @staticmethod
    def _nms(boxes: np.ndarray, iou_thres: float) -> List[int]:
        """标准 NMS，boxes 为 xyxy + score 的 (N, 5) 数组，返回保留下标。"""
        order = np.argsort(boxes[:, 4])[::-1]
        keep = []
        while order.size > 0:
            current = order[0]
            keep.append(current)
            if order.size == 1:
                break
            x1 = np.maximum(boxes[current, 0], boxes[order[1:], 0])
            y1 = np.maximum(boxes[current, 1], boxes[order[1:], 1])
            x2 = np.minimum(boxes[current, 2], boxes[order[1:], 2])
            y2 = np.minimum(boxes[current, 3], boxes[order[1:], 3])
            width = np.maximum(0, x2 - x1)
            height = np.maximum(0, y2 - y1)
            inter_area = width * height
            current_area = (boxes[current, 2] - boxes[current, 0]) * (
                boxes[current, 3] - boxes[current, 1])
            other_area = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (
                boxes[order[1:], 3] - boxes[order[1:], 1])
            union_area = current_area + other_area - inter_area
            iou = inter_area / np.maximum(union_area, 1e-6)
            remaining = np.where(iou <= iou_thres)[0]
            order = order[remaining + 1]
        return keep

    @staticmethod
    def _restore_box(boxes: np.ndarray, ratio: float, left: int, top: int) -> np.ndarray:
        """将 letterbox 坐标还原到原图：列 [0,2,5,7,9,11] 为 x，[1,3,6,8,10,12] 为 y。"""
        boxes[:, [0, 2, 5, 7, 9, 11]] -= left
        boxes[:, [1, 3, 6, 8, 10, 12]] -= top
        boxes[:, [0, 2, 5, 7, 9, 11]] /= ratio
        boxes[:, [1, 3, 6, 8, 10, 12]] /= ratio
        return boxes

    def postprocess(self, input_image: np.ndarray, output: List[np.ndarray],
                    ratio: float, left: int, top: int
                    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, int, float, np.ndarray]]]:
        """
        返回 (绘制后的图像, 检测结果列表)。
        检测结果元素: (x1, y1, x2, y2, label, score, landmarks)
        坐标均为原图上的整数坐标，landmarks 为 (4, 2) 的 float 角点数组。
        """
        image_h, image_w = input_image.shape[:2]
        dets = np.squeeze(output[0])
        dets = dets[dets[:, 4] > self.confidence_thres]
        if dets.size == 0:
            return input_image, []

        dets[:, 13:15] *= dets[:, 4:5]  # 类别得分乘目标置信度
        boxes = self._xywh_to_xyxy(dets[:, :4])
        scores = np.max(dets[:, 13:15], axis=-1, keepdims=True)
        labels = np.argmax(dets[:, 13:15], axis=-1).reshape(-1, 1)
        output = np.concatenate((boxes, scores, dets[:, 5:13], labels), axis=1)
        output = output[self._nms(output, self.iou_thres)]
        output = self._restore_box(output, ratio, left, top)

        detections = []
        for o in output:
            x1, y1, x2, y2 = o[:4]
            score = float(o[4])
            label = int(o[-1])
            landmarks = o[5:13].reshape(4, 2)
            box = (int(max(0, min(image_w - 1, x1))), int(max(0, min(image_h - 1, y1))),
                   int(max(0, min(image_w - 1, x2))), int(max(0, min(image_h - 1, y2))))
            detections.append((box[0], box[1], box[2], box[3], label, score, landmarks))
            if self.draw_boxes:
                self.draw_detections(input_image, box, score, label, landmarks)
        return input_image, detections

    def run(self, img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, int, float, np.ndarray]]]:
        """执行推理，返回 (绘制后的图像, 检测结果列表)。"""
        img_copy = img.copy()
        img_data, ratio, left, top = self.preprocess(img)
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
        return self.postprocess(img_copy, outputs, ratio, left, top)


class PlateRec:
    """车牌文字识别模型（plate_rec_color.onnx）的 ONNXRuntime 推理封装。

    识别模型输出两个分支：字符序列（CTC 解码）+ 车牌颜色；兼容单输出模型（颜色为"未知"）。
    """

    def __init__(self, onnx_model: str, rec_width: int = 168, rec_height: int = 48):
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(onnx_model, providers=providers or available)
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.rec_width = rec_width
        self.rec_height = rec_height

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """预处理：缩放至 (rec_width, rec_height) + 归一化 + HWC->CHW。"""
        img = cv2.resize(img, (self.rec_width, self.rec_height))
        img = img.astype(np.float32)
        img = (img / 255.0 - MEAN_VALUE) / STD_VALUE
        img = img.transpose(2, 0, 1)
        img = img[None].astype(np.float32)
        return img

    @staticmethod
    def _decode_plate(preds: np.ndarray) -> str:
        """CTC 解码：去掉空白符(索引 0)并合并连续重复字符。"""
        previous = 0
        decoded = []
        for pred in preds:
            pred = int(pred)
            if pred != 0 and pred != previous and pred < len(PLATE_CHARS):
                decoded.append(PLATE_CHARS[pred])
            previous = pred
        return "".join(decoded)

    def postprocess(self, output: List[np.ndarray]) -> Tuple[str, str]:
        """解码输出，返回 (plate_no, plate_color)。"""
        index = np.argmax(output[0][0], axis=-1)
        plate_no = self._decode_plate(index)
        if len(output) >= 2:
            color_logits = output[1]
            if color_logits.ndim > 1:
                color_logits = color_logits[0]
            plate_color = PLATE_COLORS[int(np.argmax(color_logits))]
        else:
            plate_color = "未知"
        return plate_no, plate_color

    def run(self, img: np.ndarray) -> Tuple[str, str]:
        """执行推理，输入为矫正后的车牌图像，返回 (plate_no, plate_color)。"""
        img_data = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.session.get_inputs()[0].name: img_data})
        return self.postprocess(outputs)

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """将 4 个角点按 左上、右上、右下、左下 排序。"""
        rect = np.zeros((4, 2), dtype="float32")
        sums = pts.sum(axis=1)
        rect[0] = pts[np.argmin(sums)]
        rect[2] = pts[np.argmax(sums)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """根据 4 个角点做透视矫正，返回正向摆正的车牌图像。"""
        rect = PlateRec._order_points(pts.astype("float32"))
        tl, tr, br, bl = rect
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b), 1)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b), 1)
        dst = np.array(
            [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, matrix, (max_width, max_height))

    @staticmethod
    def _split_merge(img: np.ndarray) -> np.ndarray:
        """双层车牌：上/下层文本区各自截取后左右拼接成单行。"""
        height, _, _ = img.shape
        upper = img[0: int(5 / 12 * height), :]
        lower = img[int(1 / 3 * height):, :]
        upper = cv2.resize(upper, (lower.shape[1], lower.shape[0]))
        return np.hstack((upper, lower))

    def recognize(self, img: np.ndarray, landmarks: np.ndarray, label: int = 0
                  ) -> Tuple[str, str, int]:
        """从原图中按角点裁剪、矫正（label=1 时拆合双层），返回 (plate_no, plate_color, roi_height)。"""
        roi_img = self._four_point_transform(img, landmarks)
        if label == 1:
            roi_img = self._split_merge(roi_img)
        plate_no, plate_color = self.run(roi_img)
        return plate_no, plate_color, roi_img.shape[0]


class PlateOCR:
    """车牌 OCR 统一调用入口（即拿即用，检测 + 识别）。

    detector / recognizer 直接传实例即可；也可传模型路径字符串（用默认模型加载该路径）
    或不传（使用默认模型路径），方便快速测试。
    """

    def __init__(self, detector: Union[str, Any, None] = None,
                 recognizer: Union[str, Any, None] = None,
                 min_score: float = 0.3):
        self.detector = self._build(detector, default_model=DEFAULT_DET_MODEL,
                                    factory=lambda p: PlateDet(p))
        self.recognizer = self._build(recognizer, default_model=DEFAULT_REC_MODEL,
                                      factory=lambda p: PlateRec(p))
        self.min_score = min_score

    @staticmethod
    def _build(spec: Union[str, Any, None], default_model: str, factory) -> Any:
        if spec is None:
            return factory(str(_repo_root() / default_model))
        if isinstance(spec, str):
            path = Path(spec)
            if not path.is_absolute():
                path = _repo_root() / path
            return factory(str(path))
        return spec  # 已传入实例，直接使用

    def ocr(self, img: np.ndarray, min_score: Optional[float] = None) -> List[Dict]:
        """执行 检测 + 识别。

        返回列表，每项:
            {"rect": [x1,y1,x2,y2], "landmarks": [[x,y]x4], "plate_no": str,
             "plate_color": str, "roi_height": int, "score": float,
             "type": "single_layer" / "double_layer"}
        """
        min_score = self.min_score if min_score is None else min_score
        _, detections = self.detector.run(img)
        results = []
        for det in detections:
            x1, y1, x2, y2, label, score, quad = det
            if score < min_score:
                continue
            plate_no, plate_color, roi_height = self._recognize(img, det)
            results.append({
                "rect": [int(x1), int(y1), int(x2), int(y2)],
                "landmarks": np.round(quad, 2).tolist(),
                "plate_no": plate_no,
                "plate_color": plate_color,
                "roi_height": roi_height,
                "score": round(float(score), 4),
                "type": "double_layer" if label == 1 else "single_layer",
            })
        return results

    def _recognize(self, img: np.ndarray, det: tuple) -> Tuple[str, str, int]:
        x1, y1, x2, y2, label, score, quad = det
        crop = None
        # 车牌式识别器用角点透视矫正；否则退化为矩形裁剪喂给通用识别器
        if quad is not None and hasattr(self.recognizer, "recognize"):
            res = self.recognizer.recognize(img, quad, label)
        else:
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            res = self.recognizer.run(crop)
        if isinstance(res, (tuple, list)):
            plate_no = str(res[0])
            plate_color = str(res[1]) if len(res) > 1 else "未知"
            roi_height = int(res[2]) if len(res) > 2 else (crop.shape[0] if crop is not None else -1)
        else:
            plate_no = str(res)
            plate_color = "未知"
            roi_height = crop.shape[0] if crop is not None else -1
        return plate_no, plate_color, roi_height

    def detect(self, img: np.ndarray):
        """仅检测，返回 (绘制图, detections)，detections 元素见模块说明。"""
        return self.detector.run(img)

    @staticmethod
    def _put_chinese_text(img: np.ndarray, text: str, left: int, top: int,
                          color: tuple, text_size: int, font_path: Optional[str] = None) -> None:
        """用 PIL 绘制中文文本；字体文件或 PIL 不可用时回退 cv2.putText。"""
        if font_path is None or not Path(font_path).exists():
            cv2.putText(img, text, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.5, text_size / 40), color, 2, cv2.LINE_AA)
            return
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            cv2.putText(img, text, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.5, text_size / 40), color, 2, cv2.LINE_AA)
            return
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype(str(font_path), text_size, encoding="utf-8")
        draw.text((left, top), text, color, font=font)
        img[...] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

    @classmethod
    def draw_results(cls, img: np.ndarray, results: List[Dict],
                     font_path: Optional[str] = None) -> np.ndarray:
        """绘制检测框 + 角点 + 车牌文本/颜色，返回可视化图。

        font_path 默认取仓库根目录 fonts/platech.ttf，缺失时中文无法用 cv2 显示。
        """
        if font_path is None:
            font_path = str(_repo_root() / "fonts" / "platech.ttf")
        vis = img.copy()
        for r in results:
            x1, y1, x2, y2 = r["rect"]
            w, h = x2 - x1, y2 - y1
            pad_w, pad_h = 0.05 * w, 0.11 * h
            x1 = int(max(0, x1 - pad_w))
            y1 = int(max(0, y1 - pad_h))
            x2 = int(min(vis.shape[1], x2 + pad_w))
            y2 = int(min(vis.shape[0], y2 + pad_h))

            color = (64, 128, 255) if r["type"] == "single_layer" else (0, 200, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            for p in np.array(r["landmarks"], dtype=np.int32):
                cv2.circle(vis, (int(p[0]), int(p[1])), 5, color, -1)

            text = f"{r['plate_no']} {r['plate_color']}"
            text_size = max(int(r.get("roi_height") or 20), 12)
            cls._put_chinese_text(vis, text, x1 - text_size, y1 - text_size - 10,
                                  (0, 255, 0), text_size, font_path)
        return vis


if __name__ == "__main__":
    detect_model = str(_repo_root() / DEFAULT_DET_MODEL)
    rec_model = str(_repo_root() / DEFAULT_REC_MODEL)
    image_path = str(_repo_root() / "assets" / "license_plate.jpg")

    det = PlateDet(detect_model)
    rec = PlateRec(rec_model)
    ocr = PlateOCR(detector=det, recognizer=rec)

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"读取图片失败: {image_path}")

    results = ocr.ocr(img)
    print(f"识别到 {len(results)} 个车牌：")
    for r in results:
        print(f"  {r['plate_no']} {r['plate_color']}  rect={r['rect']}  score={r['score']}  type={r['type']}")

    cv2.imshow("Plate OCR", ocr.draw_results(img, results))
    cv2.waitKey(0)
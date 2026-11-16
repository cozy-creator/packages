"""Motion transfer for Union Control: people's pose as Lightricks' Union Control workflow draws it (`DWPreprocessor`
with body, hands and face, resolution 512, `yolox_l.onnx` and `dw-ll_ucoco_384`): YOLOX-L finds the people, DWPose-l
reads their 133 whole-body keypoints, drawn as OpenPose's 18-point body, hands and face on black. It follows
Fannovel16/comfyui_controlnet_aux@b9abafa1 (src/custom_controlnet_aux/dwpose: `DwposeDetector.__call__`,
wholebody.py, dw_onnx/cv_ox_det.py, dw_torchscript/jit_pose.py, whose numerics are IDEA-Research/DWPose's onnxdet.py
and onnxpose.py), with each network given the channel order it was trained on: BGR to YOLOX, RGB to DWPose (its
config's `bgr_to_rgb`). The drawing reproduces OpenPose's rendering as that annotator draws it (util.py
`draw_bodypose`, `draw_handpose`, `draw_facepose`), written here. Both checkpoints are their authors' state dicts as
they are, in F32."""

from __future__ import annotations

import colorsys
import math

import cv2
import numpy as np
import torch
from cozy_runtime.accel import sequence_parallel
from cozy_runtime.author import Config, Loader, Model, Serving, uses_components
from PIL import Image

from ..model import stored
from .reference import scaled
from .rtmpose import INPUT_SIZE, SPLIT, RTMPose
from .yolox import YOLOX

COMPONENT = "model"
#: The detector's square input; frames and crops per network call (fixed, so each call replays one CUDA graph).
DETECT_SIZE, DETECT_BATCH, POSE_BATCH = 640, 8, 16
#: The keypoint score a part is drawn from (wholebody.py `format_result`), and the drawn map's short side.
SCORE, RESOLUTION = 0.3, 512
#: OpenPose's body limbs (1-based) and part colors, and its hand edges (util.py:82-202).
LIMBS = ((2, 3), (2, 6), (3, 4), (4, 5), (6, 7), (7, 8), (2, 9), (9, 10), (10, 11), (2, 12), (12, 13), (13, 14),
         (2, 1), (1, 15), (15, 17), (1, 16), (16, 18))
COLORS = ((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
          (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
          (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85))
HAND_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20))
EPS = 0.01
#: COCO-WholeBody's body order to OpenPose's, a neck inserted at 17 (wholebody.py:120-138).
MMPOSE, OPENPOSE = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3], [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
MEAN, STD = np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375])


def _anchors() -> tuple[np.ndarray, np.ndarray]:
    grids, strides = [], []
    for stride in (8, 16, 32):
        side = DETECT_SIZE // stride
        xv, yv = np.meshgrid(np.arange(side), np.arange(side))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((side * side, 1), stride))
    return np.concatenate(grids), np.concatenate(strides)


GRIDS, STRIDES = _anchors()


def _letterboxed(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """cv_ox_det.py:78-94 `preprocess`: scaled to fit 640x640 (bilinear), top-left on gray 114, CHW."""
    padded = np.ones((DETECT_SIZE, DETECT_SIZE, 3), dtype=np.uint8) * 114
    r = min(DETECT_SIZE / bgr.shape[0], DETECT_SIZE / bgr.shape[1])
    width, height = int(bgr.shape[1] * r), int(bgr.shape[0] * r)
    padded[:height, :width] = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), r


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """cv_ox_det.py:4-31."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        w = np.maximum(0.0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]) + 1)
        inter = w * h
        order = order[np.where(inter / (areas[i] + areas[order[1:]] - inter) <= threshold)[0] + 1]
    return keep


def _people(prediction: np.ndarray, ratio: float) -> np.ndarray:
    """cv_ox_det.py:56-76, 96-129 for the person class, the only one the annotator keeps: anchors decoded, boxes at
    objectness times class score over 0.1 suppressed at IoU 0.45, those over 0.3 kept. [n, 4] x0 y0 x1 y1, in pixels."""
    prediction = prediction.copy()
    prediction[:, :2] = (prediction[:, :2] + GRIDS) * STRIDES
    prediction[:, 2:4] = np.exp(prediction[:, 2:4]) * STRIDES
    boxes, scores = prediction[:, :4], prediction[:, 4] * prediction[:, 5]
    xyxy = np.ones_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    xyxy /= ratio
    valid = scores > 0.1
    keep = _nms(xyxy[valid], scores[valid], 0.45)
    found, confidence = xyxy[valid][keep].astype(np.float64), scores[valid][keep].astype(np.float64)
    return found[confidence > 0.3]


def _third(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def _crop(rgb: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """jit_pose.py:7-49, 119-289 (`bbox_xyxy2cs`, `_fix_aspect_ratio`, `get_warp_matrix`, `top_down_affine`): the box
    grown 1.25x about its center and widened to 3:4, warped to 288x384 (bilinear), normalized. Returns the crop and
    the box's center and scale."""
    x0, y0, x1, y1 = np.asarray(box, dtype=np.float64)
    center = np.hstack([x0 + x1, y0 + y1]) * 0.5
    w, h = np.hstack([x1 - x0, y1 - y0]) * 1.25
    width, height = INPUT_SIZE
    aspect = width / height
    scale = np.hstack([w, w / aspect]) if w > h * aspect else np.hstack([h * aspect, h])
    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + np.array([0.0, scale[0] * -0.5])
    src[2] = _third(src[0], src[1])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = [width * 0.5, height * 0.5]
    dst[1] = np.array([width * 0.5, height * 0.5]) + np.array([0.0, width * -0.5])
    dst[2] = _third(dst[0], dst[1])
    crop = cv2.warpAffine(rgb, cv2.getAffineTransform(src, dst), (width, height), flags=cv2.INTER_LINEAR)
    return ((crop - MEAN) / STD).astype(np.float32), center, scale


def _decoded(simcc_x: np.ndarray, simcc_y: np.ndarray, center: np.ndarray,
             scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """jit_pose.py:84-116, 292-354: each keypoint at its x and y SimCC maxima, scored by the lower; back in pixels."""
    locs = np.stack((np.argmax(simcc_x, axis=1), np.argmax(simcc_y, axis=1)), axis=-1).astype(np.float32)
    best_x, best_y = np.amax(simcc_x, axis=1), np.amax(simcc_y, axis=1)
    scores = np.where(best_x > best_y, best_y, best_x)
    locs[scores <= 0.] = -1
    locs /= SPLIT
    return locs / INPUT_SIZE * scale + center - scale / 2, scores


def _openpose(keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """wholebody.py:120-140: [n, 133, 3] COCO-WholeBody to [n, 134, 3] OpenPose order, the neck the shoulders' mean,
    scored 1 where both shoulders score over 0.3."""
    info = np.concatenate((keypoints, scores[..., None]), axis=-1)
    neck = np.mean(info[:, [5, 6]], axis=1)
    neck[:, 2:4] = np.logical_and(info[:, 5, 2:4] > 0.3, info[:, 6, 2:4] > 0.3).astype(int)
    info = np.insert(info, 17, neck, axis=1)
    info[:, OPENPOSE] = info[:, MMPOSE]
    return info


def _part(points: np.ndarray) -> list[np.ndarray | None] | None:
    """wholebody.py `format_keypoint_part`: points scoring under 0.3 dropped; a part with none is absent."""
    kept = [point if point[2] >= SCORE else None for point in points]
    return None if all(point is None for point in kept) else kept


def _body(canvas: np.ndarray, points: list[np.ndarray | None]) -> None:
    for (a, b), color in zip(LIMBS, COLORS, strict=False):
        p, q = points[a - 1], points[b - 1]
        if p is None or q is None:
            continue
        ys, xs = np.array([p[0], q[0]]), np.array([p[1], q[1]])
        length = ((xs[0] - xs[1]) ** 2 + (ys[0] - ys[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(xs[0] - xs[1], ys[0] - ys[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(ys)), int(np.mean(xs))), (int(length / 2), 4), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])
    for point, color in zip(points, COLORS, strict=True):
        if point is not None:
            cv2.circle(canvas, (int(point[0]), int(point[1])), 4, color, thickness=-1)


def _hand(canvas: np.ndarray, points: list[np.ndarray | None] | None) -> None:
    if not points:
        return
    for index, (a, b) in enumerate(HAND_EDGES):
        p, q = points[a], points[b]
        if p is None or q is None:
            continue
        x1, y1, x2, y2 = int(p[0]), int(p[1]), int(q[0]), int(q[1])
        if x1 > EPS and y1 > EPS and x2 > EPS and y2 > EPS:
            color = [c * 255 for c in colorsys.hsv_to_rgb(index / float(len(HAND_EDGES)), 1.0, 1.0)]
            cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)
    for point in points:
        if point is not None and int(point[0]) > EPS and int(point[1]) > EPS:
            cv2.circle(canvas, (int(point[0]), int(point[1])), 4, (0, 0, 255), thickness=-1)


def _face(canvas: np.ndarray, points: list[np.ndarray | None] | None) -> None:
    for point in points or ():
        if point is not None and int(point[0]) > EPS and int(point[1]) > EPS:
            cv2.circle(canvas, (int(point[0]), int(point[1])), 3, (255, 255, 255), thickness=-1)


def _drawn(people: np.ndarray, height: int, width: int) -> Image.Image:
    """`DwposeDetector.__call__` from its poses: each person's body, hands and face (wholebody.py `format_result`;
    the face's last two points are the body's eyes) on black at the frame's size, then scaled so the short side is
    512 (`resize_image_with_pad`: area down, cubic up)."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for person in people:
        body = _part(person[:18]) or [None] * 18
        face = _part(person[24:92])
        if face is not None:
            face += [body[14], body[15]]
        _body(canvas, body)
        _hand(canvas, _part(person[92:113]))
        _hand(canvas, _part(person[113:134]))
        _face(canvas, face)
    k = RESOLUTION / min(height, width)
    size = (int(np.round(width * k)), int(np.round(height * k)))
    return Image.fromarray(cv2.resize(canvas, size, interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA))


class DetectorGraph:
    def __init__(self, config: Config) -> None:
        self.components = {COMPONENT: stored(YOLOX(), COMPONENT, config)}


@sequence_parallel(degrees=(2, 4, 8))
class PersonDetector(Model[DetectorGraph]):
    """YOLOX-L (Megvii's `yolox_l.pth`), for the people DWPose reads."""

    serving = Serving(encoded_leaves=False, min_bits=16)
    graph: DetectorGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(DetectorGraph)

    @uses_components(COMPONENT)
    def people(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Each RGB frame's people, [n, 4] boxes in its pixels (`_people`), read as BGR as YOLOX was trained."""
        model = self.graph.components[COMPONENT]
        device = torch.device(str(self.device))
        found: list[np.ndarray] = []
        for start in range(0, len(frames), DETECT_BATCH):
            chunk = [_letterboxed(np.ascontiguousarray(frame[:, :, ::-1])) for frame in frames[start:start + DETECT_BATCH]]
            batch = np.stack([image for image, _ in chunk] + [chunk[-1][0]] * (DETECT_BATCH - len(chunk)))
            with torch.inference_mode():
                raw = model(torch.from_numpy(batch).to(device)).float().cpu().numpy()
            found += [_people(raw[i], ratio) for i, (_, ratio) in enumerate(chunk)]
        return found


class EstimatorGraph:
    def __init__(self, config: Config) -> None:
        self.components = {COMPONENT: stored(RTMPose(), COMPONENT, config)}


@sequence_parallel(degrees=(2, 4, 8))
class WholeBody(Model[EstimatorGraph]):
    """DWPose-l (`dw-ll_ucoco_384`): 133 whole-body keypoints per person."""

    serving = Serving(encoded_leaves=False, min_bits=16)
    graph: EstimatorGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(EstimatorGraph)

    @uses_components(COMPONENT)
    def keypoints(self, frames: list[np.ndarray], boxes: list[np.ndarray]) -> list[np.ndarray]:
        """Each RGB frame's people in its boxes, [n, 134, 3] x, y, score in OpenPose's order (`_openpose`)."""
        model = self.graph.components[COMPONENT]
        device = torch.device(str(self.device))
        wanted = [(index, box) for index, found in enumerate(boxes) for box in found]
        people: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in frames]
        for start in range(0, len(wanted), POSE_BATCH):
            part = wanted[start:start + POSE_BATCH]
            crops = [_crop(frames[index], box) for index, box in part]
            batch = np.stack([crop for crop, _, _ in crops] + [np.zeros_like(crops[0][0])] * (POSE_BATCH - len(crops)))
            with torch.inference_mode():
                logits = model(torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2))).to(device))
            simcc_x, simcc_y = (each.float().cpu().numpy() for each in logits)
            for i, ((index, _), (_, center, scale)) in enumerate(zip(part, crops, strict=True)):
                people[index].append(_decoded(simcc_x[i], simcc_y[i], center, scale))
        return [_openpose(np.array([k for k, _ in mine]), np.array([s for _, s in mine])) if mine
                else np.zeros((0, 134, 3)) for mine in people]


def drawn(frames: list[Image.Image], detector: PersonDetector, estimator: WholeBody) -> list[Image.Image]:
    """`frames` as the workflow's pose maps: each at 544 on its short side (`reference.scaled`), its people found and
    read, drawn at that size and scaled to 512. A frame with no one is black."""
    rgb = [np.asarray(frame) for frame in scaled(frames)]
    people = estimator.keypoints(rgb, detector.people(rgb))
    return [_drawn(found, *frame.shape[:2]) for frame, found in zip(rgb, people, strict=True)]

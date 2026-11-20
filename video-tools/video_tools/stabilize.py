"""Estimate and smooth one rigid camera path per shot in two streaming passes."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from cozy_runtime.author import InvalidRequest
from cozy_runtime.media import DecodedMediaEvent, DecodedVideoFrame

FloatArray = NDArray[np.float64]
GrayFrame = NDArray[np.uint8]


def _rgb(frame: DecodedVideoFrame) -> GrayFrame:
    return np.frombuffer(frame.rgb, dtype=np.uint8).reshape(frame.height, frame.width, 3)


def _gray(frame: DecodedVideoFrame) -> GrayFrame:
    picture = _rgb(frame)
    ratio = min(1.0, 640 / max(frame.width, frame.height))
    if ratio < 1:
        size = (max(1, round(frame.width * ratio)), max(1, round(frame.height * ratio)))
        picture = np.asarray(
            cv2.resize(picture, size, interpolation=cv2.INTER_AREA), dtype=np.uint8
        )
    return np.asarray(cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY), dtype=np.uint8)


def _estimate(previous: GrayFrame, current: GrayFrame) -> FloatArray | None:
    corners = cv2.goodFeaturesToTrack(previous, 240, 0.01, 8)
    if corners is None or len(corners) < 8:
        return None
    tracked, status, _ = cast(
        "tuple[NDArray[np.float32] | None, GrayFrame | None, NDArray[np.float32] | None]",
        cv2.calcOpticalFlowPyrLK(previous, current, corners, None),
    )
    if tracked is None or status is None:
        return None
    good = status.ravel().astype(bool)
    if int(good.sum()) < 8:
        return None
    affine, inliers = cv2.estimateAffinePartial2D(
        corners[good], tracked[good], ransacReprojThreshold=2
    )
    if affine is None or inliers is None or int(inliers.sum()) < max(8, len(inliers) // 2):
        return None
    # Large scale changes with weak geometric overlap are usually unrelated shots.
    scale = float(np.hypot(affine[0, 0], affine[1, 0]))
    return np.asarray(affine, dtype=np.float64) if 0.8 <= scale <= 1.25 else None


def _rotations(angles: FloatArray) -> FloatArray:
    cos, sin = np.cos(angles), np.sin(angles)
    return np.stack((cos, -sin, sin, cos), axis=-1).reshape(-1, 2, 2)


def _corrections(
    positions: list[FloatArray], radius: int, half: FloatArray, crop: bool
) -> FloatArray:
    path = np.asarray(positions, dtype=np.float64)
    window = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-((window / max(1, radius / 2)) ** 2))
    # Fit each shot independently and renormalize the kernel where the shot ends.
    support = np.convolve(np.ones(len(path)), weights)[radius : radius + len(path)]
    smoothed = np.column_stack(
        [np.convolve(axis, weights)[radius : radius + len(path)] / support for axis in path.T]
    )
    rotations = _rotations(smoothed[:, 2] - path[:, 2])
    shifts = smoothed[:, :2] - (rotations @ path[:, :2, None])[:, :, 0]

    if crop:
        inverse = rotations.transpose(0, 2, 1)
        reach = half - np.abs((inverse @ shifts[:, :, None])[:, :, 0])
        if (reach <= 0).any():
            raise InvalidRequest(
                "Crop to fit would lose the frame center. Use less smoothing or turn it off.",
                fields=["crop_to_fit"],
            )
        zoom = max(1.0, float((np.abs(inverse) @ half / reach).max()))
        rotations *= zoom
        shifts *= zoom

    matrices = np.empty((len(path), 2, 3), dtype=np.float64)
    matrices[:, :, :2] = rotations
    matrices[:, :, 2] = shifts + half - rotations @ half
    return matrices


def transforms(
    events: Iterator[DecodedMediaEvent],
    radius: int,
    crop: bool,
    checkpoint: Callable[[], None],
) -> FloatArray:
    previous = None
    half: FloatArray | None = None
    size: FloatArray | None = None
    shots: list[FloatArray] = []
    path: list[FloatArray] = []
    for frame in events:
        if not isinstance(frame, DecodedVideoFrame):
            continue
        checkpoint()
        gray = _gray(frame)
        if previous is None:
            half = (np.asarray(gray.shape[::-1], dtype=np.float64) - 1) / 2
            size = np.asarray([frame.width / gray.shape[1], frame.height / gray.shape[0]])
            path.append(np.zeros(3, dtype=np.float64))
        else:
            assert half is not None
            motion = _estimate(previous, gray)
            distance = float(cv2.absdiff(previous, gray).mean()) if motion is None else 0.0
            if motion is None and distance >= 28:
                shots.append(_corrections(path, radius, half, crop))
                path = [np.zeros(3, dtype=np.float64)]
            elif motion is None:
                path.append(path[-1])
            else:
                angle = float(np.arctan2(motion[1, 0], motion[0, 0]))
                center_shift = motion[:, :2] @ half + motion[:, 2] - half
                rotation = _rotations(np.asarray([angle]))[0]
                shift = rotation @ path[-1][:2] + center_shift
                path.append(np.asarray([shift[0], shift[1], path[-1][2] + angle]))
        previous = gray

    if half is None or size is None:
        raise InvalidRequest("The input has no video frames.", fields=["video"])
    shots.append(_corrections(path, radius, half, crop))
    result = np.concatenate(shots)
    result[:, :, 2] *= size
    result[:, :, :2] *= size[:, None] / size[None, :]
    return result


def stabilized(
    events: Iterator[DecodedMediaEvent],
    matrices: FloatArray,
    checkpoint: Callable[[], None],
) -> Iterator[DecodedMediaEvent]:
    index = 0
    for event in events:
        if not isinstance(event, DecodedVideoFrame):
            yield event
            continue
        checkpoint()
        if index >= len(matrices):
            raise InvalidRequest(
                "The source changed between stabilization passes.", fields=["video"]
            )
        picture = cv2.warpAffine(
            _rgb(event),
            matrices[index],
            (event.width, event.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        index += 1
        yield replace(event, rgb=picture.tobytes())
    if index != len(matrices):
        raise InvalidRequest("The source changed between stabilization passes.", fields=["video"])

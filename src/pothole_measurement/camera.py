"""Explicit camera calibration and cached rectification; no dataset paths."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np


def _key(array):
    array = np.ascontiguousarray(array, dtype=np.float64)
    return array.shape, array.tobytes()


def _array(key):
    shape, data = key
    return np.frombuffer(data, dtype=np.float64).reshape(shape)


@lru_cache(maxsize=8)
def _maps(model, size, raw_key, distortion_key, rect_key, map_type):
    raw, distortion, rect = map(_array, (raw_key, distortion_key, rect_key))
    if model == 'fisheye':
        return cv2.fisheye.initUndistortRectifyMap(raw, distortion, np.eye(3), rect, size, map_type)
    return cv2.initUndistortRectifyMap(raw, distortion, np.eye(3), rect, size, map_type)


@dataclass
class Camera:
    raw_k: np.ndarray
    distortion: np.ndarray
    model: str = 'fisheye'
    image_size: tuple[int, int] | None = None
    rect_k: np.ndarray | None = None
    balance: float = 0.0

    def __post_init__(self):
        self.raw_k = np.asarray(self.raw_k, dtype=np.float64)
        self.distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        if self.rect_k is not None:
            self.rect_k = np.asarray(self.rect_k, dtype=np.float64)
        for matrix in (self.raw_k, self.rect_k):
            if matrix is not None and (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0):
                raise ValueError('camera matrices must be finite 3 by 3 arrays with positive focal lengths')
        if self.model not in ('fisheye', 'pinhole'):
            raise ValueError('camera model must be fisheye or pinhole')
        if not np.isfinite(self.distortion).all() or (self.model == 'fisheye' and self.distortion.size != 4):
            raise ValueError('fisheye distortion must contain four finite coefficients')
        if self.model == 'pinhole' and self.distortion.size not in (4, 5, 8, 12, 14):
            raise ValueError('pinhole distortion needs 4, 5, 8, 12, or 14 coefficients')
        if not 0 <= self.balance <= 1:
            raise ValueError('camera balance must lie between zero and one')
        if self.image_size is not None:
            self.image_size = tuple(int(n) for n in self.image_size)
            if len(self.image_size) != 2 or min(self.image_size) < 2:
                raise ValueError('image_size must be [width, height], both at least two')

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(**{key: data[key] for key in ('raw_k', 'distortion', 'model', 'image_size', 'rect_k', 'balance') if key in data})

    def check_size(self, size):
        if self.image_size is not None and tuple(size) != self.image_size:
            raise ValueError(f'camera expects {self.image_size[0]}x{self.image_size[1]}, received {size[0]}x{size[1]}; supply calibration for this image size')

    def matrix(self, size):
        size = tuple(map(int, size))
        self.check_size(size)
        if self.rect_k is not None:
            return self.rect_k.copy()
        if self.model == 'fisheye':
            return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.raw_k, self.distortion, size, np.eye(3), balance=self.balance)
        if not self.distortion.any():
            return self.raw_k.copy()
        return cv2.getOptimalNewCameraMatrix(self.raw_k, self.distortion, size, self.balance, size)[0]

    def maps(self, size, k=None, *, map_type=cv2.CV_32FC1):
        size = tuple(map(int, size))
        self.check_size(size)
        return _maps(self.model, size, _key(self.raw_k), _key(self.distortion), _key(self.matrix(size) if k is None else k), int(map_type))

    def rectify_image(self, image, k=None):
        h, w = image.shape[:2]
        return cv2.remap(image, *self.maps((w, h), k, map_type=cv2.CV_16SC2), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def rectify_bbox(self, box, k):
        """Sample all four edges: four transformed corners miss fisheye extrema."""
        x0, y0, x1, y1 = box
        t = np.linspace(0, 1, 64)
        points = np.vstack([
            np.c_[x0+(x1-x0)*t, np.full_like(t,y0)],
            np.c_[x0+(x1-x0)*t, np.full_like(t,y1)],
            np.c_[np.full_like(t,x0), y0+(y1-y0)*t],
            np.c_[np.full_like(t,x1), y0+(y1-y0)*t],
        ]).reshape(-1, 1, 2)
        if self.model == 'fisheye':
            points = cv2.fisheye.undistortPoints(points, self.raw_k, self.distortion, R=np.eye(3), P=k)
        else:
            points = cv2.undistortPoints(points, self.raw_k, self.distortion, R=np.eye(3), P=k)
        points = points.reshape(-1, 2)
        if not np.isfinite(points).all():
            raise ValueError('box boundary rectification produced non-finite coordinates')
        return float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())

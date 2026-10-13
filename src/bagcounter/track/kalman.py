from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.linalg

F64 = npt.NDArray[np.float64]

_POS_W = 1.0 / 20
_VEL_W = 1.0 / 160


class KalmanXYAH:
    def __init__(self) -> None:
        self._F = np.eye(8)
        self._F[:4, 4:] = np.eye(4)
        self._H = np.eye(4, 8)

    def initiate(self, z: F64) -> tuple[F64, F64]:
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = np.array([2 * _POS_W * h, 2 * _POS_W * h, 1e-2, 2 * _POS_W * h,
                        10 * _VEL_W * h, 10 * _VEL_W * h, 1e-5, 10 * _VEL_W * h])
        return mean, np.diag(std**2)

    def predict(self, mean: F64, cov: F64) -> tuple[F64, F64]:
        h = mean[3]
        std = np.array([_POS_W * h, _POS_W * h, 1e-2, _POS_W * h, _VEL_W * h, _VEL_W * h, 1e-5, _VEL_W * h])
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + np.diag(std**2)
        return mean, cov

    def update(self, mean: F64, cov: F64, z: F64) -> tuple[F64, F64]:
        h = mean[3]
        r = np.diag(np.array([_POS_W * h, _POS_W * h, 1e-1, _POS_W * h]) ** 2)
        s = self._H @ cov @ self._H.T + r
        chol = scipy.linalg.cho_factor(s, lower=True, check_finite=False)
        gain = scipy.linalg.cho_solve(chol, (cov @ self._H.T).T, check_finite=False).T
        mean = mean + gain @ (z - self._H @ mean)
        cov = cov - gain @ s @ gain.T
        return mean, cov

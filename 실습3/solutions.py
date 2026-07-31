# -*- coding: utf-8 -*-
"""
solutions.py — 실습 3 빈칸 정답 (막히면 여기서 가져다 쓰고 계속 진행)

빈칸 1. refine_vp        : 직선 다발로부터 소실점 (SVD 최소 특이벡터)
빈칸 2. R_from_vps       : 소실점 2개 -> 회전행렬
빈칸 3. solve_t          : 대응점으로 평행이동 t
"""
import numpy as np

EPS = 1e-12


def _n(v):
    return np.asarray(v, float) / (np.linalg.norm(v) + EPS)


def _skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], float)


# --------------------------------------------------------------- 빈칸 1
def refine_vp(lines):
    """
    lines : (M,3) 정규화된 직선들 (같은 세계 방향을 공유)
    소실점 v 는 모든 l 에 대해 l^T v = 0  ->  min ||L v||, ||v||=1
    -> L 의 최소 특이값에 대응하는 우특이벡터
    """
    _, _, Vt = np.linalg.svd(np.asarray(lines, float))
    return Vt[-1]


# --------------------------------------------------------------- 빈칸 2
def R_from_vps(v1, v2, K):
    """
    v1 ~ K r1,  v2 ~ K r2   (r1, r2 = R 의 1,2 열)
      r1 = normalize(K^-1 v1)
      r2 = normalize(K^-1 v2)
      r3 = r1 x r2
    측정 잡음으로 완전 직교가 아니므로 SVD 로 SO(3) 에 투영.
    """
    Ki = np.linalg.inv(K)
    r1 = _n(Ki @ np.asarray(v1, float))
    r2 = _n(Ki @ np.asarray(v2, float))
    M = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


# --------------------------------------------------------------- 빈칸 3
def solve_t(R, K, img_pts, gnd_pts):
    """
    m = K^-1 x  (정규화 광선) 라 하면 m 은 (r1 X + r2 Y + t) 와 평행.
      m x (r1 X + r2 Y + t) = 0
      -> [m]_x t = -[m]_x (r1 X + r2 Y)
    [m]_x 는 rank 2 이므로 점당 독립식 2개. 미지수 3개 -> 최소 2점.
    """
    Ki = np.linalg.inv(K)
    A, b = [], []
    for (u, v), (X, Y) in zip(np.asarray(img_pts, float).reshape(-1, 2),
                              np.asarray(gnd_pts, float).reshape(-1, 2)):
        m = _n(Ki @ np.array([u, v, 1.0]))
        S = _skew(m)
        A.append(S)
        b.append(-S @ (R[:, 0] * X + R[:, 1] * Y))
    t, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
    return t

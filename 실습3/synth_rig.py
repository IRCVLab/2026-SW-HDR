# -*- coding: utf-8 -*-
"""
synth_rig.py — 실습용 합성 4카메라 Around-View 리그 생성기

외부 데이터 없이 실습이 자립하도록, 지면 텍스처와 카메라 영상을
전부 코드로 생성한다. 실데이터로 바꿀 때는 이 모듈만 교체하면 된다.

좌표계
  세계: 지면 Z=0, +X 전방, +Y 좌측, +Z 위
  카메라: x 우, y 아래, z 광축  (OpenCV 관례)
"""
import numpy as np
import cv2

TEX_PPM = 40                      # 지면 텍스처 해상도 (pixels per meter)
TEX_X = (-16.0, 16.0)
TEX_Y = (-16.0, 16.0)


# ------------------------------------------------------------------ 회전 도구
def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


# +X 를 바라보는 수평 카메라의 world->cam 회전
R_BASE = np.array([[0, -1, 0],
                   [0, 0, -1],
                   [1, 0, 0]], float)


def bev_matrix(x_range, y_range, ppm):
    """지면(X,Y,1) -> BEV 픽셀. +X=위, +Y=왼쪽"""
    x0, x1 = x_range
    y0, y1 = y_range
    W = int(round((y1 - y0) * ppm))
    H = int(round((x1 - x0) * ppm))
    M = np.array([[0.0, -ppm, y1 * ppm],
                  [-ppm, 0.0, x1 * ppm],
                  [0.0, 0.0, 1.0]])
    return M, (W, H)


def tex_matrix():
    return bev_matrix(TEX_X, TEX_Y, TEX_PPM)


# ------------------------------------------------------------------ 지면 텍스처
def make_ground_texture(seed=7):
    """주차장 느낌: 세계 X방향/Y방향 평행선 격자 + 아스팔트 노이즈 + 얼룩"""
    rng = np.random.default_rng(seed)
    M, (W, H) = tex_matrix()
    img = np.full((H, W, 3), 62, np.uint8)
    img = np.clip(img + rng.normal(0, 7, (H, W, 1)), 0, 255).astype(np.uint8)

    def g2p(X, Y):
        p = M @ np.array([X, Y, 1.0])
        return int(round(p[0])), int(round(p[1]))

    for Y in np.arange(-15, 15.01, 2.5):          # 세계 X 방향으로 뻗는 선들
        cv2.line(img, g2p(TEX_X[0], Y), g2p(TEX_X[1], Y), (225, 225, 225), 5, cv2.LINE_AA)
    for X in np.arange(-15, 15.01, 2.5):          # 세계 Y 방향으로 뻗는 선들
        cv2.line(img, g2p(X, TEX_Y[0]), g2p(X, TEX_Y[1]), (205, 210, 235), 5, cv2.LINE_AA)
    for _ in range(120):
        c = g2p(rng.uniform(-15, 15), rng.uniform(-15, 15))
        cv2.circle(img, c, int(rng.integers(3, 10)), (int(rng.integers(40, 90)),) * 3, -1)
    return img


# ------------------------------------------------------------------ 카메라
def make_cam(C, yaw_deg, pitch_deg, W=960, H=640, fov_deg=110.0, name=""):
    """
    C         : 카메라 중심 (월드)
    yaw_deg   : 광축 방위각 (0=전방, 90=좌측)
    pitch_deg : 하향 틸트 (+ 가 아래)
    """
    f = (W / 2) / np.tan(np.radians(fov_deg / 2))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], float)
    R = Rx(np.radians(pitch_deg)) @ R_BASE @ Rz(np.radians(yaw_deg)).T
    C = np.asarray(C, float)
    t = -R @ C
    Hg = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return dict(name=name, K=K, R=R, t=t, C=C, size=(W, H),
                yaw=yaw_deg, pitch=pitch_deg, H=Hg / Hg[2, 2])


def make_rig(seed=7):
    """차량 4카메라 리그 (전/좌/후/우). 높이·틸트를 조금씩 다르게."""
    rng = np.random.default_rng(seed)
    spec = [("front", 0, 3.4, 0.0), ("left", 90, 0.0, 0.9),
            ("rear", 180, -1.0, 0.0), ("right", -90, 0.0, -0.9)]
    cams = {}
    for name, yaw, x_off, y_off in spec:
        h = 0.95 + 0.15 * rng.random()
        cams[name] = make_cam([x_off, y_off, h], yaw,
                              pitch_deg=38 + 6 * rng.random(), name=name)
    return cams


def render(cam, tex, seed=0):
    """지면 텍스처 -> 카메라 영상. 카메라 뒤쪽(depth<=0)은 검게 마스킹."""
    rng = np.random.default_rng(seed)
    W, H = cam["size"]
    M, (TW, TH) = tex_matrix()
    Hi = np.linalg.inv(cam["H"])                  # 영상 -> 지면(X,Y)

    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)])
    g = Hi @ pix
    g = g / (g[2] + 1e-12)
    Xw = np.stack([g[0], g[1], np.zeros_like(g[0])])
    depth = (cam["R"] @ Xw + cam["t"][:, None])[2]

    tp = M @ np.stack([g[0], g[1], np.ones_like(g[0])])
    mx = (tp[0] / tp[2]).reshape(H, W).astype(np.float32)
    my = (tp[1] / tp[2]).reshape(H, W).astype(np.float32)
    img = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    bad = (depth.reshape(H, W) <= 0.1) | (mx < 0) | (my < 0) | (mx >= TW) | (my >= TH)
    img[bad] = 0
    return np.clip(img.astype(np.float32) + rng.normal(0, 3, img.shape),
                   0, 255).astype(np.uint8)


# ------------------------------------------------------------------ 대응점
def project(cam, gnd):
    from vp_avm import to_hom, from_hom
    return from_hom((cam["H"] @ to_hom(gnd).T).T)


def in_view(cam, pts_img, margin=20):
    W, H = cam["size"]
    return ((pts_img[:, 0] > margin) & (pts_img[:, 0] < W - margin) &
            (pts_img[:, 1] > margin) & (pts_img[:, 1] < H - margin))


def make_correspondences(cam, mode="cluster", noise=0.7, seed=0):
    """
    mode='wide'    : 전 영역에 분산된 4점  (DLT 이상 조건 — 현실에선 만들기 어려움)
    mode='cluster' : 카메라 앞 1.2m 사방 보드 4점 (현실적)

    주의: cluster 정사각 배치는 90° 회전에 대칭이라 4중 모호성 해소에
    불리하다. 실습에서는 앞의 2점만 사용해 대칭을 깬다.
    """
    rng = np.random.default_rng(seed)
    fwd = np.array([np.cos(np.radians(cam["yaw"])), np.sin(np.radians(cam["yaw"]))])
    org = cam["C"][:2] + fwd * 2.2
    if mode == "cluster":
        s = 0.6
        gnd = org + np.array([[-s, -s], [s, -s], [s, s], [-s, s]])
    else:
        gnd = []
        for r in (2.0, 7.0):
            for a in (-32, 32):
                d = np.array([np.cos(np.radians(cam["yaw"] + a)),
                              np.sin(np.radians(cam["yaw"] + a))])
                gnd.append(cam["C"][:2] + d * r)
        gnd = np.array(gnd)
    img = project(cam, gnd) + rng.normal(0, noise, (len(gnd), 2))
    keep = in_view(cam, img)
    return gnd[keep], img[keep]


# ------------------------------------------------------------------ 평가
def ground_rmse(H_est, cam, x_range=(-9, 9), y_range=(-9, 9), n=40):
    """영상->지면 역투영 오차 (미터). 카메라 시야 안 격자점만 평가."""
    from vp_avm import to_hom, from_hom
    X, Y = np.meshgrid(np.linspace(*x_range, n), np.linspace(*y_range, n))
    gnd = np.stack([X.ravel(), Y.ravel()], 1)
    pix = project(cam, gnd)
    m = in_view(cam, pix)
    Xc = (cam["R"] @ np.column_stack([gnd, np.zeros(len(gnd))]).T + cam["t"][:, None])[2]
    m &= Xc > 0.5
    if m.sum() < 10:
        return float("nan")
    back = from_hom((np.linalg.inv(H_est) @ to_hom(pix[m]).T).T)
    return float(np.sqrt(np.mean(np.sum((back - gnd[m]) ** 2, axis=1))))

# -*- coding: utf-8 -*-
"""
vp_avm.py — 소실점(Vanishing Point) 기반 지면 호모그래피 추정 + 멀티카메라 Around-View

핵심 아이디어
-------------
지면(Z=0) -> 영상 호모그래피는
        H_{g->i} = K [ r1  r2  t ]
로 분해된다. 여기서
  * 세계좌표 +X 방향 무한원점의 상  = K r1  = v_X  (소실점)
  * 세계좌표 +Y 방향 무한원점의 상  = K r2  = v_Y  (소실점)

즉 **소실점 2개가 H의 앞 두 열을 그대로 결정**한다.
  - 일반 DLT : 8-DoF -> 대응점 4개 (전 영역 분산 + non-collinear 필요)
  - VP 사용  : 4-DoF -> 대응점 2개
  - VP + 장착높이 : 2-DoF -> 대응점 1개

소실점은 "이 선들이 세계에서 평행하다"는 것만 알면 되고, 그 선의 실제 좌표는
몰라도 된다(주차선, 차선, 연석, 타일 줄눈 ...). 대응점보다 훨씬 얻기 쉽다.

모든 입력 점은 **왜곡 보정된(undistorted) 픽셀 좌표** 기준이다.
"""

import numpy as np
import cv2

EPS = 1e-12


# =============================================================================
# 0. 소도구
# =============================================================================
def to_hom(p):
    """(N,2) -> (N,3) 동차좌표"""
    p = np.asarray(p, float).reshape(-1, 2)
    return np.hstack([p, np.ones((len(p), 1))])


def from_hom(p):
    """(N,3) -> (N,2)"""
    p = np.asarray(p, float).reshape(-1, 3)
    return p[:, :2] / (p[:, 2:3] + EPS)


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], float)


def normalize(v):
    return np.asarray(v, float) / (np.linalg.norm(v) + EPS)


def project_to_SO3(M):
    """가장 가까운 회전행렬 (det=+1)"""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


# =============================================================================
# 1. 선분 검출 -> 소실점 추정
# =============================================================================
def detect_segments(img_gray, min_len=40, canny=(50, 150)):
    """
    선분 검출. LSD(있으면) -> HoughLinesP 폴백.
    반환: (N,4) [x1,y1,x2,y2]
    """
    segs = None
    try:  # OpenCV LSD (버전에 따라 없을 수 있음)
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines = lsd.detect(img_gray)[0]
        if lines is not None:
            segs = lines.reshape(-1, 4)
    except Exception:
        segs = None

    if segs is None or len(segs) < 8:
        edges = cv2.Canny(img_gray, canny[0], canny[1], apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=60,
                                minLineLength=min_len, maxLineGap=8)
        segs = np.zeros((0, 4)) if lines is None else lines.reshape(-1, 4).astype(float)

    if len(segs) == 0:
        return segs
    L = np.linalg.norm(segs[:, 2:4] - segs[:, 0:2], axis=1)
    return segs[L >= min_len]


def segments_to_lines(segs):
    """선분 -> 정규화된 직선 l = p1 x p2 (l[:2] 단위벡터)"""
    p1 = to_hom(segs[:, 0:2])
    p2 = to_hom(segs[:, 2:4])
    L = np.cross(p1, p2)
    n = np.linalg.norm(L[:, :2], axis=1, keepdims=True) + EPS
    return L / n


def _vp_angular_inliers(vp, segs, tol_deg=2.0):
    """선분 방향 vs (중점->VP) 방향의 각도 오차로 인라이어 판정."""
    mid = 0.5 * (segs[:, 0:2] + segs[:, 2:4])
    d = segs[:, 2:4] - segs[:, 0:2]
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + EPS)

    # VP가 무한원점(vp[2]~0)이면 방향벡터 자체가 vp[:2]
    if abs(vp[2]) < 1e-8:
        r = np.tile(vp[:2], (len(segs), 1))
    else:
        r = vp[:2] / vp[2] - mid
    r = r / (np.linalg.norm(r, axis=1, keepdims=True) + EPS)

    cos = np.abs(np.sum(d * r, axis=1)).clip(0, 1)
    ang = np.degrees(np.arccos(cos))
    return ang < tol_deg


def _refine_vp(lines):
    """인라이어 직선들로부터 min ||L v||, ||v||=1"""
    _, _, Vt = np.linalg.svd(lines)
    return Vt[-1]


def estimate_vps(segs, n_vp=3, tol_deg=2.0, iters=800, min_inliers=6, rng=None,
                 refine_fn=None):
    """
    RANSAC + greedy 로 지배적인 소실점들을 순차 검출.
    refine_fn : 직선다발 -> 소실점 함수 (실습 빈칸① 주입용). None 이면 내장 사용.
    반환: [(vp(3,), inlier_mask(N,)), ...]  (인라이어 수 내림차순)
    """
    _ref = refine_fn or _refine_vp
    rng = np.random.default_rng(0 if rng is None else rng)
    segs = np.asarray(segs, float)
    if len(segs) < 4:
        return []
    lines = segments_to_lines(segs)
    alive = np.ones(len(segs), bool)
    out = []

    for _ in range(n_vp):
        idx = np.flatnonzero(alive)
        if len(idx) < min_inliers:
            break
        best_n, best_mask = -1, None
        for _ in range(iters):
            i, j = rng.choice(idx, 2, replace=False)
            vp = np.cross(lines[i], lines[j])
            if np.linalg.norm(vp) < 1e-9:
                continue
            m = np.zeros(len(segs), bool)
            m[idx] = _vp_angular_inliers(vp, segs[idx], tol_deg)
            if m.sum() > best_n:
                best_n, best_mask = m.sum(), m
        if best_mask is None or best_n < min_inliers:
            break
        # 리파인 (2회 반복)
        vp = _ref(lines[best_mask])
        for _ in range(2):
            m = np.zeros(len(segs), bool)
            m[idx] = _vp_angular_inliers(vp, segs[idx], tol_deg)
            if m.sum() < min_inliers:
                break
            vp = _ref(lines[m])
            best_mask = m
        out.append((vp / (np.linalg.norm(vp) + EPS), best_mask.copy()))
        alive &= ~best_mask

    out.sort(key=lambda t: -t[1].sum())
    return out


def pick_orthogonal_pair(vps, K):
    """
    후보 소실점들 중 세계좌표에서 **직교**하는 방향쌍을 고른다.
    K^{-1}v 를 정규화한 방향벡터끼리 내적이 0에 가장 가까운 쌍.
    반환: (v_a, v_b, angle_deg)
    """
    Ki = np.linalg.inv(K)
    cand = [v for v, _ in vps]
    best = None
    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            da = normalize(Ki @ cand[a])
            db = normalize(Ki @ cand[b])
            ang = np.degrees(np.arccos(abs(np.dot(da, db)).clip(0, 1)))
            score = abs(90.0 - ang)
            if best is None or score < best[0]:
                best = (score, cand[a], cand[b], ang)
    if best is None:
        return None
    return best[1], best[2], best[3]


# =============================================================================
# 2. 소실점 -> R,  대응점 -> t
# =============================================================================
def R_from_vps(v1, v2, K):
    """v1 ~ K r1, v2 ~ K r2 로부터 회전행렬 (부호 미정 상태의 기본해)"""
    Ki = np.linalg.inv(K)
    r1 = normalize(Ki @ v1)
    r2 = normalize(Ki @ v2)
    return project_to_SO3(np.column_stack([r1, r2, np.cross(r1, r2)]))


def solve_t(R, K, img_pts, gnd_pts):
    """
    R,K 기지. m = K^{-1}x 라 할 때  m x (r1 X + r2 Y + t) = 0.
    -> [m]_x t = -[m]_x (r1 X + r2 Y)  : 점당 2개 독립식, 미지수 3
    최소 2점 필요.
    """
    Ki = np.linalg.inv(K)
    A, b = [], []
    for (u, v), (X, Y) in zip(np.asarray(img_pts, float).reshape(-1, 2),
                              np.asarray(gnd_pts, float).reshape(-1, 2)):
        m = normalize(Ki @ np.array([u, v, 1.0]))
        S = skew(m)
        A.append(S)
        b.append(-S @ (R[:, 0] * X + R[:, 1] * Y))
    A = np.vstack(A)
    b = np.concatenate(b)
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    return t


def solve_t_known_height(R, K, img_pts, gnd_pts, cam_height):
    """
    장착 높이 h 기지: 카메라 중심 C = -R^T t, C_z = h 고정.
    미지수는 (C_x, C_y) 2개  -> **대응점 1개면 충분**.
    """
    Ki = np.linalg.inv(K)
    A, b = [], []
    for (u, v), (X, Y) in zip(np.asarray(img_pts, float).reshape(-1, 2),
                              np.asarray(gnd_pts, float).reshape(-1, 2)):
        m = normalize(Ki @ np.array([u, v, 1.0]))
        S = skew(m)
        M = -S @ R                       # t = -R C 이므로 S t = -S R C
        A.append(M[:, :2])
        b.append(-S @ (R[:, 0] * X + R[:, 1] * Y) - M[:, 2] * cam_height)
    A = np.vstack(A)
    b = np.concatenate(b)
    Cxy, *_ = np.linalg.lstsq(A, b, rcond=None)
    C = np.array([Cxy[0], Cxy[1], cam_height])
    return -R @ C


def H_from_Rt(K, R, t):
    """지면(X,Y,1) -> 영상"""
    H = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return H / (H[2, 2] + EPS)


def reproj_error(H, img_pts, gnd_pts):
    pred = from_hom((H @ to_hom(gnd_pts).T).T)
    return float(np.mean(np.linalg.norm(pred - np.asarray(img_pts, float).reshape(-1, 2), axis=1)))


# =============================================================================
# 3. 전체 파이프라인
# =============================================================================
def estimate_H_vp(K, v_a, v_b, img_pts, gnd_pts, cam_height=None,
                  min_height=0.2, max_height=5.0,
                  yaw_prior_deg=None, yaw_tol_deg=50.0, refine=None):
    """
    소실점 2개 + 소수 대응점으로 H_{g->i} 추정.

    v_a, v_b 각각이 세계 +X / +Y 중 무엇인지 + 부호(±)가 미정 -> 8가지 후보.
    이를 아래 순서로 걸러낸다.
      (1) 지면법선 조건 : r3 = R e3 (= 카메라 좌표계에서 본 세계 +Z).
          카메라 y축이 아래를 향하므로 r3[1] < 0  -> 8 -> 4
      (2) cheirality    : 카메라 높이 범위, 관측점 depth > 0
      (3) 장착 방위 prior(yaw_prior_deg) : 차량 리그는 어느 카메라가
          전/후/좌/우인지 이미 알고 있음 -> 남은 4중 모호성 제거
      (4) 재투영오차 최소

    주의: 대응점 1개 + 높이 조건은 미지수=식 수라 잔차가 항상 0이 되어
    (4)로는 구분이 불가능하다. 이 경우 yaw_prior_deg 지정이 사실상 필수.

    refine : (R_from_vps_fn, solve_t_fn) 튜플. 실습 빈칸②③ 주입용.
    반환: dict(H, R, t, C, err, label_swapped, signs, n_candidates)
    """
    _Rfn, _tfn = refine if refine else (R_from_vps, solve_t)
    img_pts = np.asarray(img_pts, float).reshape(-1, 2)
    gnd_pts = np.asarray(gnd_pts, float).reshape(-1, 2)
    n = len(img_pts)
    if cam_height is None and n < 2:
        raise ValueError("소실점 방식은 대응점 2개 이상 필요 (높이를 주면 1개)")
    if cam_height is not None and n < 2 and yaw_prior_deg is None:
        raise ValueError("대응점 1개 모드는 yaw_prior_deg(장착 방위)가 필요")

    cands = []
    for swap in (False, True):
        va, vb = (v_b, v_a) if swap else (v_a, v_b)
        for s1 in (1, -1):
            for s2 in (1, -1):
                R = _Rfn(s1 * va, s2 * vb, K)
                # (1) 지면법선: 세계 +Z 는 카메라에서 '위쪽' = y 음수
                if R[1, 2] >= 0:
                    continue
                try:
                    if cam_height is not None:
                        t = solve_t_known_height(R, K, img_pts, gnd_pts, cam_height)
                    else:
                        t = _tfn(R, K, img_pts, gnd_pts)
                except np.linalg.LinAlgError:
                    continue
                C = -R.T @ t
                # (2) cheirality
                if not (min_height < C[2] < max_height):
                    continue
                Xc = (R @ np.column_stack([gnd_pts[:, 0], gnd_pts[:, 1],
                                           np.zeros(n)]).T).T + t
                if np.any(Xc[:, 2] <= 0.1):
                    continue
                # (3) 장착 방위 prior : 광축의 세계 XY 방위각
                fwd = R.T @ np.array([0.0, 0.0, 1.0])
                yaw = np.degrees(np.arctan2(fwd[1], fwd[0]))
                dyaw = abs((yaw - (yaw_prior_deg or 0.0) + 180) % 360 - 180)
                if yaw_prior_deg is not None and dyaw > yaw_tol_deg:
                    continue
                H = H_from_Rt(K, R, t)
                cands.append(dict(H=H, R=R, t=t, C=C,
                                  err=reproj_error(H, img_pts, gnd_pts),
                                  yaw=yaw, dyaw=dyaw,
                                  label_swapped=swap, signs=(s1, s2)))

    if not cands:
        raise RuntimeError("유효한 해 없음 (소실점/대응점/prior 확인)")
    # (4) 잔차가 사실상 동률이면 prior 근접도로 결정
    cands.sort(key=lambda c: (round(c["err"], 6), c["dyaw"]))
    best = cands[0]
    best["n_candidates"] = len(cands)
    return best


def estimate_H_dlt(img_pts, gnd_pts):
    """비교용 기준선: 일반 DLT (대응점 4개 이상, non-collinear, 넓게 분산 필요)"""
    img_pts = np.asarray(img_pts, np.float32).reshape(-1, 1, 2)
    gnd_pts = np.asarray(gnd_pts, np.float32).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(gnd_pts, img_pts, 0)
    return H / (H[2, 2] + EPS)


# =============================================================================
# 4. BEV 렌더링 / 블렌딩
# =============================================================================
def bev_matrix(x_range, y_range, ppm):
    """
    지면(X,Y) -> BEV 픽셀.  ppm = pixels per meter.
    BEV: +X(전방) = 위쪽, +Y(좌측) = 왼쪽  (차량 좌표계 관례)
    """
    x0, x1 = x_range
    y0, y1 = y_range
    W = int(round((y1 - y0) * ppm))
    H = int(round((x1 - x0) * ppm))
    # u = (y1 - Y)*ppm ,  v = (x1 - X)*ppm
    M = np.array([[0.0, -ppm, y1 * ppm],
                  [-ppm, 0.0, x1 * ppm],
                  [0.0, 0.0, 1.0]])
    return M, (W, H)


def warp_to_bev(img, H_g2i, M_bev, size):
    """영상 -> BEV.  H_{bev->i} = H_{g->i} @ M_bev^{-1}"""
    H_b2i = H_g2i @ np.linalg.inv(M_bev)
    return cv2.warpPerspective(img, np.linalg.inv(H_b2i), size,
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def feather_weight(mask, blur=31):
    """거리변환 기반 페더링 가중치"""
    d = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    d = cv2.GaussianBlur(d, (blur | 1, blur | 1), 0)
    return d


def render_around_view(images, Hs, x_range=(-8, 8), y_range=(-8, 8), ppm=40,
                       feather=41):
    """
    여러 카메라 영상을 하나의 Around-View 로 합성.
    images: [HxWx3], Hs: [지면->영상 호모그래피]
    """
    M, size = bev_matrix(x_range, y_range, ppm)
    W, Hh = size
    acc = np.zeros((Hh, W, 3), np.float32)
    wsum = np.zeros((Hh, W), np.float32)

    for img, Hg in zip(images, Hs):
        warped = warp_to_bev(img, Hg, M, size)
        valid = warp_to_bev(np.full(img.shape[:2], 255, np.uint8), Hg, M, size)
        w = feather_weight(valid, feather)
        acc += warped.astype(np.float32) * w[..., None]
        wsum += w

    out = acc / np.maximum(wsum, 1e-6)[..., None]
    out[wsum < 1e-6] = 0
    return np.clip(out, 0, 255).astype(np.uint8), M, size

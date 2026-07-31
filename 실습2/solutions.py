# solutions.py — 막힐 때만 여세요.
# 노트북의 빈칸(①②③)을 채우지 못했을 때 임시로 사용되는 정답 모음 +
# 선택 구현(A: RANSAC, B: 원통 투영)의 참조 구현입니다.
import numpy as np
import cv2

# ── 빈칸 ① : ratio test 임계값 ────────────────────────────────────────
DEFAULT_RATIO = 0.75

# ── 빈칸 ② : DLT의 A 행렬 2행 ────────────────────────────────────────
def build_A_rows(x, y, xp, yp):
    """대응 (x, y) → (x', y') 하나가 만드는 A의 두 행 (각 길이 9).

    x' × Hx = 0  (HZ 4.1, w = w' = 1) 에서:
      row1 = [ 0, 0, 0,  -x, -y, -1,   yp*x,  yp*y,  yp ]
      row2 = [ x, y, 1,   0,  0,  0,  -xp*x, -xp*y, -xp ]
    """
    row1 = [0.0, 0.0, 0.0, -x, -y, -1.0,  yp * x,  yp * y,  yp]
    row2 = [x,   y,   1.0,  0.0, 0.0, 0.0, -xp * x, -xp * y, -xp]
    return row1, row2

# ── 빈칸 ③ : 블렌딩 두 번째 가중치 ───────────────────────────────────
def second_weight(w1):
    """겹침 영역에서 영상 2의 가중치. 두 가중치의 합은 1이어야 한다."""
    return 1.0 - w1

# ── 선택 구현 A : RANSAC (참조) ──────────────────────────────────────
def my_ransac(src, dst, estimate_fn, iters=2000, thresh=3.0, seed=0):
    """최소 표본 4점 → 가설 H → 합의 집합 심사 → 반복.
    src, dst: (N, 2). estimate_fn(src4, dst4) -> H(3x3).
    반환: (best_H, inlier_mask)"""
    rng = np.random.default_rng(seed)
    N = len(src)
    best_inl, best_H = None, None
    src_h = np.hstack([src, np.ones((N, 1))])
    for _ in range(iters):
        idx = rng.choice(N, 4, replace=False)
        try:
            H = estimate_fn(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        if H is None or not np.isfinite(H).all():
            continue
        p = src_h @ H.T
        w = p[:, 2:3]
        w[np.abs(w) < 1e-12] = 1e-12
        err = np.linalg.norm(p[:, :2] / w - dst, axis=1)
        inl = err < thresh
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best_H = inl, H
    # 인라이어 전체로 재적합(정제)
    if best_inl is not None and best_inl.sum() >= 4:
        best_H = estimate_fn(src[best_inl], dst[best_inl])
    return best_H, best_inl

# ── 선택 구현 B : 원통 투영 (참조) ───────────────────────────────────
def cylindrical_warp(img, f):
    """초점거리 f[px] 기준 원통 좌표로 재투영 — 넓은 파노라마에서 늘어짐 완화."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    theta = (xs - cx) / f
    hcoord = (ys - cy) / f
    X, Y, Z = np.sin(theta), hcoord, np.cos(theta)
    map_x = (f * X / Z + cx).astype(np.float32)
    map_y = (f * Y / Z + cy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)

# ── 선택 구현 D : 대응 좌표 국소 미세조정 (참조) ─────────────────────
def refine_dst_local(src, dst, old_img, pano_img, steps=(8, 5, 3, 1.5),
                     top_frac=0.70):
    """육안 초깃값 dst를, 마스크된 엣지 상관을 목적함수로 힐클라임 정제.
    '선형/수동 초기값 -> 국소 최적화'라는 이 과정의 단골 패턴을 그대로 재현한다."""
    import numpy as np, cv2
    Ho, Wo = old_img.shape[:2]; Hp, Wp = pano_img.shape[:2]
    def edges(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
        g = cv2.GaussianBlur(g, (0, 0), 1.0)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
        m = np.sqrt(gx * gx + gy * gy); return (m / (m.max() + 1e-9)).astype(np.float32)
    Eo, Ep = edges(old_img), edges(pano_img)
    Mo = np.zeros((Ho, Wo), np.float32); Mo[:int(Ho * top_frac), :] = 1
    def score(d):
        H, _ = cv2.findHomography(np.float64(src), np.float64(d), 0)
        if H is None: return -1
        w = cv2.warpPerspective(Eo * Mo, H, (Wp, Hp))
        m = cv2.warpPerspective(Mo, H, (Wp, Hp)) > 0.5
        if m.sum() < 5000: return -1
        a, b = w[m], Ep[m]
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))
    dst = np.float64(dst).copy(); cur = score(dst)
    for step in steps:
        improved = True
        while improved:
            improved = False
            for i in range(len(dst)):
                for dx, dy in [(step,0),(-step,0),(0,step),(0,-step),
                               (step,step),(-step,-step),(step,-step),(-step,step)]:
                    cand = dst.copy(); cand[i] += (dx, dy)
                    s = score(cand)
                    if s > cur + 1e-4:
                        cur, dst = s, cand; improved = True
    return dst, cur

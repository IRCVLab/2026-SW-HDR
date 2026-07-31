# -*- coding: utf-8 -*-
"""
demo_vp_avm.py — 소실점 기반 Around-View 엔드투엔드 데모 (독립 실행)

  A) DLT / 4점 / 전 영역 분산     : 이상적 (현실에선 구축 곤란)
  B) DLT / 4점 / 근거리 한 곳 집중 : 현실적 -> 원거리 붕괴
  C) VP  / 2점 / 같은 근거리 패치  : 정상 복원
  D) VP  / 1점 + 장착높이          : 정상 복원

  python demo_vp_avm.py
"""
import os
import numpy as np
import cv2
import vp_avm as V
import synth_rig as S

OUT = "out"


def main():
    os.makedirs(OUT, exist_ok=True)
    tex = S.make_ground_texture()
    cv2.imwrite(f"{OUT}/00_ground_texture.png", tex)
    cams = S.make_rig()
    imgs = {n: S.render(c, tex, seed=i) for i, (n, c) in enumerate(cams.items())}

    keys = ["A DLT wide-4", "B DLT cluster-4", "C VP + 2pt", "D VP + 1pt + h"]
    print("=" * 92)
    print(" 지면 역투영 RMSE (m)  — 낮을수록 좋음")
    print("=" * 92)
    print(f"{'cam':<7}" + "".join(f"{k:>18}" for k in keys) + f"{'VP각':>8}")

    results = {}
    for name, cam in cams.items():
        cv2.imwrite(f"{OUT}/10_cam_{name}.png", imgs[name])
        gray = cv2.cvtColor(imgs[name], cv2.COLOR_BGR2GRAY)
        segs = V.detect_segments(gray, min_len=45)
        vps = V.estimate_vps(segs, n_vp=3, tol_deg=1.5)
        pair = V.pick_orthogonal_pair(vps, cam["K"])
        if pair is None:
            print(f"{name:<7} 소실점 검출 실패 (선분 {len(segs)}개)")
            continue
        v_a, v_b, ang = pair

        gw, iw = S.make_correspondences(cam, "wide", noise=0.7, seed=2)
        gc, ic = S.make_correspondences(cam, "cluster", noise=0.7, seed=1)
        yp = cam["yaw"]                     # 리그 장착 방위 prior (설계값)

        H = {}
        H[keys[0]] = V.estimate_H_dlt(iw, gw) if len(gw) >= 4 else None
        H[keys[1]] = V.estimate_H_dlt(ic, gc) if len(gc) >= 4 else None
        H[keys[2]] = V.estimate_H_vp(cam["K"], v_a, v_b, ic[:2], gc[:2],
                                     yaw_prior_deg=yp)["H"]
        H[keys[3]] = V.estimate_H_vp(cam["K"], v_a, v_b, ic[:1], gc[:1],
                                     cam_height=cam["C"][2], yaw_prior_deg=yp)["H"]
        e = {k: (S.ground_rmse(v, cam) if v is not None else np.nan)
             for k, v in H.items()}
        results[name] = (H, e)
        print(f"{name:<7}" + "".join(f"{e[k]:>18.3f}" for k in keys) + f"{ang:>7.1f}°")

        vis = imgs[name].copy()
        for (vp, mask), col in zip(vps[:2], [(0, 255, 255), (255, 120, 0)]):
            for s in segs[mask]:
                cv2.line(vis, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), col, 2)
        cv2.imwrite(f"{OUT}/20_vp_{name}.png", vis)

    if results:
        print("-" * 92)
        print(f"{'평균':<7}" + "".join(
            f"{np.nanmean([e[k] for _, e in results.values()]):>18.3f}" for k in keys))

    print("\nAround-View 렌더링...")
    for tag, key in [("GT", None), ("B_dlt_cluster", keys[1]), ("C_vp_2pt", keys[2]),
                     ("D_vp_1pt", keys[3])]:
        ims, Hs = [], []
        for name in results:
            Hh = cams[name]["H"] if key is None else results[name][0][key]
            if Hh is None:
                continue
            ims.append(imgs[name])
            Hs.append(Hh)
        avm, _, _ = V.render_around_view(ims, Hs, (-9, 9), (-9, 9), ppm=45, feather=61)
        cv2.imwrite(f"{OUT}/30_avm_{tag}.png", avm)
        print(f"  {OUT}/30_avm_{tag}.png")


if __name__ == "__main__":
    main()

# 실습 1 · 캘리브레이션 조건 비교 실험

실제 AVM 리그 **front_right 어안 카메라**의 ChArUco 촬영본으로, 촬영 설계(자세·거리 다양성, 커버리지)가
캘리브레이션 품질을 어떻게 좌우하는지 직접 검증합니다.

## 구성
```
lab1_calib_compare.ipynb   실습 노트북 (완성 코드 — 실행·관찰 중심)
board.json                 ChArUco 8x7 · DICT_5X5_1000 규격
lab1_data/
  setA/   10장  정면·거리 고정 (가설 1: f–Z 모호성)
  setB/   15장  중앙 관측만 — 노트북에서 중심 반경 0.6 코너만 사용 (가설 2: 외삽)
  setC/   15장  권장 — 자세·거리·위치 다양
  master/ 60장  마스터
reference/master_params.json  전체 233장 사전 캘리브레이션 (의사 참값)
requirements.txt
```

## 준비
전용 가상환경을 새로 만들어 쓰는 것을 권장합니다. (전역 파이썬에 OpenCV 5.x가 깔려 있으면
`aruco`·`fisheye` API가 바뀌어 노트북이 동작하지 않습니다.)
```bash
python3.10 -m venv .venv             # Python 3.10 권장 (3.9–3.12 동작)
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # OpenCV 4.10 등 검증된 버전 고정 설치
python -m ipykernel install --user --name lab1-calib \
       --display-name "Python 3.10 (lab1-calib)"

jupyter lab lab1_calib_compare.ipynb # 또는 VS Code에서 열기
```
- Jupyter/VS Code에서 **커널을 `lab1-calib`(또는 워크스페이스 `.venv`)로 선택**한 뒤 실행하세요.
- 셀 1의 출력이 `OpenCV 4.x`인지 먼저 확인하면 환경 오선택을 바로 잡을 수 있습니다.

셀 1부터 순서대로 실행하면 됩니다. 전체 소요 약 20–30초 (검출 캐시 이후 재실행은 수 초).
저사양 노트북은 셀 1의 `DOWNSCALE = 0.5` 주석을 해제하세요.
`det_cache/`는 검출 캐시이므로, OpenCV 버전을 바꾸면 삭제하고 재생성하세요.

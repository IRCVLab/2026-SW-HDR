# 실습 2 · 영상 정합과 파노라마 생성

**내 폰을 캘리브레이션하고, 내가 찍은 사진으로 파노라마를 만드는** 실습입니다.

## 구성
```
lab2_part1_calibration.ipynb   파트 1 — 내 카메라 캘리브레이션 → out/my_params.json
lab2_part23_panorama.ipynb     파트 2~3 — 매칭 ① · DLT ② · RANSAC · 워핑 · 블렌딩 ③ → 내 파노라마 · 모듈화
lab2_part4_timewarp.ipynb      파트 4 (보너스) — 시간을 잇는 호모그래피 "Dear Photograph"
lab2_pano.ipynb        (구버전) 위 세 노트북을 한 파일에 담은 통합본
solutions.py           빈칸 정답 + 선택 구현(RANSAC·원통 투영) 참조 — 막힐 때만!
board_screen.html      화면 표시용 ChArUco 보드 (전체화면 F, 체크리스트 내장) — 기본 사용
board/                 백업: A4 인쇄 PDF (반사 심한 환경용) — 실습 1과 동일 보드
my_calib/              <- 내 폰 보드 사진 10~12장 (안내문 참조)
my_pano/               <- 내 폰 파노라마 사진 3~4장 (안내문 참조)
data/hyu_*.jpeg        대체 데이터 (촬영이 여의치 않을 때 자동 사용)
data/timewarp/         보너스: 과거·현재 본관 사진 + 강사 예시 대응(corr_default.json)
corr_picker.html       보너스: 확대 클릭 대응점 툴 (브라우저 더블클릭 실행)
my_corr/               <- 툴에서 Export한 timewarp_corr.json 저장 위치
```

## 준비
```bash
pip install -r requirements.txt
jupyter lab lab2_part1_calibration.ipynb    # 이어서 파트 2~3 → 파트 4
```
- 세 노트북은 **각각 독립 실행**됩니다 (파트 1 → 파트 2~3 순서로 진행하는 것이 기본).
  - **빈칸 3개는 모두 파트 2~3**에 있습니다 (① 매칭 · ② DLT · ③ 블렌딩)
  - 파트 2~3은 `out/my_params.json`(파트 1 산출물)이 있으면 내 K로 왜곡 보정까지 적용, 없으면 생략하고 진행
  - 파트 4(보너스)는 파노라마 파이프라인이 필요 없습니다 — 첫 셀이 매칭기와 DLT만 준비하므로 언제든 단독 실행
- 빈칸이 비어 있어도 노트북은 solutions로 **임시 진행**됩니다(⚠ 표시). 빈칸을 채워 ⚠를 지우는 것이 목표.
- 보드는 board_screen.html을 노트북/태블릿 전체화면으로 띄워 촬영 (밝기 최대·반사 주의)
- 셀 2a 스모크테스트로 첫 1장을 판정한 뒤 나머지를 채우는 흐름
- 폰 사진은 EXIF 회전을 자동 반영하며, 처리 해상도는 긴 변 1600px로 통일됩니다.

## 체크포인트 (수업 진행)
- 파트 1 · 셀 2b — 내 폰 K, fx±σ 출력 (RMS<1, σ/fx<1% 목표)
- 파트 2~3 · 셀 4 — 매칭 시각화
- 파트 2~3 · 셀 6 — 검증 게이트: 내 DLT ≈ OpenCV
- 파트 2~3 · 셀 8 — 정합 겹침 확인
- 파트 2~3 · 셀 9 — 내 파노라마 완성

## 제출물 (LMS)
`out/` 폴더째: my_params.json · matches.png · inlier_layout.png · my_panorama.jpg · 검증 게이트 스크린샷

## 파트 4 (보너스) · 시간을 잇는 호모그래피
`corr_picker.html`을 브라우저로 열고 15점 프로토콜(페디먼트 3 + 기둥 6x상·하)대로 확대 클릭
→ JSON 내보내기 → `my_corr/timewarp_corr.json` 저장 → `lab2_part4_timewarp.ipynb` 실행.
과거 사진이 오늘의 본관 위에 정합됩니다. (툴은 실습 3 avm_picker의 미리보기판)

## 선택 과제 (테이크홈)
A. RANSAC 직접 구현 (solutions.my_ransac 참조)
B. 원통 투영 파노라마 — 내 fx를 f로 사용
C. 4장 이상 체인 파노라마
D. 시간 정합 다듬기 — 재클릭 또는 solutions.refine_dst_local

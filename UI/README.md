# UI: 3D 공간 탐색 뷰어

`lift3d`가 계산한 공구 3D 위치(`objects.json`)와 3DGS 결과(`point_cloud.ply`)를 브라우저에서 보면서,
공구 이름을 검색/선택하면 입구 시점에서 그 공구까지 카메라가 이동하는 뷰어. [viser](https://viser.studio/) 사용.

## 1. 환경 설치 (한 번만)

```bash
conda create -n viewer python=3.10 -y
conda activate viewer
pip install viser numpy scipy opencv-python-headless pyyaml
```

## 2. 필요한 입력

| 입력 | 예시 경로 | 비고 |
|---|---|---|
| 3DGS point cloud | `gs_data/point_cloud.ply` | `lift3d`와 같은 ply |
| COLMAP 모델 | `colmap/` (`cameras.bin`+`images.bin`, 또는 `.txt`) | 3DGS용 사진(`gs_data/images`) 기준 포즈 |
| 3D 위치 결과 | `runs/objects_3d_v2/objects.json` | `lift3d/lift_to_3d.py` 출력 |

## 3. 실행

```bash
python UI/app.py --ply gs_data/point_cloud.ply --colmap colmap --objects runs/objects_3d/objects.json \
  --location "1공학관 C302호_연구실창고"
```
터미널에 뜨는 `http://localhost:8080`을 브라우저로 열면 된다.

### 주요 옵션
- `--location`: 좌측 상단에 표시할 장소 이름
- `--home-image`: "처음 시점"으로 쓸 이미지 파일명 (안 주면 파일명 번호가 제일 작은 프레임 = 촬영 시작 지점)
- `--max-points` (기본 400000), `--min-opacity` (기본 0.05): 렌더링 성능용 가우시안 수 조절. 화질이 떨어져 보이면
  `--max-points 2000000 --min-opacity 0.0`로 전체 가우시안을 다 써보고 비교할 것 (viser의 `add_gaussian_splats`는
  아직 실험적 기능이라 깊이 정렬/시점별 색(SH 고차항)을 쓰지 않아, 전용 3DGS 뷰어보다 화질이 낮게 보일 수 있음).
- `--port` (기본 8080)

## 4. 사용법

- **목록에서 찾기 / 이름으로 검색**: 공구 선택 → 카메라가 입구 시점이 아니라 **현재 위치에서** 해당 공구의
  클로즈업 시점까지 이동 (약간 떠올랐다 내려오는 곡선 + 느려졌다 빨라지는 가속)
- **처음 시점**: `--home-image` 기준 위치로 복귀
- **이동 시간(초)**: 카메라 이동에 걸리는 시간
- **도착 후 좌우로 살짝 둘러보기**: 체크하면 도착 후 공구를 바라본 채로 ±17.5도 좌우로 천천히 흔들림 (끄면 도착한 자리에 고정)
- 도착하면 공구 위치에 노란 박스 + 이름표가 표시됨

## app_v2.py (재고관리 포함 버전)

같은 입력(`--ply`, `--colmap`, `--objects`)에 `--inventory` 하나가 추가된다. 공구별 수량을
기록해서 전부/일부/전량 반출 상태를 박스 색(초록/주황/빨강)과 배터리 모양 칸으로 표시하고,
반출/반납 기록을 남길 수 있다.

```bash
python UI/app_v2.py --ply gs_data/point_cloud.ply --colmap colmap --objects runs/objects_3d_v2/objects.json \
  --inventory runs/inventory.json --location "1공학관 C302호 연구실 창고"
```

`--inventory`로 준 파일이 없으면 공구마다 수량 1개로 새로 만든다. 수량은 그 파일의
`items.<공구>.total` 값을 직접 고치면 된다.

## 참고
- 3D 위치 계산(`lift3d`) 쪽 코드/방법은 `lift3d/README.md` 참고.
- 카메라/가우시안 수학은 `UI/app.py`, `UI/app_v2.py` 안에 전부 있고 (COLMAP 리더는 `lift3d/lift_to_3d.py`의 것을 재사용), 외부 라이브러리는 viser/numpy/scipy/opencv/pyyaml뿐이다.

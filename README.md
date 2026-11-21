# 공구 3D 위치 탐색 시스템

YOLO-Seg로 공구 위치를 찾고, CNN으로 종류(drill/hammer/clamp)를 분류하고, 3DGS 공간에
3D 위치를 계산해서 검색 UI로 보여주는 프로젝트. 전체 구조와 결과는 `pipeline.md` 참고.

## 폴더 구조

```
src/                   CNN 모델/데이터 파이프라인
train.py, eval.py      CNN 학습/평가
infer_2d.py            YOLO 결과 + CNN -> results.jsonl
yolo_export.py         YOLO만 실행해서 JSON으로 내보내기 (YOLO 담당자용)
prepare_single_class_dataset.py   3클래스 라벨 -> YOLO 학습용 1클래스 변환
evaluate_integration.py           YOLO+CNN 통합 결과를 정답과 비교
labeling/              박스+SAM으로 라벨 만드는 도구
lift3d/                2D 결과 + COLMAP + 3DGS -> 3D 위치(objects.json)
UI/                    3D 검색 뷰어 (app.py 기본형 / app_v2.py 재고관리 포함)
configs/               설정 파일
runs/                  학습/평가 결과물 (버전별로 폴더 구분)
```

## 실행 환경

conda 환경 여러 개로 나눠서 씀 (TensorFlow와 ultralytics를 같은 환경에 안 섞음). 자세한 건
`environment.md`, `UI/README.md`, `lift3d/README.md` 참고.

## 빠른 시작

```bash
python train.py --config configs/cnn.yaml
python eval.py --config configs/cnn.yaml --ckpt runs/cnn_rgb128_classroom_v2_final/best.h5 --split test
python infer_2d.py --source <이미지폴더> --yolo-weights runs/yolo_v2_final/best.pt \
  --config configs/cnn.yaml --ckpt runs/cnn_rgb128_classroom_v2_final/best.h5 --out <결과폴더>
```

## 구글 드라이브에서 따로 받아야 하는 파일

깃허브 용량 제한(100MB) 때문에 아래 파일들은 저장소에 없음. 구글 드라이브 링크는 별도 공유.

| 파일 | 용도 | 놓을 위치 |
|---|---|---|
| `point_cloud.ply` | 3DGS 모델 | `lift3d`/`UI` 실행 시 `--ply` |
| `cameras.bin`, `images.bin` | COLMAP 카메라 포즈 | `lift3d`/`UI` 실행 시 `--colmap` |
| 원본 라벨링 이미지셋 | CNN/YOLO 학습 데이터 원본 | `datasets/` 등 |

## 결과 요약

- CNN: test accuracy 95.6%, k-fold 평균 95.0%
- YOLO(단일클래스) + CNN 통합: GT 대비 97.22%
- 자세한 수치/이미지/아키텍처 설명은 `pipeline.md`

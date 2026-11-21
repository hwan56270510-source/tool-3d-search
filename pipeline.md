# 파이프라인 구조와 결과

## 1. 전체 구조: YOLO / CNN / 3DGS가 어떻게 연결되는가

```
사진 한 장
   │
   ▼
┌─────────────┐   박스 + 마스크 + YOLO conf (JSON)
│  YOLO-Seg   │ ─────────────────────────────────┐
│ (1클래스     │                                   │
│  "tool")     │   "여기 공구 같은 게 있다"만 찾음      │
└─────────────┘                                   ▼
                                           ┌─────────────┐
                                           │     CNN      │  ← 이 수업의 핵심
                                           │ drill/hammer │
                                           │ /clamp/other │
                                           └─────────────┘
                                                   │
                                   종류 + 확신도 (results.jsonl)
                                                   ▼
                                           ┌─────────────┐
                           COLMAP 카메라 ─▶│   lift3d     │◀─ 3DGS point_cloud.ply
                           (여러 시점)      │ (3D 위치계산) │
                                           └─────────────┘
                                                   │
                                        공구별 3D 위치 (objects.json)
                                                   ▼
                                              검색 UI
```

**YOLO는 "뭔지" 모른다.** 1클래스(`tool`)로만 학습해서 위치(박스)와 모양(마스크)만 찾는다.
공구 종류를 drill/hammer/clamp 중 무엇으로 볼지는 **전부 CNN이 결정**한다. YOLO가 찾은
crop 한 장 한 장이 CNN의 입력이고, CNN이 내놓는 클래스와 확신도가 그대로 `results.jsonl`에
들어가서 3D 단계로 넘어간다 — 즉 이 시스템에서 "이게 드릴이다/망치다/클램프다"라고 말하는
유일한 주체가 CNN이다. YOLO를 바꿔도, 3DGS를 바꿔도 종류 판단 로직은 그대로 CNN 안에 있다.

lift3d는 같은 공구가 여러 시점(여러 사진)에서 검출된 결과를 모아서, CNN이 매긴 확신도가
낮은 검출은 버리고 여러 시점에서 일치하는 것만 모아 3D 위치를 계산한다. 즉 3D 단계의 입력
품질은 CNN이 얼마나 정확히 분류했는지에 그대로 달려 있다.

### 왜 YOLO를 1클래스로 바꿨나
원래는 YOLO도 drill/hammer/clamp 3클래스로 학습했다. 그런데 YOLO의 역할은 애초에 "위치
찾기"뿐이고 종류 판단은 CNN이 하므로, YOLO를 3클래스로 학습하는 건 불필요한 일이었다.
1클래스로 바꾸니 YOLO가 공구를 놓치는 비율이 줄었고(특히 hammer), 결과적으로 CNN까지 이어진
전체 성공률이 올라갔다. CNN 자체는 두 경우 모두 건드리지 않았고, 매칭된 검출에 대한 분류
정확도도 100%로 동일했다 — 즉 이번 개선은 CNN의 입력 품질(검출률)을 높인 것이지 CNN의
성능을 바꾼 게 아니다.

## 2. 버전 구분

| 버전 | YOLO | 비고 |
|---|---|---|
| `objects_3d_legacy_v1` | 3클래스 YOLO | 처음 만든 3D 결과, 이후 버전과 비교용으로 남김 |
| `yolo_v2_final` | 1클래스 "tool", batch=2 | **최종 채택** |
| `yolo_v3_batch1_compare` | 1클래스 "tool", batch=1 | batch 비교용, V2와 거의 동일한 결과 |
| `objects_3d_v2` | yolo_v2_final 기준 | 최종 3D 위치 결과 |

CNN은 모든 버전에서 `runs/cnn_rgb128_classroom_v2_final/best.h5` 하나만 썼다(재학습 없음).

## 3. CNN 결과

### 최종 모델 (`runs/cnn_rgb128_classroom_v2_final/`)
| 이미지 | 의미 |
|---|---|
| `figures/class_distribution.png` | train/val/test 클래스별 개수. clamp가 제일 적어서 class_weight로 보정함 |
| `figures/training_curves.png` | 학습 곡선. val 데이터가 적어서(43장) 그래프 자체는 흔들리지만, 실제 성능은 아래 k-fold로 따로 확인함 |
| `figures/confusion_matrix_test_gt.png` | test 93장 기준 혼동행렬. drill 97%, hammer 96%, clamp 100%, other 93% |
| `figures/sample_grid_train.png`, `sample_grid_val.png` | 실제 학습/검증에 쓰인 crop 샘플 |
| `figures/augmentation_before_after.png` | bbox 흔들기/마스크 침식·팽창 증강 전후 비교 |
| `figures/filters_conv1.png`, `filters_conv2.png` | conv 레이어 필터 시각화 |
| `figures/feature_maps.png` | 입력 한 장에 대한 특징맵 |
| `figures/pred_correct_test_gt.png`, `pred_wrong_test_gt.png` | 맞은/틀린 예측 샘플 |

### k-fold 교차검증 (`runs/cnn_rgb128_margin05_noearlystop_fold1~4/`)
val 데이터가 적어 학습곡선만으로는 과적합 여부를 판단하기 어려워서, 분할을 4번 바꿔가며
같은 설정으로 재학습해 확인함.

| fold | test accuracy |
|---|---|
| 1 | 97.4% |
| 2 | 97.4% |
| 3 | 95.6% |
| 4 | 89.5% |
| 평균 | **95.0%** |

`fold1/figures/confusion_matrix_test_gt.png`가 대표 예시 — 분할을 바꿔도 drill/hammer/clamp가
꾸준히 잘 분류되는 걸 보여줌.

## 4. YOLO 결과

### 최종 채택 (`runs/yolo_v2_final/`, 1클래스 "tool", batch=2)
| 이미지 | 의미 |
|---|---|
| `results.png` | 100 epoch 학습 곡선. train/val loss가 끝까지 같이 떨어지고 안 벌어짐 → 과적합 아님 |
| `confusion_matrix.png`, `confusion_matrix_normalized.png` | val 기준 tool vs 배경 혼동행렬 |
| `test_confusion_matrix.png` | test 93장 기준 |
| `BoxPR_curve.png`, `MaskPR_curve.png` | Precision-Recall 곡선 |
| `best.pt` | 이 버전의 YOLO 가중치 |
| `integration/results.jsonl` | 이 YOLO + CNN을 test 93장에 돌린 결과 |
| `integration/eval_summary.txt` | 위 결과를 정답과 비교한 수치 (`evaluate_integration.py` 출력) |

test 93장(정답 72개) 기준 YOLO+CNN 통합 성공률: **97.22%** (70/72)

### batch 비교 (`runs/yolo_v3_batch1_compare/`, batch=1)
같은 데이터·설정으로 batch만 1로 바꿔 재학습. `results.png`/`confusion_matrix.png`를 V2와
비교하면 거의 차이가 없다 — 통합 성공률도 97.22%로 동일(실패한 이미지까지 같음). 학습 속도
때문에 batch=2를 최종으로 채택함.

### 1클래스 전환 전/후 비교 (`runs/comparisons/yolo_old_vs_new_comparison.png`)
3클래스로 학습했던 예전 YOLO와 이번 1클래스 YOLO를 같은 CNN에 연결해서 비교한 그래프.
drill/clamp는 차이 없고, hammer 검출률이 84% → 96%로 올라가면서 전체 성공률이
93.06% → 97.22%로 올랐다.

## 5. 3D 위치 결과

### 최종 (`runs/objects_3d_v2/`)
`drill_check.jpg`, `hammer_check.jpg`, `clamp_check.jpg` — 3DGS 포인트클라우드 위에서 각
공구에 해당하는 Gaussian을 찾아 박스로 표시한 확인용 이미지. `objects.json`에 공구별 3D
중심좌표/크기/대표 시점이 들어있다.

### 예전 버전 (`runs/objects_3d_legacy_v1/`)
3클래스 YOLO로 만들었던 첫 3D 결과. 지금은 `objects_3d_v2`로 대체됐지만 비교용으로 남겨둠.

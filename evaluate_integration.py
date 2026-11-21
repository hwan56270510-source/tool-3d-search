import argparse
import json
from pathlib import Path
from collections import Counter

# infer_2d.py가 만든 results.jsonl(YOLO 검출 + CNN 분류)을 원본 3클래스 정답 라벨과 비교해
# "YOLO가 찾고 CNN이 맞게 분류한 비율"을 계산한다.

parser = argparse.ArgumentParser()
parser.add_argument("--results", required=True, help="infer_2d.py 결과 results.jsonl")
parser.add_argument("--gt-labels", required=True, help="원본 3클래스 정답 라벨 폴더 (YOLO-seg 폴리곤)")
parser.add_argument("--iou", type=float, default=0.5)
args = parser.parse_args()

PRED = Path(args.results)
GT_DIR = Path(args.gt_labels)
CLASSES = ["drill", "hammer", "clamp"]


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(path, w, h):
    objects = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        values = line.split()
        if not values:
            continue
        cls_id = int(float(values[0]))
        coords = list(map(float, values[1:]))
        xs = [x * w for x in coords[0::2]]
        ys = [y * h for y in coords[1::2]]
        objects.append((CLASSES[cls_id], [min(xs), min(ys), max(xs), max(ys)]))
    return objects


total, detected, correct, confusion = Counter(), Counter(), Counter(), Counter()
missed, wrong, extra = [], [], []
images = 0

for line in PRED.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    record = json.loads(line)
    name = Path(record["image"]).name
    w, h = record["image_wh"]
    gt_path = GT_DIR / (Path(name).stem + ".txt")
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    gts = load_ground_truth(gt_path, w, h)
    preds = record["detections"]
    images += 1
    for cls, _ in gts:
        total[cls] += 1

    candidates = []
    for i, (cls, box) in enumerate(gts):
        for j, pred in enumerate(preds):
            iou = box_iou(box, pred["bbox_xyxy"])
            if iou >= args.iou:
                candidates.append((iou, i, j))

    used_gt, used_pred = set(), set()
    for iou, i, j in sorted(candidates, reverse=True):
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
        true_cls, pred_cls = gts[i][0], preds[j]["pred_class"]
        detected[true_cls] += 1
        confusion[(true_cls, pred_cls)] += 1
        if true_cls == pred_cls:
            correct[true_cls] += 1
        else:
            wrong.append((name, true_cls, pred_cls, round(iou, 3)))

    for i, (cls, _) in enumerate(gts):
        if i not in used_gt:
            missed.append((name, cls))
    for j, pred in enumerate(preds):
        if j not in used_pred:
            extra.append((name, pred["pred_class"], round(pred["yolo_conf"], 3)))

print("\n=== YOLO + CNN 통합 평가 ===")
print("평가 이미지:", images)
print("정답 객체:", sum(total.values()))
print("정답과 매칭된 검출:", sum(detected.values()))
print("올바르게 분류한 객체:", sum(correct.values()))
for cls in CLASSES:
    n, d, c = total[cls], detected[cls], correct[cls]
    if n and d:
        print(f"{cls}: 정답 {n}, 검출 {d}, 분류 정답 {c}, "
              f"검출 후 분류 정확도 {c/d:.2%}, 전체 정답 대비 성공률 {c/n:.2%}")
    else:
        print(f"{cls}: 정답 {n}, 검출 {d}, 분류 정답 {c}")

print("\n=== 혼동행렬 (매칭된 검출 기준) ===")
print("실제\\예측", *CLASSES, "other", sep="\t")
for true in CLASSES:
    print(true, *[confusion[(true, pred)] for pred in CLASSES + ["other"]], sep="\t")

print("\n=== 오분류 ===")
for item in wrong:
    print(item)
print("\n=== 미검출 ===")
for item in missed:
    print(item)
print("\n=== 추가/중복 검출 후보 ===")
for item in extra:
    print(item)

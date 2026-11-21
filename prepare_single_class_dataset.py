import argparse
from pathlib import Path
from collections import Counter
import shutil

# CNN용 3클래스(drill/hammer/clamp) 라벨 데이터셋을 YOLO 학습용 1클래스("tool") 데이터셋으로 변환.
# YOLO는 위치/마스크만 찾고 종류 분류는 CNN이 전담하므로, YOLO 학습은 1클래스로 충분하다.

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True, help="원본 3클래스 데이터셋 (train/val/test 각각 images/labels)")
parser.add_argument("--dst", required=True, help="출력 폴더 (이미 있으면 에러)")
args = parser.parse_args()

SRC = Path(args.src)
DST = Path(args.dst)
EXPECTED_SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_split(path):
    parts = [x.lower() for x in path.relative_to(SRC).parts]
    for split in EXPECTED_SPLITS:
        if split in parts:
            return split
    return None


def collect_files():
    images, labels = {}, {}
    for p in SRC.rglob("*"):
        if not p.is_file():
            continue
        split = get_split(p)
        if split is None:
            continue
        key = (split, p.stem)
        parents = [x.lower() for x in p.parent.parts]
        if p.suffix.lower() in IMAGE_EXTS and "images" in parents:
            if key in images:
                raise RuntimeError(f"중복 이미지: {key}")
            images[key] = p
        elif p.suffix.lower() == ".txt" and "labels" in parents:
            if key in labels:
                raise RuntimeError(f"중복 라벨: {key}")
            labels[key] = p
    return images, labels


def convert_label(path):
    result = []
    original_classes = Counter()
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) < 7 or (len(tokens) - 1) % 2 != 0:
            raise ValueError(f"잘못된 폴리곤: {path}:{line_no}")
        cls = int(tokens[0])
        if cls not in (0, 1, 2):
            raise ValueError(f"예상하지 못한 클래스 {cls}: {path}")
        coords = [float(v) for v in tokens[1:]]
        if not all(0 <= v <= 1 for v in coords):
            raise ValueError(f"좌표 범위 오류: {path}:{line_no}")
        original_classes[cls] += 1
        result.append("0 " + " ".join(tokens[1:]))  # drill/hammer/clamp 전부 tool(0)로 통합
    text = "\n".join(result)
    if text:
        text += "\n"
    return text, original_classes


def main():
    if not SRC.is_dir():
        raise FileNotFoundError(f"원본 폴더 없음: {SRC}")
    if DST.exists():
        raise FileExistsError(f"출력 폴더가 이미 존재합니다: {DST}")

    images, labels = collect_files()
    counts = Counter(split for split, _ in images)
    print("원본 이미지 수:", dict(counts))

    if images.keys() != labels.keys():
        missing = images.keys() - labels.keys()
        extra = labels.keys() - images.keys()
        raise RuntimeError(f"라벨 누락: {sorted(missing)[:10]}\n대응 이미지 없음: {sorted(extra)[:10]}")

    converted, class_counts, empty_count = {}, Counter(), Counter()
    for key, label_path in labels.items():
        text, classes = convert_label(label_path)
        converted[key] = text
        class_counts.update(classes)
        if not text:
            empty_count[key[0]] += 1

    for split in EXPECTED_SPLITS:
        (DST / "images" / split).mkdir(parents=True)
        (DST / "labels" / split).mkdir(parents=True)

    for (split, stem), image_path in sorted(images.items()):
        shutil.copy2(image_path, DST / "images" / split / image_path.name)
        (DST / "labels" / split / f"{stem}.txt").write_text(converted[(split, stem)], encoding="utf-8")

    yaml_text = (
        f'path: "{DST.as_posix()}"\n'
        "train: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames:\n  0: tool\n"
    )
    (DST / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("원본 클래스별 객체 수:", dict(class_counts))
    print("빈 라벨 이미지 수:", dict(empty_count))
    print("출력:", DST)


if __name__ == "__main__":
    main()

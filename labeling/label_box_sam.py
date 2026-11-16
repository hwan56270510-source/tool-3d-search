#!/usr/bin/env python3
"""
박스 → SAM 마스크 → YOLO-seg 라벨 GUI 도구

마우스로 물체에 박스를 그리면 SAM 2.1이 박스 안의 정확한 마스크를 만들고,
YOLO-seg 형식 라벨(polygon)로 저장한다. 한 이미지에 여러 개 박스 가능.

자동 제안 (기본 꺼짐, --auto 또는 실행 중 a 키로 켜기): 저장한 박스를 다음 사진으로 넘겨서, 라벨이 없는 다음 사진을 열면
  직전 박스를 넓혀(--propose_margin) SAM에 넣어 마스크를 미리 잡아준다 (하늘색, "auto").
  맞으면 Space로 저장하고 넘기고, 틀리면 u/r로 지우고 다시 그린다.
  크기가 직전 대비 1/4~4배를 벗어나는 제안은 자동으로 버린다 (엉뚱한 물체를 잡은 경우).
  연속 프레임(0.5~1.3초 간격)에서 물체가 화면 안에서 크게 안 움직일 때 효과적.

조작:
  왼쪽 드래그        박스 그리기 → SAM 마스크 생성
  Space / Enter / n  저장하고 다음 이미지 (박스 없으면 저장 안 하고 넘어감)
  e                  "물체 없음"으로 저장하고 다음 (음성 샘플, 빈 라벨 파일)
  b                  저장하고 이전 이미지
  u                  마지막 박스 취소      r  이 이미지 박스 전부 지우기
  0~9                현재 클래스 선택 (--names 순서)
  a                  자동 제안 켜기/끄기
  q / Esc            저장하고 종료 (다음 실행 때 라벨 안 된 이미지부터 이어서 시작)

출력 (--out 아래):
  labels/<stem>.txt    YOLO-seg 라벨: "cls x1 y1 x2 y2 ..." (0~1 정규화)
  ann/<stem>.json      박스 원본 (다시 열면 박스로 마스크 재생성 → 수정 가능)
  masks/<stem>.png     마스크 (0=배경, 255=물체) - 확인용
  previews/<stem>.jpg  오버레이 미리보기 - 확인용

사용 예 (모니터 앞 터미널에서, 자세한 설명은 labeling/README.md):
  python labeling/label_box_sam.py \
    --images <데이터>/yolo/images \
    --list labeling/lists/label_list.txt \
    --out <데이터>/det/labels \
    --names drill,hammer,clamp
"""
import argparse
import json
import os

import cv2
import numpy as np

COLORS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0)]


def mask_to_polygon(mask, eps=1.0):
    """마스크의 가장 큰 외곽선을 polygon으로. (YOLO-seg는 인스턴스당 polygon 1개)"""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, 0
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    poly = cv2.approxPolyDP(cnts[0], eps, True).reshape(-1, 2)
    # 가장 큰 조각 외에 의미 있는 조각(5% 이상)이 몇 개인지 - 가림으로 쪼개진 경우 경고용
    big = sum(cv2.contourArea(c) > 0.05 * cv2.contourArea(cnts[0]) for c in cnts[1:])
    return (poly if len(poly) >= 3 else None), big


class Labeler:
    def __init__(self, args):
        from ultralytics import SAM
        self.args = args
        self.sam = SAM(args.sam)
        self.names = args.names.split(",")
        self.out = args.out
        for d in ("labels", "ann", "masks", "previews"):
            os.makedirs(os.path.join(self.out, d), exist_ok=True)
        if args.list:
            self.files = [l.strip() for l in open(args.list) if l.strip()]
        else:
            self.files = sorted(f for f in os.listdir(args.images) if f.lower().endswith((".jpg", ".png")))
        self.cls = 0
        self.drag = None
        self.auto = args.auto  # 기본 꺼짐 (V3와 같은 동작). 실행 중 a 키로 켜기
        self.auto_note = ""  # 자동 제안을 버렸을 때 화면 안내
        self.carry = []  # 직전에 저장한 사진의 박스 [{"cls", "box", "area"}] → 다음 사진 자동 제안용

    @staticmethod
    def mask_box(m):
        ys, xs = np.nonzero(m)
        if len(xs) < 50:
            return None
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

    def grow(self, box, r):
        x1, y1, x2, y2 = box
        dx, dy = (x2 - x1) * r, (y2 - y1) * r
        return [max(0.0, x1 - dx), max(0.0, y1 - dy), min(self.W - 1.0, x2 + dx), min(self.H - 1.0, y2 + dy)]

    def propose(self):
        """직전 사진의 박스를 넓혀 SAM에 넣어 이번 사진의 마스크를 제안.
        크기가 급변하면(=뒤의 더 큰 물체를 잡음, V4 드릴 → 뒤 공구 케이스) 버리고 직접 그리게 함.
        그대로 두면 큰 박스가 다음 사진으로 넘어가며 눈덩이처럼 커진다."""
        self.auto_note = ""
        rejected = 0
        r0 = self.args.propose_margin
        for c in self.carry:
            # 넓히는 정도를 3가지로 시도 (카메라가 다가가 물체가 커지면 작은 박스는 물체를 다 못 덮음)
            #   → 크기 검사를 통과한 것 중 직전 크기와 가장 비슷한 마스크 선택
            masks = self.segment_multi(self.img, [self.grow(c["box"], r) for r in (r0, r0 * 2, r0 * 3.3)])
            pw, ph = c["box"][2] - c["box"][0], c["box"][3] - c["box"][1]
            best = None
            for m in masks:
                tb = self.mask_box(m)
                if tb is None:
                    continue
                ratio = m.sum() / max(1, c["area"])
                rw, rh = (tb[2] - tb[0]) / max(1.0, pw), (tb[3] - tb[1]) / max(1.0, ph)
                if 0.5 <= ratio <= 2.0 and 0.6 <= rw <= 1.7 and 0.6 <= rh <= 1.7:
                    score = abs(np.log(ratio))
                    if best is None or score < best[0]:
                        best = (score, m, tb)
            if best is None:
                rejected += 1
                continue
            self.inst.append({"cls": c["cls"], "box": self.grow(best[2], 0.03), "mask": best[1], "auto": True})
        if rejected:
            self.auto_note = f"auto: {rejected} proposal(s) rejected (size jump) - draw the box"

    def segment(self, img, box):
        return self.segment_multi(img, [box])[0]

    def segment_multi(self, img, boxes):
        """박스 여러 개를 한 번에 (이미지 분석은 1회) → 박스별 마스크"""
        r = self.sam(img, bboxes=boxes, verbose=False)[0]
        out = []
        for k in range(len(boxes)):
            if r.masks is None or k >= len(r.masks.data):
                out.append(np.zeros(img.shape[:2], bool))
                continue
            m = r.masks.data[k].cpu().numpy() > 0.5
            if m.shape != img.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            out.append(m)
        return out

    def stem(self, i):
        return os.path.splitext(self.files[i])[0]

    def load(self, i):
        self.img = cv2.imread(os.path.join(self.args.images, self.files[i]))
        self.H, self.W = self.img.shape[:2]
        self.scale = min(1.0, self.args.display_width / self.W)
        self.inst = []  # [{"cls", "box", "mask"}]
        self.status = None
        self.auto_note = ""
        p = os.path.join(self.out, "ann", self.stem(i) + ".json")
        if os.path.exists(p):
            a = json.load(open(p))
            self.status = a.get("status")
            for it in a.get("instances", []):
                self.inst.append({"cls": it["cls"], "box": it["box"], "mask": self.segment(self.img, it["box"])})
        elif self.auto and self.carry:
            self.propose()

    def save(self, i, status):
        s = self.stem(i)
        lines, union, warn = [], np.zeros((self.H, self.W), np.uint8), []
        prev = self.img.copy()
        for it in self.inst:
            poly, extra = mask_to_polygon(it["mask"])
            if poly is None:
                continue
            if extra:
                warn.append(f"cls{it['cls']}: 마스크가 {extra + 1}조각 - 가장 큰 조각만 라벨에 저장")
            lines.append(f"{it['cls']} " + " ".join(f"{x / self.W:.6f} {y / self.H:.6f}" for x, y in poly))
            union[it["mask"]] = 255
            col = COLORS[it["cls"] % len(COLORS)]
            prev[it["mask"]] = (0.5 * prev[it["mask"]] + 0.5 * np.array(col)).astype(np.uint8)
            cv2.polylines(prev, [poly.astype(np.int32)], True, (0, 255, 255), 2)
        with open(os.path.join(self.out, "labels", s + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        json.dump({"image": self.files[i], "width": self.W, "height": self.H, "status": status,
                   "instances": [{"cls": it["cls"], "box": [int(v) for v in it["box"]], "auto": bool(it.get("auto"))}
                                 for it in self.inst]},
                  open(os.path.join(self.out, "ann", s + ".json"), "w"), indent=1)
        # 다음 사진 자동 제안용으로 이번 박스를 넘겨줌 (물체 없음이면 비움)
        #   박스는 마스크에 딱 맞는 박스로 넘김 (넓힌 박스를 넘기면 매 장 조금씩 커지는 효과가 누적)
        self.carry = [{"cls": it["cls"], "box": self.mask_box(it["mask"]), "area": int(it["mask"].sum())}
                      for it in self.inst if self.mask_box(it["mask"]) is not None]
        cv2.imwrite(os.path.join(self.out, "masks", s + ".png"), union)
        cv2.imwrite(os.path.join(self.out, "previews", s + ".jpg"), cv2.resize(prev, None, fx=0.5, fy=0.5))
        for w in warn:
            print("  [warn]", w)
        print(f"[saved] {self.files[i]}  status={status}  instances={len(lines)}")

    def n_done(self):
        return sum(os.path.exists(os.path.join(self.out, "ann", self.stem(k) + ".json")) for k in range(len(self.files)))

    def draw(self, i, msg=""):
        d = self.img.copy()
        for it in self.inst:
            col = COLORS[it["cls"] % len(COLORS)]
            d[it["mask"]] = (0.55 * d[it["mask"]] + 0.45 * np.array(col)).astype(np.uint8)
            x1, y1, x2, y2 = map(int, it["box"])
            bc = (255, 255, 0) if it.get("auto") else (0, 255, 0)  # 자동 제안 = 하늘색
            cv2.rectangle(d, (x1, y1), (x2, y2), bc, 2)
            cv2.putText(d, self.names[it["cls"]] + (" (auto)" if it.get("auto") else ""), (x1, max(20, y1 - 8)), 0, 0.9, bc, 2)
        d = cv2.resize(d, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        if self.drag and self.drag[2] is not None:
            (x0, y0), (x1, y1) = self.drag[0], self.drag[2]
            cv2.rectangle(d, (x0, y0), (x1, y1), (0, 255, 255), 2)
        st = f"[{i + 1}/{len(self.files)}] {self.files[i]}  class={self.cls}:{self.names[self.cls]}  boxes={len(self.inst)}"
        st += f"  saved={self.status or '-'}  done={self.n_done()}/{len(self.files)}  auto={'ON' if self.auto else 'OFF'}"
        cv2.rectangle(d, (0, 0), (d.shape[1], 52), (0, 0, 0), -1)
        cv2.putText(d, st, (8, 20), 0, 0.55, (255, 255, 255), 1)
        msg = msg or self.auto_note
        cv2.putText(d, msg or "drag=box  space/n=save+next  e=no object  b=back  u=undo  r=reset  0-9=class  a=auto  q=quit",
                    (8, 42), 0, 0.5, (0, 255, 255), 1)
        return d

    def on_mouse(self, ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.drag = [(x, y), True, None]
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag and self.drag[1]:
            self.drag[2] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and self.drag:
            (x0, y0) = self.drag[0]
            self.drag = None
            if abs(x - x0) > 5 and abs(y - y0) > 5:
                s = self.scale
                box = [min(x0, x) / s, min(y0, y) / s, max(x0, x) / s, max(y0, y) / s]
                box = [max(0, box[0]), max(0, box[1]), min(self.W - 1, box[2]), min(self.H - 1, box[3])]
                self.inst.append({"cls": self.cls, "box": box, "mask": self.segment(self.img, box)})

    def run(self):
        i = 0
        if not self.args.start_from_first:
            while i < len(self.files) - 1 and os.path.exists(os.path.join(self.out, "ann", self.stem(i) + ".json")):
                i += 1
        win = "label_box_sam"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self.on_mouse)
        self.load(i)
        msg = ""
        while True:
            cv2.imshow(win, self.draw(i, msg))
            k = cv2.waitKey(30) & 0xFF
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                k = ord("q")
            if k == 255:
                continue
            msg = ""
            if k in (ord("q"), 27):
                if self.inst:
                    self.save(i, "labeled")
                break
            elif k in (ord(" "), 13, ord("n")):
                if self.inst:
                    self.save(i, "labeled")
                elif self.status is None:
                    print(f"[skip] {self.files[i]} (박스 없음, 저장 안 함 - 물체가 없으면 e)")
                if i < len(self.files) - 1:
                    i += 1
                    self.load(i)
                else:
                    msg = "Last image. Press q to quit."
            elif k == ord("e"):
                self.inst = []
                self.save(i, "empty")
                if i < len(self.files) - 1:
                    i += 1
                    self.load(i)
            elif k == ord("b"):
                if self.inst:
                    self.save(i, "labeled")
                if i > 0:
                    i -= 1
                    self.load(i)
            elif k == ord("u") and self.inst:
                self.inst.pop()
            elif k == ord("r"):
                self.inst = []
            elif ord("0") <= k <= ord("9") and k - ord("0") < len(self.names):
                self.cls = k - ord("0")
            elif k == ord("a"):
                self.auto = not self.auto
                msg = f"auto-propose {'ON' if self.auto else 'OFF'}"
        cv2.destroyAllWindows()
        print(f"[done] {self.n_done()}/{len(self.files)} images have annotations -> {self.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--list", default=None, help="라벨링할 파일명 목록 (없으면 폴더 전체)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default="drill", help="클래스 이름, 쉼표 구분 (예: drill,hammer)")
    local_sam = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "sam2.1_b.pt")
    ap.add_argument("--sam", default=local_sam if os.path.exists(local_sam) else "sam2.1_b.pt",
                    help="SAM 가중치. weights/sam2.1_b.pt 가 없으면 ultralytics가 처음 실행 때 자동으로 받는다")
    ap.add_argument("--display_width", type=int, default=1500, help="화면 표시 폭(px), 원본보다 크면 원본 크기")
    ap.add_argument("--start_from_first", action="store_true", help="이어하기 대신 첫 이미지부터")
    ap.add_argument("--auto", action="store_true", help="자동 제안 켜기 (기본 꺼짐, 실행 중 a 키로도 전환)")
    ap.add_argument("--propose_margin", type=float, default=0.15, help="자동 제안 시 직전 박스를 넓히는 비율")
    Labeler(ap.parse_args()).run()


if __name__ == "__main__":
    main()

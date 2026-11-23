import argparse
import datetime
import html as html_lib
import itertools
import os
import re
import sys
import threading
import time

import numpy as np
import viser
import viser.transforms as vtf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lift3d.lift_to_3d import read_colmap, qvec2rotmat  # noqa: E402
from src import utils  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--ply", required=True, help="3DGS point_cloud.ply")
parser.add_argument("--colmap", required=True, help="COLMAP 모델 폴더 (gs_data/sparse/0, .bin 또는 .txt)")
parser.add_argument("--objects", required=True, help="lift3d가 만든 objects.json")
parser.add_argument("--inventory", default="runs/inventory.json",
                    help="수량/반출 이력 저장 파일 (없으면 공구마다 수량 1개로 새로 만듦)")
parser.add_argument("--location", default="1공학관 C302호 연구실 창고", help="UI 상단에 표시할 장소 이름")
parser.add_argument("--home-image", default=None, help="'전체 보기'로 쓸 이미지 파일명 (안 주면 제일 앞 번호 프레임)")
parser.add_argument("--max-points", type=int, default=400000, help="렌더링할 최대 가우시안 수 (성능용 서브샘플링)")
parser.add_argument("--min-opacity", type=float, default=0.05)
parser.add_argument("--port", type=int, default=8080)
args = parser.parse_args()

SH_C0 = 0.28209479177387814

# 상태 색 (초록 = 전부 있음 / 주황 = 일부 반출 / 빨강 = 전량 반출)
COLOR_FULL = (99, 153, 34)
COLOR_PARTIAL = (239, 159, 39)
COLOR_EMPTY = (226, 75, 74)
COLOR_SEG_OFF = (180, 178, 169)
MAX_SEGMENTS = 10  # 배터리 칸 최대 개수 (수량이 더 많으면 비율로 10칸에 나눠 표시)
ARC_HEIGHT = 0.35  # 이동 중 떠오르는 높이 (이동 거리 대비). 원래 0.15

# 3D 라벨 카드 (시안 A)
COLOR_BRAND = (37, 99, 235)  # 검색/선택한 물품 강조색 (configure_theme 의 brand_color 와 같음)
COLOR_LEADER = (148, 163, 184)  # 선택 안 된 물품의 연결선 색
BRACKET_FRAC = 0.28  # 모서리 표시 길이 (박스 변 길이 대비)
CARD_GAP = 0.15  # 물체 오른쪽 모서리에서 카드까지 띄우는 거리 (물체 크기 대비)


# ----------------------------------------------------------------- 3DGS ply (xyz+회전+스케일+불투명도+색) 읽기
def read_gaussians_full(path):
    with open(path, "rb") as f:
        n, props = 0, []
        while (line := f.readline().decode().strip()) != "end_header":
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                t, name = line.split()[1:3]
                props.append((name, {"float": "<f4", "double": "<f8", "uchar": "u1"}[t]))
        a = np.fromfile(f, dtype=np.dtype(props), count=n)
    xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
    rot = np.stack([a["rot_0"], a["rot_1"], a["rot_2"], a["rot_3"]], 1).astype(np.float64)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True) + 1e-9
    scale = np.exp(np.stack([a["scale_0"], a["scale_1"], a["scale_2"]], 1).astype(np.float64))
    opacity = 1 / (1 + np.exp(-a["opacity"].astype(np.float64)))
    rgb = np.clip(0.5 + SH_C0 * np.stack([a["f_dc_0"], a["f_dc_1"], a["f_dc_2"]], 1), 0, 1)
    return xyz, rot, scale, opacity, rgb


def quats_to_covariances(rot_wxyz, scale):
    # cov = R @ diag(scale^2) @ R^T, 가우시안마다 3x3
    n = rot_wxyz.shape[0]
    w, x, y, z = rot_wxyz[:, 0], rot_wxyz[:, 1], rot_wxyz[:, 2], rot_wxyz[:, 3]
    R = np.stack([
        1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y,
        2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x,
        2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y,
    ], axis=1).reshape(n, 3, 3)
    S2 = scale ** 2
    cov = np.einsum("nij,nj,nkj->nik", R, S2, R)
    return cov


# ----------------------------------------------------------------- 로드
print("[load] ply ...")
xyz, rot, scale, opacity, rgb = read_gaussians_full(args.ply)
keep = opacity >= args.min_opacity
xyz, rot, scale, opacity, rgb = xyz[keep], rot[keep], scale[keep], opacity[keep], rgb[keep]
if len(xyz) > args.max_points:
    idx = np.random.default_rng(0).choice(len(xyz), args.max_points, replace=False)
    xyz, rot, scale, opacity, rgb = xyz[idx], rot[idx], scale[idx], opacity[idx], rgb[idx]
covariances = quats_to_covariances(rot, scale)
print(f"[load] 가우시안 {len(xyz)}개 렌더링")

print("[load] colmap ...")
cams, imgs = read_colmap(args.colmap)


def frame_num(name):
    m = re.findall(r"(\d+)", name)
    return int(m[-1]) if m else 1 << 30


home_name = args.home_image or min(imgs, key=frame_num)
scene_up = np.mean([-qvec2rotmat(q).T[:, 1] for q, _, _ in imgs.values()], axis=0)
scene_up /= np.linalg.norm(scene_up)


def image_pose(name):
    q, t, _ = imgs[name]
    R = qvec2rotmat(q)
    C = -R.T @ t
    look_at = C + R.T @ np.array([0.0, 0.0, 1.0])
    return C, look_at


home_pos, home_lookat = image_pose(home_name)
print(f"[load] 처음 시점 = {home_name}")

# 이동 중 떠오르는 높이가 천장을 뚫지 않도록, 촬영 당시 카메라가 실제로 도달했던 최대 높이로 제한
cam_centers = np.array([-qvec2rotmat(q).T @ t for q, t, _ in imgs.values()])
max_cam_height = float(np.percentile(cam_centers @ scene_up, 97))  # 이상치 약간 제외

print("[load] objects.json ...")
objects = utils.load_json(args.objects)["objects"]


def quat_from_axes(axes_rows):
    # axes_rows: 지역->세계 변환의 행벡터(axes.T 가 세계 기준 회전행렬)
    return vtf.SO3.from_matrix(np.asarray(axes_rows).T).wxyz


# ----------------------------------------------------------------- 재고 / 반출 이력 (파일에 저장)
inv_lock = threading.RLock()


def load_inventory():
    if os.path.exists(args.inventory):
        inv = utils.load_json(args.inventory)
    else:
        inv = {"items": {}, "logs": []}
    inv.setdefault("items", {})
    inv.setdefault("logs", [])
    for k in objects:
        inv["items"].setdefault(k, {"total": 1})
    return inv


def save_inventory():
    os.makedirs(os.path.dirname(os.path.abspath(args.inventory)), exist_ok=True)
    utils.save_json(inventory, args.inventory)


inventory = load_inventory()
save_inventory()
print(f"[load] 재고 파일 = {args.inventory}  (수량은 이 파일의 items.<공구>.total 을 고치면 됨)")


def total_of(key):
    return int(inventory["items"][key]["total"])


def open_logs(key):
    return [g for g in inventory["logs"] if g["key"] == key and not g.get("returned")]


def remaining_of(key):
    return total_of(key) - sum(int(g["qty"]) for g in open_logs(key))


def status_of(key):
    rem, tot = remaining_of(key), total_of(key)
    if rem <= 0:
        return "empty"
    if rem < tot:
        return "partial"
    return "full"


STATUS_COLOR = {"full": COLOR_FULL, "partial": COLOR_PARTIAL, "empty": COLOR_EMPTY}
STATUS_TEXT = {"full": "🟢 전부 있음", "partial": "🟠 일부 반출", "empty": "🔴 전량 반출"}


def battery_cells(key):
    """(채운 칸 수, 전체 칸 수). 수량이 MAX_SEGMENTS 보다 많으면 비율로 줄여서 표시."""
    rem, tot = max(remaining_of(key), 0), max(total_of(key), 1)
    if tot <= MAX_SEGMENTS:
        return rem, tot
    return int(round(MAX_SEGMENTS * rem / tot)), MAX_SEGMENTS


def today_str(days=0):
    d = datetime.date.today() + datetime.timedelta(days=days)
    return f"{d.month}월 {d.day}일"


def name_of(key):
    return objects[key].get("name_ko") or key


def josa(word, with_final, without_final):
    """받침 있으면 with_final(은/을), 없으면 without_final(는/를)."""
    ch = word[-1]
    if "가" <= ch <= "힣":
        return with_final if (ord(ch) - 0xAC00) % 28 else without_final
    return f"{with_final}({without_final})"


# ----------------------------------------------------------------- viser 서버
server = viser.ViserServer(port=args.port)
server.gui.configure_theme(show_logo=False, show_share_button=False, control_width="medium",
                           brand_color=(37, 99, 235))
server.scene.add_gaussian_splats("/gaussians", centers=xyz.astype(np.float32), covariances=covariances.astype(np.float32),
                                  rgbs=rgb.astype(np.float32), opacities=opacity[:, None].astype(np.float32))


def hex_color(rgb):
    return "#{:02X}{:02X}{:02X}".format(*rgb)


STATUS_LABEL = {"full": "보관 중", "partial": "일부 반출", "empty": "전량 반출"}
CHIP_STYLE = {"full": ("#EEF6E4", "#3B6D11"), "partial": ("#FEF3E2", "#8A4B08"), "empty": ("#FDECEC", "#B42318")}
ICON_PATHS = {
    "drill": '<path d="M4 6h10a3 3 0 0 1 0 6H4z"/><path d="M17 8h3M17 10h3"/>'
             '<path d="M8 12l-1 6h4l1-6"/><path d="M5 18h8v3H5z"/>',
    "hammer": '<path d="M10 5h7l3 3-2 2-2-2h-6z"/><path d="M12 8l-8 8 2 2 8-8"/>',
    "clamp": '<path d="M15 5H6v14h9"/><path d="M15 5v3"/><path d="M15 21V11"/><path d="M12 21h6"/>',
    "box": '<path d="M4 7l8-4 8 4v10l-8 4-8-4z"/><path d="M4 7l8 4 8-4M12 11v10"/>',
}
ICON_WORDS = {"drill": ("drill", "드릴"), "hammer": ("hammer", "망치"), "clamp": ("clamp", "클램프")}
USE_HTML_CARD = hasattr(server.gui, "add_html")  # 구버전 viser 면 마크다운 카드로 대체


def icon_kind(key):
    obj = objects[key]
    words = " ".join([key, str(obj.get("name_ko", ""))] + [str(a) for a in obj.get("aliases", [])]).lower()
    for kind, kws in ICON_WORDS.items():
        if any(w in words for w in kws):
            return kind
    return "box"


def obb_parts(obj):
    obb = obj["obb"]
    # axes: 행마다 박스의 지역 축을 세계 좌표로 쓴 것 (quat_from_axes 와 같은 규칙)
    return (np.asarray(obb["center"], dtype=np.float64), np.asarray(obb["extents"], dtype=np.float64),
            np.asarray(obb["axes"], dtype=np.float64))


def half_extent_along(obj, d):
    """방향 d 로 잰 박스의 반 길이."""
    _, ext, axes = obb_parts(obj)
    return 0.5 * float(np.sum(np.abs(axes @ d) * ext))


def bracket_segments(obj):
    """박스 꼭짓점 8개마다 세 변을 따라 짧게 그은 선 24개. 와이어프레임 대신 모서리만 표시."""
    center, ext, axes = obb_parts(obj)
    segs = []
    for s in itertools.product((-1.0, 1.0), repeat=3):
        s = np.asarray(s)
        p = center + axes.T @ (s * ext / 2)
        for i in range(3):
            segs.append([p, p - s[i] * axes[i] * ext[i] * BRACKET_FRAC])
    return np.asarray(segs)


def card_anchor(obj):
    """(물체 오른쪽 위 모서리, 카드 왼쪽 위 위치). '오른쪽'은 그 물체의 클로즈업 시점 기준.
    3D 카드는 기준점에서 오른쪽 아래로 펼쳐지므로, 물체 오른쪽에 두면 물체를 가리지 않음."""
    center, ext, _ = obb_parts(obj)
    cam = np.asarray(obj["closeup"]["position"], dtype=np.float64)
    look = np.asarray(obj["closeup"]["look_at"], dtype=np.float64)
    fwd = look - cam
    fwd /= np.linalg.norm(fwd) + 1e-9
    right = np.cross(fwd, scene_up)
    if np.linalg.norm(right) < 1e-6:  # 위에서 똑바로 내려다보는 시점일 때
        right = np.cross([0.0, 0.0, 1.0], scene_up)
    right /= np.linalg.norm(right)
    gap = float(np.clip(CARD_GAP * ext.max(), 0.01, 0.15))
    corner = center + right * half_extent_along(obj, right) + scene_up * half_extent_along(obj, scene_up)
    return corner, corner + right * gap + scene_up * gap * 0.5


def add_lines(name, segs, color, width_px):
    """viser 버전에 따라 선 굵기 인자 이름이 달라서(line_width → thickness) 둘 다 시도."""
    segs = np.asarray(segs, dtype=np.float32)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (segs.shape[0], 2, 1))
    try:
        return server.scene.add_line_segments(name, points=segs, colors=colors,
                                              thickness=width_px, thickness_units="screen")
    except TypeError:
        return server.scene.add_line_segments(name, points=segs, colors=colors, line_width=width_px)


def card_content(key):
    """시안 A 카드 내용. 재고 함수(remaining_of 등)를 오른쪽 패널과 똑같이 써서 숫자가 항상 일치함."""
    st = status_of(key)
    rem, tot = max(remaining_of(key), 0), total_of(key)
    logs = open_logs(key)
    note = ""
    if logs:
        latest = logs[-1]
        parts = [f"{sum(int(g['qty']) for g in logs)}개 반출"]
        if latest.get("who"):
            parts.append(html_lib.escape(str(latest["who"])))
        if latest.get("due"):
            parts.append(f"{html_lib.escape(str(latest['due']))} 반납 예정")
        note = " · ".join(parts)

    if not USE_HTML_CARD:
        md = f"### {name_of(key)}\n**{STATUS_TEXT[st]}** · 남은 수량 **{rem} / {tot}개**"
        return md + (f"\n\n{note}" if note else "")

    brand = hex_color(COLOR_BRAND)
    dot = hex_color(STATUS_COLOR[st])
    seg_off = hex_color(COLOR_SEG_OFF)
    chip_bg, chip_fg = CHIP_STYLE[st]
    on, n = battery_cells(key)
    bar = "".join(f'<div style="flex:1;height:8px;border-radius:4px;background:{dot if i < on else seg_off}"></div>'
                  for i in range(n))
    top = (f'<div style="height:4px;border-radius:2px;background:{brand};margin-bottom:12px"></div>'
           if selected["key"] == key else "")
    note_html = (f'<div style="margin-top:10px;font-size:13px;line-height:1.4;color:#475569">{note}</div>'
                 if note else "")
    rem_color = "#B42318" if st == "empty" else "#0F172A"
    return (
        f'<div style="padding:4px 2px 2px">{top}'
        f'<div style="display:flex;align-items:center;gap:10px">'
        f'<div style="width:44px;height:44px;flex:0 0 44px;border-radius:12px;background:#EFF4FF;'
        f'display:flex;align-items:center;justify-content:center">'
        f'<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="{brand}" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round">{ICON_PATHS[icon_kind(key)]}</svg></div>'
        f'<div style="flex:1;min-width:0">'
        f'<div style="font-size:26px;font-weight:700;line-height:1.15;color:#0F172A;white-space:nowrap;'
        f'overflow:hidden;text-overflow:ellipsis">{html_lib.escape(name_of(key))}</div>'
        f'<div style="display:flex;align-items:center;gap:6px;margin-top:4px">'
        f'<span style="display:inline-flex;align-items:center;gap:5px;padding:3px 9px;border-radius:999px;'
        f'background:{chip_bg};color:{chip_fg};font-size:13px;font-weight:700;white-space:nowrap">'
        f'<span style="width:7px;height:7px;border-radius:50%;background:{dot}"></span>{STATUS_LABEL[st]}</span>'
        f'<span style="font-size:12px;color:#64748B;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
        f'{html_lib.escape(key)}</span></div></div></div>'
        f'<div style="height:1px;background:#E2E8F0;margin:12px 0"></div>'
        f'<div style="display:flex;align-items:baseline;justify-content:space-between">'
        f'<span style="font-size:13px;font-weight:500;color:#475569">남은 수량</span>'
        f'<span style="font-size:22px;font-weight:700;color:{rem_color}">{rem}'
        f'<span style="font-size:15px;font-weight:500;color:#64748B"> / {tot}개</span></span></div>'
        f'<div style="display:flex;gap:4px;margin-top:8px">{bar}</div>{note_html}</div>'
    )


selected = {"key": None}  # 검색/선택한 물품. 오른쪽 패널·모서리 강조가 바로 씀
card_state = {"key": None}  # 카메라가 도착해서 카드를 띄워도 되는 물품. 이동 중엔 None
fill_handles, bracket_handles, leader_handles, card_handles, card_container_handles = {}, {}, {}, {}, {}


def draw_object(key):
    """공구 하나의 3D 표시(모서리 표시 + 연결선 + 라벨 카드)를 현재 재고·선택 상태로 다시 그림.
    반출/반납/검색 때마다 refresh_all 이 불러서, 카드 숫자가 오른쪽 패널과 같이 바뀜.
    모서리 강조(is_sel)는 선택하자마자, 카드(is_card)는 도착(카메라 이동 끝)한 뒤에만 보임."""
    obj = objects[key]
    center, ext, _ = obb_parts(obj)
    st = status_of(key)
    is_sel = selected["key"] == key
    is_card = card_state["key"] == key

    # 모서리 표시: 선택한 물품은 강조색으로 굵게, 나머지는 상태색
    if key in bracket_handles:
        bracket_handles[key].remove()
    bracket_handles[key] = add_lines(f"/obj_{key}/brackets", bracket_segments(obj),
                                     COLOR_BRAND if is_sel else STATUS_COLOR[st], 4.0 if is_sel else 2.5)

    # 전량 반출이면 반투명 빨간 박스를 겹쳐서 "화면엔 보이지만 실제론 없음"을 표시
    if key in fill_handles:
        fill_handles[key].visible = (st == "empty")
    else:
        fill_handles[key] = server.scene.add_box(f"/obj_{key}/fill", color=COLOR_EMPTY, dimensions=tuple(ext * 1.02),
                                                  opacity=0.85, cast_shadow=False, receive_shadow=False,
                                                  position=tuple(center), wxyz=quat_from_axes(obj["obb"]["axes"]),
                                                  visible=(st == "empty"))

    # 물체 오른쪽 위 모서리 → 카드를 잇는 선. 선택된 물품일 때만 보이게 함
    corner, anchor = card_anchor(obj)
    if key in leader_handles:
        leader_handles[key].remove()
    leader_handles[key] = add_lines(f"/obj_{key}/leader", [[corner, anchor]],
                                    COLOR_BRAND if is_sel else COLOR_LEADER, 2.0)
    leader_handles[key].visible = is_card

    # 라벨 카드: 처음 한 번 만들고, 그 뒤로는 내용·보임여부만 바꿈 (도착한 물품만 보임)
    content = card_content(key)
    if key in card_handles:
        card_handles[key].content = content
        card_container_handles[key].visible = is_card
    else:
        with server.scene.add_3d_gui_container(f"/obj_{key}/card", position=tuple(anchor), visible=is_card) as container:
            card_container_handles[key] = container
            card_handles[key] = (server.gui.add_html(content) if USE_HTML_CARD
                                 else server.gui.add_markdown(content))


for k in objects:
    draw_object(k)

server.scene.set_up_direction(tuple(scene_up))
server.initial_camera.position = tuple(home_pos)
server.initial_camera.look_at = tuple(home_lookat)

# ----------------------------------------------------------------- GUI
title_md = server.gui.add_markdown(f"### {args.location}")
summary_md = server.gui.add_markdown("")

key_by_name = {name_of(k): k for k in objects}
item_buttons = server.gui.add_button_group("물품", list(key_by_name.keys()))
home_btn = server.gui.add_button("전체 보기", color="gray", icon=viser.Icon.HOME)
status_md = server.gui.add_markdown("")

card_folder = server.gui.add_folder("선택한 물품", visible=False)
with card_folder:
    card_md = server.gui.add_markdown("")
    stock_bar = server.gui.add_progress_bar(100.0, color="green")
    history_md = server.gui.add_markdown("")

out_folder = server.gui.add_folder("반출하기", visible=False)
with out_folder:
    out_date = server.gui.add_text("반출일", today_str())
    out_qty = server.gui.add_number("수량", 1, min=1, step=1)
    out_who = server.gui.add_text("반출자", "")
    out_due = server.gui.add_text("반납 예정일", today_str(3))
    out_btn = server.gui.add_button("반출하기", color="red", icon=viser.Icon.LOGOUT)

ret_folder = server.gui.add_folder("반납하기", visible=False)
with ret_folder:
    ret_choice = server.gui.add_dropdown("반납할 기록", ["반출 중인 기록 없음"])
    ret_btn = server.gui.add_button("반납하기", color="green", icon=viser.Icon.LOGIN)

search_folder = server.gui.add_folder("이름으로 검색", expand_by_default=False)
with search_folder:
    search_text = server.gui.add_text("검색어", "", hint="드릴, hammer 처럼 한글/영문 둘 다 됨")
    search_btn = server.gui.add_button("검색", color="gray", icon=viser.Icon.SEARCH)

settings_folder = server.gui.add_folder("설정", expand_by_default=False)
with settings_folder:
    duration_slider = server.gui.add_slider("이동 시간(초)", min=1.0, max=10.0, step=0.5, initial_value=2.5)
    sway_checkbox = server.gui.add_checkbox("도착 후 좌우로 살짝 둘러보기", True)

ret_option_to_id = {}


def refresh_summary():
    out_count = sum(1 for k in objects if status_of(k) != "full")
    line = f"물품 {len(objects)}종 인식됨"
    if out_count:
        line += f" · 반출 중 {out_count}종"
    summary_md.content = line


def refresh_card():
    key = selected["key"]
    visible = key is not None
    card_folder.visible = out_folder.visible = ret_folder.visible = visible
    if not visible:
        return
    st = status_of(key)
    rem, tot = max(remaining_of(key), 0), total_of(key)
    card_md.content = (f"### {name_of(key)} ({key})\n"
                       f"**{STATUS_TEXT[st]}** · {rem}/{tot}개 보관 중")
    stock_bar.value = 100.0 * rem / max(tot, 1)
    stock_bar.color = {"full": "green", "partial": "orange", "empty": "red"}[st]

    logs = [g for g in inventory["logs"] if g["key"] == key][-5:][::-1]
    if logs:
        lines = ["**최근 이력**"]
        for g in logs:
            if g.get("returned"):
                tail = f"{g['returned']} 반납"
            else:
                tail = f"**반출 중** · ~{g['due']}" if g.get("due") else "**반출 중**"
            who = f" · {g['who']}" if g.get("who") else ""
            lines.append(f"- {g['out_date']} · {g['qty']}개{who} → {tail}")
        history_md.content = "\n".join(lines)
    else:
        history_md.content = "아직 반출 이력이 없어요."

    out_qty.max = max(rem, 1)
    out_btn.disabled = rem <= 0

    ret_option_to_id.clear()
    for g in open_logs(key):
        who = f" · {g['who']}" if g.get("who") else ""
        ret_option_to_id[f"{g['out_date']} · {g['qty']}개{who}"] = g["id"]
    if ret_option_to_id:
        ret_choice.options = list(ret_option_to_id.keys())
        ret_btn.disabled = False
    else:
        ret_choice.options = ["반출 중인 기록 없음"]
        ret_btn.disabled = True


def refresh_all(key=None, redraw_all=False):
    """오른쪽 패널과 3D 카드를 같은 재고 상태로 함께 갱신.
    key: 재고가 바뀐 물품(반출/반납) / redraw_all: 선택이 바뀌어서 카드 강조를 옮겨야 할 때."""
    with inv_lock:
        for k in (list(objects) if redraw_all else [key] if key is not None else []):
            draw_object(k)
        refresh_summary()
        refresh_card()


refresh_all()

# ----------------------------------------------------------------- 애니메이션 상태
state_lock = threading.Lock()
state = {"mode": "idle", "t0": 0.0, "dur": 2.5,
          "pos0": home_pos, "look0": home_lookat, "pos1": home_pos, "look1": home_lookat,
          "sway_center": None, "sway_offset": None, "sway_t0": 0.0, "active_key": None}
SWAY_DEG = 17.5
SWAY_PERIOD = 6.0


def ease(t):
    return t * t * (3 - 2 * t)


def start_move(pos1, look1, key=None):
    clients = list(server.get_clients().values())
    pos0 = clients[0].camera.position.copy() if clients else state["pos1"]
    look0 = clients[0].camera.look_at.copy() if clients else state["look1"]
    with state_lock:
        state.update(mode="moving", t0=time.time(), dur=duration_slider.value,
                      pos0=pos0, look0=look0, pos1=np.asarray(pos1, dtype=np.float64),
                      look1=np.asarray(look1, dtype=np.float64), active_key=key)


def goto_object(key):
    selected["key"] = key
    card_state["key"] = None  # 이동 중엔 카드 숨김, 도착하면 animate_loop 가 다시 켬
    refresh_all(redraw_all=True)
    c = objects[key]["closeup"]
    start_move(c["position"], c["look_at"], key=key)
    if status_of(key) == "empty":
        # 날아가서야 없다는 걸 알지 않도록 출발할 때 먼저 알려줌
        due = ", ".join(sorted({g["due"] for g in open_logs(key) if g.get("due")}))
        msg = f"⚠ **{name_of(key)}**{josa(name_of(key), '은', '는')} 지금 전부 반출 중이에요"
        status_md.content = msg + (f" (반납 예정 {due})" if due else "") + ". 원래 자리로 이동할게요."
    else:
        status_md.content = f"**{name_of(key)}** 위치로 이동 중"


def goto_home():
    selected["key"] = None
    card_state["key"] = None
    refresh_all(redraw_all=True)
    start_move(home_pos, home_lookat, key=None)
    status_md.content = "전체 보기로 이동 중"


@item_buttons.on_click
def _(event):
    goto_object(key_by_name[event.target.value])


@search_btn.on_click
def _(_):
    q = search_text.value.strip().lower()
    if not q:  # 빈 검색어는 모든 이름에 '포함'돼서 항상 첫 공구로 가던 버그 방지
        status_md.content = "검색어를 입력해 주세요"
        return
    hit = next((k for k in objects if q in k.lower() or q in str(objects[k].get("name_ko", "")).lower()
                or any(q in a.lower() for a in objects[k].get("aliases", []))), None)
    if hit:
        goto_object(hit)
    else:
        status_md.content = f"'{search_text.value}'{josa(search_text.value, '을', '를')} 찾을 수 없어요"


@home_btn.on_click
def _(_):
    goto_home()


@out_btn.on_click
def _(_):
    key = selected["key"]
    if key is None:
        return
    with inv_lock:
        qty = int(out_qty.value)
        rem = remaining_of(key)
        if qty < 1 or qty > rem:
            status_md.content = f"지금 남은 수량은 {rem}개예요. 그보다 많이 반출할 수 없어요"
            return
        if not out_date.value.strip():
            status_md.content = "반출일을 입력해 주세요"
            return
        new_id = max([g["id"] for g in inventory["logs"]], default=0) + 1
        inventory["logs"].append({"id": new_id, "key": key, "qty": qty, "who": out_who.value.strip(),
                                  "out_date": out_date.value.strip(), "due": out_due.value.strip(),
                                  "returned": None})
        save_inventory()
    out_qty.value = 1
    refresh_all(key)
    status_md.content = f"**{name_of(key)}** {qty}개 반출 기록됨"


@ret_btn.on_click
def _(_):
    key = selected["key"]
    log_id = ret_option_to_id.get(ret_choice.value)
    if key is None or log_id is None:
        return
    with inv_lock:
        for g in inventory["logs"]:
            if g["id"] == log_id:
                g["returned"] = today_str()
                qty = g["qty"]
        save_inventory()
    refresh_all(key)
    status_md.content = f"**{name_of(key)}** {qty}개 반납 기록됨"


@sway_checkbox.on_update
def _(_):
    if not sway_checkbox.value and state["mode"] == "sway":
        with state_lock:
            state["mode"] = "idle"


# ----------------------------------------------------------------- 메인 루프 (카메라 이동/둘러보기 연출)
def animate_loop():
    while True:
        time.sleep(1 / 30)
        with state_lock:
            mode, t0, dur = state["mode"], state["t0"], state["dur"]
            pos0, look0, pos1, look1 = state["pos0"], state["look0"], state["pos1"], state["look1"]
            active_key = state["active_key"]

        if mode == "moving":
            t = min(1.0, (time.time() - t0) / max(dur, 1e-6))
            e = ease(t)
            bump = ARC_HEIGHT * np.linalg.norm(pos1 - pos0) * np.sin(np.pi * t)
            pos = pos0 + (pos1 - pos0) * e + scene_up * bump
            over = float(pos @ scene_up) - max_cam_height  # 천장 위로 못 뚫게 누르기
            if over > 0:
                pos = pos - scene_up * over
            look = look0 + (look1 - look0) * e
            for c in server.get_clients().values():
                c.camera.position = pos
                c.camera.look_at = look
            if t >= 1.0:
                with state_lock:
                    if active_key is not None and sway_checkbox.value:
                        state["mode"] = "sway"
                        state["sway_center"] = objects[active_key]["closeup"]["look_at"]
                        state["sway_offset"] = pos1 - np.asarray(objects[active_key]["closeup"]["look_at"])
                        state["sway_t0"] = time.time()
                    else:
                        state["mode"] = "idle"
                if active_key is not None:
                    # 도착(둘러보기 시작)한 지금에서야 카드를 띄움
                    with inv_lock:
                        card_state["key"] = active_key
                        draw_object(active_key)
                    if status_of(active_key) == "empty":
                        status_md.content = f"**{name_of(active_key)}** 원래 자리 (현재 반출 중)"
                    else:
                        status_md.content = f"**{name_of(active_key)}** 도착"
                else:
                    status_md.content = "전체 보기"

        elif mode == "sway":
            with state_lock:
                center, offset, t0s = state["sway_center"], state["sway_offset"], state["sway_t0"]
            ang = np.radians(SWAY_DEG) * np.sin(2 * np.pi * (time.time() - t0s) / SWAY_PERIOD)
            k = scene_up / (np.linalg.norm(scene_up) + 1e-9)
            v = offset
            v_rot = v * np.cos(ang) + np.cross(k, v) * np.sin(ang) + k * (k @ v) * (1 - np.cos(ang))
            pos = np.asarray(center) + v_rot
            for c in server.get_clients().values():
                c.camera.position = pos
                c.camera.look_at = center


threading.Thread(target=animate_loop, daemon=True).start()
print(f"[viewer] http://localhost:{args.port} 에서 접속하세요")
server.sleep_forever()

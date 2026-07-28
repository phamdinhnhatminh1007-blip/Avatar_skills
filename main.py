"""
main.py — FILE CHINH: Hand Magic
================================
Chi lo phan "khung": webcam, nhan dien tay, ML doan chieu, dung scene 3D,
va DIEU PHOI chieu -> hieu ung. Toan bo hieu ung + logic tung chieu nam o
cac module rieng:

    Fireball3D.py  -> chieu "Xoe ban tay" (phun lua)
    lightning.py   -> chieu "Nam dam"    (set)

Muon them chieu moi: viet module hieu ung co ham cast(lm, hand_to_world),
roi them 1 dong vao bang SKILLS ben duoi. Khong can sua gi khac.

Chay: python main.py   (phim 'b' bat/tat bloom, ESC de thoat)
"""

import atexit
import os
import time
import math
import threading
from collections import deque, Counter

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAM_W, CAM_H = 960, 540
DETECT_W = 512
CAM_VIEW_SCALE = 0.78   # thu tron khung de thay rong den hai vai, khong crop mat noi dung


class _LatestFrameReader:
    """Mo + doc cam o thread rieng, song song voi luc thu vien/model dang tai."""

    def __init__(self):
        self.capture = None
        self.frame = None
        self.frame_id = 0
        self.error = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._read_loop,
            name="webcam-reader",
            daemon=True,
        )

    def start(self):
        self._thread.start()

    def _read_loop(self):
        # Dung camera 0 + backend mac dinh cua he dieu hanh. Khong ep DSHOW,
        # MJPG hay 16:9 de tranh driver crop/zoom khung hinh goc.
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            self.error = "Khong mo duoc webcam. Hay dong app dang dung camera."
            capture.release()
            return

        self.capture = capture
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self._stop.is_set():
            capture.release()
            return

        while not self._stop.is_set():
            ok, frame = capture.read()
            if ok:
                with self._lock:
                    self.frame = frame
                    self.frame_id += 1
            elif not self._stop.is_set():
                time.sleep(0.01)

    def latest(self):
        with self._lock:
            return self.frame_id, self.frame

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=0.5)
        if self.capture is not None:
            self.capture.release()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)


# Mo camera truoc cac import nang. MediaPipe, model ML va Ursina se duoc nap
# song song voi thoi gian driver camera khoi dong/lay frame dau tien.
_camera_reader = _LatestFrameReader()
_camera_reader.start()

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from panda3d.core import Texture as PandaTexture
from pathlib import Path
from ursina import (
    Ursina, Entity, Texture as UrsinaTexture, camera, window, color,
    application, Vec3, Text, destroy,
)

from gesture_utils import (
    FastMLP, GESTURES, is_lightning_pose, is_spiderman_pose,
    normalize_landmarks,
)
import Fireball3D
import lightning
import earth
import air
import ice

# Ursina tim texture/am thanh trong asset_folder -> tro ve thu muc du an
# (de sounds/*.wav luon tim thay du chay tu cwd nao).
application.asset_folder = Path(BASE_DIR)


# ======================================================================
# 1. WEBCAM + MEDIAPIPE (lay 21 landmarks ban tay)
# ======================================================================
detector = None
_cleanup_done = False
_t0 = time.monotonic()


def _cleanup_resources():
    """Dong webcam va MediaPipe dung mot lan, ke ca khi khoi tao app bi loi."""
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    if detector is not None:
        try:
            detector.close()
        except Exception as exc:
            print("Loi khi dong MediaPipe:", exc)
    try:
        if _camera_reader is not None:
            _camera_reader.stop()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


atexit.register(_cleanup_resources)

_hand_model = os.path.join(BASE_DIR, "hand_landmarker.task")
if not os.path.exists(_hand_model):
    raise FileNotFoundError(f"Khong tim thay model: {_hand_model}")

detector = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_hand_model),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,          # nhan dien ca 2 tay
        min_hand_detection_confidence=0.62,
        min_tracking_confidence=0.5,
    )
)


# ======================================================================
# 2. ML MODEL — doan chieu tu landmarks (co lam muot qua nhieu frame)
# ======================================================================
_fast_model = os.path.join(BASE_DIR, "gesture_model_fast.npz")
if os.path.exists(_fast_model):
    # Ban NumPy nay cho ket qua giong model sklearn, nhung khoi dong nhanh hon.
    clf = FastMLP.load(_fast_model)
else:
    import joblib
    clf = joblib.load(os.path.join(BASE_DIR, "gesture_model.joblib"))
CONF_THRESHOLD = 0.8       # duoi nguong nay coi nhu khong ro chieu
LIGHTNING_CONF_THRESHOLD = 0.64
SMOOTH_FRAMES = 7          # so frame de binh chon (chong nhap nhay)
LIGHTNING_GESTURE = 3
ICE_GESTURE = 5
_histories = {}            # lich su vote RIENG cho tung tay: {hand_id: deque}
_lightning_streaks = {}     # gate nhanh: chi can 2 frame hinh hoc lien tiep


def current_gesture(lm, hand_id):
    """Tra ve index chieu on dinh cho 1 tay, hoac None neu chua ro."""
    vec = normalize_landmarks(lm)
    # thu ca ban goc va ban lat guong, chon ket qua tin cay hon
    mirrored = vec.reshape(21, 2).copy()
    mirrored[:, 0] *= -1
    variants = np.stack((vec, mirrored.reshape(-1)))
    probs = clf.predict_proba(variants)
    vi = int(probs.max(axis=1).argmax())
    ci = int(probs[vi].argmax())
    label = int(clf.classes_[ci])
    conf = float(probs[vi, ci])

    exact_ice_pose = is_spiderman_pose(lm)
    exact_lightning_pose = is_lightning_pose(lm)
    threshold = (
        LIGHTNING_CONF_THRESHOLD
        if label == LIGHTNING_GESTURE
        else CONF_THRESHOLD
    )
    candidate = label if conf >= threshold else -1
    # Model cu hoc the tro + ut. The Spider-Man moi duoc gate hinh hoc nhan
    # truc tiep, nen khong can thu data/train lai chi vi them ngon cai.
    if exact_ice_pose:
        candidate = ICE_GESTURE
    elif candidate == ICE_GESTURE:
        candidate = -1
    if exact_lightning_pose:
        candidate = LIGHTNING_GESTURE

    hist = _histories.setdefault(hand_id, deque(maxlen=SMOOTH_FRAMES))
    hist.append(candidate)

    streak = _lightning_streaks.get(hand_id, 0)
    streak = min(3, streak + 1) if exact_lightning_pose else max(0, streak - 1)
    _lightning_streaks[hand_id] = streak
    if streak >= 2:
        return LIGHTNING_GESTURE

    vote, n = Counter(hist).most_common(1)[0]
    if vote == ICE_GESTURE and not exact_ice_pose:
        return None
    return vote if (vote != -1 and n >= SMOOTH_FRAMES // 2 + 1) else None


# ======================================================================
# 3. SCENE 3D (Ursina) — webcam lam nen, bat glow bloom
# ======================================================================
app = Ursina()
window.title = "Hand Magic"
window.color = color.black
camera.fov = 60         # goc 3D rong, thay tron ven cac luong phep lon

# Bloom de tat mac dinh vi mot so GPU/driver lam nen camera bi den.
# Hieu ung lua van phat sang nho additive blending; bam B neu muon bat bloom.
_bloom_enabled = False
BG_BRIGHTNESS = 1.0

Fireball3D.init_effects()  # tao texture hieu ung (sau khi Ursina() khoi dong)

# Nen 3D dan webcam
aspect = CAM_W / CAM_H
_fov = math.radians(camera.fov)
_bg_h = 2 * 50 * math.tan(_fov / 2)
background = Entity(parent=camera, model="quad",
                    scale=(_bg_h * aspect, _bg_h), z=50, double_sided=True)
_webcam_texture = PandaTexture("webcam")
_texture_size = (0, 0)
# Entity cua Ursina can wrapper Texture (co thuoc tinh `_texture`), khong nhan
# truc tiep panda3d.core.Texture.
background.texture = UrsinaTexture(_webcam_texture, filtering="bilinear")

# Mat phang xuat hien hieu ung
SPAWN_Z = 16
_spawn_h = 2 * SPAWN_Z * math.tan(_fov / 2)
_spawn_w = _spawn_h * aspect


def hand_to_world(hx, hy):
    """Toa do tay chuan hoa (0..1) -> toa do 3D tren mat phang spawn."""
    # Frame webcam duoc thu quanh tam; bien doi landmarks cung ti le de
    # hieu ung van bam chinh xac vao ban tay tren hinh da zoom-out.
    hx = 0.5 + (hx - 0.5) * CAM_VIEW_SCALE
    hy = 0.5 + (hy - 0.5) * CAM_VIEW_SCALE
    return Vec3((hx - 0.5) * _spawn_w, (0.5 - hy) * _spawn_h, SPAWN_Z)


def _update_background(frame_bgr):
    """Cap nhat cung mot texture Panda3D, tranh tao PIL/Texture moi moi frame."""
    global _texture_size, _spawn_w
    height, width = frame_bgr.shape[:2]
    if _texture_size != (width, height):
        _webcam_texture.setup2dTexture(
            width,
            height,
            PandaTexture.T_unsigned_byte,
            PandaTexture.F_rgba,
        )
        _texture_size = (width, height)
        # Ton trong aspect goc cua webcam (thuong 4:3 hoac 16:9) thay vi ep
        # frame vao 16:9. Mat nen khong bi phong/crop va phep van bam dung tay.
        frame_aspect = width / max(1, height)
        background.scale = (_bg_h * frame_aspect, _bg_h)
        _spawn_w = _spawn_h * frame_aspect

    if CAM_VIEW_SCALE < 1.0:
        view_w = max(2, int(width * CAM_VIEW_SCALE))
        view_h = max(2, int(height * CAM_VIEW_SCALE))
        view = cv2.resize(
            frame_bgr,
            (view_w, view_h),
            interpolation=cv2.INTER_AREA,
        )
        # Vien toi nhe giup khung zoom-out gon, khong ton chi phi blur moi frame.
        framed = np.empty_like(frame_bgr)
        framed[:] = (10, 8, 6)
        x0 = (width - view_w) // 2
        y0 = (height - view_h) // 2
        framed[y0:y0 + view_h, x0:x0 + view_w] = view
        frame_bgr = framed

    if BG_BRIGHTNESS != 1.0:
        frame_bgr = cv2.convertScaleAbs(frame_bgr, alpha=BG_BRIGHTNESS)
    pixels = cv2.cvtColor(
        np.ascontiguousarray(frame_bgr[::-1]),
        cv2.COLOR_BGR2BGRA,
    )
    _webcam_texture.setRamImage(pixels.tobytes())


def _set_bloom(enabled):
    """Bat/tat bloom an toan voi Ursina 7."""
    global _bloom_enabled
    if enabled:
        camera.shader = Fireball3D.bloom_shader
        camera.set_shader_input("threshold", 0.90)
        camera.set_shader_input("intensity", 1.7)
        camera.set_shader_input("blur_size", 0.013)
    elif _bloom_enabled:
        # Khong gan None luc khoi dong: Ursina se cleanup filter_manager chua tao.
        camera.shader = None
    _bloom_enabled = enabled


# Chu bao dang khoi dong (tu an khi co khung hinh dau tien)
loading_text = Text("Dang khoi dong camera...", origin=(0, 0), scale=1.8,
                    color=color.white, background=True)
_started = False
_last_frame_id = 0


# ======================================================================
# 4. BANG CHIEU: gesture index -> ham tung chieu
#    (moi module tu lo hieu ung + cooldown ben trong ham cast)
# ======================================================================
SKILLS = {
    0: Fireball3D.cast,    # Chieu 1 - Xoe ban tay  -> luong phun lua (Hoa)
    1: earth.cast,         # Chieu 2 - Nam dam      -> khien da (Tho)
    3: lightning.cast,     # Chieu 4 - Tro hoac chu V  -> set giang tu troi
    4: air.cast,           # Chieu 5 - Do ngon cai     -> loc xoay (Khi)
    5: ice.cast,           # Chieu 6 - The Spider-Man  -> cot bang chui len (Thuy)
    # them chieu moi: <index>: <module>.cast
}


# ======================================================================
# 5. VONG LAP CHINH
# ======================================================================
def update():
    global _started, _last_frame_id

    frame_id, frame = _camera_reader.latest()
    if frame is None:
        if _camera_reader.error and not _started:
            loading_text.text = _camera_reader.error
            loading_text.color = color.red
        return
    if frame_id == _last_frame_id:
        return
    _last_frame_id = frame_id
    frame = cv2.flip(frame, 1)

    # Nen 540p de mo/upload nhanh; MediaPipe dung ban 512p nhe hon.
    _update_background(frame)
    detect_h = max(2, int(round((DETECT_W * frame.shape[0] / frame.shape[1]) / 2) * 2))
    detect_frame = cv2.resize(
        frame, (DETECT_W, detect_h), interpolation=cv2.INTER_AREA,
    )
    rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)

    # Co khung hinh dau tien -> an chu "dang khoi dong"
    if not _started:
        destroy(loading_text)
        _started = True

    # Nhan dien tay
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect_for_video(mp_img, int((time.monotonic() - _t0) * 1000))

    hands = res.hand_landmarks or []
    handed = res.handedness or []
    if not hands:
        _histories.clear()
        _lightning_streaks.clear()
        return

    # Xu ly TUNG TAY doc lap
    seen = set()
    for i, lm in enumerate(hands):
        # khoa phan biet tay: dung Left/Right; neu trung thi them chi so
        label = handed[i][0].category_name if i < len(handed) and handed[i] else str(i)
        key = label if label not in seen else f"{label}{i}"
        seen.add(key)

        gesture = current_gesture(lm, key)
        skill = SKILLS.get(gesture)      # moi module tu xu ly cooldown theo key
        if skill:
            skill(lm, hand_to_world, key)

    # Xoa lich su cua tay khong con thay
    for k in list(_histories):
        if k not in seen:
            _histories[k].clear()
            _lightning_streaks.pop(k, None)


def input(key):
    if key == "escape":
        application.quit()
    elif key == "b":
        _set_bloom(not _bloom_enabled)


if __name__ == "__main__":
    print("Xoe=phun lua | Dam=khien | Tro/V=set | Cai=loc | Spider-Man=bang | ESC thoat.")
    try:
        app.run()
    finally:
        # Thoat "cung" bang os._exit de tranh loi destructor MediaPipe (Python 3.13)
        _cleanup_resources()
        os._exit(0)

"""
ice.py — MODULE HIEU UNG THUY / BANG (Icebending) (file phu)
============================================================
The Spider-Man (cai + tro + ut) -> mot DAY COT BANG dam tu long dat CHUI LEN quanh
tay: cac khoi bang sac moc hon loan theo ca chieu ngang va chieu sau, rung
man hinh cho uy luc, giu mot lat roi bi keo chim xuong.

Cach dung (trong main.py):
    import ice
    SKILLS = { 5: ice.cast, ... }     # 5 = "The Spider-Man"
"""

import math
import os
import random
import time
from copy import copy

from ursina import (
    Entity, camera, color, destroy, time as utime, Vec3, Audio, Mesh,
)

_SFX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")

# Ursina/Panda3D khong dam bao giu strong reference cho Entity Python tu tao.
# Neu chi goi IceSpike(...) ma khong luu lai, garbage collector co the xoa gan
# het cac chong va tren man hinh chi con Entity duoc tao cuoi cung.
_ACTIVE_ICE = {}


def _retain(entity):
    _ACTIVE_ICE[id(entity)] = entity


def _finish(entity):
    _ACTIVE_ICE.pop(id(entity), None)
    destroy(entity)


def _make_ice_crystal():
    """Tinh the bang 6 mat, than thuon dai va chop nhon ro tu moi goc nhin."""
    sides = 6
    rings = []
    ring_specs = (
        (0.00, 0.54),
        (0.34, 0.46),
        (0.68, 0.30),
    )
    phase_offsets = (0.00, 0.05, -0.035)
    for (height, radius), phase in zip(ring_specs, phase_offsets):
        ring = []
        for i in range(sides):
            angle = 2 * math.pi * i / sides + phase
            wobble = 1.0 if i % 2 == 0 else 0.88
            ring.append(Vec3(
                math.cos(angle) * radius * wobble,
                height,
                math.sin(angle) * radius * wobble,
            ))
        rings.append(ring)

    tip = Vec3(0.045, 1.0, -0.035)
    face_colors = (
        color.rgba32(92, 184, 230, 255),
        color.rgba32(205, 248, 255, 255),
        color.rgba32(126, 211, 244, 255),
        color.rgba32(232, 253, 255, 255),
        color.rgba32(105, 194, 236, 255),
        color.rgba32(174, 232, 250, 255),
    )

    vertices = []
    colors = []
    # Than thuon gom hai tang, moi mat co sac do rieng de bat canh tinh the.
    for ring_index in range(len(rings) - 1):
        lower = rings[ring_index]
        upper = rings[ring_index + 1]
        for i in range(sides):
            j = (i + 1) % sides
            face = [
                lower[i], lower[j], upper[j],
                lower[i], upper[j], upper[i],
            ]
            vertices.extend(face)
            colors.extend([face_colors[i]] * len(face))

    # Chop dai chiem 32% chieu cao, tao dang chong bang thay vi khoi da.
    for i in range(sides):
        j = (i + 1) % sides
        point = [rings[-1][i], rings[-1][j], tip]
        vertices.extend(point)
        colors.extend([face_colors[(i + 1) % sides]] * len(point))

        base = [rings[0][j], rings[0][i], Vec3(0, 0, 0)]
        vertices.extend(base)
        colors.extend([color.rgba32(76, 164, 218, 255)] * len(base))

    return Mesh(vertices=vertices, colors=colors, mode="triangle")


_CRYSTAL_MODEL = _make_ice_crystal()


def _make_ice_shard():
    """Manh vo dang kim tu thap nho, sac hon cube."""
    tip = Vec3(0, 0.95, 0)
    bottom = Vec3(0, -0.55, 0)
    rim = (
        Vec3(-0.32, 0, -0.20),
        Vec3(0.25, 0, -0.25),
        Vec3(0.30, 0, 0.19),
        Vec3(-0.22, 0, 0.27),
    )
    tints = (
        color.rgba32(225, 252, 255, 245),
        color.rgba32(122, 211, 246, 245),
        color.rgba32(184, 239, 255, 245),
        color.rgba32(91, 180, 230, 245),
    )
    vertices = []
    colors = []
    for i in range(len(rim)):
        j = (i + 1) % len(rim)
        vertices.extend((tip, rim[i], rim[j], bottom, rim[j], rim[i]))
        colors.extend([tints[i]] * 6)
    return Mesh(vertices=vertices, colors=colors, mode="triangle")


_SHARD_MODEL = _make_ice_shard()


def play_sound(name, volume=0.8):
    if not os.path.exists(os.path.join(_SFX_DIR, name)):
        return
    try:
        Audio(name, autoplay=True, loop=False, volume=volume)
    except Exception as e:
        print("Sound error:", e)


def _ice_color():
    return color.rgb32(random.randint(150, 185),
                       random.randint(205, 228),
                       random.randint(225, 240))    # xanh bang nhat


class IceShard(Entity):
    """Manh bang nhon toe ra cung luc tung dot chong moc len."""

    def __init__(self, position, delay=0.0, direction=None):
        self.full_scale = Vec3(
            random.uniform(0.18, 0.42),
            random.uniform(0.55, 1.15),
            random.uniform(0.16, 0.34),
        )
        super().__init__(
            # Mesh la NodePath: phai copy, neu dung chung no se bi reparent va
            # chi manh duoc tao cuoi cung con hien tren scene.
            model=copy(_SHARD_MODEL),
            color=color.white,
            scale=0.001,
            position=position,
            unlit=True,
            alpha=0,
            rotation=Vec3(
                random.uniform(-40, 40),
                random.uniform(0, 360),
                random.uniform(-40, 40),
            ),
        )
        _retain(self)
        direction = direction or Vec3(
            random.uniform(-1, 1), 0, random.uniform(-0.7, 0.7)
        )
        self.velocity = Vec3(
            direction.x * random.uniform(4.0, 8.5) + random.uniform(-1.5, 1.5),
            random.uniform(5.0, 10.5),
            direction.z * random.uniform(3.0, 7.0) + random.uniform(-1.0, 1.0),
        )
        self.spin = Vec3(random.uniform(-360, 360), random.uniform(-360, 360),
                         random.uniform(-360, 360))
        self.delay = delay
        self.life = random.uniform(0.55, 0.95)
        self.active = False

    def update(self):
        if not self.active:
            self.delay -= utime.dt
            if self.delay > 0:
                return
            self.active = True
            self.scale = self.full_scale
            self.alpha = 1

        self.velocity += Vec3(0, -16, 0) * utime.dt      # roi
        self.position += self.velocity * utime.dt
        self.rotation += self.spin * utime.dt
        self.life -= utime.dt
        if self.life < 0.25:
            self.alpha = max(0, self.life / 0.25)
            self.scale *= 0.92
        if self.life <= 0:
            _finish(self)


# ----------------------------------------------------------------------
# COT BANG — moc tu day len, giu, roi tut xuong + vo
# ----------------------------------------------------------------------
class IceSpike(Entity):
    def __init__(self, base_pos, height, thin, delay=0.0, hold=0.9,
                 tilt_x=None, tilt_z=None, depth_scale=None):
        # Ursina reparent Mesh truc tiep vao Entity, nen moi chong can mot
        # NodePath copy rieng de tat ca cung duoc render.
        super().__init__(
            model=copy(_CRYSTAL_MODEL),
            color=color.white,
            unlit=True,
        )
        _retain(self)
        self.base = Vec3(*base_pos)
        self.H = height
        self.thin = thin
        self.depth_scale = depth_scale or thin * random.uniform(0.72, 1.08)
        self.delay = delay
        self.hold = hold
        self.rise_dur = random.uniform(0.13, 0.21)
        self.sink_dur = random.uniform(0.42, 0.58)
        self.rotation = Vec3(
            tilt_x if tilt_x is not None else random.uniform(-12, 12),
            random.uniform(0, 360),
            tilt_z if tilt_z is not None else random.uniform(-12, 12),
        )
        self.state = "wait"
        self.t = 0.0
        self._set_h(0.001)

    def _set_h(self, h):
        """Khoi crystal co goc tai day, than day moc len va giu chop sac."""
        h = max(0.001, h)
        self.scale = Vec3(self.thin, h, self.depth_scale)
        self.position = Vec3(self.base.x, self.base.y, self.base.z)

    def update(self):
        dt = utime.dt
        self.t += dt
        if self.state == "wait":
            if self.t >= self.delay:
                self.state, self.t = "rise", 0.0
        elif self.state == "rise":
            k = min(1, self.t / self.rise_dur)
            # Ease-out-back: bung vuot nhe roi dung lai, tao cam giac dam xuyen dat.
            q = k - 1.0
            eased = 1.0 + 2.25 * q ** 3 + 1.25 * q ** 2
            self._set_h(self.H * eased)
            if k >= 1:
                self.state, self.t = "hold", 0.0
        elif self.state == "hold":
            if self.t >= self.hold:
                self.state, self.t = "fall", 0.0
        elif self.state == "fall":
            # Giu nguyen hinh nhon va keo ca tinh the chim xuong, nhanh dan.
            k = min(1, self.t / self.sink_dur)
            eased = k * k
            self.position = Vec3(
                self.base.x,
                self.base.y - self.H * 1.08 * eased,
                self.base.z,
            )
            self.alpha = max(0.0, 1.0 - max(0.0, k - 0.68) / 0.32)
            if k >= 1:
                _finish(self)


# ----------------------------------------------------------------------
# TUNG CHIEU
# ----------------------------------------------------------------------
def cast_ice(center):
    """Tao mot song chong bang nhieu lop moc lan tu tam ra ngoai."""
    center = Vec3(*center)
    ground_y = center.y - 5.15

    camera.shake(duration=0.38, magnitude=0.92, speed=0.018)

    # 8 hang tao ca mot "rung bang" trong CHI MOT LAN cast. Moi hang moc tre
    # hon hang truoc de thay ro
    # chuyen dong lan tren mat dat, thay vi tat ca chong chong len mot diem.
    rows = 8
    for row in range(rows):
        count = 7 + row * 2
        row_width = 2.2 + row * 1.25
        row_depth = (row - 3.25) * 0.62
        row_delay = row * 0.047
        height_falloff = 1.0 - row / (rows * 1.35)

        for slot in range(count):
            u = 0.0 if count == 1 else slot / (count - 1) * 2.0 - 1.0
            x = center.x + u * row_width + random.uniform(-0.34, 0.34)
            z = center.z + row_depth + random.uniform(-0.42, 0.42)
            y = ground_y + random.uniform(-0.18, 0.16)

            dx = x - center.x
            dz = z - center.z
            length = max(0.001, math.sqrt(dx * dx + dz * dz))
            outward_x, outward_z = dx / length, dz / length
            tilt = random.uniform(7, 19) + abs(u) * random.uniform(5, 12)
            delay = row_delay + abs(u) * 0.018 + random.uniform(0.0, 0.025)
            height = (
                random.uniform(4.2, 6.8) * height_falloff
                + (1.0 - abs(u)) * random.uniform(1.0, 2.7)
            )
            width = random.uniform(0.48, 0.84) * (0.92 + height / 11.0)

            IceSpike(
                Vec3(x, y, z),
                height,
                width,
                delay=delay,
                hold=random.uniform(1.0, 1.35),
                tilt_x=outward_z * tilt + random.uniform(-4, 4),
                tilt_z=-outward_x * tilt + random.uniform(-4, 4),
                depth_scale=width * random.uniform(0.68, 0.94),
            )

            # Manh vo toe ra co chon loc de day dan nhung khong qua nang FPS.
            if (row + slot) % 2 == 0:
                shard_origin = Vec3(
                    x,
                    y + random.uniform(0.15, 0.65),
                    z,
                )
                direction = Vec3(outward_x, 0, outward_z)
                IceShard(shard_origin, delay=delay + random.uniform(0.02, 0.09),
                         direction=direction)

    # Vong chong nho ben ngoai lam vien "toe" khong deu, tranh dang luoi qua ngay ngan.
    for _ in range(36):
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(5.2, 11.8)
        dx = math.cos(angle) * radius
        dz = math.sin(angle) * radius * 0.36
        delay = random.uniform(0.20, 0.38)
        height = random.uniform(1.8, 3.8)
        width = random.uniform(0.26, 0.52)
        IceSpike(
            Vec3(
                center.x + dx,
                ground_y + random.uniform(-0.25, 0.12),
                center.z + dz,
            ),
            height,
            width,
            delay=delay,
            hold=random.uniform(0.82, 1.18),
            tilt_x=math.sin(angle) * random.uniform(18, 34),
            tilt_z=-math.cos(angle) * random.uniform(18, 34),
            depth_scale=width * random.uniform(0.65, 0.9),
        )


# ----------------------------------------------------------------------
# LOGIC TUNG CHIEU — main.py chi goi cast(lm, hand_to_world)
# ----------------------------------------------------------------------
ICE_COOLDOWN = 1.3           # giay giua 2 lan dung bang
_last_cast = {}


def cast(lm, hand_to_world, hand_id=""):
    """The Spider-Man: day cot bang nhon chui len (cooldown rieng moi tay)."""
    if time.time() - _last_cast.get(hand_id, 0.0) < ICE_COOLDOWN:
        return
    center = hand_to_world(lm[9].x, lm[9].y)
    cast_ice(center)
    play_sound("ice.wav", volume=0.8)
    _last_cast[hand_id] = time.time()

"""
Fireball3D.py — MODULE HIEU UNG PHUN LUA (file phu)
===================================================
Chi chua nhung gi lien quan den hieu ung: shader, texture, particle,
luong phun lua. KHONG chua webcam / MediaPipe / ML / vong lap game
— nhung thu do nam o main.py.

Cach dung (xem main.py):
    from Fireball3D import bloom_shader, init_effects, cast

    app = Ursina()
    camera.shader = bloom_shader          # bat glow
    init_effects()                        # tao texture (goi SAU Ursina())
    ...
    cast(lm, hand_to_world, hand_id)      # phun lua khi con giu gesture
"""

import math
import os
import random
import time
from copy import copy

import numpy as np
from PIL import Image
from panda3d.core import ColorBlendAttrib
from ursina import (
    Entity, camera, color, destroy, time as utime, Vec3, Vec2, Shader, Texture,
    Audio, Mesh,
)

# --- Am thanh: dat file .wav/.ogg vao thu muc sounds/ ---
_SFX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")

# Giu particle Python song den het animation. Neu khong co strong reference,
# Ursina/Panda3D co the thu gom particle giua luong phun.
_ACTIVE_FIRE = {}


def _retain_fire(entity):
    _ACTIVE_FIRE[id(entity)] = entity


def _finish_fire(entity):
    _ACTIVE_FIRE.pop(id(entity), None)
    destroy(entity)


def play_sound(name, volume=0.7):
    """Phat 1 file am thanh trong sounds/. Thieu file thi bo qua (khong crash).

    LUU Y: Ursina Audio tim file theo TEN trong application.asset_folder (quet
    de quy), KHONG nhan duong dan tuyet doi -> chi truyen ten (vd "fireball.wav").
    """
    if not os.path.exists(os.path.join(_SFX_DIR, name)):
        return
    try:
        Audio(name, autoplay=True, loop=False, volume=volume)
    except Exception as e:
        print("Sound error:", e)


def load_effect_texture(path):
    """Doc PNG (giu kenh alpha/trong suot) tu Canva thanh Ursina Texture."""
    return Texture(Image.open(path).convert("RGBA"))

# ----------------------------------------------------------------------
# BLOOM SHADER (screen-space post-processing, gan vao camera.shader)
# ----------------------------------------------------------------------
bloom_shader = Shader(fragment='''
#version 430
uniform sampler2D tex;
in vec2 uv;
out vec4 color;

uniform float threshold;   // nguong sang: chi pixel sang hon moi glow
uniform float intensity;   // do manh cua glow
uniform float blur_size;   // ban kinh lan toa (theo uv)

vec3 bright(vec3 c) {
    float b = max(c.r, max(c.g, c.b));
    return c * clamp((b - threshold) / (1.0 - threshold), 0.0, 1.0);
}

void main() {
    vec3 original = texture(tex, uv).rgb;
    vec3 glow = vec3(0.0);
    float total = 0.0;
    for (int i = 0; i < 16; i++) {
        float a = float(i) * 0.39269908;      // = 2*pi/16
        for (int r = 1; r <= 2; r++) {
            vec2 off = vec2(cos(a), sin(a)) * blur_size * float(r);
            float w = 1.0 / float(r);
            glow  += bright(texture(tex, uv + off).rgb) * w;
            total += w;
        }
    }
    glow /= total;
    color = vec4(original + glow * intensity, 1.0);
}
''',
default_input=dict(
    threshold=0.90,
    intensity=1.7,
    blur_size=0.013,
))


# ----------------------------------------------------------------------
# FIRE SHADER (procedural): sinh lua bang noise + time.
# heat: 0..1 (cao = nong/trang), softness: 0..1 (cao = canh tan nhu khoi)
# ----------------------------------------------------------------------
fire_shader = Shader(name='fire_shader', language=Shader.GLSL,
vertex='''
#version 130
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
uniform vec2 texture_scale;
uniform vec2 texture_offset;
out vec2 texcoords;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    texcoords = (p3d_MultiTexCoord0 * texture_scale) + texture_offset;
}
''',
fragment='''
#version 140
uniform vec4 p3d_ColorScale;
in vec2 texcoords;
out vec4 fragColor;

uniform float time;
uniform float heat;
uniform float softness;

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { v += a * vnoise(p); p *= 2.0; a *= 0.5; }
    return v;
}
vec3 firePalette(float t) {
    vec3 c1 = vec3(0.6, 0.0, 0.0);
    vec3 c2 = vec3(1.0, 0.25, 0.0);
    vec3 c3 = vec3(1.0, 0.6, 0.05);
    vec3 c4 = vec3(1.0, 0.9, 0.35);
    vec3 c5 = vec3(1.0, 1.0, 0.92);
    t = clamp(t, 0.0, 1.0);
    if (t < 0.25) return mix(c1, c2, t / 0.25);
    if (t < 0.5)  return mix(c2, c3, (t - 0.25) / 0.25);
    if (t < 0.75) return mix(c3, c4, (t - 0.5) / 0.25);
    return mix(c4, c5, (t - 0.75) / 0.25);
}
void main() {
    vec2 uv = texcoords;
    float n1 = fbm(uv * 4.0 + vec2(time * 0.25, -time * 1.1));
    float n2 = fbm(uv * 8.0 - vec2(0.0, time * 1.9));
    float f = n1 * 0.6 + n2 * 0.4;

    float t = clamp(f * 0.75 + heat * 0.55, 0.0, 1.0);
    float hot = smoothstep(0.48, 0.95, f + heat * 0.28);
    vec3 col = firePalette(t) * (1.20 + heat * 0.55);
    col += vec3(1.0, 0.34, 0.03) * hot * hot * 0.75;
    col *= p3d_ColorScale.rgb;

    float a = p3d_ColorScale.a;
    a *= mix(1.0, smoothstep(0.24, 0.70, f), softness);
    fragColor = vec4(col, a);
}
''',
default_input=dict(
    texture_scale=Vec2(1, 1),
    texture_offset=Vec2(0, 0),
    time=0.0,
    heat=0.5,
    softness=0.0,
))


# Shader rieng cho luong phun: cat alpha thanh luoi lua, day noise chay tu
# goc toi mui va doi mau theo nhiet do. Khac fire_shader cua qua cau, shader
# nay duoc toi uu cho billboard dai cua sung phun lua.
flame_jet_shader = Shader(name='flame_jet_shader', language=Shader.GLSL,
vertex='''
#version 130
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 uv;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    uv = p3d_MultiTexCoord0;
}
''',
fragment='''
#version 140
uniform vec4 p3d_ColorScale;
in vec2 uv;
out vec4 fragColor;

uniform float time;
uniform float phase;
uniform float heat;
uniform float turbulence;

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
        mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x),
        f.y
    );
}

float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.55;
    for (int i = 0; i < 5; i++) {
        value += noise(p) * amplitude;
        p = p * 2.03 + vec2(7.1, 3.7);
        amplitude *= 0.48;
    }
    return value;
}

vec3 fire_color(float temperature) {
    vec3 deep_red = vec3(0.72, 0.015, 0.0);
    vec3 orange = vec3(1.0, 0.20, 0.005);
    vec3 gold = vec3(1.0, 0.66, 0.045);
    vec3 hot = vec3(1.0, 0.96, 0.60);
    vec3 white_hot = vec3(1.0, 1.0, 0.96);
    if (temperature < 0.22)
        return mix(deep_red, orange, temperature / 0.22);
    if (temperature < 0.48)
        return mix(orange, gold, (temperature - 0.22) / 0.26);
    if (temperature < 0.76)
        return mix(gold, hot, (temperature - 0.48) / 0.28);
    return mix(hot, white_hot, (temperature - 0.76) / 0.24);
}

void main() {
    float y = clamp(uv.y, 0.0, 1.0);
    float x = uv.x * 2.0 - 1.0;
    float travel = time * 2.7 + phase;

    float coarse = fbm(vec2(x * 1.7 + phase, y * 4.0 - travel));
    float detail = fbm(vec2(x * 4.8 - phase, y * 9.0 - travel * 1.75));
    float flow = coarse * 0.68 + detail * 0.32;

    // Bien dang ngang tang dan ve mui, giong dong khi nong xe luoi lua.
    float warp = (flow - 0.5) * turbulence * (0.16 + y * 0.48);
    warp += sin(y * 16.0 - travel * 2.1 + phase) * turbulence * y * 0.075;
    float cone_width = mix(0.62, 0.055, pow(y, 0.82));
    cone_width *= 0.83 + flow * 0.40;

    float edge_distance = 1.0 - abs(x + warp) / max(cone_width, 0.01);
    float body = smoothstep(0.035, 0.27, edge_distance);
    float sharp_core = smoothstep(0.20, 0.78, edge_distance);
    float base_fade = smoothstep(0.0, 0.075, y);
    float torn_tip = 1.0 - smoothstep(0.76 + (flow - 0.5) * 0.24, 1.0, y);
    float alpha = body * base_fade * torn_tip;
    float combustion = smoothstep(0.30, 0.67, flow + edge_distance * 0.24);
    alpha *= (0.52 + combustion * 0.48) * (0.68 + flow * 0.28);

    float temperature = (
        heat * 0.50
        + sharp_core * 0.24
        + flow * 0.21
        - y * 0.30
    );
    vec3 col = fire_color(clamp(temperature, 0.0, 1.0));
    col *= 0.86 + sharp_core * 0.30 + heat * 0.18;
    col *= p3d_ColorScale.rgb;

    fragColor = vec4(col, alpha * p3d_ColorScale.a);
}
''',
default_input=dict(
    time=0.0,
    phase=0.0,
    heat=0.65,
    turbulence=0.7,
))


def _make_flame_cone(rings=16, sides=28):
    """Mat non 3D mo rong theo -Z, UV lien tuc de noise chay doc luong."""
    vertices = []
    uvs = []
    for ring in range(rings - 1):
        t0 = ring / (rings - 1)
        t1 = (ring + 1) / (rings - 1)
        r0 = 0.075 + 0.49 * (t0 ** 0.78)
        r1 = 0.075 + 0.49 * (t1 ** 0.78)
        for side in range(sides):
            u0 = side / sides
            u1 = (side + 1) / sides
            a0 = u0 * math.tau
            a1 = u1 * math.tau
            p00 = Vec3(math.cos(a0) * r0, math.sin(a0) * r0, -t0)
            p01 = Vec3(math.cos(a1) * r0, math.sin(a1) * r0, -t0)
            p10 = Vec3(math.cos(a0) * r1, math.sin(a0) * r1, -t1)
            p11 = Vec3(math.cos(a1) * r1, math.sin(a1) * r1, -t1)
            vertices.extend((p00, p11, p01, p00, p10, p11))
            uvs.extend((
                Vec2(u0, t0), Vec2(u1, t1), Vec2(u1, t0),
                Vec2(u0, t0), Vec2(u0, t1), Vec2(u1, t1),
            ))
    return Mesh(vertices=vertices, uvs=uvs, mode="triangle", static=True)


_FLAME_CONE_MODEL = _make_flame_cone()


# ----------------------------------------------------------------------
# TEXTURE NOISE (dung cho khoi + vet lua)
# ----------------------------------------------------------------------
fire_tex = None       # noise cho khoi
glow_tex = None       # dia sang mem cho loi/trail
ring_tex = None       # vong xung kich khi no
flame_tex = None      # giot lua nhon dung cho luong phun


def make_fire_texture(size=256):
    """Noise dam may TILEABLE (FFT low-pass)."""
    rng = np.random.default_rng(7)
    white = rng.random((size, size))
    f = np.fft.fft2(white)
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    radius = np.sqrt(fx ** 2 + fy ** 2)
    filt = 1.0 / (radius * size * 0.15 + 1) ** 2
    img = np.fft.ifft2(f * filt).real
    img -= img.min()
    img /= (img.max() + 1e-9)
    img = img ** 1.4
    img = 0.5 + 0.5 * img
    arr = (img * 255).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Texture(Image.fromarray(rgb))


def make_radial_texture(size=256, ring=False):
    """Tao texture RGBA mem cho glow hoac vong xung kich."""
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    radius = np.sqrt(xx * xx + yy * yy)

    if ring:
        alpha = np.exp(-((radius - 0.58) / 0.09) ** 2)
        alpha *= np.clip((1.0 - radius) * 8.0, 0.0, 1.0)
    else:
        alpha = np.clip(1.0 - radius, 0.0, 1.0) ** 2.4

    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = 255
    rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
    return Texture(Image.fromarray(rgba, mode="RGBA"), filtering="bilinear")


def make_flame_texture(size=256):
    """Tao sprite ngon lua dai, canh gon va mui rat nhon."""
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # Than rong o goc, thu nhanh ve mui; bien uon song de giong lua bi gio xe.
    center = (
        0.085 * np.sin(yy * 12.0) * yy
        + 0.035 * np.sin(yy * 25.0) * yy * yy
    )
    width = 0.49 * (1.0 - yy) ** 0.72 + 0.002
    edge = np.clip(1.0 - np.abs(xx - center) / width, 0.0, 1.0)
    vertical = np.clip(
        np.sin(np.pi * np.clip(yy, 0.0, 1.0)),
        0.0,
        1.0,
    ) ** 0.28
    # Luy thua cao lam bien lua sac net; loi van dac de cac sprite noi lien.
    alpha = (edge ** 2.25) * vertical

    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = 255
    rgba[:, :, 3] = (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
    return Texture(Image.fromarray(rgba, mode="RGBA"), filtering="bilinear")


def enable_additive(entity):
    """Cong mau particle vao background de glow ma khong can bloom."""
    entity.setAttrib(ColorBlendAttrib.make(
        ColorBlendAttrib.M_add,
        ColorBlendAttrib.O_incoming_alpha,
        ColorBlendAttrib.O_one,
    ))
    entity.setDepthWrite(False)


def init_effects():
    """Goi 1 lan SAU app = Ursina() de tao texture dung chung."""
    global fire_tex, glow_tex, ring_tex, flame_tex
    if fire_tex is None:
        fire_tex = make_fire_texture()
    if glow_tex is None:
        glow_tex = make_radial_texture()
    if ring_tex is None:
        ring_tex = make_radial_texture(ring=True)
    if flame_tex is None:
        flame_tex = make_flame_texture()


# ----------------------------------------------------------------------
# PARTICLES: TAN LUA, KHOI, VET LUA
# ----------------------------------------------------------------------
class Ember(Entity):
    """Tan lua nho, sang va sac net."""

    def __init__(self, position, size=1.0, velocity=None):
        super().__init__(
            model="quad",
            texture=glow_tex,
            billboard=True,
            color=random.choice([
                color.rgba32(255, 245, 180, 255),
                color.rgba32(255, 155, 30, 255),
                color.rgba32(255, 70, 8, 255),
            ]),
            scale=size * random.uniform(0.12, 0.30),
            position=position,
            unlit=True,
            double_sided=True,
        )
        enable_additive(self)
        self.velocity = velocity or Vec3(
            random.uniform(-2.8, 2.8),
            random.uniform(-2.2, 3.5),
            random.uniform(2.0, 7.0),
        )
        self.life = random.uniform(0.28, 0.62)
        self.max_life = self.life

    def update(self):
        self.position += self.velocity * utime.dt
        self.velocity.y -= 1.2 * utime.dt
        self.life -= utime.dt
        self.scale *= max(0.0, 1.0 - 2.5 * utime.dt)
        self.alpha = max(0.0, self.life / self.max_life) ** 1.4
        if self.life <= 0:
            destroy(self)


class Smoke(Entity):
    """Khoi mem, no dan va troi len."""

    def __init__(self, position, size=1.0):
        super().__init__(
            model="quad",
            texture=glow_tex,
            billboard=True,
            color=color.rgba32(62, 52, 48, 72),
            scale=size * random.uniform(0.55, 0.95),
            position=position,
            unlit=True,
            double_sided=True,
        )
        self.velocity = Vec3(
            random.uniform(-0.7, 0.7),
            random.uniform(0.8, 1.8),
            random.uniform(1.0, 3.0),
        )
        self.life = random.uniform(0.65, 1.10)
        self.max_life = self.life
        self.spin = random.uniform(-40, 40)

    def update(self):
        self.position += self.velocity * utime.dt
        self.life -= utime.dt
        self.scale *= 1.0 + 1.1 * utime.dt
        self.rotation_z += self.spin * utime.dt
        ratio = max(0.0, self.life / self.max_life)
        self.alpha = 0.22 * ratio * ratio
        if self.life <= 0:
            destroy(self)


class TrailPuff(Entity):
    """Vet lua additive nho gon, de lai duong bay lien tuc."""

    def __init__(self, position, size=1.0):
        super().__init__(
            model="quad",
            texture=glow_tex,
            billboard=True,
            color=random.choice([
                color.rgba32(255, 175, 35, 210),
                color.rgba32(255, 82, 8, 190),
            ]),
            scale=size * random.uniform(0.65, 0.95),
            position=position,
            unlit=True,
            double_sided=True,
        )
        enable_additive(self)
        self.life = random.uniform(0.22, 0.34)
        self.max_life = self.life
        self.velocity = Vec3(
            random.uniform(-0.30, 0.30),
            random.uniform(0.15, 0.65),
            random.uniform(0.25, 1.1),
        )

    def update(self):
        self.position += self.velocity * utime.dt
        self.life -= utime.dt
        self.scale *= max(0.0, 1.0 - 2.0 * utime.dt)
        ratio = max(0.0, self.life / self.max_life)
        self.alpha = ratio * ratio
        if self.life <= 0:
            destroy(self)


class ImpactBurst(Entity):
    """Vu no bom lua: chop sang, hai vong xung kich, tan lua va khoi toa tron."""

    def __init__(self, position, size=1.0):
        super().__init__(position=position)
        self.life = 0.52
        self.max_life = self.life
        self.size = size

        self.flash = Entity(
            parent=self,
            model="quad",
            texture=glow_tex,
            billboard=True,
            color=color.rgba32(255, 205, 75, 240),
            scale=size * 2.0,
            unlit=True,
            double_sided=True,
        )
        self.ring = Entity(
            parent=self,
            model="quad",
            texture=ring_tex,
            billboard=True,
            color=color.rgba32(255, 92, 12, 230),
            scale=size * 1.2,
            unlit=True,
            double_sided=True,
        )
        self.inner_ring = Entity(
            parent=self,
            model="quad",
            texture=ring_tex,
            billboard=True,
            color=color.rgba32(255, 220, 92, 245),
            scale=size * 0.75,
            unlit=True,
            double_sided=True,
        )
        enable_additive(self.flash)
        enable_additive(self.ring)
        enable_additive(self.inner_ring)

        # Chia deu goc de vu no luon toe tron, them jitter de khong bi may moc.
        spark_count = max(28, int(34 * size))
        for i in range(spark_count):
            angle = 2 * math.pi * i / spark_count + random.uniform(-0.11, 0.11)
            speed = random.uniform(7.0, 16.0) * size
            direction = Vec3(
                math.cos(angle) * speed,
                math.sin(angle) * speed,
                random.uniform(-3.0, 6.0),
            )
            ember = Ember(position, size=size * random.uniform(0.65, 1.15),
                          velocity=direction)
            ember.life = random.uniform(0.38, 0.82)
            ember.max_life = ember.life

        # Khoi cung bung theo cac huong, sau do cham dan va boc len.
        for i in range(9):
            angle = 2 * math.pi * i / 9 + random.uniform(-0.18, 0.18)
            smoke = Smoke(
                position + Vec3(
                    random.uniform(-0.25, 0.25),
                    random.uniform(-0.25, 0.25),
                    random.uniform(-0.15, 0.15),
                ),
                size=size * random.uniform(0.75, 1.25),
            )
            smoke.velocity = Vec3(
                math.cos(angle) * random.uniform(1.8, 4.2) * size,
                math.sin(angle) * random.uniform(1.8, 4.2) * size + 1.0,
                random.uniform(0.5, 3.0),
            )

    def update(self):
        self.life -= utime.dt
        progress = 1.0 - max(0.0, self.life / self.max_life)
        self.flash.scale = self.size * (2.0 + progress * 4.5)
        self.ring.scale = self.size * (1.2 + progress * 7.5)
        self.inner_ring.scale = self.size * (0.75 + progress * 5.2)
        self.flash.alpha = max(0.0, 1.0 - progress * 1.8)
        self.ring.alpha = max(0.0, (1.0 - progress) ** 1.3)
        self.inner_ring.alpha = max(0.0, 1.0 - progress * 1.35) ** 1.5
        if self.life <= 0:
            destroy(self)


# ----------------------------------------------------------------------
# FIREBALL
# ----------------------------------------------------------------------
class Fireball(Entity):
    """Cau lua co loi nong, glow additive, trail va vu no ket thuc."""

    # (scale, heat, softness, alpha) cho tung lop
    BASE = [
        (0.72, 1.00, 0.00, 1.00),   # loi trang nong
        (1.10, 0.76, 0.18, 0.92),   # vang
        (1.58, 0.42, 0.52, 0.58),   # cam
        (2.05, 0.18, 0.82, 0.24),   # vien do mem
    ]

    def __init__(self, position, velocity, power=1.0):
        super().__init__(position=position)
        self.velocity = velocity
        self.life = 2.4
        self.timer = 0.0
        self.size = 0.65 + power * 0.75
        self.trail_timer = 0.0
        self.smoke_timer = 0.0
        self.ember_timer = 0.0

        self.layers = []
        self.base_scales = []
        for (sc, heat, soft, a) in Fireball.BASE:
            e = Entity(parent=self, model="sphere", color=color.white,
                       shader=fire_shader, unlit=True)
            e.set_shader_input("heat", heat)
            e.set_shader_input("softness", soft)
            e.set_shader_input("texture_scale", Vec2(4, 4))
            e.set_shader_input("texture_offset",
                               Vec2(random.random(), random.random()))
            e.alpha = a
            self.layers.append(e)
            self.base_scales.append(sc * self.size)

        self.glows = []
        for scale, tint in [
            (2.25, color.rgba32(255, 100, 12, 115)),
            (3.20, color.rgba32(255, 38, 4, 48)),
        ]:
            glow = Entity(
                parent=self,
                model="quad",
                texture=glow_tex,
                billboard=True,
                color=tint,
                scale=self.size * scale,
                unlit=True,
                double_sided=True,
            )
            enable_additive(glow)
            self.glows.append((glow, scale, tint.a))

    def update(self):
        self.position += self.velocity * utime.dt
        self.life -= utime.dt
        self.timer += utime.dt

        flick = 1 + 0.08 * math.sin(self.timer * 38)
        flick2 = 1 + 0.05 * math.sin(self.timer * 23 + 1.5)
        for i, e in enumerate(self.layers):
            fl = flick if i % 2 == 0 else flick2
            e.scale = self.base_scales[i] * fl
            e.set_shader_input("time", self.timer * 1.6 + i * 3.3)
            e.rotation_y += (18 + i * 7) * utime.dt

        for i, (glow, scale, base_alpha) in enumerate(self.glows):
            pulse = 1.0 + 0.07 * math.sin(self.timer * (17 + i * 4) + i)
            glow.scale = self.size * scale * pulse
            glow.alpha = base_alpha * (0.88 + 0.12 * math.sin(self.timer * 21 + i))

        self.trail_timer -= utime.dt
        if self.trail_timer <= 0:
            TrailPuff(self.world_position, self.size)
            self.trail_timer = 0.035

        self.smoke_timer -= utime.dt
        if self.smoke_timer <= 0:
            Smoke(self.world_position, self.size)
            self.smoke_timer = 0.13

        self.ember_timer -= utime.dt
        if self.ember_timer <= 0:
            Ember(
                self.world_position + Vec3(
                    random.uniform(-0.35, 0.35),
                    random.uniform(-0.35, 0.35),
                    random.uniform(-0.20, 0.20),
                ),
                size=self.size,
            )
            self.ember_timer = 0.065

        if self.z < 0.8:            # dap vao man hinh -> no tia lua
            self._impact()
            destroy(self)
        elif self.life <= 0:        # chay het -> tat, khong no
            destroy(self)

    def _impact(self):
        """No nhu bom va toe tan lua theo moi huong khi dap vao man hinh."""
        p = self.world_position
        ImpactBurst(p, self.size * 1.15)
        camera.shake(
            duration=0.22,
            magnitude=min(1.0, 0.45 + self.size * 0.24),
            speed=0.018,
        )


# ----------------------------------------------------------------------
# FLAMETHROWER — cac luoi lua nhon ghep thanh luong phun lien tuc
# ----------------------------------------------------------------------
class FlameTongue(Entity):
    """Mot luoi lua nhon trong luong phun lien tuc tu long ban tay."""

    def __init__(self, position, velocity, size=1.0, angle=0.0, hot=False):
        tint = random.choice([
            color.rgba32(255, 54, 2, 250),
            color.rgba32(255, 102, 3, 255),
            color.rgba32(255, 176, 18, 255),
        ])
        super().__init__(
            model="quad",
            texture=flame_tex,
            billboard=True,
            color=tint,
            position=position,
            rotation_z=angle + random.uniform(-36, 36),
            unlit=True,
            double_sided=True,
        )
        _retain_fire(self)
        self.setDepthWrite(False)
        self.velocity = Vec3(*velocity)
        self.life = random.uniform(0.38, 0.56)
        self.max_life = self.life
        self.age = 0.0
        self.phase = random.uniform(0, 2 * math.pi)
        self.spin = random.uniform(-95, 95)
        self.base_width = size * random.uniform(0.62, 0.98)
        self.base_length = size * random.uniform(1.25, 2.20)
        self.scale = Vec3(self.base_width, self.base_length, 1)

        # Loi vang-trang hep lam luong lua nong va sac, khong thanh dom tron.
        self.core = Entity(
            parent=self,
            model="quad",
            texture=flame_tex,
            color=(color.white if hot else color.rgba32(255, 224, 92, 245)),
            scale=Vec3(0.50, 0.84, 1),
            z=-0.01,
            unlit=True,
            double_sided=True,
        )
        enable_additive(self.core)

    def _die(self):
        p = self.world_position
        if random.random() < 0.42:
            Ember(
                p,
                size=self.base_width * 1.4,
                velocity=Vec3(
                    random.uniform(-4.5, 4.5),
                    random.uniform(-2.0, 5.0),
                    random.uniform(0.5, 4.0),
                ),
            )
        if random.random() < 0.12:
            Smoke(p, size=self.base_width * 1.3)
        _finish_fire(self)

    def update(self):
        dt = utime.dt
        self.age += dt
        self.life -= dt

        # Nhip dao nhanh nhung bien do hep: luong phun du doi ma khong bi dut.
        self.velocity.x += math.sin(self.age * 22.0 + self.phase) * 2.8 * dt
        self.velocity.y += math.cos(self.age * 18.0 + self.phase) * 2.2 * dt
        self.position += self.velocity * dt
        self.rotation_z += self.spin * dt

        progress = min(1.0, self.age / self.max_life)
        flicker = 0.82 + 0.18 * math.sin(self.age * 43.0 + self.phase)
        self.scale = Vec3(
            self.base_width * (1.0 + progress * 0.88),
            self.base_length * (1.0 + progress * 0.52) * flicker,
            1,
        )
        fade = max(0.0, 1.0 - progress)
        self.alpha = min(1.0, (fade ** 0.48) * 1.12)
        self.core.alpha = fade ** 0.9
        self.core.scale_x = 0.46 + progress * 0.18

        if random.random() < 1.6 * dt:
            Ember(self.world_position, size=self.base_width * 0.9)
        if random.random() < 0.38 * dt and progress > 0.45:
            Smoke(self.world_position, size=self.base_width)

        if self.z < 0.9 or self.life <= 0:
            self._die()


class JetSpark(Entity):
    """Tia lua dai, lao nhanh quanh loi luong phun."""

    def __init__(self, position, velocity, angle):
        super().__init__(
            model="quad",
            texture=glow_tex,
            billboard=True,
            color=random.choice([
                color.rgba32(255, 252, 205, 255),
                color.rgba32(255, 190, 38, 255),
                color.rgba32(255, 78, 4, 245),
            ]),
            position=position,
            rotation_z=angle + random.uniform(-8, 8),
            scale=Vec3(
                random.uniform(0.045, 0.105),
                random.uniform(0.58, 1.22),
                1,
            ),
            unlit=True,
            double_sided=True,
        )
        _retain_fire(self)
        enable_additive(self)
        self.velocity = Vec3(*velocity)
        self.life = random.uniform(0.30, 0.52)
        self.max_life = self.life

    def update(self):
        dt = utime.dt
        self.position += self.velocity * dt
        self.life -= dt
        ratio = max(0.0, self.life / self.max_life)
        self.alpha = ratio ** 0.75
        self.scale_y *= max(0.0, 1.0 - 1.5 * dt)
        if self.z < 0.75 or self.life <= 0:
            _finish_fire(self)


class FlamePressureWave(Entity):
    """Khoi ap suat lua lao doc luong, bung rong bat doi xung roi tan."""

    def __init__(self, position, velocity, angle, size=1.0):
        super().__init__(position=position)
        _retain_fire(self)
        self.velocity = Vec3(*velocity)
        self.life = 0.46
        self.max_life = self.life
        self.size = size
        self.angle = angle

        self.outer_bloom = Entity(
            parent=self,
            model="quad",
            texture=flame_tex,
            billboard=True,
            color=color.rgba32(255, 62, 3, 225),
            rotation_z=angle,
            unlit=True,
            double_sided=True,
        )
        self.hot_bloom = Entity(
            parent=self,
            model="quad",
            texture=flame_tex,
            billboard=True,
            color=color.rgba32(255, 225, 105, 245),
            rotation_z=angle - 21,
            z=-0.018,
            unlit=True,
            double_sided=True,
        )
        self.flash = Entity(
            parent=self,
            model="quad",
            texture=glow_tex,
            billboard=True,
            color=color.rgba32(255, 132, 18, 150),
            z=0.015,
            unlit=True,
            double_sided=True,
        )
        enable_additive(self.outer_bloom)
        enable_additive(self.hot_bloom)
        enable_additive(self.flash)

    def update(self):
        dt = utime.dt
        self.position += self.velocity * dt
        self.life -= dt
        progress = 1.0 - max(0.0, self.life / self.max_life)
        eased = 1.0 - (1.0 - progress) ** 3
        self.outer_bloom.scale = Vec3(
            self.size * (0.75 + eased * 4.4),
            self.size * (1.4 + eased * 7.8),
            1,
        )
        self.hot_bloom.scale = Vec3(
            self.size * (0.38 + eased * 2.5),
            self.size * (0.9 + eased * 5.5),
            1,
        )
        self.flash.scale = self.size * (0.85 + eased * 4.0)
        self.outer_bloom.rotation_z += 72 * dt
        self.hot_bloom.rotation_z -= 105 * dt
        self.outer_bloom.alpha = max(0.0, (1.0 - progress) ** 1.2)
        self.hot_bloom.alpha = max(0.0, 1.0 - progress * 1.55) ** 1.45
        self.flash.alpha = max(0.0, 1.0 - progress * 1.9)
        if self.z < 0.70 or self.life <= 0:
            _finish_fire(self)


class FlameIgnitionBurst(Entity):
    """Vu bung lua o long ban tay khi bat dau mot lan phun."""

    def __init__(self, position, base_velocity, angle):
        super().__init__(position=position)
        _retain_fire(self)
        self.life = 0.38
        self.max_life = self.life

        self.flash = Entity(
            parent=self,
            model="quad",
            texture=glow_tex,
            billboard=True,
            color=color.rgba32(255, 245, 180, 245),
            unlit=True,
            double_sided=True,
        )
        self.fire_bloom = Entity(
            parent=self,
            model="quad",
            texture=flame_tex,
            billboard=True,
            color=color.rgba32(255, 74, 3, 245),
            rotation_z=angle,
            z=0.018,
            unlit=True,
            double_sided=True,
        )
        enable_additive(self.flash)
        enable_additive(self.fire_bloom)

        # Tia bung uu tien huong phun, nhung van toe xung quanh diem moi lua.
        for i in range(14):
            radial = 2 * math.pi * i / 14 + random.uniform(-0.16, 0.16)
            velocity = Vec3(*base_velocity) * random.uniform(0.16, 0.34)
            velocity += Vec3(
                math.cos(radial) * random.uniform(2.8, 7.0),
                math.sin(radial) * random.uniform(2.8, 7.0),
                random.uniform(-2.0, 2.5),
            )
            JetSpark(position, velocity, angle + math.degrees(radial))

        camera.shake(duration=0.20, magnitude=0.44, speed=0.014)

    def update(self):
        self.life -= utime.dt
        progress = 1.0 - max(0.0, self.life / self.max_life)
        eased = 1.0 - (1.0 - progress) ** 3
        self.flash.scale = 0.75 + eased * 4.8
        self.fire_bloom.scale = Vec3(
            1.3 + eased * 4.0,
            2.8 + eased * 8.2,
            1,
        )
        self.flash.alpha = max(0.0, 1.0 - progress * 1.75)
        self.fire_bloom.alpha = max(0.0, (1.0 - progress) ** 1.2)
        if self.life <= 0:
            _finish_fire(self)


class FlameStream(Entity):
    """Emitter giu than lua lien mach, khong phu thuoc FPS nhan dien tay."""

    EMIT_INTERVAL = 0.020
    SPARK_INTERVAL = 0.072
    REFRESH_TIMEOUT = 0.26

    def __init__(self, hand_id):
        super().__init__()
        self.hand_id = hand_id
        self.nozzle = Vec3(0, 0, 0)
        self.target_nozzle = Vec3(0, 0, 0)
        self.initialized = False
        self.base_velocity = Vec3(0, 0, -19)
        self.ux = 0.0
        self.uy = 1.0
        self.angle = 0.0
        self.last_refresh = time.time()
        self.emit_timer = 0.0
        self.spark_timer = 0.0
        self.surge_timer = random.uniform(0.32, 0.48)
        self.surge = 0.0
        self.timer = 0.0

        # Hai lop gan tay luon ton tai de noi cac particle thanh mot than lua.
        self.outer_source = Entity(
            parent=self,
            model="quad",
            origin_y=-0.5,
            billboard=True,
            color=color.white,
            shader=flame_jet_shader,
            unlit=True,
            double_sided=True,
        )
        self.hot_source = Entity(
            parent=self,
            model="quad",
            origin_y=-0.5,
            billboard=True,
            color=color.white,
            shader=flame_jet_shader,
            z=-0.012,
            unlit=True,
            double_sided=True,
        )
        self.outer_source.set_shader_input("phase", random.uniform(0, 10))
        self.outer_source.set_shader_input("heat", 0.66)
        self.outer_source.set_shader_input("turbulence", 0.48)
        self.hot_source.set_shader_input("phase", random.uniform(0, 10))
        self.hot_source.set_shader_input("heat", 1.0)
        self.hot_source.set_shader_input("turbulence", 0.30)
        self.outer_source.setDepthWrite(False)
        enable_additive(self.hot_source)

        self.source_glow = Entity(
            parent=self,
            model="quad",
            texture=glow_tex,
            billboard=True,
            color=color.rgba32(255, 188, 35, 235),
            z=0.02,
            unlit=True,
            double_sided=True,
        )
        enable_additive(self.source_glow)

        # Ong lua hinh non gom nhieu tiet dien chong lap. Moi tiet dien co vo
        # do-cam va loi vang-trang rieng, tao mot khoi phun lien tuc thay vi
        # cac sprite roi rac.
        self.bridges = []
        bridge_specs = [
            (0.70, 0.58, 4.2, color.white),
            (2.60, 0.78, 4.7, color.white),
            (4.55, 1.02, 5.2, color.white),
            (6.50, 1.28, 5.7, color.white),
            (8.45, 1.56, 6.2, color.white),
            (10.40, 1.86, 6.7, color.white),
        ]
        for depth, width, length, _tint in bridge_specs:
            progress = depth / bridge_specs[-1][0]
            outer = Entity(
                parent=self,
                model="quad",
                billboard=True,
                color=color.white,
                shader=flame_jet_shader,
                unlit=True,
                double_sided=True,
            )
            core = Entity(
                parent=self,
                model="quad",
                billboard=True,
                color=color.white,
                shader=flame_jet_shader,
                unlit=True,
                double_sided=True,
            )
            outer.set_shader_input("phase", random.uniform(0, 20))
            outer.set_shader_input("heat", 0.58 - progress * 0.36)
            outer.set_shader_input("turbulence", 0.52 + progress * 0.46)
            core.set_shader_input("phase", random.uniform(0, 20))
            core.set_shader_input("heat", 1.0 - progress * 0.42)
            core.set_shader_input("turbulence", 0.30 + progress * 0.28)
            outer.setDepthWrite(False)
            enable_additive(core)
            self.bridges.append((outer, core, depth, width, length))

        # Mot mesh non lien tuc tao the tich 3D that tu tay toi dau luong.
        # Lop vo alpha-blend giu mau cam/do; loi nho hon additive de phat nong.
        self.volume_shell = Entity(
            parent=self,
            model=copy(_FLAME_CONE_MODEL),
            color=color.white,
            shader=flame_jet_shader,
            unlit=True,
            double_sided=False,
        )
        self.volume_shell.set_shader_input("heat", 0.48)
        self.volume_shell.set_shader_input("turbulence", 0.72)
        self.volume_shell.set_shader_input("phase", random.uniform(0, 20))
        self.volume_shell.setDepthWrite(True)

        self.volume_core = Entity(
            parent=self,
            model=copy(_FLAME_CONE_MODEL),
            color=color.white,
            shader=flame_jet_shader,
            unlit=True,
            double_sided=False,
        )
        self.volume_core.set_shader_input("heat", 0.94)
        self.volume_core.set_shader_input("turbulence", 0.38)
        self.volume_core.set_shader_input("phase", random.uniform(0, 20))
        enable_additive(self.volume_core)
        self.volume_core.setDepthWrite(True)

    def refresh(self, nozzle, velocity, ux, uy, angle):
        self.target_nozzle = Vec3(*nozzle)
        if not self.initialized:
            self.nozzle = Vec3(*nozzle)
            self.initialized = True
            FlameIgnitionBurst(self.nozzle, velocity, angle)
        self.base_velocity = Vec3(*velocity)
        self.ux = ux
        self.uy = uy
        self.angle = angle
        self.last_refresh = time.time()
        self.position = self.nozzle

    def _emit(self):
        # Ba dong lap day: mot loi trang nong va hai lop cam-do bung rong.
        # Tong mat do cao hon nhung van bi khoa theo thoi gian, khong phu thuoc FPS.
        for i, spread in enumerate((0.30, 0.72, 1.18)):
            velocity = self.base_velocity + Vec3(
                random.uniform(-2.5, 2.5) * spread,
                random.uniform(-2.2, 3.0) * spread,
                random.uniform(-1.2, 0.7),
            )
            spawn = self.nozzle + Vec3(
                random.uniform(-0.11, 0.11),
                random.uniform(-0.10, 0.10),
                random.uniform(-0.07, 0.07),
            )
            FlameTongue(
                spawn,
                velocity,
                size=random.uniform(1.24, 1.78),
                angle=self.angle,
                hot=(i == 0),
            )

    def update(self):
        dt = utime.dt
        self.timer += dt
        idle = time.time() - self.last_refresh
        if idle > self.REFRESH_TIMEOUT:
            if _streams.get(self.hand_id) is self:
                _streams.pop(self.hand_id, None)
            _finish_fire(self)
            return

        # Loc rung tracking nhe de goc phun bam tay nhung khong giat tung frame.
        follow = 1.0 - math.exp(-22.0 * dt)
        self.nozzle += (self.target_nozzle - self.nozzle) * follow
        self.position = self.nozzle

        self.surge = max(0.0, self.surge - dt * 3.25)
        self.surge_timer -= dt
        if self.surge_timer <= 0.0:
            self.surge = random.uniform(0.82, 1.0)
            self.surge_timer = random.uniform(0.36, 0.58)
            wave_velocity = self.base_velocity * random.uniform(0.42, 0.58)
            FlamePressureWave(
                self.nozzle,
                wave_velocity,
                self.angle,
                size=random.uniform(1.02, 1.34),
            )
            # Mot xung nhien lieu day them nhieu luoi lua vao cung mot frame.
            self._emit()
            self._emit()
            self._emit()
            self._emit()
            camera.shake(
                duration=0.11,
                magnitude=0.22 + self.surge * 0.14,
                speed=0.012,
            )

        flicker = (
            0.92
            + 0.055 * math.sin(self.timer * 43.0)
            + 0.025 * math.sin(self.timer * 71.0)
            + self.surge * 0.22
        )
        # Chan hai lop nam dung tai long ban tay. Hai lop dao lech pha de phan
        # dau lua cong, vặn va khong con thanh mot tia thang cung.
        source_sway = (
            math.sin(self.timer * 7.2) * 8.5
            + math.sin(self.timer * 13.7 + 1.4) * 3.5
        )
        self.outer_source.rotation_z = self.angle + source_sway
        self.hot_source.rotation_z = (
            self.angle + source_sway * 0.58 - math.sin(self.timer * 17.0) * 3.8
        )
        self.outer_source.set_shader_input("time", self.timer)
        self.hot_source.set_shader_input("time", self.timer * 1.08 + 2.7)
        # Rut ngan loi gan tay: bridge cong va the tich 3D se tao chieu dai,
        # tranh vet vang thang nhu laser chay xuyen ca luong.
        self.outer_source.scale = Vec3(2.42 * flicker, 6.8, 1)
        self.hot_source.scale = Vec3(1.12 * flicker, 5.2, 1)
        self.outer_source.alpha = 0.62
        self.hot_source.alpha = 0.49
        self.source_glow.scale = 3.9 * flicker
        self.source_glow.alpha = 0.68 + 0.16 * flicker
        self.outer_source.position = Vec3(0, 0, 0)
        self.hot_source.position = Vec3(0, 0, -0.02)

        bridge_count = max(1, len(self.bridges) - 1)
        for i, (outer, core, depth, width, length) in enumerate(self.bridges):
            progress = i / bridge_count
            # Song nhiet chay tu gan tay ra dau luong, cho cam giac khi bi ep.
            phase = self.timer * 18.0 - depth * 1.42
            # Duong tam uon mem: bien do tang dan ve dau luong, gom song lon
            # cham va song nho nhanh de lua cong vẹo tu nhien thay vi thang cung.
            macro_bend = (
                math.sin(self.timer * 4.1 - progress * 3.6)
                * (0.04 + progress * 0.62)
            )
            tip_whip = (
                math.sin(self.timer * 2.35 + progress * 5.2 + 1.1)
                * (progress ** 1.7) * 0.46
            )
            cross_wave = (
                macro_bend
                + tip_whip
                + math.sin(phase) * (0.035 + progress * 0.13)
            )
            lift_wave = (
                math.cos(self.timer * 3.25 - progress * 2.8)
                * (0.025 + progress * 0.24)
                + math.cos(phase * 0.83) * (0.018 + progress * 0.075)
            )
            pulse = (
                0.94
                + 0.085 * math.sin(self.timer * 32.0 - depth * 2.05)
                + 0.025 * math.sin(self.timer * 57.0 + i)
            )
            position = Vec3(
                self.ux * depth * 0.205 - self.uy * cross_wave,
                self.uy * depth * 0.205 + self.ux * cross_wave + lift_wave,
                -depth,
            )
            rotation = (
                self.angle
                + math.sin(self.timer * 4.1 - progress * 3.6)
                * (4.0 + progress * 13.0)
                + math.sin(self.timer * 12.5 - depth * 0.58)
                * (1.5 + progress * 4.5)
            )
            outer.position = position
            outer.rotation_z = rotation
            outer.scale = Vec3(width * 1.68 * pulse, length * 1.38 * pulse, 1)
            outer.alpha = (0.25 - progress * 0.065) * pulse
            outer.set_shader_input("time", self.timer + depth * 0.045)

            core.position = Vec3(position.x, position.y, position.z - 0.025)
            core.rotation_z = rotation - math.sin(phase * 1.35) * 2.0
            core.scale = Vec3(
                width * (0.68 - progress * 0.15) * pulse,
                length * (1.25 - progress * 0.12) * pulse,
                1,
            )
            core.alpha = max(0.025, 0.12 - progress * 0.075) * pulse
            core.set_shader_input("time", self.timer * 1.06 + depth * 0.065)

        cone_depth = 12.6
        pressure = (
            0.94
            + 0.055 * math.sin(self.timer * 25.0)
            + 0.025 * math.sin(self.timer * 43.0)
            + self.surge * 0.30
        )
        target_x = self.ux * cone_depth * 0.205
        target_y = self.uy * cone_depth * 0.205
        pitch = math.degrees(math.atan2(target_y, cone_depth))
        yaw = -math.degrees(math.atan2(target_x, cone_depth))

        self.volume_shell.position = Vec3(0, 0, -0.08)
        volume_sway = math.sin(self.timer * 4.1) * 2.8
        self.volume_shell.rotation = Vec3(
            pitch + math.sin(self.timer * 5.4) * 2.2,
            yaw + math.cos(self.timer * 4.7) * 2.2,
            self.timer * 9.0 + volume_sway,
        )
        self.volume_shell.scale = Vec3(
            3.58 * pressure,
            3.58 * pressure,
            cone_depth * 1.08 * (0.98 + 0.02 * pressure),
        )
        self.volume_shell.alpha = (0.39 + self.surge * 0.13) * pressure
        self.volume_shell.set_shader_input("time", self.timer * 1.55)

        self.volume_core.position = Vec3(0, 0, -0.10)
        self.volume_core.rotation = Vec3(
            pitch - math.sin(self.timer * 5.4) * 1.4,
            yaw - math.cos(self.timer * 4.7) * 1.4,
            -self.timer * 12.0 + volume_sway * 0.55,
        )
        self.volume_core.scale = Vec3(
            1.82 * pressure,
            1.82 * pressure,
            cone_depth * 1.01,
        )
        self.volume_core.alpha = (0.15 + self.surge * 0.085) * pressure
        self.volume_core.set_shader_input("time", self.timer * 1.85 + 2.4)

        self.emit_timer -= dt
        while self.emit_timer <= 0:
            self._emit()
            self.emit_timer += self.EMIT_INTERVAL

        self.spark_timer -= dt
        while self.spark_timer <= 0:
            # Moi nhip ban ba tia: mot tia bam loi va hai tia xe rong hai ben.
            # Cac vet dai, nhanh va lech goc nhe lam luong lua du doi hon.
            for spark_index in range(3):
                spread = 0.48 if spark_index == 0 else 1.0
                spark_velocity = self.base_velocity * random.uniform(1.20, 1.62)
                spark_velocity += Vec3(
                    random.uniform(-5.4, 5.4) * spread,
                    random.uniform(-4.2, 5.8) * spread,
                    random.uniform(-2.0, 1.2),
                )
                JetSpark(
                    self.nozzle + Vec3(
                        random.uniform(-0.26, 0.26),
                        random.uniform(-0.22, 0.22),
                        random.uniform(-0.10, 0.10),
                    ),
                    spark_velocity,
                    self.angle + random.uniform(-7.0, 7.0),
                )
            self.spark_timer += self.SPARK_INTERVAL


FIRE_SOUND_COOLDOWN = 0.55
_streams = {}
_last_sound = {}


def cast(lm, hand_to_world, hand_id=""):
    """Chieu "Xoe ban tay": phun mot luong lua lien tuc ve phia man hinh.

    - lm           : 21 landmarks ban tay (tu MediaPipe)
    - hand_to_world: ham doi toa do tay -> toa do 3D (main.py truyen vao)
    - hand_id      : khoa phan biet tay (vd "Left"/"Right") de cooldown doc lap
    """
    now = time.time()
    # Tam long ban tay nam giua co tay va hang bon khop MCP. Dat trong so hoi
    # ve co tay de nguon khong bi troi len sat cac ngon khi ban tay xoe rong.
    mcp_ids = (5, 9, 13, 17)
    mcp_x = sum(lm[i].x for i in mcp_ids) / len(mcp_ids)
    mcp_y = sum(lm[i].y for i in mcp_ids) / len(mcp_ids)
    palm_x = lm[0].x * 0.52 + mcp_x * 0.48
    palm_y = lm[0].y * 0.52 + mcp_y * 0.48
    pos = hand_to_world(palm_x, palm_y)
    dx = lm[12].x - lm[0].x
    dy = lm[0].y - lm[12].y
    n = math.hypot(dx, dy) or 1
    ux, uy = dx / n, dy / n
    # Nguon trung dung long ban tay; cac quad duoc neo o chan (origin_y=-0.5)
    # nen lua chi phun ra ngoai, khong chay nguoc xuyen qua canh tay.
    nozzle = pos
    base_velocity = Vec3(ux * 9.8, uy * 9.8, -28.0)
    screen_angle = -math.degrees(math.atan2(ux, uy))

    stream = _streams.get(hand_id)
    if stream is None:
        stream = FlameStream(hand_id)
        _streams[hand_id] = stream
    stream.refresh(nozzle, base_velocity, ux, uy, screen_angle)

    if now - _last_sound.get(hand_id, 0.0) >= FIRE_SOUND_COOLDOWN:
        play_sound("fireball.wav", volume=0.55)
        _last_sound[hand_id] = now


# ----------------------------------------------------------------------
# SPRITE EFFECT (BILLBOARD) — dung anh tu Canva, van giu do sau 3D
# ----------------------------------------------------------------------
class SpriteEffect(Entity):
    """Tam anh (quad) luon quay mat ve camera, bay trong khong gian 3D.

    - texture : Ursina Texture (dung load_effect_texture("file.png"))
    - Anh tinh: de frames=1.
    - Sprite sheet (nhieu frame trong 1 anh): dat frames, columns, fps.
      Vi du sheet 4x4 = 16 frame -> frames=16, columns=4.
    """

    def __init__(self, position, velocity, texture, size=3.0, life=3.0,
                 frames=1, columns=1, fps=20, spin=0.0, fade=False,
                 grow=1.0):
        super().__init__(model="quad", texture=texture, position=position,
                         double_sided=True, scale=size)
        self.velocity = velocity
        self.life = life
        self.max_life = life
        self.spin = spin                       # xoay quanh truc nhin (do/giay)
        self.fade = fade                       # True -> mo dan theo life
        self.grow = grow                       # >1 to dan, <1 nho dan moi giay
        self.timer = 0.0

        # Sprite sheet
        self.frames = frames
        self.columns = max(1, columns)
        self.rows = math.ceil(frames / self.columns)
        self.fps = fps
        if frames > 1:
            self.texture_scale = Vec2(1 / self.columns, 1 / self.rows)

    def update(self):
        self.position += self.velocity * utime.dt
        self.life -= utime.dt
        self.timer += utime.dt

        # Luon quay mat ve phia camera (billboard)
        self.look_at(camera.world_position)
        if self.spin:
            self.rotation_z += self.spin * utime.dt
        if self.grow != 1.0:
            self.scale *= (1 + (self.grow - 1) * utime.dt)
        if self.fade:
            self.alpha = max(0.0, self.life / self.max_life)

        # Chay animation neu la sprite sheet
        if self.frames > 1:
            idx = int(self.timer * self.fps) % self.frames
            col = idx % self.columns
            row = idx // self.columns
            self.texture_offset = Vec2(col / self.columns,
                                       1 - (row + 1) / self.rows)

        if self.z < 0.8 or self.life <= 0:
            ImpactBurst(self.world_position, self.size)
            destroy(self)

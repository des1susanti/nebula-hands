#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEBULA HANDS — sistem partikel 3D yang dikendalikan gestur tangan (computer vision)

Teknologi: OpenCV (kamera + rendering), MediaPipe Hand Landmarker (21 titik tangan, 2 tangan),
           NumPy (simulasi fisika partikel + proyeksi perspektif 3D), efek bloom/glow & motion trail.

MULTI-TANGAN: setiap tangan mendapat bentuknya sendiri (2 tangan 3 jari = 2 hati),
  kecuali 5 jari & finger heart yang tetap berupa efek layar penuh. (--hands 3 untuk 3 tangan)
TAMPILAN: layar terbagi dua -> kiri kamera + kerangka tangan, kanan kanvas partikel (tombol L = gabung/pisah)

GESTUR SATU TANGAN
  Telapak terbuka (5 jari)        -> Jaringan node kecil memenuhi layar (tangan menyibak node)
  Telunjuk saja (1 jari)          -> Otak 3D dengan jaringan neuron yang menyala
  4 jari (jempol dilipat)         -> Planet bercincin (Saturnus) berputar
  Finger heart ala Korea          -> Hujan hati kecil bertaburan di seluruh layar
  Peace (2 jari)                  -> Partikel menyusun teks 3D (default "I LOVE YOU")
  Tiga jari                       -> Hati 3D berdetak
  Kepalan                         -> Bola energi; TAHAN untuk mengisi daya, lalu BUKA tangan -> LEDAKAN
  Cubit (jempol + telunjuk)       -> Lubang hitam: partikel tersedot & mengorbit
  Metal (telunjuk + kelingking)   -> Kubus 3D berputar
GESTUR DUA TANGAN
  Ujung telunjuk & jempol kedua tangan saling bertemu (bentuk hati) -> Hati besar di tengah

TOMBOL
  Q/ESC keluar | L tampilan terbagi/gabung | S kerangka tangan | C palet warna (auto -> tetap)
  G glow on/off | [ ] kecerahan | F layar penuh | P screenshot | H bantuan

Jalankan:  python nebula_hands_v6.py --text "I LOVE YOU"
"""
import argparse
import math
import os
import sys
import time
import urllib.request
from collections import deque

import cv2
import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17)]
TIPS = (4, 8, 12, 16, 20)


# =====================================================================================
# 1. HAND TRACKING
# =====================================================================================
def _download(url, dest):
    """Unduh file; tahan terhadap masalah sertifikat SSL di Python macOS."""
    import shutil
    import ssl
    import subprocess
    tmp = dest + ".part"
    errors = []
    # 1) urllib dengan bundel sertifikat certifi (jika ada) atau bawaan sistem
    try:
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
        with urllib.request.urlopen(url, context=ctx, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(tmp, dest)
        return
    except Exception as e:  # noqa: BLE001
        errors.append(f"urllib: {e}")
    # 2) curl bawaan sistem (macOS/Linux/Windows 10+) memakai sertifikat milik OS
    if shutil.which("curl"):
        try:
            subprocess.run(["curl", "-fL", "--retry", "2", "-o", tmp, url], check=True)
            os.replace(tmp, dest)
            return
        except Exception as e:  # noqa: BLE001
            errors.append(f"curl: {e}")
    if os.path.exists(tmp):
        os.remove(tmp)
    raise RuntimeError(
        "Gagal mengunduh model.\n  " + "\n  ".join(errors) +
        f"\n\nSolusi manual: unduh file berikut lewat browser\n  {url}\n"
        f"lalu simpan dengan nama 'hand_landmarker.task' di folder:\n  {os.path.dirname(dest)}")


def ensure_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        print("[i] Mengunduh model hand landmarker (sekali saja, ~8 MB)...")
        _download(MODEL_URL, MODEL_PATH)
        print("[i] Model tersimpan:", MODEL_PATH)
    return MODEL_PATH


class HandTracker:
    """Memakai MediaPipe Tasks API (baru); otomatis jatuh ke mp.solutions (lama) bila perlu."""

    def __init__(self, max_hands=2):
        import mediapipe as mp
        self.mp = mp
        self._last_ts = -1
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            opts = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            self.detector = vision.HandLandmarker.create_from_options(opts)
            self.backend = "tasks"
        except Exception as e:  # noqa: BLE001
            if not hasattr(mp, "solutions"):     # mediapipe >= 0.10.30 tidak punya API lama
                sys.exit(f"[x] Hand tracker gagal dimuat:\n{e}")
            print(f"[!] Tasks API gagal ({e}); mencoba mp.solutions.hands ...")
            self.detector = mp.solutions.hands.Hands(
                max_num_hands=max_hands, model_complexity=1,
                min_detection_confidence=0.6, min_tracking_confidence=0.5)
            self.backend = "solutions"
        print(f"[i] Backend MediaPipe: {self.backend}")

    def process(self, rgb):
        hands = []
        if self.backend == "tasks":
            ts = int(time.monotonic() * 1000)
            if ts <= self._last_ts:
                ts = self._last_ts + 1
            self._last_ts = ts
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                data=np.ascontiguousarray(rgb))
            res = self.detector.detect_for_video(img, ts)
            for lms in res.hand_landmarks:
                hands.append(np.array([[p.x, p.y, p.z] for p in lms], np.float32))
        else:
            res = self.detector.process(rgb)
            if res.multi_hand_landmarks:
                for hl in res.multi_hand_landmarks:
                    hands.append(np.array([[p.x, p.y, p.z] for p in hl.landmark], np.float32))
        return hands

    def close(self):
        try:
            self.detector.close()
        except Exception:  # noqa: BLE001
            pass


# =====================================================================================
# 2. GESTURE RECOGNITION
# =====================================================================================
class HandInfo:
    __slots__ = ("P", "size", "palm", "gesture", "roll", "tip", "thumb", "pinch_pt")


def analyze_hand(lm, W, H):
    P = lm[:, :2] * np.array([W, H], np.float32)
    h = HandInfo()
    h.P = P
    h.size = float(np.linalg.norm(P[9] - P[0])) + 1e-6          # pergelangan -> pangkal jari tengah
    h.palm = P[[0, 5, 9, 13, 17]].mean(0)
    h.tip, h.thumb = P[8].copy(), P[4].copy()
    h.pinch_pt = (P[4] + P[8]) / 2
    h.roll = math.atan2(P[9, 0] - P[0, 0], P[0, 1] - P[9, 1])   # kemiringan tangan dari vertikal
    wrist = P[0]

    def extended(tip, pip, ratio=1.12):
        return np.linalg.norm(P[tip] - wrist) > np.linalg.norm(P[pip] - wrist) * ratio

    idx, mid, ring = extended(8, 6), extended(12, 10), extended(16, 14)
    pky = extended(20, 18, 1.06)                      # kelingking lebih pendek
    tipd = np.linalg.norm(P[[8, 12, 16, 20]] - h.palm, axis=1) / h.size
    curl = float(tipd.max())                          # kepalan = SEMUA ujung jari dekat telapak
    pinch = float(np.linalg.norm(P[4] - P[8])) / h.size
    # finger heart ala Korea: jempol menyilang di ujung/ruas telunjuk, 3 jari lain menggenggam
    thumb_on_index = min(np.linalg.norm(P[4] - P[j]) for j in (7, 8)) / h.size
    others_curl = float(np.mean(np.linalg.norm(P[[12, 16, 20]] - h.palm, axis=1))) / h.size
    index_out = float(np.linalg.norm(P[8] - h.palm)) / h.size
    thumb_folded = float(np.linalg.norm(P[4] - P[17])) / h.size < 0.85

    if thumb_on_index < 0.3 and others_curl < 0.95 and index_out > 0.75 and not (mid or ring or pky):
        g = "KHEART"
    elif curl < 1.0 and not (idx or mid or ring or pky):
        g = "FIST"
    elif pinch < 0.25:
        g = "PINCH"
    elif idx and mid and ring and pky:
        g = "FOUR" if thumb_folded else "OPEN"
    elif idx and mid and ring:
        g = "THREE"
    elif idx and mid:
        g = "PEACE"
    elif idx and pky and not mid and not ring:
        g = "ROCK"
    elif idx:
        g = "POINT"
    else:
        g = "NONE"
    h.gesture = g
    return h


class Stabilizer:
    """Gestur baru harus konsisten beberapa frame agar tidak berkedip-kedip."""

    def __init__(self, frames=4):
        self.cur, self.cand, self.cnt, self.n = "IDLE", None, 0, frames

    def update(self, g):
        if g == self.cur:
            self.cand, self.cnt = None, 0
        elif g == self.cand:
            self.cnt += 1
            if self.cnt >= self.n:
                self.cur, self.cand, self.cnt = g, None, 0
        else:
            self.cand, self.cnt = g, 1
        return self.cur


class Smooth:
    def __init__(self, a=0.35):
        self.v, self.a = None, a

    def __call__(self, x):
        x = np.asarray(x, np.float32)
        self.v = x.copy() if self.v is None else self.v + (x - self.v) * self.a
        return self.v


# =====================================================================================
# 3. BENTUK 3D (koordinat lokal, y ke atas, skala ~1)
# =====================================================================================
def shape_sphere(n, rng):
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    p = np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)
    p *= 1 + rng.normal(0, 0.03, (n, 1))
    return p[rng.permutation(n)]


def shape_galaxy(n, rng, arms=3):
    core = rng.random(n) < 0.18
    r = np.clip(rng.exponential(0.38, n), 0.03, 1.25)
    arm = rng.integers(0, arms, n)
    th = arm * 2 * np.pi / arms + r * 4.0 + rng.normal(0, 1, n) * (0.12 + 0.25 * r)
    p = np.stack([r * np.cos(th), rng.normal(0, 1, n) * 0.04 * (1.2 - r), r * np.sin(th)], 1)
    p[core] = rng.normal(0, 0.11, (core.sum(), 3)) * [1, 0.6, 1]
    return p


def shape_heart(n, rng):
    t = rng.uniform(0, 2 * np.pi, n)
    r = np.where(rng.random(n) < 0.55, rng.uniform(0.93, 1.0, n), np.sqrt(rng.random(n)))
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    z = rng.normal(0, 1, n) * 0.28 * np.sqrt(np.clip(1 - r ** 2, 0.02, 1))
    return np.stack([x * r / 17, (y * r + 3) / 17, z], 1)


def shape_text(text, n, rng):
    font, scale, th = cv2.FONT_HERSHEY_DUPLEX, 4.0, 14
    (w, h), base = cv2.getTextSize(text, font, scale, th)
    img = np.zeros((h + base + 40, w + 40), np.uint8)
    cv2.putText(img, text, (20, h + 20), font, scale, 255, th, cv2.LINE_AA)
    ys, xs = np.nonzero(img > 128)
    k = rng.integers(0, len(xs), n)
    half = img.shape[1] / 2
    x = (xs[k] + rng.uniform(-0.5, 0.5, n) - half) / half
    y = -(ys[k] + rng.uniform(-0.5, 0.5, n) - img.shape[0] / 2) / half
    return np.stack([x, y, rng.normal(0, 0.025, n)], 1)


def shape_dna(n, rng):
    t = rng.uniform(-1, 1, n)
    rung = rng.random(n) < 0.3
    t[rung] = np.round((t[rung] + 1) * 12) / 12 - 1
    ang = t * 3.2 * np.pi
    side = rng.integers(0, 2, n) * np.pi
    u = np.where(rung, rng.uniform(-1, 1, n), 1.0)
    a2 = np.where(rung, ang, ang + side)
    p = np.stack([0.38 * u * np.cos(a2), t * 1.2, 0.38 * u * np.sin(a2)], 1)
    return p + rng.normal(0, 0.012, (n, 3))


def shape_portal(n, rng):
    ring = rng.random(n) < 0.72
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    rr = 0.1 * np.sqrt(rng.random(n))
    torus = np.stack([(1 + rr * np.cos(v)) * np.cos(u), (1 + rr * np.cos(v)) * np.sin(u),
                      rr * np.sin(v)], 1)
    rd = np.sqrt(rng.random(n)) * 0.9                      # pusaran di dalam cincin
    sw = u + rd * 6
    disk = np.stack([rd * np.cos(sw), rd * np.sin(sw), rng.normal(0, 0.02, n)], 1)
    return np.where(ring[:, None], torus, disk)


def shape_saturn(n, rng):
    ring = rng.random(n) < 0.45
    planet = shape_sphere(n, rng) * 0.48
    planet *= rng.uniform(0.9, 1.0, (n, 1))                   # sedikit volume
    a = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(0.68, 1.15, n)
    r = np.where(rng.random(n) < 0.15, rng.uniform(0.68, 0.74, n), r)   # celah cincin tipis
    disk = np.stack([r * np.cos(a), rng.normal(0, 0.012, n), r * np.sin(a)], 1)
    return np.where(ring[:, None], disk, planet), ring


def saturn_colors(shape, ring):
    """Planet oranye bergaris-garis, cincin keemasan (BGR)."""
    n = len(shape)
    band = 0.5 + 0.5 * np.sin(shape[:, 1] * 38)
    planet = np.stack([0.05 + 0.08 * band, 0.35 + 0.25 * band, np.full(n, 1.0)], 1)
    rr = np.linalg.norm(shape[:, [0, 2]], axis=1)
    shade = 0.75 + 0.25 * np.sin(rr * 40)
    ringc = np.stack([0.45 * shade, 0.82 * shade, np.full(n, 1.0) * shade], 1)
    return np.where(ring[:, None], ringc, planet).astype(np.float32)


def shape_cloud(n, rng):
    p = rng.uniform(-1, 1, (n, 3)) * [1.0, 0.85, 0.6]
    return p + rng.normal(0, 0.02, (n, 3))


def shape_cube(n, rng):
    """Kubus: 60% partikel di 12 rusuk (tajam), 40% di permukaan sisi."""
    edge = rng.random(n) < 0.6
    p = rng.uniform(-1, 1, (n, 3))
    # rusuk: dua sumbu dikunci ke +-1
    ax = rng.integers(0, 3, n)
    sgn = rng.choice([-1.0, 1.0], (n, 3))
    e = sgn.copy()
    e[np.arange(n), ax] = p[np.arange(n), ax]
    # sisi: satu sumbu dikunci ke +-1
    f = p.copy()
    f[np.arange(n), ax] = sgn[np.arange(n), ax]
    out = np.where(edge[:, None], e, f) * 0.62
    return out + rng.normal(0, 0.008, (n, 3))


def shape_brain(n, rng):
    """Otak: dua belahan (hemisfer) bergyri + otak kecil + batang otak."""
    kind = rng.random(n)
    u = rng.normal(0, 1, (n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    th = np.arctan2(u[:, 2], u[:, 0])
    ph = np.arccos(np.clip(u[:, 1], -1, 1))
    gyri = 1 + 0.07 * np.sin(13 * th + 3 * np.sin(5 * ph)) * np.sin(11 * ph)
    shell = rng.uniform(0.86, 1.0, n) * gyri
    side = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    hemi = np.stack([np.abs(u[:, 0]) * 0.42 * shell * side + side * 0.05,
                     u[:, 1] * 0.48 * shell + 0.08,
                     u[:, 2] * 0.72 * shell], 1)
    cereb = np.stack([u[:, 0] * 0.32, u[:, 1] * 0.16 - 0.38, u[:, 2] * 0.2 - 0.48], 1)
    stem_t = rng.random(n)
    stem = np.stack([rng.normal(0, 0.05, n), -0.3 - stem_t * 0.45,
                     -0.22 + rng.normal(0, 0.05, n) - stem_t * 0.05], 1)
    out = np.where((kind < 0.82)[:, None], hemi, np.where((kind < 0.94)[:, None], cereb, stem))
    return out


def rot(yaw=0.0, pitch=0.0, roll=0.0):
    cy, sy, cp, sp, cr, sr = (math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch),
                              math.cos(roll), math.sin(roll))
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return (Rz @ Rx @ Ry).astype(np.float32)



# =====================================================================================
# 4. PALET WARNA (BGR, 0..1)
# =====================================================================================
PALETTES = {
    "ocean":   [(1.0, 0.43, 0.16), (1.0, 0.62, 0.30), (0.9, 0.27, 0.12)],
    "neuro":   [(0.78, 0.31, 1.0), (1.0, 0.35, 0.55), (1.0, 0.86, 0.31)],
    "sakura":  [(0.70, 0.41, 1.0), (0.85, 0.75, 1.0), (0.39, 0.08, 1.0)],
    "saturn":  [(0.08, 0.47, 1.0), (0.24, 0.70, 1.0), (0.04, 0.31, 0.9)],
    "ice":     [(1.0, 0.78, 0.16), (1.0, 0.88, 0.35), (0.94, 0.67, 0.08)],
    "rose":    [(0.35, 0.12, 1.0), (0.50, 0.25, 1.0), (0.27, 0.08, 0.9)],
    "inferno": [(0.3, 0.9, 1.0), (0.1, 0.45, 1.0), (0.15, 0.1, 0.9)],
    "violet":  [(1.0, 0.2, 0.5), (1.0, 0.5, 0.2), (0.8, 0.4, 1.0)],
    "cyber":   [(1.0, 0.9, 0.2), (1.0, 0.4, 0.15), (1.0, 0.2, 0.9)],
}
PALETTE_ORDER = list(PALETTES)


def palette_colors(name, h):
    c = np.array(PALETTES[name], np.float32)
    x = h * (len(c) - 1)
    i = np.clip(x.astype(int), 0, len(c) - 2)
    f = (x - i)[:, None]
    return (c[i] * (1 - f) + c[i + 1] * f).astype(np.float32)


# =====================================================================================
# 5. MODE / PERILAKU
# =====================================================================================
MODES = {
    #          palet       kekakuan   redaman   turbulensi   (sangat responsif)
    "IDLE":   dict(pal="ocean",   k=0.045, damp=0.74, turb=0.10),
    "OPEN":   dict(pal="ocean",   k=0.080, damp=0.70, turb=0.05),
    "POINT":  dict(pal="neuro",   k=0.100, damp=0.68, turb=0.0),
    "FOUR":   dict(pal="saturn",  k=0.100, damp=0.68, turb=0.0),
    "PEACE":  dict(pal="ice",     k=0.090, damp=0.68, turb=0.0),
    "THREE":  dict(pal="rose",    k=0.100, damp=0.68, turb=0.0),
    "KHEART": dict(pal="sakura",  k=0.100, damp=0.68, turb=0.0),
    "FIST":   dict(pal="inferno", k=0.130, damp=0.64, turb=0.0),
    "PINCH":  dict(pal="violet",  k=0.0,   damp=0.88, turb=0.04),
    "ROCK":   dict(pal="cyber",   k=0.100, damp=0.68, turb=0.0),
    "HEART2": dict(pal="rose",    k=0.100, damp=0.68, turb=0.0),
    "PORTAL": dict(pal="cyber",   k=0.100, damp=0.68, turb=0.0),
}
LABELS = {
    "IDLE": "Menunggu tangan...", "OPEN": "5 jari : Jaringan node",
    "POINT": "1 jari : Otak neuron", "FOUR": "4 jari : Planet bercincin",
    "PEACE": "2 jari : Teks 3D", "THREE": "3 jari : Hati 3D",
    "KHEART": "Finger heart : Hujan hati", "FIST": "Kepalan : Mengisi energi",
    "PINCH": "Cubit : Lubang hitam", "ROCK": "Metal : Kubus 3D",
    "HEART2": "Dua tangan : Hati besar", "PORTAL": "Dua telapak : Portal",
}


# =====================================================================================
# 6. PARTIKEL & RENDERER
# =====================================================================================
class Particles:
    def __init__(self, n, W, H, rng):
        self.n, self.rng = n, rng
        self.pos = (rng.normal(0, 1, (n, 3)) * [W * 0.3, H * 0.3, 200]).astype(np.float32)
        self.vel = np.zeros((n, 3), np.float32)
        self.hue = rng.random(n).astype(np.float32)
        self.k_mul = rng.uniform(0.8, 1.2, (n, 1)).astype(np.float32)
        self.bright = rng.uniform(0.6, 1.0, n).astype(np.float32)
        self.jit = rng.normal(0, 1, (n, 3)).astype(np.float32)
        self._pal_cache = {k: palette_colors(k, self.hue) for k in PALETTES}
        self.col = self._pal_cache["ocean"].copy()
        self.speed = np.zeros(n, np.float32)

    def blend_colors(self, target_col, rate=0.15):
        self.col += (target_col - self.col) * rate

    def burst(self, center, power, idx=slice(None)):
        d = self.pos[idx] - center
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-3
        d += self.rng.normal(0, 0.35, d.shape).astype(np.float32)
        self.vel[idx] += d * self.rng.uniform(8, 42, (len(d), 1)).astype(np.float32) * power

    def repel(self, pt, radius, strength):
        d = self.pos[:, :2] - pt[None, :2]
        dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-3
        f = np.clip(1 - dist / radius, 0, 1) * strength
        self.vel[:, :2] += d / dist * f

    def step(self, target, k, damp, turb, t, attractors=()):
        """k, damp, turb berupa array (N,1) -> tiap kelompok partikel bisa berperilaku berbeda."""
        acc = (target - self.pos) * (k * self.k_mul)
        if np.any(turb > 0):
            p = self.pos * 0.008
            acc += turb * np.stack([
                np.sin(p[:, 1] * 1.7 + t) + np.cos(p[:, 2] * 1.3 - t * 0.7),
                np.sin(p[:, 2] * 1.5 + t * 1.1) + np.cos(p[:, 0] * 1.9),
                np.sin(p[:, 0] * 1.3 - t) + np.cos(p[:, 1] * 1.1 + t * 0.5)], 1).astype(np.float32)
        for pt, strength, swirl, ring_r, idx in attractors:
            pos = self.pos[idx]
            d = pt[None, :] - pos
            dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-3
            a = d / dist * strength * np.tanh((dist - ring_r) / 40.0)
            a += np.stack([-d[:, 1], d[:, 0], -pos[:, 2] * 0.3], 1) / dist * swirl
            acc[idx] += a
        self.vel *= damp
        self.vel += acc
        sp = np.linalg.norm(self.vel, axis=1)
        over = sp > 90
        if over.any():
            self.vel[over] *= (90 / sp[over])[:, None]
            sp[over] = 90
        self.speed = sp
        self.pos += self.vel


class Renderer:
    """Titik partikel tajam (2x2 px) + glow tipis opsional, agar bentuk terlihat jelas."""

    def __init__(self, W, H, focal=900.0):
        self.W, self.H, self.f = W, H, focal
        self.trail = np.zeros((H, W, 3), np.float32)
        self.kernel = np.ones((2, 2), np.uint8)

    def resize(self, W, H):
        self.W, self.H = W, H
        self.trail = np.zeros((H, W, 3), np.float32)

    def draw(self, ps, exposure=1.0, boost=1.0, glow=True, decay=0.25):
        W, H = self.W, self.H
        x, y, z = ps.pos[:, 0], ps.pos[:, 1], ps.pos[:, 2]
        denom = self.f + z
        s = self.f / np.maximum(denom, 60)
        sx = (W / 2 + x * s).astype(np.int32)
        sy = (H / 2 - y * s).astype(np.int32)
        ok = (denom > 60) & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
        heat = np.clip(ps.speed / 35, 0, 1)[:, None]
        col = ps.col * (1 - heat * 0.4) + heat * 0.6                 # partikel cepat = lebih putih
        depth = np.clip(0.55 + 0.45 * (s - 0.7) / 0.6, 0.35, 1.3)    # yang jauh lebih redup
        col *= (ps.bright * depth * 0.9 * boost)[:, None]
        idx = sy[ok] * W + sx[ok]
        layer = np.empty((H, W, 3), np.float32)
        for c in range(3):
            layer[..., c] = np.bincount(idx, weights=col[ok, c], minlength=W * H).reshape(H, W)
        layer = cv2.dilate(layer, self.kernel)                       # titik 2x2 piksel
        hdr = layer
        if glow:
            hdr = layer + cv2.GaussianBlur(layer, (0, 0), 3.0) * 1.6
        self.trail *= decay
        self.trail += hdr
        return 1.0 - np.exp(-self.trail * exposure)                   # tone mapping


# =====================================================================================
# 7. UI
# =====================================================================================
def put(img, text, org, scale=0.6, color=(255, 255, 255), th=1):
    cv2.putText(img, text, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                th + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def draw_skeleton(img, info, color):
    pts = info.P.astype(int)
    for a, b in HAND_EDGES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(img, tuple(p), 5 if i in TIPS else 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(p), 5 if i in TIPS else 3, color, 1, cv2.LINE_AA)


def fit_frame(frame, W, H):
    """Potong bagian tengah kamera agar rasio sama dengan panel, lalu ubah ukuran."""
    fh, fw = frame.shape[:2]
    ta = W / H
    if fw / fh > ta:
        nw = int(fh * ta)
        x0 = (fw - nw) // 2
        frame = frame[:, x0:x0 + nw]
    else:
        nh = int(fw / ta)
        y0 = (fh - nh) // 2
        frame = frame[y0:y0 + nh]
    return cv2.resize(frame, (W, H))


HELP = "Q keluar | L layout | S kerangka | C palet | G glow | [ ] terang | F penuh | P foto | H bantuan"


# =====================================================================================
# 8. MAIN LOOP
# =====================================================================================
def main():
    ap = argparse.ArgumentParser(description="NEBULA HANDS - partikel 3D dengan gestur tangan")
    ap.add_argument("--text", default="I LOVE YOU", help="teks untuk gestur 2 jari")
    ap.add_argument("--particles", type=int, default=7000, help="jumlah partikel")
    ap.add_argument("--camera", type=int, default=0, help="indeks kamera")
    ap.add_argument("--width", type=int, default=800, help="lebar tiap panel")
    ap.add_argument("--height", type=int, default=600, help="tinggi panel")
    ap.add_argument("--layout", choices=["split", "overlay"], default="split")
    ap.add_argument("--hands", type=int, default=2, help="jumlah tangan maksimal (1-4)")
    args = ap.parse_args()

    W, H, N = args.width, args.height, args.particles
    rng = np.random.default_rng(7)

    saturn_shape, saturn_ring = shape_saturn(N, rng)
    shapes = {
        "cloud": shape_cloud(N, rng), "saturn": saturn_shape,
        "heart": shape_heart(N, rng), "text": shape_text(args.text, N, rng),
        "galaxy": shape_galaxy(N, rng), "sphere": shape_sphere(N, rng),
        "portal": shape_portal(N, rng),
        "cube": shape_cube(N, rng), "brain": shape_brain(N, rng),
    }
    shapes = {k: v.astype(np.float32) for k, v in shapes.items()}

    # jaringan node layar penuh (5 jari)
    field = np.stack([rng.uniform(-0.5, 0.5, N) * W * 1.05, rng.uniform(-0.5, 0.5, N) * H * 1.05,
                      rng.uniform(-150, 150, N)], 1).astype(np.float32)
    n_hub = min(180, N)                         # node besar yang saling terhubung garis
    # hujan hati kecil (finger heart)
    n_mh = 42
    mh_x = rng.uniform(-0.48, 0.48, n_mh) * W
    mh_y0 = rng.uniform(0, 1, n_mh)
    mh_spd = rng.uniform(0.06, 0.16, n_mh)
    mh_size = rng.uniform(12, 34, n_mh)
    mh_phase = rng.uniform(0, 2 * np.pi, n_mh)
    mh_id = np.arange(N) % n_mh

    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        sys.exit("[x] Kamera tidak bisa dibuka. Coba --camera 1")

    MAXH = int(np.clip(args.hands, 1, 4))
    tracker = HandTracker(max_hands=MAXH)
    ps = Particles(N, W, H, rng)
    ps._pal_cache["saturn"] = saturn_colors(saturn_shape, saturn_ring)
    ren = Renderer(W, H)

    class Slot:                                   # status terpisah untuk tiap tangan
        def __init__(self):
            self.stab = Stabilizer(2)
            self.anchor, self.roll, self.size = Smooth(0.75), Smooth(0.6), Smooth(0.5)
            self.mode, self.prev, self.charge = "IDLE", "IDLE", 0.0
            self.info = None

    slots = [Slot() for _ in range(MAXH)]
    sm_center2, sm_dist2 = Smooth(0.6), Smooth(0.5)
    burst_until = np.zeros(N, np.float32)
    node_alpha = 0.0
    ALL = np.arange(N)
    GLOBAL_MODES = ("OPEN", "KHEART")             # efek layar penuh, tidak dibagi per tangan

    layout = args.layout
    win = "NEBULA HANDS"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W * 2 if layout == "split" else W, H)

    show_skel, show_help, fullscreen, glow_on = True, True, False, True
    pal_lock = None                              # None = otomatis per mode
    exposure = 1.0
    flash = 0.0
    fps, t0 = 0.0, time.time()
    last = time.time()

    def to_world(pt):
        return np.array([pt[0] - W / 2, H / 2 - pt[1], 0.0], np.float32)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[x] Gagal membaca frame kamera. Cek izin kamera untuk Terminal.")
            break
        frame = fit_frame(cv2.flip(frame, 1), W, H)
        now = time.time()
        dt, last = min(now - last, 0.1), now
        t = now - t0
        beat = 1 + 0.12 * max(0.0, math.sin(t * 2 * math.pi * 1.1)) ** 6

        # ---------- deteksi tangan ----------
        det = cv2.cvtColor(cv2.resize(frame, (480, int(480 * H / W))), cv2.COLOR_BGR2RGB)
        infos = [analyze_hand(lm, W, H) for lm in tracker.process(det)][:MAXH]
        infos.sort(key=lambda h: h.palm[0])        # urut kiri -> kanan agar slot stabil
        nh_ = len(infos)

        for i, sl in enumerate(slots):
            if i < nh_:
                h = sl.info = infos[i]
                if h.gesture != "NONE":
                    sl.mode = sl.stab.update(h.gesture)
                sl.anchor(h.palm), sl.roll(h.roll), sl.size(h.size)
            else:
                sl.info, sl.mode, sl.stab = None, "IDLE", Stabilizer(2)
        active = slots[:nh_]

        # ---------- mode global ----------
        gmode = None
        if nh_ >= 2:
            a, b = infos[0], infos[1]
            avg = (a.size + b.size) / 2
            if (np.linalg.norm(a.tip - b.tip) < 0.7 * avg
                    and np.linalg.norm(a.thumb - b.thumb) < 0.7 * avg):
                gmode = "HEART2"
        if gmode is None:
            for gm in ("KHEART", "OPEN"):
                if any(sl.mode == gm for sl in active):
                    gmode = gm
                    break
        if gmode is None and nh_ == 0:
            gmode = "IDLE"

        # ---------- ledakan saat kepalan dilepas (per tangan) ----------
        groups = [ALL[g::nh_] for g in range(nh_)] if nh_ else []
        for g, sl in enumerate(active):
            if sl.mode != sl.prev:
                if sl.prev == "FIST" and sl.charge > 0.3:
                    cf = min(sl.charge, 3.0) / 3.0
                    idx = ALL if gmode else groups[g]
                    ps.burst(to_world(sl.anchor.v), 0.4 + cf * 1.2, idx)
                    burst_until[idx] = t + 0.9
                    flash = max(flash, cf)
                sl.prev = sl.mode
            if sl.mode != "FIST":
                sl.charge = 0.0

        # ---------- target per partikel ----------
        target = ps.pos.copy()
        k = np.zeros((N, 1), np.float32)
        damp = np.full((N, 1), 0.8, np.float32)
        turb = np.zeros((N, 1), np.float32)
        col_t = np.empty_like(ps.col)
        attractors = []
        boost = 1.0
        labels, brain_hs = [], []
        hub_on = np.zeros(n_hub, bool)

        def apply_cfg(idx, m):
            c = MODES[m]
            k[idx], damp[idx], turb[idx] = c["k"], c["damp"], c["turb"]
            col_t[idx] = ps._pal_cache[pal_lock or c["pal"]][idx]

        if gmode == "IDLE":
            target = shapes["cloud"] @ rot(yaw=t * 0.1).T * (min(W, H) * 0.45)
            apply_cfg(ALL, "IDLE")
            labels.append(LABELS["IDLE"])
        elif gmode == "OPEN":
            drift = np.stack([np.sin(t * 0.7 + ps.jit[:, 0] * 3), np.cos(t * 0.6 + ps.jit[:, 1] * 3),
                              np.zeros(N, np.float32)], 1) * 14
            target = field + drift
            apply_cfg(ALL, "OPEN")
            hub_on[:] = True
            labels.append(LABELS["OPEN"])
        elif gmode == "KHEART":
            span = H + 120
            cy = (mh_y0 * span + mh_spd * span * t) % span - span / 2   # melayang naik
            cx = mh_x + np.sin(t * 1.3 + mh_phase) * 18
            pulse = 1 + 0.15 * np.maximum(0, np.sin(t * 6 + mh_phase)) ** 4
            sz = (mh_size * pulse)[mh_id, None]
            ctr = np.stack([cx, cy, np.zeros(n_mh)], 1).astype(np.float32)[mh_id]
            target = shapes["heart"] * sz + ctr
            apply_cfg(ALL, "KHEART")
            labels.append(LABELS["KHEART"])
        elif gmode == "HEART2":
            a, b = infos[0], infos[1]
            c = to_world(sm_center2((a.tip + b.tip + a.thumb + b.thumb) / 4))
            d = float(sm_dist2(np.linalg.norm(a.palm - b.palm)))
            R = rot(yaw=math.sin(t * 0.8) * 0.5)
            target = shapes["heart"] @ R.T * (max(d, 150) * 0.7 * beat) + c
            apply_cfg(ALL, "HEART2")
            labels.append(LABELS["HEART2"])
        else:
            # ===== satu bentuk untuk SETIAP tangan =====
            for g, sl in enumerate(active):
                idx, m = groups[g], sl.mode
                anchor = to_world(sl.anchor.v)
                roll = float(sl.roll.v)
                hs = float(sl.size.v)
                if m == "IDLE":             # baru muncul, gestur belum stabil -> awan kecil di tangan
                    target[idx] = shapes["cloud"][idx] @ rot(yaw=t * 0.3).T * (hs * 1.5) + anchor
                elif m == "POINT":
                    R = rot(yaw=t * 0.7, pitch=0.25, roll=-roll * 0.4)
                    target[idx] = shapes["brain"][idx] @ R.T * (hs * 2.3) + anchor + ps.jit[idx] * 1.5
                    hub_on |= np.isin(np.arange(n_hub), idx)
                    brain_hs.append(hs)
                elif m == "FOUR":
                    R = rot(yaw=t * 0.9, pitch=0.45, roll=0.35 - roll * 0.5)
                    target[idx] = shapes["saturn"][idx] @ R.T * (hs * 1.9) + anchor
                elif m == "PEACE":
                    if nh_ == 1:
                        px, py = sl.info.palm
                        R = rot(yaw=(px - W / 2) / W * 1.4, pitch=(py - H / 2) / H * 0.8)
                        tt = shapes["text"][idx] @ R.T * (W * 0.42)
                    else:               # beberapa tangan -> teks kecil di tiap tangan
                        R = rot(yaw=math.sin(t * 0.9 + g) * 0.4)
                        tt = shapes["text"][idx] @ R.T * (W * 0.17) + anchor
                    tt[:, 1] += np.sin(tt[:, 0] * 0.012 + t * 3) * 6
                    target[idx] = tt
                elif m == "THREE":
                    R = rot(yaw=math.sin(t * 0.8 + g) * 0.6, roll=-roll * 0.5)
                    target[idx] = shapes["heart"][idx] @ R.T * (hs * 1.8 * beat) + anchor
                elif m == "FIST":
                    sl.charge += dt
                    cf = min(sl.charge, 3.0) / 3.0
                    R = rot(yaw=t * 2.5, pitch=t * 1.3)
                    target[idx] = (shapes["sphere"][idx] @ R.T * (hs * (0.95 - 0.4 * cf)) + anchor
                                   + ps.jit[idx] * (2 + 9 * cf))
                    boost = max(boost, 1.0 + cf * 0.8)
                elif m == "PINCH":
                    attractors.append((to_world(sl.info.pinch_pt), 2.6, 2.2, hs * 0.45, idx))
                elif m == "ROCK":
                    R = rot(yaw=t * 1.2 + g, pitch=t * 0.7 + 0.5, roll=-roll)
                    target[idx] = shapes["cube"][idx] @ R.T * (hs * 2.2) + anchor
                apply_cfg(idx, m)
                labels.append(LABELS.get(m, m).split(" : ")[-1])

        hot = burst_until > t                      # biarkan ledakan "terbang" dulu
        if hot.any():
            k[hot] *= 0.15
            damp[hot] = 0.965
        substeps = int(np.clip(round(dt * 45), 1, 3))    # FPS rendah -> fisika tetap cepat
        for _ in range(substeps):
            ps.step(target, k, damp, turb, t, attractors)
        if gmode == "OPEN":                       # setiap tangan menyibak jaringan node
            for sl in active:
                ps.repel(to_world(sl.anchor.v), float(sl.size.v) * 2.0, 5.0)
        ps.blend_colors(col_t)

        flash = max(0.0, flash - dt * 1.5)
        glow = ren.draw(ps, exposure, boost=boost + flash * 1.5, glow=glow_on)
        canvas = (np.clip(glow, 0, 1) * 255).astype(np.uint8)

        # garis antar node (jaringan layar penuh / neuron otak)
        node_alpha += ((1.0 if hub_on.any() else 0.0) - node_alpha) * 0.25
        if node_alpha > 0.03 and hub_on.any():
            sel = np.nonzero(hub_on)[0]
            hp = ps.pos[sel]
            sc = ren.f / np.maximum(ren.f + hp[:, 2], 60)
            hx = (W / 2 + hp[:, 0] * sc).astype(np.int32)
            hy = (H / 2 - hp[:, 1] * sc).astype(np.int32)
            dd = np.hypot(hx[:, None] - hx[None, :], hy[:, None] - hy[None, :])
            is_brain = gmode != "OPEN"
            lim = max(40.0, min(brain_hs) * 0.6) if (is_brain and brain_hs) else 115.0
            ii, jj = np.nonzero(np.triu(dd < lim, 1))
            base = np.array(PALETTES[pal_lock or ("neuro" if is_brain else "ocean")][1]) * 255
            lines = canvas.copy()
            for i, j in zip(ii, jj):
                w = 1 - dd[i, j] / lim
                cv2.line(lines, (int(hx[i]), int(hy[i])), (int(hx[j]), int(hy[j])),
                         tuple(float(v) * (0.35 + 0.65 * w) for v in base), 1, cv2.LINE_AA)
            fire = rng.random(len(sel)) < (0.04 if is_brain else 0.0)   # neuron "menyala"
            for q, (x_, y_) in enumerate(zip(hx, hy)):
                if fire[q]:
                    cv2.circle(lines, (int(x_), int(y_)), 4, (255, 255, 255), -1, cv2.LINE_AA)
                else:
                    cv2.circle(lines, (int(x_), int(y_)), 2 if is_brain else 3,
                               tuple(float(v) for v in base), -1, cv2.LINE_AA)
            canvas = cv2.addWeighted(lines, node_alpha, canvas, 1 - node_alpha, 0)

        # ---------- komposit ----------
        def sk_color(sl):
            m = gmode or sl.mode
            pal_ = pal_lock or MODES.get(m, MODES["IDLE"])["pal"]
            return tuple(int(v * 255) for v in PALETTES[pal_][0])

        if layout == "split":
            out = np.hstack([frame.copy(), canvas])
            cv2.line(out, (W, 0), (W, H), (60, 60, 60), 2)
        else:
            bg = frame.astype(np.float32) * (0.25 / 255)
            o = 1 - (1 - bg) * (1 - glow)
            out = (np.clip(o, 0, 1) * 255).astype(np.uint8)
        if show_skel:
            for sl in active:
                draw_skeleton(out, sl.info, sk_color(sl))

        # ---------- HUD ----------
        fps = fps * 0.9 + (1.0 / max(dt, 1e-3)) * 0.1
        put(out, "NEBULA HANDS", (16, 32), 0.8, (255, 240, 200), 2)
        put(out, " + ".join(labels), (16, 62), 0.65, (200, 255, 255), 2)
        put(out, f"{fps:4.1f} FPS | {N} partikel | tangan: {nh_} | palet: {pal_lock or 'auto'}",
            (16, 88), 0.45, (200, 200, 200))
        cmax = max([sl.charge for sl in active if sl.mode == "FIST"] or [0.0])
        if cmax > 0:
            cf = min(cmax, 3.0) / 3.0
            cv2.rectangle(out, (16, 100), (216, 114), (80, 80, 80), 1)
            cv2.rectangle(out, (17, 101), (17 + int(198 * cf), 113), (40, 140, 255), -1)
            put(out, "buka tangan untuk meledakkan!", (226, 113), 0.45, (40, 200, 255))
        if show_help:
            put(out, HELP, (16, H - 16), 0.45, (180, 180, 180))

        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("l"):
            layout = "overlay" if layout == "split" else "split"
            if not fullscreen:
                cv2.resizeWindow(win, W * 2 if layout == "split" else W, H)
        elif key == ord("s"):
            show_skel = not show_skel
        elif key == ord("g"):
            glow_on = not glow_on
        elif key == ord("h"):
            show_help = not show_help
        elif key == ord("c"):
            if pal_lock is None:
                pal_lock = PALETTE_ORDER[0]
            else:
                i = PALETTE_ORDER.index(pal_lock) + 1
                pal_lock = PALETTE_ORDER[i] if i < len(PALETTE_ORDER) else None
        elif key == ord("["):
            exposure = max(0.2, exposure - 0.1)
        elif key == ord("]"):
            exposure = min(3.0, exposure + 0.1)
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord("p"):
            fn = time.strftime("nebula_%Y%m%d_%H%M%S.png")
            cv2.imwrite(fn, out)
            print(f"[i] Screenshot disimpan: {fn}")

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

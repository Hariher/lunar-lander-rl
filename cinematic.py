"""Render a trained Lunar Lander agent as a 1080p / 60 fps cinematic video.

The physics, the model and the landing are the real ones from Gymnasium.
Only the drawing is replaced: a booster-style rocket, a lunar landscape,
engine plume, dust, and a launch-webcast style HUD.

Usage:
    python cinematic.py models/ppo_none_s0                # picks the best landing
    python cinematic.py models/ppo_none_s0 --seed 7       # a specific episode
"""

import argparse
import math
import os
import random
import subprocess

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import imageio_ffmpeg
import numpy as np
import pygame
from pygame import gfxdraw
from stable_baselines3 import PPO

import sound

W, H = 1920, 1080
PHYS_FPS = 50
LEG_H = 8 / 30.0          # half-height of a physics leg, in world units
METERS_PER_UNIT = 12.0    # rocket is ~3.4 units tall, drawn as a ~41 m booster

FONT_NUM = "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf"
FONT_LABEL = "/System/Library/Fonts/Avenir Next Condensed.ttc"


# --------------------------------------------------------------------------
# 1. Fly the episode with the real physics and record everything we need
# --------------------------------------------------------------------------

def fly(model, seed):
    env = gym.make("LunarLander-v3")
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped

    terrain = [p[0] for p in u.sky_polys] + [u.sky_polys[-1][1]]
    pad = (u.helipad_x1, u.helipad_x2, u.helipad_y)

    def snapshot(action, score):
        lander = u.lander
        feet = []
        for leg in u.legs:
            foot = leg.GetWorldPoint((0, -LEG_H))
            local = lander.GetLocalPoint(foot)
            feet.append((local[0], local[1]))
        return dict(
            x=lander.position[0], y=lander.position[1], a=lander.angle,
            vx=lander.linearVelocity[0], vy=lander.linearVelocity[1],
            feet=feet, action=action, score=score,
            contact=u.legs[0].ground_contact or u.legs[1].ground_contact,
        )

    frames = [snapshot(0, 0.0)]
    score, done, reward, terminated = 0.0, False, 0.0, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        score += reward
        frames.append(snapshot(int(action), score))
        done = terminated or truncated
    env.close()
    landed = terminated and reward == 100
    return dict(frames=frames, terrain=terrain, pad=pad, score=score,
                landed=landed, seed=seed)


def pick_episode(model, seed):
    if seed is not None:
        return fly(model, seed)
    best = None
    for s in range(30):
        ep = fly(model, s)
        secs = len(ep["frames"]) / PHYS_FPS
        if not ep["landed"] or secs < 5:
            continue
        if best is None or ep["score"] > best["score"]:
            best = ep
    if best is None:
        best = fly(model, 0)
    return best


# --------------------------------------------------------------------------
# 2. Small drawing helpers
# --------------------------------------------------------------------------

def lerp(a, b, t):
    return a + (b - a) * t


def lerp_col(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def clip_pts(pts):
    return [(int(max(-30000, min(30000, x))), int(max(-30000, min(30000, y)))) for x, y in pts]


def poly(surf, pts, color, aa=True):
    pts = clip_pts(pts)
    if len(pts) < 3:
        return
    gfxdraw.filled_polygon(surf, pts, color)
    if aa:
        gfxdraw.aapolygon(surf, pts, color)


_glow_cache = {}


def glow_sprite(radius, color, falloff=2.2):
    key = (int(radius), color, falloff)
    if key not in _glow_cache:
        r = max(2, int(radius))
        yy, xx = np.mgrid[-r:r, -r:r]
        d = np.sqrt(xx ** 2 + yy ** 2) / r
        k = np.clip(1 - d, 0, 1) ** falloff
        arr = np.zeros((2 * r, 2 * r, 3), np.uint8)
        for i in range(3):
            arr[..., i] = (k * color[i]).astype(np.uint8)
        s = pygame.surfarray.make_surface(arr.transpose(1, 0, 2))
        _glow_cache[key] = s
        if len(_glow_cache) > 400:
            _glow_cache.pop(next(iter(_glow_cache)))
    return _glow_cache[key]


def add_glow(surf, pos, radius, color, falloff=2.2):
    if radius < 2:
        return
    g = glow_sprite(radius, color, falloff)
    surf.blit(g, (pos[0] - g.get_width() / 2, pos[1] - g.get_height() / 2),
              special_flags=pygame.BLEND_RGB_ADD)


_dot_cache = {}


def soft_dot(radius, color):
    key = (int(radius), color)
    if key not in _dot_cache:
        r = max(2, int(radius))
        s = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
        yy, xx = np.mgrid[-r:r, -r:r]
        d = np.sqrt(xx ** 2 + yy ** 2) / r
        a = (np.clip(1 - d, 0, 1) ** 1.6 * 255).astype(np.uint8)
        s.fill(color + (0,))
        pygame.surfarray.pixels_alpha(s)[:] = a.T
        _dot_cache[key] = s
    return _dot_cache[key]


def value_noise(w, h, cells, rng):
    """Smooth random field in [0, 1], shape (h, w)."""
    gw, gh = cells + 1, max(2, int(cells * h / w)) + 1
    grid = rng.random((gh, gw))
    fx = np.linspace(0, gw - 1.001, w)
    fy = np.linspace(0, gh - 1.001, h)
    ix, iy = fx.astype(int), fy.astype(int)
    tx, ty = fx - ix, fy - iy
    tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
    a = grid[iy][:, ix]
    b = grid[iy][:, ix + 1]
    c = grid[iy + 1][:, ix]
    d = grid[iy + 1][:, ix + 1]
    top = a + (b - a) * tx[None, :]
    bot = c + (d - c) * tx[None, :]
    return top + (bot - top) * ty[:, None]


def text(surf, s, font, color, pos, anchor="topleft", spacing=0, alpha=255):
    if spacing:
        glyphs = [font.render(ch, True, color) for ch in s]
        width = sum(g.get_width() for g in glyphs) + spacing * (len(s) - 1)
        height = max(g.get_height() for g in glyphs)
        img = pygame.Surface((width, height), pygame.SRCALPHA)
        x = 0
        for g in glyphs:
            img.blit(g, (x, 0))
            x += g.get_width() + spacing
    else:
        img = font.render(s, True, color)
    if alpha < 255:
        img.set_alpha(alpha)
    r = img.get_rect(**{anchor: (int(pos[0]), int(pos[1]))})
    surf.blit(img, r)
    return r


def ring_arc(surf, center, r, width, a0, a1, color):
    """Anti-aliased thick arc; angles in radians, 0 = up, clockwise."""
    if a1 - a0 < 0.12:
        return
    n = max(8, int((a1 - a0) * r / 4))
    outer, inner = [], []
    for i in range(n + 1):
        a = lerp(a0, a1, i / n)
        sx, cy = math.sin(a), -math.cos(a)
        outer.append((center[0] + sx * r, center[1] + cy * r))
        inner.append((center[0] + sx * (r - width), center[1] + cy * (r - width)))
    poly(surf, outer + inner[::-1], color)


# --------------------------------------------------------------------------
# 3. Static scenery built once
# --------------------------------------------------------------------------

def build_sky(rng):
    sw, sh = W + 200, H + 200
    sky = pygame.Surface((sw, sh))
    # deep space gradient, slightly lifted near the horizon
    grad = np.zeros((sh, sw, 3), np.float32)
    t = np.linspace(0, 1, sh)[:, None]
    for i, (top, bot) in enumerate(zip((3, 4, 10), (12, 14, 24))):
        grad[..., i] = top + (bot - top) * t ** 2.2
    # milky way: a soft diagonal band of noise
    n1 = value_noise(sw, sh, 18, rng)
    n2 = value_noise(sw, sh, 60, rng)
    yy, xx = np.mgrid[0:sh, 0:sw]
    band = np.exp(-(((yy - 0.55 * xx + 120) / 260.0) ** 2))
    mw = band * (0.55 * n1 + 0.45 * n2) ** 2.0
    grad[..., 0] += mw * 38
    grad[..., 1] += mw * 34
    grad[..., 2] += mw * 48
    # dust lanes inside the band
    grad *= (1 - 0.5 * band[..., None] * np.clip(n2 - 0.55, 0, 1)[..., None] * 2)
    sky_arr = np.clip(grad, 0, 255).astype(np.uint8)
    pygame.surfarray.blit_array(sky, sky_arr.transpose(1, 0, 2))

    # stars: many faint, some bright with glow, denser in the band
    for _ in range(3200):
        x, y = rng.random() * sw, rng.random() * sh
        b = rng.random() ** 3
        if rng.random() < 0.5 and band[int(y), int(x)] < 0.3:
            continue
        tint = rng.choice([(255, 255, 255), (200, 215, 255), (255, 230, 200), (220, 230, 255)])
        c = tuple(int(30 + 225 * b * k / 255) for k in tint)
        sky.set_at((int(x), int(y)), c)
        if b > 0.55:
            add_glow(sky, (x, y), 3 + 6 * b, tuple(int(v * 0.35) for v in c))
    for _ in range(14):   # a few hero stars with spikes
        x, y = rng.random() * sw, rng.random() * sh * 0.7
        add_glow(sky, (x, y), 14, (90, 95, 110))
        for dx, dy in ((1, 0), (0, 1)):
            for k in range(1, 18):
                v = int(120 * (1 - k / 18) ** 2)
                for s in (-1, 1):
                    px, py = int(x + s * k * dx), int(y + s * k * dy)
                    if 0 <= px < sw and 0 <= py < sh:
                        old = sky.get_at((px, py))
                        sky.set_at((px, py), (min(255, old[0] + v), min(255, old[1] + v), min(255, old[2] + v)))
        sky.set_at((int(x), int(y)), (255, 255, 255))

    # Earth, lit from the right
    er = 92
    earth = build_earth(er, rng)
    ex, ey = sw - 560, 330
    add_glow(sky, (ex + 8, ey), er * 1.55, (25, 55, 120), falloff=2.5)
    sky.blit(earth, (ex - er, ey - er))

    # sun glare from off-screen top right
    add_glow(sky, (sw + 60, -80), 900, (60, 52, 40), falloff=2.0)
    add_glow(sky, (sw + 60, -80), 380, (120, 105, 80), falloff=2.4)
    return sky


def build_earth(r, rng):
    size = 2 * r
    yy, xx = np.mgrid[0:size, 0:size]
    nx, ny = (xx - r + 0.5) / r, (yy - r + 0.5) / r
    d2 = nx ** 2 + ny ** 2
    inside = d2 <= 1
    nz = np.sqrt(np.clip(1 - d2, 0, 1))
    sun = np.array([0.85, -0.35, 0.45])
    sun /= np.linalg.norm(sun)
    light = np.clip(nx * sun[0] + ny * sun[1] + nz * sun[2], 0, 1)

    def fbm(octaves):
        out, tot = 0, 0
        for cells, amp in octaves:
            out = out + value_noise(size, size, cells, rng) * amp
            tot += amp
        return out / tot

    land = fbm(((4, 1.0), (9, 0.5), (21, 0.25), (47, 0.12)))
    cloud = fbm(((6, 1.0), (14, 0.6), (33, 0.35), (70, 0.2)))
    ocean = np.array([18, 58, 140])
    ground = np.array([70, 110, 60]) * (1 - ny[..., None] * 0.1)
    desert = np.array([160, 140, 95])
    lm = np.clip((land - 0.53) / 0.03, 0, 1)[..., None]
    dm = np.clip((land - 0.62) / 0.05, 0, 1)[..., None]
    col = ocean * (1 - lm) + (ground * (1 - dm) + desert * dm) * lm
    cl = np.clip((cloud - 0.52) * 2.6, 0, 0.92)[..., None]
    col = col * (1 - cl) + np.array([235, 240, 245]) * cl
    shade = (light ** 0.8)[..., None]
    col = col * (0.04 + 0.96 * shade)
    # atmosphere rim
    rim = (np.clip(d2 - 0.82, 0, 1) / 0.18) ** 2
    glow = rim * (0.15 + light)
    col = col + np.stack([60 * glow, 120 * glow, 255 * glow], -1)
    col = np.clip(col, 0, 255).astype(np.uint8)
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.surfarray.pixels3d(surf)[:] = col.transpose(1, 0, 2)
    alpha = np.clip((1 - np.sqrt(d2)) * r * 1.5, 0, 1) * 255
    pygame.surfarray.pixels_alpha(surf)[:] = (alpha * inside).T.astype(np.uint8)
    return surf


def build_terrain(env_pts, pad, rng):
    """Extend the 20-unit Gymnasium terrain sideways and add fine detail."""
    xs = [p[0] for p in env_pts]
    ys = [p[1] for p in env_pts]
    lx, ly = list(xs), list(ys)
    for i in range(1, 26):
        lx.insert(0, xs[0] - 2 * i)
        ly.insert(0, rng.uniform(0.4, 6.2))
        lx.append(xs[-1] + 2 * i)
        ly.append(rng.uniform(0.4, 6.2))
    gx = np.arange(lx[0], lx[-1], 0.05)
    gy = np.interp(gx, lx, ly)
    x1, x2, py = pad
    flat = (gx > x1 - 1.0) & (gx < x2 + 1.0)
    k = np.ones(21) / 21
    smooth = np.convolve(np.pad(gy, 10, mode="edge"), k, mode="valid")
    gy = np.where(flat, gy, smooth)
    bumps = np.zeros_like(gx)
    for f, a in ((0.7, 0.10), (1.9, 0.05), (4.3, 0.025), (9.1, 0.012)):
        bumps += a * np.sin(gx * f + rng.uniform(0, 6.28))
    blend = np.clip(np.minimum(np.abs(gx - x1 + 0.5), np.abs(gx - x2 - 0.5)) / 1.0, 0, 1)
    blend = np.where(flat, 0, blend)
    gy = gy + bumps * blend
    gy[(gx >= x1 - 0.5) & (gx <= x2 + 0.5)] = py
    return gx, gy


# --------------------------------------------------------------------------
# 4. The renderer
# --------------------------------------------------------------------------

class Renderer:
    def __init__(self, ep, fps, grain=9):
        self.ep = ep
        self.grain_amt = grain
        self.fps = fps
        self.rng = np.random.default_rng(ep["seed"] + 11)
        self.prng = random.Random(ep["seed"] + 5)
        self.frames = ep["frames"]
        self.x1, self.x2, self.pad_y = ep["pad"]
        self.pad_cx = (self.x1 + self.x2) / 2

        pygame.init()
        pygame.display.set_mode((1, 1))
        self.screen = pygame.Surface((W, H))
        self.addl = pygame.Surface((W, H))
        self.hud = pygame.Surface((W, H), pygame.SRCALPHA)

        self.sky = build_sky(self.rng)
        self.gx, self.gy = build_terrain(ep["terrain"], ep["pad"], self.rng)
        self.mountains = [self.make_ridge(0.4, 2.6, 22), self.make_ridge(0.65, 1.7, 14)]
        self.craters = self.make_craters()
        self.rocks = self.make_rocks()
        self.vignette = self.make_vignette()
        self.grain = [self.make_grain() for _ in range(4)] if grain > 0 else []

        self.f_num_big = pygame.font.Font(FONT_NUM, 40)
        self.f_num = pygame.font.Font(FONT_NUM, 30)
        self.f_num_small = pygame.font.Font(FONT_NUM, 22)
        self.f_label = pygame.font.Font(FONT_LABEL, 17)
        self.f_label_big = pygame.font.Font(FONT_LABEL, 30)
        self.f_title = pygame.font.Font(FONT_LABEL, 64)

        self.particles = []   # dust and RCS gas
        self.flame_level = 0.0
        self.cam = None
        self.events = self.find_events()

    # ---- scenery generators -------------------------------------------------

    def make_ridge(self, parallax, height, seed):
        r = np.random.default_rng(seed)
        xs = np.arange(-80, 120, 0.1)
        h = np.zeros_like(xs)
        for f, a in ((0.08, 1.0), (0.21, 0.55), (0.53, 0.25), (1.3, 0.1)):
            h += a * np.abs(np.sin(xs * f + r.uniform(0, 6.28)))
        h = h / h.max() * height
        return parallax, xs, h

    def make_craters(self):
        out = []
        for _ in range(140):
            x = self.rng.uniform(-40, 60)
            if self.x1 - 1.5 < x < self.x2 + 1.5:
                continue
            out.append((x, self.rng.uniform(0.15, 1.6), self.rng.uniform(0.12, 0.55)))
        return out

    def make_rocks(self):
        out = []
        for _ in range(220):
            x = self.rng.uniform(-40, 60)
            if self.x1 - 0.6 < x < self.x2 + 0.6:
                continue
            out.append((x, self.rng.uniform(0.03, 0.12), self.rng.uniform(0.6, 1.0)))
        return out

    def make_vignette(self):
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
        v = (np.clip(d - 0.55, 0, 1) ** 1.6 * 230).astype(np.uint8)
        s = pygame.Surface((W, H), pygame.SRCALPHA)
        s.fill((0, 0, 0, 0))
        pygame.surfarray.pixels_alpha(s)[:] = v.T
        return s

    def make_grain(self):
        n = (self.rng.random((W // 2, H // 2)) * self.grain_amt).astype(np.uint8)
        s = pygame.surfarray.make_surface(np.stack([n] * 3, -1))
        return pygame.transform.scale(s, (W, H))

    def find_events(self):
        fr = self.frames
        contact = next((i for i, f in enumerate(fr) if f["contact"]), len(fr) - 1)
        low = next((i for i, f in enumerate(fr) if f["y"] - self.pad_y < 3.4), contact)
        deploy = max(0, min(low, contact - 50))
        burn = next((i for i in range(deploy) if fr[i]["y"] - self.pad_y < 6 and fr[i]["action"] == 2), deploy)
        return dict(deploy=deploy, contact=contact, burn=min(burn, deploy), end=len(fr) - 1)

    # ---- coordinate helpers -------------------------------------------------

    def ground_at(self, x):
        return float(np.interp(x, self.gx, self.gy))

    def w2s(self, x, y):
        cx, cy, z = self.cam
        return (W / 2 + (x - cx) * z, H / 2 - (y - cy) * z)

    def local(self, st, lx, ly):
        c, s = math.cos(st["a"]), math.sin(st["a"])
        return (st["x"] + lx * c - ly * s, st["y"] + lx * s + ly * c)

    def ls(self, st, pts):
        return [self.w2s(*self.local(st, x, y)) for x, y in pts]

    # ---- state interpolation ------------------------------------------------

    def state_at(self, t):
        fr = self.frames
        f = t * PHYS_FPS
        i = int(f)
        if i >= len(fr) - 1:
            s = dict(fr[-1])
            s["idx"] = len(fr) - 1
            s["action"] = 0
            return s
        a = f - i
        p, q = fr[i], fr[i + 1]
        s = {k: lerp(p[k], q[k], a) for k in ("x", "y", "a", "vx", "vy", "score")}
        s["feet"] = [(lerp(p["feet"][j][0], q["feet"][j][0], a),
                      lerp(p["feet"][j][1], q["feet"][j][1], a)) for j in range(2)]
        s["action"] = q["action"]
        s["contact"] = q["contact"]
        s["idx"] = f
        return s

    # ---- camera ---------------------------------------------------------------

    def update_camera(self, st, first):
        top = st["y"] + 3.9
        bot = self.pad_y - 2.9
        need = top - bot
        z = min(100.0, H / need)
        cy = (top + bot) / 2
        cx = lerp(st["x"], self.pad_cx, 0.35)
        target = (cx, cy, z)
        if first or self.cam is None:
            self.cam = target
        else:
            k = 0.06
            self.cam = tuple(lerp(c, t, k) for c, t in zip(self.cam, target))

    # ---- scenery drawing -------------------------------------------------------

    def draw_sky(self):
        cx, cy, z = self.cam
        ox = -100 - (cx - self.pad_cx) * 6
        oy = -100 + (cy - self.pad_y - 5) * 4
        self.screen.blit(self.sky, (max(-200, min(0, ox)), max(-200, min(0, oy))))

    def draw_mountains(self):
        cx, cy, z = self.cam
        _, horizon = self.w2s(0, self.pad_y)
        for (par, xs, h), col, edge in zip(self.mountains,
                                           [(38, 39, 44), (62, 62, 66)],
                                           [(70, 70, 76), (105, 104, 104)]):
            zl = z * (0.45 + 0.4 * par)
            base = horizon - 30 * par + 30
            left = cx * par - (W / 2) / zl - 1
            right = cx * par + (W / 2) / zl + 1
            m = (xs >= left) & (xs <= right)
            pts = [(W / 2 + (x - cx * par) * zl, base - hh * zl) for x, hh in zip(xs[m][::2], h[m][::2])]
            if not pts:
                continue
            poly(self.screen, pts + [(pts[-1][0], H + 10), (pts[0][0], H + 10)], col)
            pygame.draw.aalines(self.screen, edge, False, clip_pts(pts))

    def draw_ground(self):
        cx, cy, z = self.cam
        left = cx - W / 2 / z - 1
        right = cx + W / 2 / z + 1
        m = (self.gx >= left) & (self.gx <= right)
        xs, ys = self.gx[m][::2], self.gy[m][::2]
        top = [self.w2s(x, y) for x, y in zip(xs, ys)]
        # smooth gradient by depth below the landing-pad horizon (no banding)
        horizon = self.w2s(0, self.pad_y + 1.0)[1]
        depth = (np.arange(H) - horizon) / z
        f = np.clip(depth / 7.0, 0, 1) ** 0.8
        stops = np.array([(160, 156, 150), (126, 123, 118), (92, 90, 88), (62, 61, 61), (40, 39, 41)], float)
        fi = f * (len(stops) - 1)
        j = np.minimum(fi.astype(int), len(stops) - 2)
        cols = stops[j] + (stops[j + 1] - stops[j]) * (fi - j)[:, None]
        strip = pygame.surfarray.make_surface(cols.astype(np.uint8)[None, :, :])
        grad = pygame.transform.scale(strip, (W, H))
        mask = pygame.Surface((W, H), pygame.SRCALPHA)
        poly(mask, top + [(top[-1][0], H + 20), (top[0][0], H + 20)], (255, 255, 255, 255))
        mask.blit(grad, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.screen.blit(mask, (0, 0))
        # sunlit top layer that follows the surface
        for dd, col in ((0.55, (205, 200, 192, 35)), (0.25, (215, 210, 202, 45)), (0.08, (225, 220, 212, 60))):
            poly(self.screen, top + [(px, py + dd * z) for px, py in top][::-1], col, aa=False)

        def col_at(sy):
            return tuple(int(v) for v in cols[int(max(0, min(H - 1, sy)))])

        # craters in the regolith
        for x, dd, r in self.craters:
            if not (left - 1 < x < right + 1):
                continue
            gy = self.ground_at(x)
            sx, sy = self.w2s(x, gy - dd)
            rx, ry = r * z, r * z * 0.32
            base = col_at(sy)
            dark = tuple(int(c * 0.62) for c in base)
            light = tuple(min(255, int(c * 1.22)) for c in base)
            rect = pygame.Rect(0, 0, int(2 * rx) + 1, int(2 * ry) + 1)
            rect.center = (int(sx), int(sy))
            if rect.width > 2 and rect.height > 1:
                gfxdraw.filled_ellipse(self.screen, rect.centerx, rect.centery, int(rx), max(1, int(ry)), dark)
                gfxdraw.aaellipse(self.screen, rect.centerx, rect.centery, int(rx), max(1, int(ry)), light)
                inner = pygame.Rect(0, 0, int(rx * 1.6), int(ry * 1.2))
                inner.center = (int(sx - rx * 0.12), int(sy - ry * 0.18))
                gfxdraw.filled_ellipse(self.screen, inner.centerx, inner.centery,
                                       max(1, int(rx * 0.8)), max(1, int(ry * 0.6)),
                                       tuple(int(c * 0.5) for c in base))
        # rocks on the surface
        for x, r, b in self.rocks:
            if not (left < x < right):
                continue
            gy = self.ground_at(x)
            sx, sy = self.w2s(x, gy)
            rr = r * z
            if rr < 1.2:
                continue
            col = tuple(int(c * b) for c in (120, 117, 112))
            gfxdraw.filled_ellipse(self.screen, int(sx), int(sy), int(rr), max(1, int(rr * 0.7)), col)
            gfxdraw.filled_ellipse(self.screen, int(sx + rr * 0.3), int(sy - rr * 0.3),
                                   max(1, int(rr * 0.45)), max(1, int(rr * 0.3)),
                                   tuple(min(255, int(c * 1.35)) for c in col))
        # sunlit rim
        pygame.draw.aalines(self.screen, (196, 192, 184), False, clip_pts(top))
        pygame.draw.lines(self.screen, (176, 172, 165), False, clip_pts([(x, y + 1) for x, y in top]), 2)

    def draw_pad(self, t):
        z = self.cam[2]
        x1, x2, y = self.x1 - 0.15, self.x2 + 0.15, self.pad_y
        cxp = (x1 + x2) / 2
        rx = (x2 - x1) / 2
        depth = 0.3
        # front face
        a, b = self.w2s(x1, y), self.w2s(x2, y - depth)
        face = [a, (b[0], a[1]), b, (a[0], b[1])]
        poly(self.screen, face, (58, 60, 64))
        pygame.draw.line(self.screen, (34, 35, 38), (a[0], b[1]), b, 2)
        # top surface seen at a grazing angle
        scx, scy = self.w2s(cxp, y)
        ry = max(3, rx * z * 0.085)
        gfxdraw.filled_ellipse(self.screen, int(scx), int(scy), int(rx * z), int(ry), (112, 113, 115))
        gfxdraw.aaellipse(self.screen, int(scx), int(scy), int(rx * z), int(ry), (150, 150, 150))
        gfxdraw.aaellipse(self.screen, int(scx), int(scy), int(rx * z * 0.72), int(ry * 0.72), (230, 228, 220))
        gfxdraw.aaellipse(self.screen, int(scx), int(scy), int(rx * z * 0.70), int(ry * 0.70), (230, 228, 220))
        for sgn in (-1, 1):
            p = (scx - rx * z * 0.45, scy + sgn * ry * 0.45)
            q = (scx + rx * z * 0.45, scy - sgn * ry * 0.45)
            pygame.draw.line(self.screen, (235, 232, 222), p, q, max(2, int(z * 0.05)))
        # edge lights, blinking
        for i in range(9):
            lx = lerp(x1 + 0.1, x2 - 0.1, i / 8)
            sx, sy = self.w2s(lx, y - depth * 0.45)
            on = (int(t * 2.2) + i) % 2 == 0
            col = (255, 60, 40) if i % 4 else (255, 240, 220)
            pygame.draw.circle(self.screen, col if on else (90, 40, 35), (int(sx), int(sy)), max(2, int(z * 0.025)))
            if on:
                add_glow(self.addl, (sx, sy), z * 0.22, tuple(int(c * 0.55) for c in col))

    def draw_shadow(self, st):
        alt = st["y"] - self.pad_y
        gx = st["x"] - alt * 0.35 - 0.5
        gy = self.ground_at(gx)
        if alt > 9:
            return
        strength = smoothstep(1 - alt / 9)
        sx, sy = self.w2s(gx, gy)
        z = self.cam[2]
        w, h = int(z * (1.2 + alt * 0.15)), int(z * 0.16)
        s = pygame.Surface((2 * w, 2 * h), pygame.SRCALPHA)
        pygame.draw.ellipse(s, (0, 0, 0, int(120 * strength)), s.get_rect())
        s = pygame.transform.smoothscale(pygame.transform.smoothscale(s, (max(1, w // 3), max(1, h // 3))), (2 * w, 2 * h))
        self.screen.blit(s, (sx - w, sy - h))

    # ---- the rocket -----------------------------------------------------------

    def deploy(self, st):
        return smoothstep((st["idx"] - self.events["deploy"]) / 42.0)

    def draw_rocket(self, st, t):
        R = 0.25
        y0, y1 = -0.33, 2.72        # white body
        top = 3.18                  # top of the black interstage
        dep = self.deploy(st)
        firing = self.flame_level

        # legs (behind the body edge)
        for j, side in enumerate((-1, 1)):
            hinge = (side * (R - 0.02), -0.26)
            fx, fy = st["feet"][j]
            # physics leg order is [-1, +1] in LEG_AWAY terms: leg 0 is on the +x side
            if (fx > 0) != (side > 0):
                fx, fy = st["feet"][1 - j]
            vdx, vdy = fx - hinge[0], fy - hinge[1]
            L = math.hypot(vdx, vdy)
            a_dep = math.atan2(vdy, vdx)
            if side < 0 and a_dep < 0:
                a_dep += 2 * math.pi
            a_st = math.pi / 2
            a = lerp(a_st, a_dep, dep)
            foot = (hinge[0] + math.cos(a) * L, hinge[1] + math.sin(a) * L)
            if dep < 0.02:
                foot = (hinge[0] + side * 0.03, hinge[1] + L)
            w = 0.055
            nx, ny = -math.sin(a) * w, math.cos(a) * w
            leg = [(hinge[0] + nx, hinge[1] + ny), (foot[0] + nx, foot[1] + ny),
                   (foot[0] - nx, foot[1] - ny), (hinge[0] - nx, hinge[1] - ny)]
            poly(self.screen, self.ls(st, leg), (28, 28, 31))
            # telescoping strut
            anchor = (side * (R - 0.01), 0.95)
            mid = (lerp(hinge[0], foot[0], 0.55), lerp(hinge[1], foot[1], 0.55))
            pa, pb = self.ls(st, [anchor, mid])
            pygame.draw.line(self.screen, (55, 55, 58), pa, pb, max(2, int(self.cam[2] * 0.035)))
            # foot pad
            if dep > 0.5:
                fp = [(foot[0] - 0.11, foot[1]), (foot[0] + 0.11, foot[1]),
                      (foot[0] + 0.07, foot[1] + 0.06), (foot[0] - 0.07, foot[1] + 0.06)]
                poly(self.screen, self.ls(st, fp), (40, 40, 44))
            # carbon fibre highlight
            hl = [(hinge[0] + nx * 0.2, hinge[1] + ny * 0.2), (foot[0] + nx * 0.2, foot[1] + ny * 0.2)]
            pa, pb = self.ls(st, hl)
            pygame.draw.aaline(self.screen, (80, 80, 86), pa, pb)

        # engine bells
        for bx in (-0.15, 0.0, 0.15):
            bell = [(bx - 0.045, -0.33), (bx + 0.045, -0.33), (bx + 0.075, -0.56), (bx - 0.075, -0.56)]
            hot = (min(255, 70 + int(110 * firing)), min(255, 62 + int(35 * firing)), 58)
            poly(self.screen, self.ls(st, bell), hot)
            lip = self.ls(st, [(bx - 0.075, -0.56), (bx + 0.075, -0.56)])
            pygame.draw.line(self.screen, (140, 120, 100), lip[0], lip[1], 2)

        # body as vertical strips: cylinder shading + soot gradient
        strips, rows = 18, 8
        white = np.array([236, 236, 232])
        soot = np.array([92, 82, 72])
        for i in range(strips):
            u0, u1 = -1 + 2 * i / strips, -1 + 2 * (i + 1) / strips
            um = (u0 + u1) / 2
            nxv = um
            nzv = math.sqrt(max(0.0, 1 - um * um))
            lit = 0.30 + 0.70 * max(0.0, nxv * 0.75 + nzv * 0.55)
            spec = max(0.0, nxv * 0.6 + nzv * 0.8) ** 18 * 0.35
            for r in range(rows):
                ya, yb = lerp(y0, y1, r / rows), lerp(y0, y1, (r + 1) / rows)
                ym = (ya + yb) / 2
                sootk = smoothstep((1.55 - ym) / 1.9) * 0.78
                streak = 0.12 * math.sin(um * 23 + 1.3) * sootk
                base = white * (1 - sootk - streak) + soot * (sootk + streak)
                col = np.clip(base * lit + 255 * spec, 0, 255).astype(int)
                quad = [(u0 * R, ya), (u1 * R, ya), (u1 * R, yb), (u0 * R, yb)]
                poly(self.screen, self.ls(st, quad), tuple(col), aa=False)
        # interstage
        for i in range(strips):
            u0, u1 = -1 + 2 * i / strips, -1 + 2 * (i + 1) / strips
            um = (u0 + u1) / 2
            lit = 0.35 + 0.65 * max(0.0, um * 0.75 + math.sqrt(max(0, 1 - um * um)) * 0.55)
            c = int(18 + 34 * lit)
            quad = [(u0 * R, y1), (u1 * R, y1), (u1 * R, top), (u0 * R, top)]
            poly(self.screen, self.ls(st, quad), (c, c, c + 3), aa=False)
        outline = self.ls(st, [(-R, y0), (R, y0), (R, top), (-R, top)])
        gfxdraw.aapolygon(self.screen, clip_pts(outline), (40, 40, 42))
        # octaweb (engine section skirt)
        skirt = [(-R, y0), (R, y0), (R * 0.92, y0 - 0.05), (-R * 0.92, y0 - 0.05)]
        poly(self.screen, self.ls(st, skirt), (48, 46, 44))
        # panel seams
        for ys in (0.55, 1.45, 2.2):
            pa, pb = self.ls(st, [(-R, ys), (R, ys)])
            pygame.draw.aaline(self.screen, (120, 118, 112), pa, pb)
        # grid fins (stowed until the descent, then they are already out)
        for side in (-1, 1):
            fin = [(side * R, 2.92), (side * (R + 0.15), 2.95), (side * (R + 0.15), 3.11), (side * R, 3.12)]
            poly(self.screen, self.ls(st, fin), (64, 64, 68))
            for k in range(1, 4):
                gx = side * (R + 0.15 * k / 4)
                pa, pb = self.ls(st, [(gx, 2.93), (gx, 3.11)])
                pygame.draw.aaline(self.screen, (30, 30, 32), pa, pb)
            pa, pb = self.ls(st, [(side * R, 3.02), (side * (R + 0.15), 3.03)])
            pygame.draw.aaline(self.screen, (30, 30, 32), pa, pb)
        # cold-gas thruster pods
        for side in (-1, 1):
            pod = [(side * R, 2.78), (side * (R + 0.035), 2.79), (side * (R + 0.035), 2.86), (side * R, 2.87)]
            poly(self.screen, self.ls(st, pod), (90, 90, 94))
        # top cap bevel
        pa, pb = self.ls(st, [(-R, top), (R, top)])
        pygame.draw.aaline(self.screen, (95, 95, 100), pa, pb)

    def draw_flame(self, st, t):
        lvl = self.flame_level
        if lvl < 0.02:
            return
        z = self.cam[2]
        noz_y = -0.56
        nw = self.local(st, 0, noz_y)
        dist = nw[1] - self.ground_at(nw[0])
        flick = 1 + 0.12 * math.sin(t * 61) + 0.08 * math.sin(t * 97 + 1) + self.prng.uniform(-0.06, 0.06)
        L = min(1.9 * flick * lvl, max(0.3, dist + 0.25))
        layers = [
            (0.62, L * 1.08, (110, 45, 12)),
            (0.42, L * 0.95, (160, 85, 30)),
            (0.26, L * 0.75, (200, 150, 90)),
            (0.12, L * 0.42, (230, 220, 200)),
        ]
        for wmax, ln, col in layers:
            pts = [(-0.2, noz_y)]
            n = 10
            for i in range(1, n + 1):
                f = i / n
                w = lerp(0.2, wmax, math.sin(min(1, f * 1.35) * math.pi / 2)) * (1 - f ** 3)
                pts.append((w + self.prng.uniform(-0.01, 0.01), noz_y - ln * f))
            pts += [(-x, y) for x, y in pts[::-1][:-1]]
            pts.append((0.2, noz_y))
            c = tuple(int(v * lvl) for v in col)
            poly(self.addl, self.ls(st, pts), c, aa=False)
        # mach-diamond like bright knots in the core
        for k in range(3):
            yk = noz_y - (0.12 + 0.17 * k) * L / 1.9 * 1.9
            if -yk + noz_y > L * 0.5:
                break
            p = self.w2s(*self.local(st, 0, yk))
            add_glow(self.addl, p, z * (0.16 - 0.03 * k), (int(150 * lvl), int(140 * lvl), int(120 * lvl)))
        add_glow(self.addl, self.w2s(*self.local(st, 0, noz_y - 0.05)), z * 0.9 * lvl, (120, 70, 25))
        # light on the ground under the plume
        if dist < 4.5:
            g = self.w2s(nw[0], self.ground_at(nw[0]))
            k = (1 - dist / 4.5) * lvl
            add_glow(self.addl, g, z * (1.4 + 2.6 * k), (int(140 * k), int(80 * k), int(30 * k)))

    # ---- particles --------------------------------------------------------------

    def spawn(self, st, dt):
        if self.flame_level > 0.3:
            nw = self.local(st, 0, -0.56)
            gy = self.ground_at(nw[0])
            dist = nw[1] - gy
            if dist < 3.2:
                k = 1 - dist / 3.2
                for _ in range(int(14 * k * self.flame_level) + 1):
                    side = self.prng.choice((-1, 1))
                    self.particles.append(dict(
                        kind="dust", x=nw[0] + self.prng.uniform(-0.3, 0.3), y=gy + self.prng.uniform(0.0, 0.12),
                        vx=side * self.prng.uniform(2.0, 7.5) * (0.4 + k), vy=self.prng.uniform(0.1, 1.2) * k,
                        life=0, ttl=self.prng.uniform(1.2, 2.8), size=self.prng.uniform(0.12, 0.3),
                        grow=self.prng.uniform(0.25, 0.7)))
        act = st["action"]
        if act in (1, 3) and st["idx"] < len(self.frames) - 1:
            side = -1 if act == 1 else 1
            for _ in range(3):
                p0 = self.local(st, side * 0.29, 2.83)
                d = self.local(st, side * 1.0, 2.83)
                ox, oy = d[0] - st["x"], d[1] - st["y"]
                p1 = self.local(st, 0, 2.83)
                dx, dy = p0[0] - p1[0], p0[1] - p1[1]
                n = math.hypot(dx, dy) or 1
                v = self.prng.uniform(1.5, 3.0)
                self.particles.append(dict(
                    kind="gas", x=p0[0], y=p0[1], vx=st["vx"] + dx / n * v + self.prng.uniform(-0.3, 0.3),
                    vy=st["vy"] + dy / n * v + self.prng.uniform(-0.3, 0.3),
                    life=0, ttl=self.prng.uniform(0.25, 0.5), size=0.05, grow=0.9))

    def step_particles(self, dt):
        alive = []
        for p in self.particles:
            p["life"] += dt
            if p["life"] > p["ttl"]:
                continue
            drag = 0.985 if p["kind"] == "dust" else 0.95
            p["vx"] *= drag
            p["vy"] = p["vy"] * drag - (0.6 if p["kind"] == "dust" else 0) * dt
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["size"] += p["grow"] * dt
            if p["kind"] == "dust":
                gy = self.ground_at(p["x"])
                if p["y"] < gy:
                    p["y"], p["vy"] = gy, abs(p["vy"]) * 0.2
            alive.append(p)
        self.particles = alive[-1600:]

    def draw_particles(self):
        z = self.cam[2]
        for p in self.particles:
            f = p["life"] / p["ttl"]
            r = max(2, int(p["size"] * z))
            if p["kind"] == "dust":
                a = int(95 * (1 - f) ** 1.5)
                lit = (176, 168, 156)
            else:
                a = int(140 * (1 - f) ** 2)
                lit = (235, 240, 250)
            if a <= 2:
                continue
            dot = soft_dot(min(r, 140), lit)
            dot.set_alpha(a)
            sx, sy = self.w2s(p["x"], p["y"])
            self.screen.blit(dot, (sx - dot.get_width() / 2, sy - dot.get_height() / 2))

    # ---- HUD ---------------------------------------------------------------------

    def draw_hud(self, st, t, total_t):
        hud = self.hud
        hud.fill((0, 0, 0, 0))
        # bottom gradient for legibility
        grad = pygame.Surface((1, 260), pygame.SRCALPHA)
        for i in range(260):
            grad.set_at((0, i), (0, 0, 0, int(150 * (i / 260) ** 1.6)))
        hud.blit(pygame.transform.scale(grad, (W, 260)), (0, H - 260))

        white = (240, 242, 245)
        dim = (150, 155, 165)

        # title, top left
        text(hud, "LUNAR LANDER", self.f_label_big, white, (60, 48), spacing=6)
        text(hud, f"PPO AGENT  ·  STABLE-BASELINES3  ·  EPISODE {self.ep['seed']}", self.f_label, dim, (62, 92), spacing=3)

        # score, top right
        text(hud, "EPISODE SCORE", self.f_label, dim, (W - 60, 50), anchor="topright", spacing=3)
        text(hud, f"{st['score']:+.1f}", self.f_num_big, white, (W - 60, 74), anchor="topright")

        # gauges, bottom left
        feet_y = min(self.local(st, *st["feet"][0])[1], self.local(st, *st["feet"][1])[1])
        alt_m = max(0.0, (feet_y - self.ground_at(st["x"])) * METERS_PER_UNIT)
        speed = math.hypot(st["vx"], st["vy"]) * METERS_PER_UNIT * 3.6
        if st["idx"] >= len(self.frames) - 1:
            alt_m, speed = 0.0, 0.0
        for i, (label, val, unit, vmax) in enumerate((("SPEED", speed, "KM/H", 300), ("ALTITUDE", alt_m, "M", 160))):
            c = (130 + 190 * i, H - 128)
            r = 66
            ring_arc(hud, c, r, 5, -2.4, 2.4, (255, 255, 255, 55))
            frac = max(0.0, min(1.0, val / vmax))
            ring_arc(hud, c, r, 5, -2.4, -2.4 + 4.8 * frac, white)
            text(hud, f"{val:0.0f}", self.f_num, white, (c[0], c[1] - 4), anchor="center")
            text(hud, unit, self.f_label, dim, (c[0], c[1] + 24), anchor="center", spacing=2)
            text(hud, label, self.f_label, white, (c[0], c[1] + r + 18), anchor="center", spacing=4)

        # timeline, bottom center
        x0, x1 = W / 2 - 430, W / 2 + 430
        ty = H - 78
        ev = self.events
        marks = [("DESCENT", 0), ("LANDING BURN", ev["burn"]), ("LEG DEPLOY", ev["deploy"]),
                 ("TOUCHDOWN", ev["contact"]), ("SECURED", ev["end"])]
        span = ev["end"] / PHYS_FPS + 0.01
        now = min(t, span)
        pygame.draw.line(hud, (255, 255, 255, 70), (x0, ty), (x1, ty), 2)
        pygame.draw.line(hud, white, (x0, ty), (lerp(x0, x1, now / span), ty), 3)
        last_x = -999
        for name, idx in marks:
            tm = idx / PHYS_FPS
            mx = lerp(x0, x1, tm / span)
            done = t >= tm
            pygame.draw.circle(hud, white if done else (90, 92, 98), (int(mx), ty), 8 if done else 6)
            pygame.draw.circle(hud, (0, 0, 0), (int(mx), ty), 3 if done else 0)
            if mx - last_x > 120:
                text(hud, name, self.f_label, white if done else dim, (mx, ty + 28), anchor="center", spacing=2)
                last_x = mx
        secs = int(t)
        text(hud, f"T+ 00:00:{secs:02d}", self.f_num, white, (W / 2, H - 140), anchor="center")

        # center event callouts
        callouts = [("LEGS DEPLOYED", (ev["deploy"] + 40) / PHYS_FPS), ("TOUCHDOWN", ev["contact"] / PHYS_FPS)]
        for i, (name, tm) in enumerate(callouts):
            nxt = callouts[i + 1][1] if i + 1 < len(callouts) else 1e9
            dur = min(2.2, nxt - tm)
            if dur > 0.3 and tm <= t < tm + dur:
                k = min(1, (t - tm) / 0.25) * min(1, (tm + dur - t) / 0.4)
                text(hud, name, self.f_label_big, white, (W / 2, 150), anchor="center", spacing=10, alpha=int(255 * k))

        # end card
        if t > span + 0.6:
            k = smoothstep((t - span - 0.6) / 0.8)
            text(hud, "TOUCHDOWN CONFIRMED", self.f_title, white, (W / 2, 150),
                 anchor="center", spacing=14, alpha=int(255 * k))
            text(hud, f"SCORE {self.ep['score']:.1f}   ·   SOLVED ≥ 200   ·   PPO, 1M STEPS",
                 self.f_label_big, dim, (W / 2, 215), anchor="center", spacing=4, alpha=int(255 * k))

        self.screen.blit(hud, (0, 0))

    # ---- one output frame -------------------------------------------------------

    def render(self, t, total_t, first):
        dt = 1 / self.fps
        st = self.state_at(t)
        self.update_camera(st, first)

        target = 1.0 if st["action"] == 2 and st["idx"] < len(self.frames) - 1 else 0.0
        self.flame_level += (target - self.flame_level) * (0.55 if target else 0.35)

        self.addl.fill((0, 0, 0))
        self.draw_sky()
        self.draw_mountains()
        self.draw_ground()
        self.draw_pad(t)
        self.draw_shadow(st)
        self.spawn(st, dt)
        self.step_particles(dt)
        self.draw_rocket(st, t)
        self.draw_flame(st, t)
        self.draw_particles()

        # additive light + bloom
        self.screen.blit(self.addl, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        small = pygame.transform.smoothscale(self.addl, (W // 6, H // 6))
        small = pygame.transform.smoothscale(small, (W // 12, H // 12))
        bloom = pygame.transform.smoothscale(small, (W, H))
        self.screen.blit(bloom, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        self.screen.blit(bloom, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

        self.screen.blit(self.vignette, (0, 0))
        if self.grain:
            g = self.grain[int(t * self.fps) % len(self.grain)]
            self.screen.blit(g, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
            self.screen.blit(self.grain[(int(t * self.fps) + 2) % len(self.grain)], (0, 0),
                             special_flags=pygame.BLEND_RGB_SUB)

        self.draw_hud(st, t, total_t)

        # fade in / out
        fade = min(1.0, t / 0.7, (total_t - t) / 0.8)
        if fade < 1:
            black = pygame.Surface((W, H))
            black.set_alpha(int(255 * (1 - max(0, fade))))
            self.screen.blit(black, (0, 0))
        return self.screen


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="videos/cinematic_1080p60.mp4")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--hold", type=float, default=4.0, help="seconds to hold after landing")
    ap.add_argument("--stills", type=str, default="", help="comma-separated times: save PNGs instead of video")
    ap.add_argument("--crf", type=int, default=16, help="video quality: lower is better and bigger")
    ap.add_argument("--grain", type=int, default=9, help="film grain strength, 0 = off")
    ap.add_argument("--poster", default="", help="also save the frame at 1.0s as a JPG/PNG here")
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    model = PPO.load(args.model, device="cpu")
    ep = pick_episode(model, args.seed)
    secs = (len(ep["frames"]) - 1) / PHYS_FPS
    print(f"Episode seed {ep['seed']}: score {ep['score']:.1f}, {secs:.1f}s, landed={ep['landed']}")

    r = Renderer(ep, args.fps, grain=args.grain)
    total_t = secs + args.hold
    n = int(total_t * args.fps)

    if args.stills:
        want = sorted(float(s) for s in args.stills.split(","))
        os.makedirs("videos/stills", exist_ok=True)
        for i in range(n):
            t = i / args.fps
            surf = r.render(t, total_t, i == 0)
            for w in want:
                if abs(t - w) < 0.5 / args.fps:
                    path = f"videos/stills/still_{w:05.2f}.png"
                    pygame.image.save(surf, path)
                    print("saved", path)
            if t > want[-1]:
                break
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    video_only = args.out + ".video.mp4"
    writer = imageio_ffmpeg.write_frames(
        video_only, (W, H), fps=args.fps, codec="libx264", pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p", quality=None, macro_block_size=1,
        output_params=["-crf", str(args.crf), "-preset", "slow"],
    )
    writer.send(None)
    poster_i = int(1.0 * args.fps)
    for i in range(n):
        surf = r.render(i / args.fps, total_t, i == 0)
        writer.send(pygame.image.tobytes(surf, "RGB"))
        if args.poster and i == poster_i:
            os.makedirs(os.path.dirname(args.poster) or ".", exist_ok=True)
            pygame.image.save(surf, args.poster)
        if i % 60 == 0:
            print(f"  frame {i}/{n}", flush=True)
    writer.close()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    if args.no_audio:
        os.replace(video_only, args.out)
    else:
        wav = args.out + ".wav"
        print("  synthesizing sound...", flush=True)
        sound.synthesize(r, n / args.fps, wav)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", video_only, "-i", wav,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                        "-movflags", "+faststart", args.out], check=True)
        os.remove(video_only)
        os.remove(wav)
    print(f"Saved {args.out}  ({W}x{H}, {args.fps} fps, {total_t:.1f}s, audio={'off' if args.no_audio else 'on'})")

if __name__ == "__main__":
    main()

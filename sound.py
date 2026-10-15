"""Synthesized soundtrack for the cinematic landing video.

Everything is generated from the episode itself (which engine fired on
which frame, the altitude, leg deploy and touchdown times), so the sound
lines up with the picture. No audio samples are used, so there is nothing
to license.
"""

import wave

import numpy as np

SR = 48000
PHYS_FPS = 50


def colored_noise(n, gain_fn, rng):
    """White noise shaped in the frequency domain by gain_fn(freqs_hz)."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = 1.0
    out = np.fft.irfft(spec * gain_fn(f), n)
    return out / (np.abs(out).max() + 1e-9)


def envelope(control, attack, release, rate=1000):
    """One-pole attack/release follower on a control-rate signal."""
    out = np.zeros_like(control)
    a_up = 1 - np.exp(-1 / (attack * rate))
    a_dn = 1 - np.exp(-1 / (release * rate))
    v = 0.0
    for i, c in enumerate(control):
        v += (c - v) * (a_up if c > v else a_dn)
        out[i] = v
    return out


def to_audio_rate(ctrl, n, rate=1000):
    return np.interp(np.arange(n) / SR, np.arange(len(ctrl)) / rate, ctrl)


def hit(n, t0, dur, rng, kind):
    """One-shot sounds placed at time t0 (seconds)."""
    out = np.zeros(n)
    s = int(t0 * SR)
    if s >= n:
        return out
    m = min(int(dur * SR), n - s)
    tt = np.arange(m) / SR
    if kind == "thud":       # touchdown: pitch-dropping body thump + crunch
        freq = 38 + 55 * np.exp(-tt / 0.07)
        body = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-tt / 0.32)
        crunch = colored_noise(m, lambda f: 1 / (1 + (f / 700) ** 2), rng) * np.exp(-tt / 0.12)
        out[s:s + m] = body * 1.0 + crunch * 0.55
    elif kind == "clunk":    # leg lock
        body = np.sin(2 * np.pi * 115 * tt) * np.exp(-tt / 0.07)
        click = colored_noise(m, lambda f: (f > 1500) / (1 + (f / 6000) ** 2), rng) * np.exp(-tt / 0.012)
        out[s:s + m] = body * 0.6 + click * 0.35
    elif kind == "whir":     # leg actuators
        freq = 120 + 70 * tt / dur
        saw = 2 * ((np.cumsum(freq) / SR) % 1) - 1
        env = np.sin(np.pi * np.clip(tt / dur, 0, 1)) ** 0.6
        out[s:s + m] = saw * env * 0.12
    elif kind == "beep":     # HUD callout
        env = np.clip(tt / 0.005, 0, 1) * np.exp(-tt / 0.09)
        out[s:s + m] = (np.sin(2 * np.pi * 1318.5 * tt) + 0.4 * np.sin(2 * np.pi * 2637 * tt)) * env * 0.25
    elif kind == "debris":   # pebbles settling after touchdown
        x = np.zeros(m)
        for _ in range(40):
            p = int(rng.uniform(0, 0.8) ** 2 * m)
            x[p:p + 1] += rng.uniform(0.2, 1.0)
        tick = colored_noise(m, lambda f: (f > 900) / (1 + (f / 5000) ** 2), rng)
        k = np.exp(-np.arange(int(0.01 * SR)) / (0.002 * SR))
        out[s:s + m] = np.convolve(x, k, "same") * tick * np.exp(-tt / 0.6)
    return out


def reverb(x, rng, seconds=1.4, wet=0.22):
    m = int(seconds * SR)
    ir = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.28 * SR))
    ir[: int(0.012 * SR)] = 0
    ir /= np.sqrt((ir ** 2).sum())
    size = len(x) + m
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x)]
    return x * (1 - wet) + y * wet * 2.2


def synthesize(r, total_t, path):
    """r is the cinematic Renderer (gives frames, events, pad)."""
    rng = np.random.default_rng(r.ep["seed"] + 99)
    n = int(total_t * SR)
    fr = r.frames
    nf = len(fr)
    ev = r.events
    span = ev["end"] / PHYS_FPS

    # ---- control signals at 1 kHz from the physics frames -------------------
    rate = 1000
    nc = int(total_t * rate) + 1
    ti = np.minimum((np.arange(nc) / rate * PHYS_FPS).astype(int), nf - 1)
    live = (np.arange(nc) / rate * PHYS_FPS) < nf - 1
    act = np.array([f["action"] for f in fr])[ti]
    alt = np.array([f["y"] - r.pad_y for f in fr])[ti]
    xpos = np.array([f["x"] for f in fr])[ti]

    main_env = to_audio_rate(envelope(((act == 2) & live).astype(float), 0.025, 0.11), n)
    left_env = to_audio_rate(envelope(((act == 1) & live).astype(float), 0.004, 0.06), n)
    right_env = to_audio_rate(envelope(((act == 3) & live).astype(float), 0.004, 0.06), n)
    near = to_audio_rate(np.clip(1 - alt / 6, 0, 1), n)
    pan = to_audio_rate(np.clip((xpos - r.pad_cx) / 8, -0.6, 0.6), n)
    t = np.arange(n) / SR

    # ---- main engine: sub rumble + roar + crackle, flutter -------------------
    rumble = colored_noise(n, lambda f: 1 / (1 + (f / 90) ** 2) ** 1.5, rng)
    roar = colored_noise(n, lambda f: np.sqrt(f / 300) / (1 + (f / 1400) ** 2) / (1 + (60 / f) ** 2), rng)
    flutter = 1 + 0.18 * colored_noise(n, lambda f: (f > 6) * (f < 22), rng)
    pops = (rng.random(n) < 0.0016).astype(float) * rng.uniform(0.3, 1, n)
    crackle = np.convolve(pops, np.exp(-np.arange(200) / 25), "same")
    crackle *= colored_noise(n, lambda f: (f > 1800) / (1 + (f / 7000) ** 2), rng)
    engine = main_env * flutter * (rumble * 0.9 + roar * (0.45 + 0.35 * near) + crackle * (0.25 + 0.3 * near))
    # ground reflection makes the engine boom louder as it gets low
    engine *= 0.7 + 0.6 * near

    # ---- cold-gas thrusters (RCS): short hiss, panned to their side ----------
    hiss = colored_noise(n, lambda f: (f > 2200) / (1 + (f / 9000) ** 2), rng)
    rcs_l = hiss * left_env * 0.22
    rcs_r = np.roll(hiss, 7919) * right_env * 0.22

    # ---- events -------------------------------------------------------------
    dep_t = ev["deploy"] / PHYS_FPS
    con_t = ev["contact"] / PHYS_FPS
    fx = np.zeros(n)
    fx += hit(n, dep_t + 0.05, 0.8, rng, "whir")
    clunk_l = hit(n, dep_t + 0.80, 0.4, rng, "clunk")
    clunk_r = hit(n, dep_t + 0.86, 0.4, rng, "clunk")
    thud = hit(n, con_t, 1.6, rng, "thud")
    debris = hit(n, con_t + 0.05, 2.0, rng, "debris") * 0.5
    beeps = hit(n, (ev["deploy"] + 40) / PHYS_FPS, 0.3, rng, "beep") + hit(n, con_t, 0.3, rng, "beep")

    # ---- ambient bed and the end-card swell -----------------------------------
    lfo = 0.6 + 0.4 * np.sin(2 * np.pi * 0.11 * t)
    drone = (np.sin(2 * np.pi * 55 * t) + 0.6 * np.sin(2 * np.pi * 82.4 * t + 1)
             + 0.35 * np.sin(2 * np.pi * 110.3 * t + 2)) * lfo * 0.035
    drone += colored_noise(n, lambda f: 1 / (1 + (f / 250) ** 2) / np.sqrt(f), rng) * 0.03
    swell_t = span + 0.6
    k = np.clip((t - swell_t) / 1.4, 0, 1) ** 2
    chord = sum(a * np.sin(2 * np.pi * f * t + ph) for f, a, ph in
                ((220.0, 1.0, 0), (277.18, 0.7, 1), (329.63, 0.8, 2), (440.0, 0.5, 3), (220.6, 0.6, 4)))
    swell = chord * k * 0.05

    # ---- mix to stereo ---------------------------------------------------------
    gl, gr = np.sqrt(0.5 - pan / 2), np.sqrt(0.5 + pan / 2)
    left = engine * gl + rcs_l * 0.9 + rcs_r * 0.25 + fx * 0.6 + clunk_l * 0.8 + clunk_r * 0.3 + (thud + debris) * gl
    right = engine * gr + rcs_r * 0.9 + rcs_l * 0.25 + fx * 0.6 + clunk_r * 0.8 + clunk_l * 0.3 + (thud + debris) * gr
    left, right = reverb(left, rng), reverb(right, rng)
    left += drone + swell + beeps * 0.7
    right += drone + np.roll(swell, 240) + beeps * 0.7

    fade = np.clip(np.minimum(t / 0.7, (total_t - t) / 0.8), 0, 1)
    stereo = np.stack([left, right], 1) * fade[:, None]
    stereo /= np.abs(stereo).max() + 1e-9
    stereo = np.tanh(stereo * 1.4) / np.tanh(1.4) * 0.89   # gentle limiter, ~-1 dBFS

    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((stereo * 32767).astype("<i2").tobytes())

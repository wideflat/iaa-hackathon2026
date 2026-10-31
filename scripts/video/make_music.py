"""Original background music for the demo video, synthesised from scratch with numpy (no samples, no downloads, no licence issues).

  python scripts/video/make_music.py --seconds 223 --out artifacts/final_submission/music.wav

96 BPM, chords Am - F - C - G, one bar each. Layers come in with the story: pad, bass, plucked arpeggio, hats, soft kick; the mix is deliberately quiet so a voice-over sits on top.
"""
import argparse
import wave
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

SR = 44100
BPM = 96
BEAT = 60.0 / BPM
CHORDS = [(57, [57, 60, 64]), (53, [53, 57, 60]), (48, [48, 55, 64]), (55, [55, 59, 62])]   # (bass MIDI, chord tones)


def mtof(m):
    return 440.0 * 2 ** ((np.asarray(m, dtype=float) - 69) / 12)


def env(n, a, d, s, r):
    t = np.arange(n) / SR
    e = np.ones(n)
    e[t < a] = t[t < a] / max(a, 1e-6)
    dm = (t >= a) & (t < a + d)
    e[dm] = 1 - (1 - s) * (t[dm] - a) / d
    e[t >= a + d] = s
    rm = t > (n / SR - r)
    e[rm] *= np.clip((n / SR - t[rm]) / r, 0, 1)
    return e


def tone(freq, dur, harmonics=(1, .5, .25, .12), detune=0.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for k, a in enumerate(harmonics, start=1):
        y += a * np.sin(2 * np.pi * freq * k * t * (1 + detune))
    return y


def add(buf, start, sig, gain):
    i = int(start * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += gain * sig[: j - i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    total = a.seconds
    n = int(total * SR)
    rng = np.random.default_rng(7)
    pad, bass, pluck, hats, kick = (np.zeros(n) for _ in range(5))
    bar = 4 * BEAT

    def level(t, a0, a1):   # smooth ramp of a layer between a0 and a1 seconds
        return float(np.clip((t - a0) / max(a1 - a0, 1e-6), 0, 1))

    nb = int(total / bar) + 2
    for b in range(nb):
        t0 = b * bar
        root, tones = CHORDS[b % 4]
        for f in tones:
            for d in (-0.002, 0.002):
                add(pad, t0, tone(mtof(f), bar * 1.15, (1, .4, .15), d) * env(int(bar * 1.15 * SR), 0.6, 0.5, 0.8, 0.8), 0.10)
        if t0 > 6:   # bass
            for k in (0, 2):
                s = tone(mtof(root - 12), BEAT * 1.8, (1, .35)) * env(int(BEAT * 1.8 * SR), 0.01, 0.3, 0.5, 0.2)
                add(bass, t0 + k * BEAT, s, 0.30 * level(t0, 6, 14))
        if t0 > 12:   # arpeggio
            for k in range(8):
                tt = t0 + k * BEAT / 2
                note = tones[[0, 1, 2, 1][k % 4]] + 12 + (12 if k % 8 == 7 else 0)
                s = tone(mtof(note), 0.5, (1, .3, .1)) * env(int(0.5 * SR), 0.004, 0.12, 0.0, 0.2)
                add(pluck, tt, s, 0.14 * level(tt, 12, 22))
        if t0 > 24:   # hats on offbeats, kick on 1 and 3
            for k in range(8):
                tt = t0 + k * BEAT / 2
                h = rng.standard_normal(int(0.05 * SR)) * env(int(0.05 * SR), 0.001, 0.02, 0.0, 0.02)
                h = np.diff(h, prepend=0)   # crude high-pass
                add(hats, tt, h, (0.10 if k % 2 else 0.05) * level(tt, 24, 34))
        if t0 > 44:
            for k in (0, 2):
                tt = t0 + k * BEAT
                m = int(0.28 * SR)
                tk = np.arange(m) / SR
                s = np.sin(2 * np.pi * (45 + 90 * np.exp(-tk * 25)) * tk) * np.exp(-tk * 9)
                add(kick, tt, s, 0.34 * level(tt, 44, 52))
    # thin out near the end (outro)
    mix = pad + bass + pluck + hats + kick
    t = np.arange(n) / SR
    outro = np.clip((total - t) / 4.0, 0, 1)
    intro = np.clip(t / 2.0, 0, 1)
    mix *= outro * intro
    # light reverb
    ir_len = int(1.6 * SR)
    ir = rng.standard_normal(ir_len) * np.exp(-np.arange(ir_len) / SR * 3.2)
    wet = fftconvolve(mix, ir, mode="full")[:n]
    wet /= max(np.abs(wet).max(), 1e-9)
    out = 0.82 * mix / max(np.abs(mix).max(), 1e-9) + 0.28 * wet
    # stereo: slight delay offset for width
    d = int(0.012 * SR)
    left = out
    right = np.concatenate([np.zeros(d), out[:-d]]) * 0.97
    st = np.stack([left, right], axis=1)
    st = st / np.abs(st).max() * 0.42   # quiet by design (about -8 dBFS peak): a voice-over goes on top
    pcm = (st * 32767).astype("<i2")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(a.out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"wrote {a.out}: {total:.1f}s, peak {np.abs(st).max():.2f}")


if __name__ == "__main__":
    main()

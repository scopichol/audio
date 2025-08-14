#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize pauses & stresses in speech:
- Estimates pitch (F0) contour
- Detects pauses from low RMS-energy segments
- Detects stress points as local F0 peaks under high energy
- Saves PNG plot and TSV with timestamps

Usage:
  python visualize_pauses_stress.py input.wav --outdir ./out
  python visualize_pauses_stress.py input.mp3 --outdir ./out --pause-percentile 25 --min-pause 0.15 --stress-percentile 75

Dependencies:
  pip install numpy matplotlib scipy
  # optional but recommended:
  pip install librosa  # для MP3/OGG та кращого Pitch-трекінгу
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, argrelextrema
from scipy.io import wavfile

# ----- optional librosa (краще читання/реземплінг, YIN/pyin) -----
try:
    import librosa
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False

# ------------------------------- Audio I/O -------------------------------

def load_audio_mono(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Завантажити аудіо як mono float32 у [-1, 1]."""
    if HAVE_LIBROSA:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y.astype(np.float32), sr
    # WAV-fallback (без librosa)
    sr, data = wavfile.read(path)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        y = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        y = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        y = (data.astype(np.float32) - 128) / 128.0
    else:
        y = data.astype(np.float32)
    if sr != target_sr:
        # простий ресемпл (лінійна інтерполяція)
        x_old = np.linspace(0, 1, num=len(y), endpoint=False)
        x_new = np.linspace(0, 1, num=int(len(y) * target_sr / sr), endpoint=False)
        y = np.interp(x_new, x_old, y).astype(np.float32)
        sr = target_sr
    return y, sr

# ------------------------------- Features -------------------------------

def rms_energy(y: np.ndarray, frame_length: int = 1024, hop_length: int = 256) -> np.ndarray:
    frames = []
    for i in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[i:i + frame_length]
        frames.append(np.sqrt(np.mean(frame**2) + 1e-12))
    return np.array(frames)

def estimate_pitch(
    y: np.ndarray, sr: int, fmin: float = 75.0, fmax: float = 400.0,
    frame_length: int = 2048, hop_length: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Оцінка F0 (Hz) та масив часу. Повертає np.nan для неозвучених кадрів."""
    if HAVE_LIBROSA:
        try:
            f0 = librosa.pyin(y, fmin=fmin, fmax=fmax, frame_length=frame_length, hop_length=hop_length)[0]
        except Exception:
            f0 = librosa.yin(y, fmin=fmin, fmax=fmax, frame_length=frame_length, hop_length=hop_length)
        times = np.arange(len(f0)) * (hop_length / sr)
        return f0, times

    # Наївний ACF fallback
    def acf_pitch(frame: np.ndarray, sr: int, fmin: float, fmax: float) -> float:
        w = np.hamming(len(frame))
        x = frame * w
        corr = np.correlate(x, x, mode='full')[len(x)-1:]
        min_lag = int(sr / fmax)
        max_lag = int(sr / fmin)
        if max_lag >= len(corr):
            return np.nan
        seg = corr[min_lag:max_lag]
        if len(seg) == 0:
            return np.nan
        lag_hat = np.argmax(seg) + min_lag
        return sr / lag_hat if lag_hat > 0 else np.nan

    f0_list = []
    for i in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[i:i + frame_length]
        f0_list.append(acf_pitch(frame, sr, fmin, fmax))
    f0 = np.array(f0_list, dtype=float)
    times = np.arange(len(f0)) * (hop_length / sr)
    return f0, times

def smooth_contour(f0: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Медіанне згладжування з коротким «замазуванням» NaN."""
    f0_s = f0.copy()
    isnan = np.isnan(f0_s)
    if np.any(~isnan):
        f0_s[isnan] = np.nanmedian(f0_s)
    if kernel_size % 2 == 0:
        kernel_size += 1
    f0_s = medfilt(f0_s, kernel_size)
    # якщо більшість кадрів були NaN — повернути їх
    if np.mean(np.isnan(f0)) > 0.5:
        f0_s[np.isnan(f0)] = np.nan
    return f0_s

# --------------------------- Detection logic ----------------------------

def detect_pauses(
    E: np.ndarray, times_E: np.ndarray, min_pause_dur: float = 0.12, low_energy_percentile: float = 20.0
) -> tuple[List[Tuple[float, float]], float]:
    """Повертає список пауз (t0, t1) та поріг енергії."""
    th = np.percentile(E, low_energy_percentile)
    low = E < th
    pauses = []
    i = 0
    while i < len(low):
        if low[i]:
            j = i
            while j < len(low) and low[j]:
                j += 1
            t0 = times_E[i]
            t1 = times_E[min(j, len(times_E)-1)]
            if (t1 - t0) >= min_pause_dur:
                pauses.append((float(t0), float(t1)))
            i = j
        else:
            i += 1
    return pauses, float(th)

def detect_stresses(
    f0_s: np.ndarray, E: np.ndarray, times: np.ndarray,
    energy_percentile: float = 80.0, f0_window: int = 5
) -> tuple[np.ndarray, np.ndarray, float]:
    """Наголос = локальний максимум F0 у кадрі з високою енергією."""
    voiced = ~np.isnan(f0_s)
    f0v = f0_s.copy()
    f0v[~voiced] = 0.0
    peaks = argrelextrema(f0v, np.greater, order=f0_window)[0]

    thE = np.percentile(E, energy_percentile)
    highE = E >= thE

    if len(E) != len(times):
        times_E = np.linspace(0, times[-1] if len(times) else 0.0, num=len(E))
        highE_interp = np.interp(times, times_E, highE.astype(float)) >= 0.5
    else:
        highE_interp = highE

    stress_idx = np.array([i for i in peaks if voiced[i] and highE_interp[i]], dtype=int)
    stress_times = times[stress_idx] if len(stress_idx) else np.array([], dtype=float)
    return stress_times, stress_idx, float(thE)

# --------------------------------- Plot ---------------------------------

def plot_with_pauses_and_stresses(
    times: np.ndarray, f0: np.ndarray, f0_s: np.ndarray,
    pauses: List[Tuple[float, float]], stress_times: np.ndarray,
    title: str, out_path: str
) -> None:
    plt.figure(figsize=(10, 5.5))
    plt.plot(times, f0, linewidth=1, alpha=0.5, label="F0 сирий (Hz)")
    plt.plot(times, f0_s, linewidth=2, label="F0 згладжений (Hz)")

    for (t0, t1) in pauses:
        plt.axvspan(t0, t1, alpha=0.2, label="Пауза" if t0 == pauses[0][0] else None)

    if len(stress_times) > 0:
        y_stress = np.interp(stress_times, times, np.nan_to_num(f0_s, nan=0.0))
        plt.scatter(stress_times, y_stress, s=50, marker='o', label="Наголос")

    plt.xlabel("Час, с")
    plt.ylabel("Частота, Гц")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# --------------------------------- Main ---------------------------------

def main():
    ap = argparse.ArgumentParser(description="Візуалізація пауз і наголосів у мовленні.")
    ap.add_argument("input", help="Шлях до аудіо (wav/mp3/ogg...).")
    ap.add_argument("--outdir", default="out", help="Каталог для збереження результатів.")
    ap.add_argument("--sr", type=int, default=16000, help="Частота дискретизації для аналізу.")
    ap.add_argument("--frame", type=int, default=2048, help="Розмір кадру для F0.")
    ap.add_argument("--hop", type=int, default=256, help="Крок кадру для F0/RMS.")
    ap.add_argument("--pause-percentile", type=float, default=20.0, help="Перцентиль енергії для пауз.")
    ap.add_argument("--min-pause", type=float, default=0.12, help="Мін. тривалість паузи, с.")
    ap.add_argument("--stress-percentile", type=float, default=80.0, help="Перцентиль енергії для наголосів.")
    ap.add_argument("--fmin", type=float, default=75.0, help="Мінімальна F0, Гц.")
    ap.add_argument("--fmax", type=float, default=400.0, help="Максимальна F0, Гц.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    y, sr = load_audio_mono(args.input, target_sr=args.sr)

    # енергія
    E = rms_energy(y, frame_length=1024, hop_length=args.hop)
    times_E = np.arange(len(E)) * (args.hop / sr)

    # F0
    f0, times = estimate_pitch(
        y, sr, fmin=args.fmin, fmax=args.fmax, frame_length=args.frame, hop_length=args.hop
    )
    f0_s = smooth_contour(f0)

    # детекція
    pauses, E_th = detect_pauses(E, times_E, min_pause_dur=args.min_pause, low_energy_percentile=args.pause_percentile)
    stress_times, stress_idx, E_high = detect_stresses(f0_s, E, times, energy_percentile=args.stress_percentile)

    # збереження графіка
    base = os.path.splitext(os.path.basename(args.input))[0]
    plot_path = os.path.join(args.outdir, f"{base}_pauses_stresses.png")
    plot_with_pauses_and_stresses(times, f0, f0_s, pauses, stress_times,
                                  title=f"{base}: F0, паузи (шэйдинг) і наголоси (кружки)", out_path=plot_path)

    # TSV з таймінгами
    tsv_path = os.path.join(args.outdir, f"{base}_pauses_stresses.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("type\tstart_s\tend_or_same_s\n")
        for (t0, t1) in pauses:
            f.write(f"pause\t{t0:.3f}\t{t1:.3f}\n")
        for tt in stress_times:
            f.write(f"stress\t{tt:.3f}\t{tt:.3f}\n")

    # короткий підсумок у консоль
    dur_s = len(y)/sr
    print("=== РЕЗУЛЬТАТ ===")
    print(f"Файл: {args.input}")
    print(f"Тривалість: {dur_s:.2f} с, SR: {sr} Гц")
    print(f"Порог енергії для пауз (P{args.pause_percentile:.0f}): {E_th:.6f}")
    print(f"Порог енергії для наголосів (P{args.stress_percentile:.0f}): {E_high:.6f}")
    print(f"К-сть пауз: {len(pauses)}; К-сть наголосів: {len(stress_times)}")
    print(f"PNG: {plot_path}")
    print(f"TSV: {tsv_path}")

if __name__ == "__main__":
    main()

# Intonation analysis demo (UA). 
# This notebook-like cell:
# 1) defines a function that computes a pitch (F0) contour and a simple intonation label,
# 2) runs a quick demo on a synthetic "rising" phrase,
# 3) optionally processes your own file if you set `user_audio_path` to a WAV/MP3 path.

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.io import wavfile
import os

# Try to use librosa if available (preferred)
try:
    import librosa
    import librosa.display  # noqa: F401
    HAVE_LIBROSA = True
except Exception as e:
    HAVE_LIBROSA = False

def load_audio_mono(path, target_sr=16000):
    """
    Load audio as mono float32 normalized to [-1,1].
    Uses librosa if available; otherwise falls back to scipy.io.wavfile (WAV only).
    """
    if HAVE_LIBROSA:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y.astype(np.float32), sr
    else:
        # WAV-only fallback
        sr, data = wavfile.read(path)
        # convert to float32 mono
        if data.ndim == 2:
            data = data.mean(axis=1)
        # normalize based on dtype
        if data.dtype == np.int16:
            y = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            y = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            y = (data.astype(np.float32) - 128) / 128.0
        else:
            y = data.astype(np.float32)
        if sr != target_sr:
            # basic resample if librosa is not available
            # simple linear resample
            x_old = np.linspace(0, 1, num=len(y), endpoint=False)
            x_new = np.linspace(0, 1, num=int(len(y) * target_sr / sr), endpoint=False)
            y = np.interp(x_new, x_old, y).astype(np.float32)
            sr = target_sr
        return y, sr

def rms_energy(y, frame_length=1024, hop_length=256):
    """
    Frame-wise RMS energy.
    """
    frames = []
    for i in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[i:i+frame_length]
        frames.append(np.sqrt(np.mean(frame**2) + 1e-12))
    return np.array(frames)

def estimate_pitch(y, sr, fmin=75.0, fmax=400.0, frame_length=2048, hop_length=256):
    """
    Estimate pitch (F0) contour in Hz.
    Preferred: librosa.yin/pyin. Fallback: naive autocorrelation-based estimator.
    Returns: f0 (Hz) with np.nan for unvoiced frames, and times array (s).
    """
    if HAVE_LIBROSA:
        try:
            # pyin is robust for voice; returns (f0, voiced_flag, voiced_probs)
            f0 = librosa.pyin(y, fmin=fmin, fmax=fmax, frame_length=frame_length, hop_length=hop_length)[0]
        except Exception:
            # fallback to yin
            f0 = librosa.yin(y, fmin=fmin, fmax=fmax, frame_length=frame_length, hop_length=hop_length)
        times = np.arange(len(f0)) * (hop_length / sr)
        return f0, times
    else:
        # Naive autocorrelation per frame
        def acf_pitch(frame, sr, fmin, fmax):
            # Hamming window
            w = np.hamming(len(frame))
            x = frame * w
            # autocorrelation
            corr = np.correlate(x, x, mode='full')[len(x)-1:]
            # search lags corresponding to fmax..fmin
            min_lag = int(sr / fmax)
            max_lag = int(sr / fmin)
            if max_lag >= len(corr):
                return np.nan
            segment = corr[min_lag:max_lag]
            if len(segment) == 0:
                return np.nan
            lag_hat = np.argmax(segment) + min_lag
            f0 = sr / lag_hat if lag_hat > 0 else np.nan
            return f0

        f0_list = []
        for i in range(0, len(y) - frame_length + 1, hop_length):
            frame = y[i:i+frame_length]
            f0_list.append(acf_pitch(frame, sr, fmin, fmax))
        f0 = np.array(f0_list, dtype=float)
        times = np.arange(len(f0)) * (hop_length / sr)
        return f0, times

def smooth_contour(f0, kernel_size=5):
    """
    Median-filter the F0 contour and inpaint short NaN gaps with local median.
    """
    f0_s = f0.copy()
    # fill short NaN gaps
    isnan = np.isnan(f0_s)
    if np.any(~isnan):
        median_val = np.nanmedian(f0_s)
        f0_s[isnan] = median_val
    # median filter
    if kernel_size % 2 == 0:
        kernel_size += 1
    f0_s = medfilt(f0_s, kernel_size)
    # restore very long unvoiced segments to NaN if the original had many NaNs
    # (simple heuristic)
    if np.mean(np.isnan(f0)) > 0.5:
        mask = np.isnan(f0)
        f0_s[mask] = np.nan
    return f0_s

def classify_intonation(f0_s, times):
    """
    Very simple rule-based intonation classifier:
    - looks at the slope of the last 30% of voiced frames
    - compares mean of last 15% vs first 15%
    Labels: 'висхідна', 'нисхідна', 'рівна/нейтральна'.
    Also returns a confidence [0..1].
    """
    voiced_mask = ~np.isnan(f0_s)
    if np.sum(voiced_mask) < 5:
        return "недостатньо даних (мало озвучених ділянок)", 0.0, {}

    f0_voiced = f0_s[voiced_mask]
    t_voiced = times[voiced_mask]

    n = len(f0_voiced)
    if n < 10:
        return "недостатньо даних", 0.2, {}

    # linear trend on last 30%
    start_idx = int(n * 0.7)
    x = t_voiced[start_idx:] - t_voiced[start_idx]
    y = f0_voiced[start_idx:]
    if len(x) < 3:
        return "недостатньо даних", 0.2, {}

    # simple linear regression slope
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    # compare means of first/last 15%
    k = max(3, int(n * 0.15))
    mean_first = np.mean(f0_voiced[:k])
    mean_last  = np.mean(f0_voiced[-k:])

    rel_change = (mean_last - mean_first) / (mean_first + 1e-6)

    # thresholds (heuristic)
    # rising if slope positive and relative change > +5%
    # falling if slope negative and relative change < -5%
    label = "рівна/нейтральна"
    conf = min(1.0, abs(rel_change) * 5.0)  # simple confidence proxy

    if slope > 0 and rel_change > 0.05:
        label = "висхідна (питальна/незавершеність)"
    elif slope < 0 and rel_change < -0.05:
        label = "нисхідна (ствердження/завершеність)"

    features = {
        "slope_last30pct_Hz_per_s": float(slope),
        "rel_change_first_last": float(rel_change),
        "f0_first_mean_Hz": float(mean_first),
        "f0_last_mean_Hz": float(mean_last),
        "voiced_ratio": float(np.mean(~np.isnan(f0_s))),
    }
    return label, float(conf), features

def analyze_intonation(path=None, y=None, sr=16000, title="Інтонація: F0 контур"):
    """
    Main helper:
    - Loads audio (if path provided) or uses (y,sr),
    - Estimates pitch contour,
    - Smooths it,
    - Classifies intonation,
    - Plots F0 vs time,
    - Returns (label, confidence, features, f0_raw, f0_smooth, times).
    """
    assert path is not None or y is not None, "Provide either path or (y,sr)."

    if path is not None:
        y, sr = load_audio_mono(path, target_sr=sr)

    # compute energy (optional; not plotted to keep a single plot)
    E = rms_energy(y)

    # pitch
    f0, times = estimate_pitch(y, sr)
    f0_s = smooth_contour(f0)

    label, conf, feats = classify_intonation(f0_s, times)

    # Plot F0
    plt.figure(figsize=(8, 4.5))
    plt.plot(times, f0, linewidth=1, alpha=0.5, label="F0 сирий (Hz)")
    plt.plot(times, f0_s, linewidth=2, label="F0 згладжений (Hz)")
    plt.xlabel("Час, с")
    plt.ylabel("Частота, Гц")
    plt.title(title + f"\nКлас: {label}, впевненість: {conf:.2f}")
    plt.legend()
    plot_path = "./pitch_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.show()

    return label, conf, feats, f0, f0_s, times, plot_path

# --------------------------
# DEMO: create synthetic "rising" phrase (~1.6s) with increasing F0 from 140->220 Hz
# --------------------------
sr = 16000
dur = 1.6
t = np.linspace(0, dur, int(sr*dur), endpoint=False)
f0_start, f0_end = 140.0, 220.0
# instantaneous frequency sweep (linear in time)
inst_freq = np.linspace(f0_start, f0_end, len(t))
phase = 2*np.pi*np.cumsum(inst_freq)/sr
y_demo = 0.2*np.sin(phase).astype(np.float32)

# add two short pauses (simulate speech breaks)
pause = np.zeros(int(0.08*sr), dtype=np.float32)
y_demo = np.concatenate([y_demo[:int(0.6*sr)], pause, y_demo[int(0.6*sr):int(1.1*sr)], pause, y_demo[int(1.1*sr):]])

# Save demo WAV for reference
demo_wav_path = "./demo_rising.wav"
wavfile.write(demo_wav_path, sr, (y_demo * 32767).astype(np.int16))

label, conf, feats, f0, f0_s, times, plot_path = analyze_intonation(y=y_demo, sr=sr, title="Демо: зростаюча інтонація")

print("=== Результат класифікації (демо) ===")
print("Мітка:", label)
print("Впевненість:", f"{conf:.2f}")
print("Ознаки:", feats)

print("\nФайли:")
print("Демо WAV:", demo_wav_path)
print("Графік F0:", plot_path)

# If you want to analyze your own file:
user_audio_path = None  # e.g., "/mnt/data/your_file.wav" or "/mnt/data/your_file.mp3"
user_audio_path = './OSR_us_000_0010_8k.wav'
user_audio_path = './Harvard list 01.wav'
user_audio_path = './sentences/sentence_3.wav'
if user_audio_path and os.path.exists(user_audio_path):
    label2, conf2, feats2, *_ = analyze_intonation(path=user_audio_path, title="Ваш файл: інтонація")
    print("\n=== Ваш файл ===")
    print("Мітка:", label2)
    print("Впевненість:", f"{conf2:.2f}")
    print("Ознаки:", feats2)

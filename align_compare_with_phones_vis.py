# align_compare_with_phones_vis.py
import os, re, math, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib
matplotlib.use("Agg")  # для headless-середовищ
import matplotlib.pyplot as plt
import torch, whisperx
from jiwer import wer
import pandas as pd

# ====================== Налаштування ======================
USE_STRICT_REF_ALIGNMENT = True   # ← строгий forced alignment під референсний текст
SENT_DIR = "sentences"            # де лежать sentence_1.wav і sentence_1.txt
OUT_DIR = "OUT"                   # куди зберігати результати
os.makedirs(OUT_DIR, exist_ok=True)

SR_RECORD = 16000                 # частота запису користувача
REC_MIN_SEC = 6                   # мінімальна тривалість запису
REC_WPM = 160                     # орієнтовна швидкість читання (слів/хв)
ASR_MODEL = "small"               # "base"/"small"/"medium"/"large-v3" (multilingual)
BEEP_FREQ = 880                   # Hz
BEEP_DUR = 0.5                    # s

# Кольори для типів операцій
COLOR_MAP = {
    "match":   "#6aaa64",  # зелений
    "replace": "#c9b458",  # жовтий
    "insert":  "#787c7e",  # сірий
    "delete":  "#d55c5c",  # червоний
    "none":    "#9aa0a6",
}

# ====================== Утиліти ======================
def read_reference_pair(base_name: str):
    wav_path = os.path.join(SENT_DIR, base_name + ".wav")
    txt_path = os.path.join(SENT_DIR, base_name + ".txt")
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Не знайдено {wav_path}")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Не знайдено {txt_path}")
    ref_audio, sr = sf.read(wav_path, dtype="float32")
    with open(txt_path, "r", encoding="utf-8") as f:
        ref_text = f.read().strip()
    return ref_audio, sr, ref_text, wav_path, txt_path

def play_audio(audio: np.ndarray, sr: int):
    sd.play(audio, sr)
    sd.wait()

def beep(sr=SR_RECORD, freq=BEEP_FREQ, dur=BEEP_DUR):
    t = np.linspace(0, dur, int(sr*dur), False)
    tone = 0.2*np.sin(2*np.pi*freq*t).astype(np.float32)
    sd.play(tone, sr); sd.wait()

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s’'-]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def words(s: str):
    return normalize_text(s).split()

def levenshtein_ops(ref_tokens, hyp_tokens):
    """
    Lевенштайн на словах.
    Повертає список: (op, ref_word, hyp_word, ref_idx, hyp_idx)
    де insert: ref_word='<eps>', ref_idx=None
       delete: hyp_word='<eps>', hyp_idx=None
    """
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if ref_tokens[i-1]==hyp_tokens[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    ops = []
    i, j = n, m
    while i>0 or j>0:
        if i>0 and j>0 and dp[i][j]==dp[i-1][j-1] and ref_tokens[i-1]==hyp_tokens[j-1]:
            ops.append(("match", ref_tokens[i-1], hyp_tokens[j-1], i-1, j-1)); i-=1; j-=1
        elif i>0 and j>0 and dp[i][j]==dp[i-1][j-1]+1:
            ops.append(("replace", ref_tokens[i-1], hyp_tokens[j-1], i-1, j-1)); i-=1; j-=1
        elif i>0 and dp[i][j]==dp[i-1][j]+1:
            ops.append(("delete", ref_tokens[i-1], "<eps>", i-1, None)); i-=1
        else:
            ops.append(("insert", "<eps>", hyp_tokens[j-1], None, j-1)); j-=1
    ops.reverse()
    return ops

def estimate_record_seconds(ref_text: str) -> int:
    w = len(words(ref_text))
    sec = max(REC_MIN_SEC, int(60 * w / max(1, REC_WPM)))
    return sec + 2  # запас

def record_user_int16(sec: int, sr: int, out_path: str) -> str:
    """
    Запис у int16 та збереження WAV як PCM_16.
    """
    print(f"▶ Починаємо запис на {sec} с (dtype=int16)…")
    audio = sd.rec(int(sec*sr), samplerate=sr, channels=1, dtype='int16')
    sd.wait()
    sf.write(out_path, audio, sr, subtype="PCM_16")
    print(f"Збережено (PCM_16): {out_path}")
    return out_path

def choose_align_lang(asr_result_lang: str) -> str:
    return asr_result_lang if asr_result_lang else "en"

def extract_words_with_times(aligned_segments):
    """
    Повертає список слів (у тій послідовності, як від aligner) з їх time-span.
    Важливо: це слова з ТЕКСТУ, який ми подавали в align (ref або ASR).
    """
    out = []
    for seg in aligned_segments:
        for w in seg.get("words", []):
            out.append({
                "word": normalize_text(w.get("word","")),
                "start": w.get("start", None),
                "end": w.get("end", None)
            })
    return out

def build_token_time_map(tokens, aligned_words):
    """
    Створює відображення: індекс токена (у списку tokens) -> (start,end) з aligned_words
    Послідовно зіставляє за нормалізованим словом.
    """
    times = []
    it = iter([w for w in aligned_words if w["word"]])
    for tok in tokens:
        t = None
        while True:
            try:
                cand = next(it)
            except StopIteration:
                break
            if cand["word"] == tok:
                t = (cand["start"], cand["end"])
                break
        times.append(t)  # може бути None, якщо не знайшли
    return times

def attach_ops_timing_by_side(ops, ref_tokens, hyp_tokens, aligned_words, strict_mode: bool):
    """
    Прив’язуємо час до ОПЕРАЦІЙ з правильного боку:
    - strict_mode=True: час беремо з РЕФЕРЕНСНИХ токенів (ref_tokens),
      тобто мапимо ref_tokens -> aligned_words.
    - strict_mode=False: час беремо з ГІПОТЕЗНИХ токенів (hyp_tokens),
      тобто мапимо hyp_tokens -> aligned_words.
    """
    if strict_mode:
        token_times = build_token_time_map(ref_tokens, aligned_words)
    else:
        token_times = build_token_time_map(hyp_tokens, aligned_words)

    rows = []
    for op, r, h, i_ref, j_hyp in ops:
        start = end = None
        word_for_timing = None
        idx_for_time = i_ref if strict_mode else j_hyp
        word_for_timing = r if strict_mode else h

        if idx_for_time is not None and 0 <= idx_for_time < len(token_times):
            t = token_times[idx_for_time]
            if t:
                start, end = t

        rows.append({
            "op": op,
            "ref_word": r,
            "hyp_word": h,
            "ref_index": i_ref,
            "hyp_index": j_hyp,
            "word_for_timing": word_for_timing,  # ← це слово ми забарвлюємо/підписуємо
            "start": start,
            "end": end
        })
    return rows

def plot_words_timeline(rows, out_png="words_timeline.png"):
    rows2 = [r for r in rows if r["start"] is not None and r["end"] is not None]
    if not rows2:
        print(f"[!] Timeline is empty, skip plot {out_png}")
        return
    rows2.sort(key=lambda r: r["start"])
    y = list(range(len(rows2)))
    starts = [r["start"] for r in rows2]
    durs = [r["end"] - r["start"] for r in rows2]
    labels = [r["word_for_timing"] for r in rows2]
    colors = [COLOR_MAP.get(r["op"], COLOR_MAP["none"]) for r in rows2]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.25*len(rows2))))
    ax.barh(y, durs, left=starts, color=colors)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Words (index)")
    ax.set_title("Words timeline with error types (timed by chosen side)")
    for i, lab in enumerate(labels):
        ax.text(starts[i] + durs[i]/2, y[i], lab, va="center", ha="center", fontsize=8, color="white")
    handles = [plt.Line2D([0],[0], color=COLOR_MAP[k], lw=6) for k in ["match","replace","insert","delete"]]
    ax.legend(handles, ["match","replace","insert","delete"], loc="lower right")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
    print(f"✔ PNG saved: {out_png}")

def plot_phones_timeline(aligned_segments, word_ops_rows, out_png="phones_timeline.png"):
    """
    Фонеми фарбуємо за типом помилки слова, час якого ми використовували (word_for_timing).
    Тобто шукаємо відповідність у aligned words за (word,start,end).
    """
    # індекс: (word_for_timing, start,end) -> op
    idx = {}
    for r in word_ops_rows:
        if r["start"] is not None and r["end"] is not None:
            key = (r["word_for_timing"], r["start"], r["end"])
            idx[key] = r["op"]

    phonelines = []
    for seg in aligned_segments:
        for w in seg.get("words", []):
            w_norm = normalize_text(w.get("word",""))
            w_start, w_end = w.get("start", None), w.get("end", None)
            if w_start is None or w_end is None:
                continue
            op = idx.get((w_norm, w_start, w_end))
            if op is None:
                # слово не використовувалось для таймінгу (наприклад, insert/delete з іншої сторони)
                continue
            for ph in w.get("phones", []):
                p_start, p_end = ph.get("start", None), ph.get("end", None)
                p = ph.get("phone","")
                if p_start is None or p_end is None:
                    continue
                phonelines.append({"phone": p, "start": p_start, "end": p_end, "op": op})

    if not phonelines:
        print(f"[!] Timeline is empty, skip plot {out_png}")
        return

    phonelines.sort(key=lambda r: r["start"])
    y = list(range(len(phonelines)))
    starts = [r["start"] for r in phonelines]
    durs = [r["end"] - r["start"] for r in phonelines]
    labels = [r["phone"] for r in phonelines]
    colors = [COLOR_MAP.get(r["op"], COLOR_MAP["none"]) for r in phonelines]

    fig, ax = plt.subplots(figsize=(14, max(4, 0.25*len(phonelines))))
    ax.barh(y, durs, left=starts, color=colors)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Phones (index)")
    ax.set_title("Phoneme timeline (colored by parent word error type)")
    step = max(1, math.ceil(len(labels)/80))
    for i in range(0, len(labels), step):
        ax.text(starts[i] + durs[i]/2, y[i], labels[i],
                va="center", ha="center", fontsize=7, color="white")
    handles = [plt.Line2D([0],[0], color=COLOR_MAP[k], lw=6) for k in ["match","replace","insert","delete"]]
    ax.legend(handles, ["match","replace","insert","delete"], loc="lower right")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
    print(f"✔ PNG saved: {out_png}")

def pick_compute_type(device: str) -> str:
    if device == "cpu":
        return "int8"
    return "float16"

# ====================== Основний сценарій ======================
def main(base_name="sentence_1"):
    prefix     = os.path.join(OUT_DIR, base_name)
    user_wav   = f"{prefix}_user.wav"
    out_csv    = f"{prefix}_errors.csv"
    out_report = f"{prefix}_report.txt"
    out_debug  = f"{prefix}_debug.json"
    png_words  = f"{prefix}_words_timeline.png"
    png_phones = f"{prefix}_phones_timeline.png"

    # 1) Референс: WAV + TXT з каталогу SENT_DIR
    ref_audio, ref_sr, ref_text, ref_wav_path, ref_txt_path = read_reference_pair(base_name)

    print("\n🔊 Програємо референсний аудіофайл…")
    play_audio(ref_audio, ref_sr)

    # 2) Показуємо текст і подаємо beep
    print("\n📄 Референсний текст (прочитайте після сигналу):")
    print("—"*60); print(ref_text); print("—"*60)
    time.sleep(0.3); beep()

    # 3) Запис користувача (PCM_16)
    rec_sec = estimate_record_seconds(ref_text)
    record_user_int16(rec_sec, SR_RECORD, out_path=user_wav)

    # 4) ASR (WhisperX) — отримаємо segments
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = pick_compute_type(device)
    print(f"\n🧠 Завантаження WhisperX ({ASR_MODEL}) на {device} | compute_type={compute_type}…")
    try:
        asr_model = whisperx.load_model(ASR_MODEL, device=device, compute_type=compute_type)
    except ValueError as e:
        if "float16" in str(e).lower() and device != "cpu":
            try:
                asr_model = whisperx.load_model(ASR_MODEL, device=device, compute_type="int8_float16")
            except Exception:
                asr_model = whisperx.load_model(ASR_MODEL, device=device, compute_type="int8")
        else:
            asr_model = whisperx.load_model(ASR_MODEL, device=device, compute_type="int8")

    asr_res = asr_model.transcribe(user_wav)
    hyp_text = asr_res.get("text", "").strip()
    hyp_text = " ".join(seg["text"].strip() for seg in asr_res.get("segments", []))
    hyp_lang = asr_res.get("language", "en")
    segs = asr_res.get("segments", []) or []
    print(f"\n———— ASR ————")
    print("Гіпотеза (ASR):", hyp_text)
    print("Мова (ASR):", hyp_lang)
    print(f"ASR segments: {len(segs)}")

    # 5) WER
    ref_norm = normalize_text(ref_text)
    hyp_norm = normalize_text(hyp_text)
    ref_tokens = words(ref_norm)
    hyp_tokens = words(hyp_norm)
    wer_score = wer(ref_norm, hyp_norm)
    print(f"\n📊 WER = {wer_score:.3f}")

    # 6) Forced alignment — STRICT під референс або під ASR-сегменти
    audio_f32, sr_check = sf.read(user_wav, dtype="float32")
    audio_dur = len(audio_f32) / sr_check
    if USE_STRICT_REF_ALIGNMENT:
        segments_for_align = [{"text": ref_text, "start": 0.0, "end": float(audio_dur)}]
        print("Режим: STRICT reference alignment.")
    else:
        if not segs:
            segs = [{"text": hyp_text, "start": 0.0, "end": float(audio_dur)}]
        segments_for_align = segs
        print(f"Режим: ASR alignment. Сегментів: {len(segments_for_align)}")

    align_lang = choose_align_lang(hyp_lang)
    align_model, metadata = whisperx.load_align_model(language_code=align_lang, device=device)
    aligned = whisperx.align(segments_for_align, align_model, metadata, user_wav, device=device)

    # 7) Слова з таймкодами з вирівнювання (це слова з того ТЕКСТУ, який подали в align)
    aligned_words = extract_words_with_times(aligned["segments"])
    print(f"Aligned words with times: {len(aligned_words)}")

    # 8) Lевенштайн-операції та прив’язка часу з правильного боку
    op_rows = attach_ops_timing_by_side(
        ops=levenshtein_ops(ref_tokens, hyp_tokens),
        ref_tokens=ref_tokens,
        hyp_tokens=hyp_tokens,
        aligned_words=aligned_words,
        strict_mode=USE_STRICT_REF_ALIGNMENT
    )

    # Діагностика: скільки рядків мають валідний час
    valid_timed = sum(1 for r in op_rows if r["start"] is not None and r["end"] is not None)
    print(f"Timed rows: {valid_timed} / {len(op_rows)} (strict={USE_STRICT_REF_ALIGNMENT})")

    # 9) CSV помилок
    out_csv = os.path.join(OUT_DIR, f"{base_name}_errors.csv")
    pd.DataFrame(op_rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"💾 Збережено деталі помилок: {out_csv}")

    # 10) Візуалізації (використовують word_for_timing + час тієї ж сторони)
    png_words  = os.path.join(OUT_DIR, f"{base_name}_words_timeline.png")
    png_phones = os.path.join(OUT_DIR, f"{base_name}_phones_timeline.png")
    plot_words_timeline(op_rows, out_png=png_words)
    plot_phones_timeline(aligned["segments"], op_rows, out_png=png_phones)

    # 11) Короткий звіт
    out_report = os.path.join(OUT_DIR, f"{base_name}_report.txt")
    report = []
    report.append(f"Reference TXT: {ref_txt_path}")
    report.append(f"Reference WAV: {ref_wav_path}")
    report.append(f"User WAV:      {os.path.join(OUT_DIR, base_name + '_user.wav')} (PCM_16)")
    report.append(f"ASR model:     {ASR_MODEL}  | device: {device}")
    report.append(f"ASR language:  {hyp_lang}")
    report.append(f"ASR segments:  {len(segs)}")
    report.append(f"WER:           {wer_score:.3f}")
    report.append(f"Strict ref alignment: {USE_STRICT_REF_ALIGNMENT}")
    report.append(f"Timed rows:    {valid_timed} / {len(op_rows)}")
    report.append("\nOperations (word-level):")
    for r in op_rows:
        span = f" @ [{r['start']:.2f}–{r['end']:.2f}] s" if r["start"] is not None else ""
        report.append(f"{r['op']:7s}  REF='{r['ref_word']}'  HYP='{r['hyp_word']}'  USING='{r['word_for_timing']}'{span}")
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"📝 Звіт: {out_report}")

    # 12) Debug JSON
    out_debug = os.path.join(OUT_DIR, f"{base_name}_debug.json")
    with open(out_debug, "w", encoding="utf-8") as f:
        json.dump({
            "ref_text": ref_text,
            "hyp_text": hyp_text,
            "ref_tokens": ref_tokens,
            "hyp_tokens": hyp_tokens,
            "ops": op_rows,
            "aligned_words": aligned_words
        }, f, ensure_ascii=False, indent=2)
    print(f"🔧 Debug JSON: {out_debug}")

    print("\n✅ Готово. PNG/CSV/звіт у папці OUT.")
    print("   Якщо PNG не створились — подивіться 'Timed rows: …'. Має бути > 0.")

# Точка входу
if __name__ == "__main__":
    main(base_name="sentence_1")

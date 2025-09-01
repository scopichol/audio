import os
import re
import sounddevice as sd
from scipy.io.wavfile import write
import torch, whisperx
from g2p_en import G2p
from nltk.metrics.distance import edit_distance  # токен-рівень
import nltk

# --- 0) Безпечне завантаження NLTK-ресурсів (опційно, якщо треба) ---
# Теггер тут не використовується; якщо знадобиться, увімкни одну з двох строк:
# nltk.download('averaged_perceptron_tagger')           # класична назва
# nltk.download('averaged_perceptron_tagger_eng')       # новіша ресурсна назва

from utilites import read_reference_text, arpabet_seq_to_ipa, display_pauses_with_symbols, analyze_pause_distribution  # додано нові функції

# --- 0.1) Hugging Face: вимкнути симлінки, щоб не ловити WinError 1314 ---
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_LOCAL_DIR_USE_SYMLINKS", "0")

# ---------- 1) Запис аудіо (опційно, приклад) ----------
# sr = 16000
# seconds = 5
# target_text = "The quick brown fox jumps over the lazy dog."
# print(f'Say the phrase: "{target_text}"')
# audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype='int16')
# sd.wait()
# write("en_record.wav", sr, audio)
# print("Saved: en_record.wav")

ref_file = "sentences/sentence_1.wav"
target_text = read_reference_text(ref_file)
sample_file = "out/sentence_1_user.wav"  # або інший твій записаний файл

# ---------- 2) Еталонні фонеми (ARPAbet) ----------
g2p = G2p()
def g2p_word_arpabet(word: str, keep_stress=False):
    ph = [p for p in g2p(word) if isinstance(p, str)]
    if not keep_stress:
        ph = [re.sub(r"\d", "", p) for p in ph]
    ph = [p for p in ph if re.fullmatch(r"[A-Z]+", p) or re.fullmatch(r"[A-Z]+\d", p)]
    return ph

# Токенізуємо слова та пунктуацію окремо
tokens = re.findall(r"\w+|[^\w\s]", target_text)
arpabet_ref = []
for tok in tokens:
    if re.fullmatch(r"\w+", tok):              # слово
        ph = g2p_word_arpabet(tok.lower(), keep_stress=False)
        arpabet_ref.extend(ph)
        arpabet_ref.append("SP")               # коротка пауза між словами
    else:                                       # пунктуація → більша пауза
        if tok in [".", "!", "?", ";", ":"]:
            arpabet_ref.append("SIL")          # довша/фразова пауза
        elif tok in [","]:
            arpabet_ref.append("SP2")          # середня пауза
# прибираємо паузу в кінці, якщо лишилась
while arpabet_ref and arpabet_ref[-1] in ("SP","SP2","SIL"):
    arpabet_ref.pop()
# прибрати цифри наголосу, залишити лише фонеми (але зберегти паузи)
arpabet_ref = [re.sub(r"\d", "", p) if re.match(r"^[A-Z]+(\d)?$", p) else p for p in arpabet_ref]
print('------------------------------------------')
print("REF phones:", arpabet_ref)
print("REF IPA:", arpabet_seq_to_ipa(arpabet_ref))

# Додаткова інформація про паузи
from utilites import display_pauses_with_symbols, analyze_pause_distribution
print("REF with pause symbols:", display_pauses_with_symbols(arpabet_ref, "brackets"))

pause_stats = analyze_pause_distribution(arpabet_ref)
print(f"Pause analysis - Total pauses: {pause_stats['total_pauses']}, Distribution: {pause_stats['pause_distribution']}")
print('------------------------------------------')

# ---------- 3) Розпізнавання + фонемний алайнмент ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Визначимо безпечний compute_type
def pick_compute_type(dev: str) -> str:
    try:
        import ctranslate2
        supported = ctranslate2.get_supported_compute_types(dev)
        # Переваги за швидкістю/якістю
        for cand in (("cuda", ["float16", "int8_float16", "int16", "int8_float32"]),
                     ("cpu",  ["int8", "int8_float32", "int16", "bfloat16", "float32"])):
            if dev == cand[0]:
                for ct in cand[1]:
                    if ct in supported:
                        return ct
        # Фолбек
        return "int8" if dev == "cpu" else "float32"
    except Exception:
        # Якщо ctranslate2 недоступний/старий — ставимо типи за замовчуванням
        return "int8" if dev == "cpu" else "float16"

compute_type = pick_compute_type(device)
print(f"Device: {device}, compute_type: {compute_type}")

# Завантаження моделі ASR (англ.)
model = whisperx.load_model(
    "small.en",          # або "Systran/faster-whisper-small.en"
    device=device,
    compute_type=compute_type
)
res = model.transcribe(sample_file)
asr_text = " ".join(seg["text"].strip() for seg in res["segments"])
print("ASR text:", asr_text)

# Алайнер (англійська)
align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
aligned = whisperx.align(res["segments"], align_model, metadata, sample_file, device=device)

# print('ALIGNED:', aligned)
# Витягнути ARPAbet фонеми з вирівняних слів
def word_to_arpabet(word: str):
    # g2p_en повертає мікс токенів, заберемо лише фонеми; приберемо цифри наголосу
    ph = [p for p in g2p(word) if isinstance(p, str)]
    ph = [re.sub(r"\d", "", p) for p in ph if re.match(r"^[A-Z]+(\d)?$", p)]
    return ph

arpabet_hyp = []
for seg in aligned.get("segments", []):
    for w in seg.get("words", []):
        # слово без пунктуації в кінці (типу 'planks.')
        clean = re.sub(r"[^\w']+$", "", w.get("word", ""))
        if not clean:
            continue
        ph = word_to_arpabet(clean.lower())
        arpabet_hyp.extend(ph)
        arpabet_hyp.append("SP")  # додаємо паузу між словами

# Прибираємо останню паузу
while arpabet_hyp and arpabet_hyp[-1] in ("SP", "SP2", "SIL"):
    arpabet_hyp.pop()

print("HYP phones:", arpabet_hyp)
print("HYP IPA:", arpabet_seq_to_ipa(arpabet_hyp))
print("HYP with pause symbols:", display_pauses_with_symbols(arpabet_hyp, "brackets"))

hyp_pause_stats = analyze_pause_distribution(arpabet_hyp)
print(f"HYP Pause analysis - Total pauses: {hyp_pause_stats['total_pauses']}, Distribution: {hyp_pause_stats['pause_distribution']}")

# ---------- 4) Phone Error Rate (PER) по токенах ----------
# ВАЖЛИВО: рахуємо редаг-відстань між СПИСКАМИ, а не між рядками
ed_tokens = edit_distance(arpabet_ref, arpabet_hyp)      # токен-рівень
per = ed_tokens / max(1, len(arpabet_ref))
print(f"\nPER = {per:.3f}  (0.0 краще)")

# ---------- 5) Phone Error Rate (PER) по IPA ----------
ed_tokens_ipa = edit_distance(arpabet_seq_to_ipa(arpabet_ref).split(), arpabet_seq_to_ipa(arpabet_hyp).split())
per_ipa = ed_tokens_ipa / max(1, len(arpabet_seq_to_ipa(arpabet_ref).split()))
print(f"PER (IPA) = {per_ipa:.3f}  (0.0 краще)")
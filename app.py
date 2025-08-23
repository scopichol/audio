import parselmouth
from pydub import AudioSegment
import numpy as np
from write_audio import record_audio
import sounddevice as sd
import soundfile as sf
import jiwer 

def analyze_pitch(audio_path):
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]  # remove unvoiced
    if len(pitch_values) < 2:
        return "невизначено"
    start_pitch = np.mean(pitch_values[:len(pitch_values)//3])
    end_pitch = np.mean(pitch_values[-len(pitch_values)//3:])
    if end_pitch > start_pitch + 10:
        return "висхідний"
    elif start_pitch > end_pitch + 10:
        return "нисхідний"
    else:
        return "рівний"

def analyze_stress(audio_path):
    snd = parselmouth.Sound(audio_path)
    intensity = snd.to_intensity()
    times = intensity.xs()
    values = intensity.values[0]
    max_idx = np.argmax(values)
    stress_time = times[max_idx]
    return stress_time

def process_sentence(audio_path, words_timestamps):
    results = []
    audio = AudioSegment.from_wav(audio_path)
    for word, (start_ms, end_ms) in words_timestamps:
        word_audio = audio[start_ms:end_ms]
        temp_path = "temp_word.wav"
        word_audio.export(temp_path, format="wav")
        pitch_type = analyze_pitch(temp_path)
        stress_time = analyze_stress(temp_path)
        results.append({
            "word": word,
            "pitch": pitch_type,
            "stress_time": stress_time
        })
    return results

def get_words_timestamps(audio_path):
    """
    Використовує whisper-timestamped для автоматичного визначення часових меж слів.
    Повертає: [(word, (start_ms, end_ms)), ...]
    """
    from whisper_timestamped import load_model, transcribe

    model = load_model("small")  # або "base", "medium", "large"
    result = transcribe(model, audio_path, language="en")

    words = []
    for segment in result["segments"]:
        for word_info in segment["words"]:
            word = word_info["text"].strip()
            start = int(word_info["start"] * 1000)
            end = int(word_info["end"] * 1000)
            words.append((word, (start, end)))
    return words

def play_audio(filepath):
    data, fs = sf.read(filepath, dtype='int16')
    sd.play(data, fs)
    sd.wait()

def play_beep(duration=0.3, freq=1000, fs=44100):
    t = np.linspace(0, duration, int(fs * duration), False)
    beep = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    sd.play(beep, fs)
    sd.wait()

def read_reference_text(audio_path):
    """
    Читає референсний текст для аудіо-файлу.
    Шукає файл з таким же ім'ям, але з розширенням .txt.
    """
    import os
    txt_path = os.path.splitext(audio_path)[0] + ".txt"
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

# Example usage:
sample_file = "sentences/sentence_1.wav"
audio_file = "output9.wav"  # записаний файл з write_audio.py

# Розпізнавання зразку
print("Розпізнавання зразку:")
sample_words_timestamps = get_words_timestamps(sample_file)
sample_results = process_sentence(sample_file, sample_words_timestamps)
for r, (w, (start, stop)) in zip(sample_results, sample_words_timestamps):
    print(f"{r['word']}: наголос на {r['stress_time']} мс, тон {r['pitch']}")

# Відтворення зразку перед записом
print("Відтворення зразку...")
play_audio(sample_file)

# Відтворення сигналу перед записом
play_beep()

# Повтор запису аудіо поки WER не менше 30%
wer_threshold = 0.3
attempt = 1
while True:
    print(f"\nСпроба запису #{attempt}")
    record_audio(audio_file, duration=15)
    words_timestamps = get_words_timestamps(audio_file)
    results = process_sentence(audio_file, words_timestamps)

    for r, (w, (start, stop)) in zip(results, words_timestamps):
        print(f"{r['word']}: наголос на {r['stress_time']} мс, тон {r['pitch']}")

    ref = read_reference_text(sample_file)
    hypUser = ' '.join([r['word'] for r in results])
    wer = jiwer.wer(ref, hypUser)
    print(f"\nREF: {ref}")
    print(f"HYP (User): {hypUser}")
    print(f"\nHyp (User) WER: {wer*100:.2f}%, CER: {jiwer.cer(ref,hypUser)*100:.2f}%")

    play_audio(audio_file)

    if wer < wer_threshold:
        print(f"\nWER < {wer_threshold*100:.0f}%. Запис прийнято.")
        break
    else:
        print(f"\nWER >= {wer_threshold*100:.0f}%. Повторіть запис.")
        play_beep()
        attempt += 1
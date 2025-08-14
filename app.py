import parselmouth
from pydub import AudioSegment
import numpy as np

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
    result = transcribe(model, audio_path, language="uk")

    words = []
    for segment in result["segments"]:
        for word_info in segment["words"]:
            word = word_info["text"].strip()
            start = int(word_info["start"] * 1000)
            end = int(word_info["end"] * 1000)
            words.append((word, (start, end)))
    return words

# Example usage:
words_timestamps = get_words_timestamps("sentences/sentence_1.wav")
results = process_sentence("sentences/sentence_1.wav", words_timestamps)
for r in results:
    print(f"{r['word']}: наголос на {r['stress_time']} мс, тон {r['pitch']}")
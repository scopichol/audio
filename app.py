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

def get_words_timestamps(audio_path, transcript_path):
    """
    Використовує aeneas для автоматичного визначення часових меж слів.
    transcript_path — текстовий файл з реченням (одне речення).
    Повертає: [(word, (start_ms, end_ms)), ...]
    """
    from aeneas.executetask import ExecuteTask
    from aeneas.task import Task
    import xml.etree.ElementTree as ET

    config_string = u"task_language=uk|is_text_type=plain|os_task_file_format=xml"
    task = Task(config_string=config_string)
    task.audio_file_path_absolute = audio_path
    task.text_file_path_absolute = transcript_path
    task.output_file_path_absolute = "temp_syncmap.xml"

    ExecuteTask(task).execute()
    task.output_sync_map_file()

    tree = ET.parse("temp_syncmap.xml")
    root = tree.getroot()
    results = []
    for fragment in root.findall(".//fragment"):
        word = fragment.attrib["lines"].strip()
        start = float(fragment.attrib["begin"]) * 1000
        end = float(fragment.attrib["end"]) * 1000
        results.append((word, (int(start), int(end))))
    return results

# Example usage:
# transcript.txt має одне речення, кожне слово на окремому рядку або через пробіл.
words_timestamps = get_words_timestamps("sentences/sentence_1.wav", "sentences/sentence_1.txt")
results = process_sentence("sentences/sentence_1.wav", words_timestamps)
for r in results:
    print(f"{r['word']}: наголос на {r['stress_time']} мс, тон {r['pitch']}")
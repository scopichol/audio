import re
import os
import json
import math
import time
import itertools
import warnings
warnings.filterwarnings("ignore")

import sounddevice as sd
from scipy.io.wavfile import write
import torch, whisperx
from g2p_en import G2p
import Levenshtein as lev
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from utilites import (
    get_output_path, play_beep_signal, arpabet_token_to_ipa, 
    arpabet_seq_to_ipa, save_confusion_matrix, save_hyp_phones_csv, 
    plot_timeline, g2p_arpabet, compute_per
)


# ------------------------ Налаштування ------------------------
SR = 16000
SECONDS = 6
REFERENCE_NAME = "sentence_11"  # Назва референсного файлу
REFERENCE_AUDIO = f"sentences/{REFERENCE_NAME}.wav"
REFERENCE_TEXT_FILE = f"sentences/{REFERENCE_NAME}.txt"
AUDIO_OUT = get_output_path(f"{REFERENCE_NAME}_record.wav")  # Зберігаємо з префіксом в каталог out
TARGET_TEXT = "The quick brown fox jumps over the lazy dog."  # Буде замінено з файлу
MODEL_NAME = "small.en"  # можна "base.en"/"medium.en"/"large-v3"
LANG_CODE = "en"         # для align-моделі


# ------------------------ Утіліти ------------------------
def read_reference_text(text_file_path):
    """Читає текст з референсного файлу"""
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        return text
    except FileNotFoundError:
        print(f"❌ Файл {text_file_path} не знайдено")
        return None
    except Exception as e:
        print(f"❌ Помилка читання файлу {text_file_path}: {e}")
        return None

def play_reference_audio(audio_file_path):
    """Відтворює референсний аудіо файл"""
    try:
        import soundfile as sf
        
        # Читаємо аудіо файл
        data, samplerate = sf.read(audio_file_path)
        
        print(f"🔊 Відтворюємо референсний аудіо: {audio_file_path}")
        print("   Слухайте уважно і готуйтесь повторити...")
        
        # Відтворюємо аудіо
        sd.play(data, samplerate)
        sd.wait()  # Чекаємо до завершення відтворення
        
        print("✅ Референсний аудіо завершено")
        return True
        
    except ImportError:
        print("❌ soundfile не встановлено. Встановіть: pip install soundfile")
        return False
    except FileNotFoundError:
        print(f"❌ Аудіо файл {audio_file_path} не знайдено")
        return False
    except Exception as e:
        print(f"❌ Помилка відтворення аудіо {audio_file_path}: {e}")
        return False

def pick_compute_type(device: str) -> str:
    """Вибирає безпечний compute_type для WhisperX в залежності від пристрою."""
    if device == "cpu":
        return "int8"
    else:  # cuda
        try:
            # Перевіряємо, чи підтримує GPU float16
            if torch.cuda.is_available():
                # Простий тест на підтримку float16
                test_tensor = torch.randn(10, 10, device=device, dtype=torch.float16)
                _ = torch.mm(test_tensor, test_tensor)
                return "float16"
        except Exception as e:
            print(f"Float16 не підтримується: {e}")
        return "int8"

def safe_load_whisperx_model(model_name, device):
    """Безпечне завантаження моделі WhisperX з автоматичним вибором compute_type."""
    compute_type = pick_compute_type(device)
    
    try:
        print(f"Спроба завантаження з compute_type={compute_type}")
        return whisperx.load_model(model_name, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"Помилка з compute_type={compute_type}: {e}")
        
        # Якщо не вдалося, спробуємо з int8
        if compute_type != "int8":
            try:
                print("Спроба з compute_type=int8")
                return whisperx.load_model(model_name, device=device, compute_type="int8")
            except Exception as e2:
                print(f"Помилка з int8: {e2}")
        
        # Якщо і int8 не працює, спробуємо без compute_type (за замовчуванням)
        try:
            print("Спроба без compute_type (за замовчуванням)")
            return whisperx.load_model(model_name, device=device)
        except Exception as e3:
            print(f"Критична помилка: {e3}")
            raise e3


def g2p_arpabet(text):
    """Конвертує текст в ARPAbet фонеми з паузами між словами"""
    g2p = G2p()
    words = text.split()
    all_phonemes = []
    
    for i, word in enumerate(words):
        # Прибираємо пунктуацію з слова
        clean_word = re.sub(r'[^\w\']+', '', word.lower())
        if clean_word:
            raw = [p for p in g2p(clean_word) if isinstance(p, str)]
            # Залишаємо фонеми як є, з наголосами
            word_phonemes = [p for p in raw if re.match(r"^[A-Z]+(\d)?$", p)]
            all_phonemes.extend(word_phonemes)
            
            # Додаємо паузу після кожного слова (крім останнього)
            if i < len(words) - 1:
                all_phonemes.append('SP')
    
    return all_phonemes

def extract_hyp_arpabet_from_whisperx(aligned_segments):
    """
    Витягує ARPAbet фонеми з вирівняних сегментів WhisperX.
    Використовує підхід з test_phonem.py - g2p для кожного слова.
    """
    from g2p_en import G2p
    g2p = G2p()
    
    def word_to_arpabet(word: str):
        # g2p_en повертає мікс токенів, заберемо лише фонеми; залишаємо наголоси
        ph = [p for p in g2p(word) if isinstance(p, str)]
        ph = [p for p in ph if re.match(r"^[A-Z]+(\d)?$", p)]
        return ph
    
    hyp = []
    rows = []  # для CSV
    
    for seg in aligned_segments:
        for w in seg.get("words", []):
            word = w.get("word", "")
            w_start = w.get("start", None)
            w_end = w.get("end", None)
            
            # слово без пунктуації в кінці (типу 'planks.')
            clean = re.sub(r"[^\w']+$", "", word)
            if not clean:
                continue
                
            # Отримуємо фонеми через g2p
            phones = word_to_arpabet(clean.lower())
            
            for phone in phones:
                hyp.append(phone)
                rows.append({
                    "word": clean,
                    "word_start": w_start,
                    "word_end": w_end,
                    "phone": phone,
                    "phone_start": w_start,  # Приблизно, оскільки немає точної інформації
                    "phone_end": w_end
                })
            
            # Додаємо паузу між словами
            hyp.append("SP")
            rows.append({
                "word": "<pause>",
                "word_start": w_end,
                "word_end": w_end,
                "phone": "SP",
                "phone_start": w_end,
                "phone_end": w_end
            })
    
    # Прибираємо останню паузу
    while hyp and hyp[-1] in ("SP", "SP2", "SIL"):
        hyp.pop()
        if rows and rows[-1]["phone"] in ("SP", "SP2", "SIL"):
            rows.pop()
    
    return hyp, rows

def levenshtein_ops(ref, hyp):
    """
    Отримати послідовність операцій (match/replace/insert/delete) на токенах.
    Повертає список кортежів (op, ref_tok, hyp_tok).
    """
    # побудуємо DP-таблицю
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # delete
                dp[i][j-1] + 1,      # insert
                dp[i-1][j-1] + cost  # replace or match
            )
    # зворотний прохід
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i>0 and j>0 and dp[i][j] == dp[i-1][j-1] and ref[i-1]==hyp[j-1]:
            ops.append(("match", ref[i-1], hyp[j-1])); i-=1; j-=1
        elif i>0 and j>0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(("replace", ref[i-1], hyp[j-1])); i-=1; j-=1
        elif i>0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(("delete", ref[i-1], "<eps>")); i-=1
        else:
            ops.append(("insert", "<eps>", hyp[j-1])); j-=1
    ops.reverse()
    return ops

def compute_per(ref, hyp):
    """Обчислює Phone Error Rate з урахуванням пауз"""
    ed = lev.distance(" ".join(ref), " ".join(hyp))
    return ed / max(1, len(ref))

def compute_wer(ref_text, hyp_text):
    """
    Обчислює Word Error Rate (WER) між референсним та розпізнаним текстом.
    WER = (S + D + I) / N
    де S = кількість замін, D = кількість видалень, I = кількість вставок, N = кількість слів у reference
    """
    # Нормалізуємо тексти - приводимо до нижнього регістру та прибираємо пунктуацію
    ref_words = re.findall(r'\b\w+\b', ref_text.lower())
    hyp_words = re.findall(r'\b\w+\b', hyp_text.lower())
    
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    
    # Використовуємо Levenshtein distance для слів
    ed = lev.distance(ref_words, hyp_words)
    wer = ed / len(ref_words)
    
    return wer

def detailed_word_comparison(ref_text, hyp_text):
    """
    Детальне порівняння слів з виділенням помилок.
    Повертає список операцій та статистику для WER.
    """
    # Нормалізуємо тексти
    ref_words = re.findall(r'\b\w+\b', ref_text.lower())
    hyp_words = re.findall(r'\b\w+\b', hyp_text.lower())
    
    if not ref_words and not hyp_words:
        return [], {"wer": 0.0, "correct": 0, "substitutions": 0, "insertions": 0, "deletions": 0}
    
    if not ref_words:
        return [], {"wer": 1.0, "correct": 0, "substitutions": 0, "insertions": len(hyp_words), "deletions": 0}
    
    # Отримуємо операції Levenshtein для слів
    ops = levenshtein_ops(ref_words, hyp_words)
    
    # Підраховуємо статистику
    correct = sum(1 for op, _, _ in ops if op == "match")
    substitutions = sum(1 for op, _, _ in ops if op == "replace")
    insertions = sum(1 for op, _, _ in ops if op == "insert")
    deletions = sum(1 for op, _, _ in ops if op == "delete")
    
    wer = (substitutions + insertions + deletions) / len(ref_words)
    
    return ops, {
        "wer": wer,
        "correct": correct,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions
    }

def print_word_alignment(ref_text, hyp_text):
    """
    Друкує вирівняне порівняння слів з виділенням помилок для WER.
    """
    ops, stats = detailed_word_comparison(ref_text, hyp_text)
    
    print("\n" + "="*80)
    print("ДЕТАЛЬНЕ ПОРІВНЯННЯ СЛІВ (WER)")
    print("="*80)
    
    print(f"Статистика: {stats['correct']} збігів, {stats['substitutions']} замін, {stats['insertions']} вставок, {stats['deletions']} видалень")
    print(f"WER (Word Error Rate): {stats['wer']:.3f} ({stats['wer']*100:.1f}%)")
    print(f"Точність слів: {(1-stats['wer'])*100:.1f}%")
    print("-"*80)
    
    if not ops:
        print("Немає даних для порівняння")
        return stats
    
    # Виводимо вирівняння слів
    ref_line = "REF: "
    hyp_line = "HYP: "
    ops_line = "OPS: "
    
    for op, ref_word, hyp_word in ops:
        # Визначаємо ширину колонки
        max_len = max(len(ref_word) if ref_word != "<eps>" else 3, 
                     len(hyp_word) if hyp_word != "<eps>" else 3, 4)
        
        # Форматуємо слова
        ref_str = ref_word.center(max_len) if ref_word != "<eps>" else "---".center(max_len)
        hyp_str = hyp_word.center(max_len) if hyp_word != "<eps>" else "---".center(max_len)
        
        # Символ операції
        op_symbol = {
            "match": "✓",
            "replace": "✗", 
            "insert": "+",
            "delete": "-"
        }[op]
        
        ops_str = op_symbol.center(max_len)
        
        ref_line += ref_str + " "
        hyp_line += hyp_str + " "
        ops_line += ops_str + " "
    
    print(ref_line)
    print(hyp_line) 
    print(ops_line)
    print("-"*80)
    
    # Виводимо детальний список помилок
    errors = [(op, ref_word, hyp_word) for op, ref_word, hyp_word in ops if op != "match"]
    if errors:
        print("ПОМИЛКИ У СЛОВАХ:")
        print("-"*40)
        for i, (op, ref_word, hyp_word) in enumerate(errors, 1):
            if op == "replace":
                print(f"{i}. '{ref_word}' → '{hyp_word}' (заміна)")
            elif op == "insert":
                print(f"{i}. вставлено '{hyp_word}'")
            elif op == "delete":
                print(f"{i}. пропущено '{ref_word}'")
    else:
        print("🎉 Помилок у словах не знайдено! Ідеальне розпізнавання!")
    
    return stats

def save_confusion_matrix(ref, hyp, out_png="phones_confusion.png", include_eps=False, prefix=""):
    """
    Будуємо матрицю лише для замін (replace) і збігів (match).
    Для insert/delete можна додати <eps>, якщо include_eps=True.
    Тепер обидва списки мають паузи між словами.
    """
    ops = levenshtein_ops(ref, hyp)
    pairs = []
    if include_eps:
        for op, r, h in ops:
            if op in ("match", "replace", "insert", "delete"):
                pairs.append((r, h))
    else:
        for op, r, h in ops:
            if op in ("match", "replace") and r != "<eps>" and h != "<eps>":
                pairs.append((r, h))

    if not pairs:
        print("Немає пар для матриці плутанин.")
        return

    y_true = [r for r,h in pairs]
    y_pred = [h for r,h in pairs]
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=None)

    fig, ax = plt.subplots(figsize=(max(6, 0.4*len(labels)), max(5, 0.4*len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=True)
    ax.set_title("Confusion Matrix (phones)")
    
    if prefix:
        filename = f"{prefix}_{out_png}"
    else:
        filename = out_png
    
    output_path = get_output_path(filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Матрицю плутанин збережено у {output_path}")

def save_hyp_phones_csv(rows, out_csv="phones_timing.csv", prefix=""):
    """Зберігає CSV з префіксом назви"""
    if prefix:
        filename = f"{prefix}_{out_csv}"
    else:
        filename = out_csv
    
    output_path = get_output_path(filename)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Таймінги фонем збережено у {output_path}")

def plot_timeline(rows, out_png="phones_timeline.png", max_labels=60, prefix=""):
    """
    Проста візуалізація: кожна фонема — горизонтальний відрізок за часом.
    Якщо фонем дуже багато, підписи обрізаються.
    """
    if not rows:
        print("Немає даних для таймлайну.")
        return
    # відкинемо None
    rows2 = [r for r in rows if r["phone_start"] is not None and r["phone_end"] is not None]
    if not rows2:
        print("Немає валідних таймінгів для таймлайну.")
        return

    # сортування за початком
    rows2.sort(key=lambda r: r["phone_start"])
    y = list(range(len(rows2)))
    starts = [r["phone_start"] for r in rows2]
    durs = [r["phone_end"] - r["phone_start"] for r in rows2]
    labels = [r["phone"] for r in rows2]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.25*len(rows2))))
    ax.barh(y, durs, left=starts)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phones (index)")
    ax.set_title("Phoneme Timeline")

    # підписи (не для всіх, щоб не захаращити)
    step = max(1, math.ceil(len(labels)/max_labels))
    for i in range(0, len(labels), step):
        ax.text(starts[i] + durs[i]/2, y[i], labels[i], va="center", ha="center", fontsize=8)

    if prefix:
        filename = f"{prefix}_{out_png}"
    else:
        filename = out_png
    
    output_path = get_output_path(filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Таймлайн фонем збережено у {output_path}")


def detailed_phoneme_comparison(ref_phones, hyp_phones):
    """
    Детальне порівняння фонем з виділенням помилок.
    Повертає список операцій з позиціями та типами помилок.
    Референсний текст є основою для вирівнювання.
    Тепер обидва списки мають паузи між словами.
    """
    ops = levenshtein_ops(ref_phones, hyp_phones)
    comparison_results = []
    
    ref_pos = 0
    hyp_pos = 0
    
    for op, ref_phone, hyp_phone in ops:
        result = {
            "operation": op,
            "ref_phone": ref_phone,
            "hyp_phone": hyp_phone,
            "ref_position": ref_pos if ref_phone != "<eps>" else None,
            "hyp_position": hyp_pos if hyp_phone != "<eps>" else None,
            "is_error": op != "match"
        }
        
        comparison_results.append(result)
        
        # Оновлюємо позиції
        if ref_phone != "<eps>":
            ref_pos += 1
        if hyp_phone != "<eps>":
            hyp_pos += 1
    
    return comparison_results

def print_phoneme_alignment(ref_phones, hyp_phones):
    """
    Друкує вирівняне порівняння фонем з виділенням помилок.
    """
    comparison = detailed_phoneme_comparison(ref_phones, hyp_phones)
    
    print("\n" + "="*80)
    print("ДЕТАЛЬНЕ ПОРІВНЯННЯ ФОНЕМ")
    print("="*80)
    
    # Підрахунок статистики
    total_ops = len(comparison)
    matches = sum(1 for c in comparison if c["operation"] == "match")
    substitutions = sum(1 for c in comparison if c["operation"] == "replace")
    insertions = sum(1 for c in comparison if c["operation"] == "insert")
    deletions = sum(1 for c in comparison if c["operation"] == "delete")
    
    print(f"Статистика: {matches} збігів, {substitutions} замін, {insertions} вставок, {deletions} видалень")
    print(f"Точність: {matches/total_ops*100:.1f}%")
    print("-"*80)
    
    # Виводимо вирівняння
    ref_line = "REF: "
    hyp_line = "HYP: "
    ops_line = "OPS: "
    pos_line = "POS: "
    
    for i, comp in enumerate(comparison):
        op = comp["operation"]
        ref_phone = comp["ref_phone"]
        hyp_phone = comp["hyp_phone"]
        
        # Визначаємо ширину колонки
        max_len = max(len(ref_phone), len(hyp_phone), 4)
        
        # Форматуємо фонеми
        ref_str = ref_phone.center(max_len) if ref_phone != "<eps>" else "---".center(max_len)
        hyp_str = hyp_phone.center(max_len) if hyp_phone != "<eps>" else "---".center(max_len)
        
        # Символ операції
        op_symbol = {
            "match": "✓",
            "replace": "✗", 
            "insert": "+",
            "delete": "-"
        }[op]
        
        ops_str = op_symbol.center(max_len)
        pos_str = str(i).center(max_len)
        
        ref_line += ref_str + " "
        hyp_line += hyp_str + " "
        ops_line += ops_str + " "
        pos_line += pos_str + " "
    
    print(pos_line)
    print(ref_line)
    print(hyp_line) 
    print(ops_line)
    print("-"*80)
    
    # Виводимо детальний список помилок
    errors = [c for c in comparison if c["is_error"]]
    if errors:
        print("ПОМИЛКИ ВИМОВИ:")
        print("-"*40)
        for i, error in enumerate(errors, 1):
            op = error["operation"]
            if op == "replace":
                print(f"{i}. Позиція {error['ref_position']}: '{error['ref_phone']}' → '{error['hyp_phone']}' (заміна)")
            elif op == "insert":
                print(f"{i}. Позиція {error['hyp_position']}: вставлено '{error['hyp_phone']}'")
            elif op == "delete":
                print(f"{i}. Позиція {error['ref_position']}: пропущено '{error['ref_phone']}'")
    else:
        print("🎉 Помилок не знайдено! Ідеальна вимова!")
    
    # Повертаємо статистику
    total_ops = len(comparison)
    matches = sum(1 for c in comparison if c["operation"] == "match")
    substitutions = sum(1 for c in comparison if c["operation"] == "replace")
    insertions = sum(1 for c in comparison if c["operation"] == "insert")
    deletions = sum(1 for c in comparison if c["operation"] == "delete")
    accuracy = (matches / total_ops * 100) if total_ops > 0 else 0
    
    return {
        "accuracy": accuracy,
        "correct": matches,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions
    }

def plot_phoneme_comparison(ref_phones, hyp_phones, out_png="phoneme_comparison.png", prefix="", ref_text="", hyp_text=""):
    """
    Створює графічну візуалізацію порівняння фонем у форматі IPA та зберігає в каталог out.
    """

    from utilites import get_output_path, arpabet_token_to_ipa
    
    if prefix:
        filename = f"{prefix}_{out_png}"
    else:
        filename = out_png
        
    out_png = get_output_path(filename)  # Зберігаємо в каталог out
    comparison = detailed_phoneme_comparison(ref_phones, hyp_phones)
    
    # Підготовка даних для графіку
    positions = list(range(len(comparison)))
    ref_labels = []
    hyp_labels = []
    colors = []
    
    for comp in comparison:
        # Конвертуємо в IPA для відображення
        ref_ipa = arpabet_token_to_ipa(comp["ref_phone"]) if comp["ref_phone"] != "<eps>" else ""
        hyp_ipa = arpabet_token_to_ipa(comp["hyp_phone"]) if comp["hyp_phone"] != "<eps>" else ""
        ref_labels.append(ref_ipa)
        hyp_labels.append(hyp_ipa)
        
        # Кольори для різних типів операцій
        if comp["operation"] == "match":
            colors.append("lightgreen")
        elif comp["operation"] == "replace":
            colors.append("lightcoral") 
        elif comp["operation"] == "insert":
            colors.append("lightblue")
        else:  # delete
            colors.append("orange")

    # Створення графіку з більшою висотою для тексту
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(15, len(comparison)*0.6), 12))
    
    # Верхній графік - REF фонеми
    bars1 = ax1.bar(positions, [1]*len(positions), color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title("Еталонні фонеми (REF) - IPA Format", fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 2.0)  # Збільшуємо висоту для тексту
    ax1.set_ylabel("REF")
    
    # Додаємо референсний текст над графіком
    if ref_text:
        ax1.text(len(positions)/2, 1.8, f"Референсний текст: {ref_text}", 
                ha='center', va='center', fontsize=16, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # Додаємо підписи REF фонем у IPA
    for i, (pos, label) in enumerate(zip(positions, ref_labels)):
        if label:
            ax1.text(pos, 0.5, label, ha='center', va='center', fontweight='bold', fontsize=16)
        # Додаємо номер позиції
        ax1.text(pos, 1.4, str(i), ha='center', va='center', fontsize=8, alpha=0.7)
    
    # Нижній графік - HYP фонеми  
    bars2 = ax2.bar(positions, [1]*len(positions), color=colors, alpha=0.7, edgecolor='black')
    ax2.set_title("Розпізнані фонеми (HYP) - IPA Format", fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 2.0)  # Збільшуємо висоту для тексту
    ax2.set_ylabel("HYP")
    ax2.set_xlabel("Позиція")
    
    # Додаємо гіпотезний текст над графіком
    if hyp_text:
        ax2.text(len(positions)/2, 1.8, f"Розпізнаний текст: {hyp_text}", 
                ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # Додаємо підписи HYP фонем у IPA
    for i, (pos, label) in enumerate(zip(positions, hyp_labels)):
        if label:
            ax2.text(pos, 0.5, label, ha='center', va='center', fontweight='bold', fontsize=16)
    
    # Легенда
    legend_elements = [
        plt.Rectangle((0,0),1,1, color='lightgreen', alpha=0.7, label='Збіг'),
        plt.Rectangle((0,0),1,1, color='lightcoral', alpha=0.7, label='Заміна'),
        plt.Rectangle((0,0),1,1, color='lightblue', alpha=0.7, label='Вставка'),
        plt.Rectangle((0,0),1,1, color='orange', alpha=0.7, label='Видалення')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Налаштування осей
    max_pos = len(positions)
    step = max(1, max_pos // 20)  # Показуємо кожну 20-ту позицію для великих списків
    
    for ax in [ax1, ax2]:
        ax.set_xlim(-0.5, max_pos-0.5)
        ax.set_xticks(range(0, max_pos, step))
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Графік порівняння фонем (IPA) збережено у: {out_png}")

def analyze_error_patterns(ref_phones, hyp_phones):
    """
    Аналізує паттерни помилок у вимові.
    """
    comparison = detailed_phoneme_comparison(ref_phones, hyp_phones)
    
    # Збираємо статистику помилок
    substitution_patterns = {}
    common_insertions = {}
    common_deletions = {}
    
    for comp in comparison:
        if comp["operation"] == "replace":
            pattern = f"{comp['ref_phone']} → {comp['hyp_phone']}"
            substitution_patterns[pattern] = substitution_patterns.get(pattern, 0) + 1
        elif comp["operation"] == "insert":
            phone = comp["hyp_phone"]
            common_insertions[phone] = common_insertions.get(phone, 0) + 1
        elif comp["operation"] == "delete":
            phone = comp["ref_phone"]
            common_deletions[phone] = common_deletions.get(phone, 0) + 1
    
    print("\n" + "="*60)
    print("АНАЛІЗ ПАТТЕРНІВ ПОМИЛОК")
    print("="*60)
    
    if substitution_patterns:
        print("Найчастіші заміни:")
        for pattern, count in sorted(substitution_patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern} (x{count})")
    
    if common_insertions:
        print("\nНайчастіші зайві звуки:")
        for phone, count in sorted(common_insertions.items(), key=lambda x: x[1], reverse=True):
            print(f"  +{phone} (x{count})")
    
    if common_deletions:
        print("\nНайчастіші пропущені звуки:")
        for phone, count in sorted(common_deletions.items(), key=lambda x: x[1], reverse=True):
            print(f"  -{phone} (x{count})")
    
    return {
        "substitutions": substitution_patterns,
        "insertions": common_insertions,
        "deletions": common_deletions
    }


# ------------------------ Основний сценарій ------------------------
def main():
    # Читаємо референсний текст
    target_text = read_reference_text(REFERENCE_TEXT_FILE)
    if not target_text:
        print(f"❌ Використовуємо стандартний текст замість файлу {REFERENCE_TEXT_FILE}")
        target_text = TARGET_TEXT
    else:
        print(f"✅ Референсний текст завантажено з {REFERENCE_TEXT_FILE}")
    
    print(f'🎯 Фраза для вимови: "{target_text}"')
    
    # Відтворюємо референсний аудіо файл
    print("\n🔊 Спочатку послухайте референсний зразок:")
    if not play_reference_audio(REFERENCE_AUDIO):
        print("⚠️ Не вдалося відтворити референсний аудіо, продовжуємо без нього...")
    
    # Пауза між референсним аудіо та записом
    print("\n⏳ Готуйтесь до запису через 2 секунди...")
    time.sleep(2)
    
    # Подаємо звуковий сигнал перед записом
    play_beep_signal(frequency=800, duration=0.3, count=3)
    
    print("🎤 Запис...")
    audio = sd.rec(int(SECONDS * SR), samplerate=SR, channels=1, dtype='int16')
    sd.wait()
    write(AUDIO_OUT, SR, audio)
    print(f"✅ Аудіо збережено: {AUDIO_OUT}")

    # 1) Еталонні фонеми
    ref_phones = g2p_arpabet(target_text)
    ref_ipa = arpabet_seq_to_ipa(ref_phones)
    print(f"📝 Еталонні фонеми (ARPAbet): {ref_phones}")
    print(f"📝 Еталонні фонеми (IPA): {ref_ipa}")

    # 2) ASR + Align
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Завантаження WhisperX ({MODEL_NAME}) на {device}...")
    
    asr_model = safe_load_whisperx_model(MODEL_NAME, device)
    res = asr_model.transcribe(AUDIO_OUT)
    
    # Правильна обробка результату ASR
    asr_text = " ".join(seg["text"].strip() for seg in res["segments"])
    print(f"🗣️ ASR розпізнав: {asr_text}")

    # Використаємо align-модель (англійська)
    align_model, metadata = whisperx.load_align_model(language_code=LANG_CODE, device=device)
    aligned = whisperx.align(res["segments"], align_model, metadata, AUDIO_OUT, device=device)

    # 3) Витяг гіпотезних фонем + CSV
    hyp_phones, phone_rows = extract_hyp_arpabet_from_whisperx(aligned.get("segments", []))
    hyp_ipa = arpabet_seq_to_ipa(hyp_phones)
    print(f"🎯 Розпізнані фонеми (ARPAbet): {hyp_phones}")
    print(f"🎯 Розпізнані фонеми (IPA): {hyp_ipa}")

    save_hyp_phones_csv(phone_rows, "phones_timing.csv", prefix=REFERENCE_NAME)
    plot_timeline(phone_rows, "phones_timeline.png", prefix=REFERENCE_NAME)

    # 4) PER + детальний аналіз
    per = compute_per(ref_phones, hyp_phones)
    print(f"\n📊 PER (Phone Error Rate) = {per:.3f} (0.0 = ідеально)")

    # 5) WER (Word Error Rate) + детальний аналіз слів
    wer = compute_wer(target_text, asr_text)
    print(f"📊 WER (Word Error Rate) = {wer:.3f} (0.0 = ідеально)")
    
    # Детальне порівняння слів
    word_comparison_results = print_word_alignment(target_text, asr_text)

    # Детальне порівняння фонем з IPA відображенням
    comparison_results = print_phoneme_alignment(ref_phones, hyp_phones)
    analyze_error_patterns(ref_phones, hyp_phones)
    plot_phoneme_comparison(ref_phones, hyp_phones, "phoneme_comparison.png", prefix=REFERENCE_NAME, 
                          ref_text=target_text, hyp_text=asr_text)

    # Матриця плутанин
    save_confusion_matrix(ref_phones, hyp_phones, out_png="phones_confusion.png", include_eps=False, prefix=REFERENCE_NAME)

    # 6) Збережемо результати у JSON
    json_path = get_output_path(f"{REFERENCE_NAME}_phones_ref_hyp.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "reference_file": REFERENCE_NAME,
            "target_text": target_text,
            "asr_text": asr_text,
            "ref_phones_arpabet": ref_phones,
            "ref_phones_ipa": ref_ipa,
            "hyp_phones_arpabet": hyp_phones,
            "hyp_phones_ipa": hyp_ipa,
            "per": per,
            "wer": wer,
            "phoneme_accuracy": comparison_results["accuracy"],
            "word_accuracy": (1 - word_comparison_results["wer"]) * 100,
            "phoneme_errors": {
                "substitutions": comparison_results["substitutions"],
                "insertions": comparison_results["insertions"],
                "deletions": comparison_results["deletions"]
            },
            "word_errors": {
                "correct": word_comparison_results["correct"],
                "substitutions": word_comparison_results["substitutions"],
                "insertions": word_comparison_results["insertions"],
                "deletions": word_comparison_results["deletions"]
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"💾 Результати збережено у: {json_path}")
    
    print(f"\n📁 Всі файли збережено в каталозі: {os.path.abspath('out')}")

if __name__ == "__main__":
    main()

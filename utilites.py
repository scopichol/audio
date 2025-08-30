import os
import re

# Каталог для збереження вихідних файлів
OUTPUT_DIR = "out"

def ensure_output_dir():
    """Забезпечує існування каталогу для вихідних файлів"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    return OUTPUT_DIR

def get_output_path(filename):
    """Повертає повний шлях до файлу в каталозі out"""
    ensure_output_dir()
    return os.path.join(OUTPUT_DIR, filename)

def play_beep_signal(frequency=1000, duration=0.5, count=3):
    """
    Подає звуковий сигнал перед записом.
    
    Args:
        frequency: частота сигналу в Гц
        duration: тривалість одного сигналу в секундах
        count: кількість сигналів
    """
    try:
        import numpy as np
        import sounddevice as sd
        import time
        
        print("🔊 Готуйтесь до запису...")
        
        for i in range(count):
            print(f"Сигнал {i+1}/{count}")
            
            # Генеруємо синусоїду
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            signal = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            # Відтворюємо сигнал
            sd.play(signal, sample_rate)
            sd.wait()  # Чекаємо завершення відтворення
            
            if i < count - 1:
                time.sleep(0.3)  # Пауза між сигналами
        
        print("🎤 ЗАПИС ПОЧАВСЯ!")
        
    except Exception as e:
        print(f"Не вдалося відтворити звуковий сигнал: {e}")
        print("📢 Готуйтесь до запису (без звуку)...")

def read_reference_text(audio_path):
    """
    Читає референсний текст для аудіо-файлу.
    Шукає файл з таким же ім'ям, але з розширенням .txt.
    """
    txt_path = os.path.splitext(audio_path)[0] + ".txt"
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

ARPABET_TO_IPA = {
    # Голосні
    "AA": "ɑ",  "AE": "æ",  "AH": "ʌ",  "AO": "ɔ",  "AW": "aʊ",
    "AY": "aɪ", "EH": "ɛ",  "ER": "ɝ",  "EY": "eɪ", "IH": "ɪ",
    "IY": "i",  "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ",  "UW": "u",
    # Приголосні
    "P": "p", "B": "b", "T": "t", "D": "d", "K": "k", "G": "ɡ",
    "CH": "tʃ", "JH": "dʒ",
    "F": "f", "V": "v", "TH": "θ", "DH": "ð",
    "S": "s", "Z": "z", "SH": "ʃ", "ZH": "ʒ",
    "HH": "h",
    "M": "m", "N": "n", "NG": "ŋ",
    "L": "l", "R": "ɹ", "Y": "j", "W": "w",
    # Слогові / r-colored варіанти (залежно від стилю можна змінити):
    "AX": "ə",    # schwa
    "EL": "l̩",   # слогове l
    "EM": "m̩",   # слогове m
    "EN": "n̩",   # слогове n
    "ER0": "ɚ",  # безнаголосне r-colored
    "ER1": "ɝ",  # наголошене r-colored (синон. ER)
    "ER2": "ɝ",
    # Паузи та спеціальні символи
    "SP": "‿",    # коротка пауза між словами
    "SP2": "|",   # середня пауза (кома)
    "SIL": "||",  # довга пауза/тиша (крапка, знак оклику тощо)
    " ": " ",     # пробіл
}

VOWELS = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW","AX"}

# Коди пауз, що використовуються у ARPAbet
PAUSE_CODES = {"SP", "SP2", "SIL"}

# Символи для пауз (короткі коди ARPAbet)
PAUSE_SYMBOLS = {
    "SP": "‿",               # коротка пауза між словами
    "SP2": "|",              # середня пауза (кома)
    "SIL": "||",             # довга пауза/тиша
    "WORD_BOUNDARY": " ",    # межа слова (звичайний пробіл)
}

def arpabet_token_to_ipa(tok: str) -> str:
    """
    Перетворює один ARPAbet токен (наприклад 'IY1' або 'DH') у IPA.
    Додає ˈ / ˌ перед голосною з наголосом 1/2.
    Також обробляє спеціальні символи пауз (SP, SP2, SIL).
    """
    # Обробка спеціальних символів пауз
    if tok in ["SP", "SP2", "SIL"]:
        return ARPABET_TO_IPA[tok]
    
    # Обробка пробілу
    if tok == " ":
        return " "
    
    m = re.fullmatch(r"([A-Z]+)([0-2])?", tok)
    if not m:
        return tok  # лишаємо як є, якщо не впізнали
    base, stress = m.group(1), m.group(2)

    # Спеціальні коди ER0/1/2 уже прописані як ключі
    if base == "ER" and stress is not None:
        ipa = ARPABET_TO_IPA.get(base + stress, ARPABET_TO_IPA.get(base, "ɝ"))
        # ER0/1/2 вже містять відповідну r-colored голосну; наголос не додаємо окремо
        return ipa

    ipa = ARPABET_TO_IPA.get(base, base)  # якщо не знайшли — повернемо base
    if stress and base in VOWELS:
        if stress == "1":
            return "ˈ" + ipa
        elif stress == "2":
            return "ˌ" + ipa
    return ipa

def arpabet_seq_to_ipa(seq):
    """seq: список фонем ARPAbet (наприклад ['DH','AH0','B','ER1','CH',...])"""
    return " ".join(arpabet_token_to_ipa(t) for t in seq)

def add_pause_to_phonemes(phonemes, pause_duration="SP"):
    """
    Додає паузи до списку фонем.
    pause_duration: тип паузи (SP, SP2, SIL)
    """
    if isinstance(phonemes, list):
        return phonemes + [pause_duration]
    return phonemes

def add_word_boundaries(text_with_phonemes):
    """
    Додає межі слів між фонемами.
    Приймає текст або список фонем і додає WORD_BOUNDARY між словами.
    """
    if isinstance(text_with_phonemes, str):
        # Якщо це строка, розділяємо слова пробілами та додаємо межі
        words = text_with_phonemes.split()
        return " WORD_BOUNDARY ".join(words)
    elif isinstance(text_with_phonemes, list):
        # Якщо це список фонем, шукаємо натуральні паузи та додаємо межі
        result = []
        for i, phoneme in enumerate(text_with_phonemes):
            result.append(phoneme)
            # Додаємо межу слова після певних фонем (можна налаштувати логіку)
            if i < len(text_with_phonemes) - 1:
                result.append("WORD_BOUNDARY")
        return result
    return text_with_phonemes

def create_pause_sequence(pause_type, count=1):
    """
    Створює послідовність пауз заданого типу.
    pause_type: тип паузи
    count: кількість пауз поспіль
    """
    return [pause_type] * count

def phonemes_with_pauses_to_ipa(phonemes_with_pauses):
    """
    Перетворює список фонем з паузами у IPA транскрипцію.
    Обробляє як звичайні фонеми, так і спеціальні символи пауз.
    """
    ipa_symbols = []
    for token in phonemes_with_pauses:
        ipa_symbol = arpabet_token_to_ipa(token)
        ipa_symbols.append(ipa_symbol)
    
    # Об'єднуємо з урахуванням особливостей пауз
    result = []
    for symbol in ipa_symbols:
        if symbol in PAUSE_SYMBOLS.values():
            result.append(symbol)
        else:
            result.append(symbol)
    
    return "".join(result)

def customize_pause_symbols(short="‿", medium="|", long="||", word_boundary=" "):
    """
    Дозволяє налаштувати символи для різних типів пауз.
    """
    global PAUSE_SYMBOLS
    PAUSE_SYMBOLS = {
        "SP": short,
        "SP2": medium,
        "SIL": long,
        "WORD_BOUNDARY": word_boundary,
    }

def format_ipa_with_pauses(ipa_text, show_pauses=True, compact=False):
    """
    Форматує IPA текст з паузами для кращого візуального відображення.
    
    Args:
        ipa_text: IPA текст з символами пауз
        show_pauses: чи показувати паузи (False приховує всі паузи)
        compact: компактний режим (без пробілів між фонемами)
    
    Returns:
        Відформатований текст
    """
    if not show_pauses:
        # Видаляємо всі символи пауз
        for pause_symbol in PAUSE_SYMBOLS.values():
            ipa_text = ipa_text.replace(pause_symbol, "")
    
    if compact:
        # Видаляємо зайві пробіли
        ipa_text = re.sub(r'\s+', '', ipa_text)
    
    return ipa_text

def analyze_pause_structure(phonemes_with_pauses):
    """
    Аналізує структуру пауз у списку фонем.
    
    Returns:
        dict з інформацією про паузи
    """
    pause_counts = {pause_type: 0 for pause_type in PAUSE_SYMBOLS.keys()}
    total_phonemes = len(phonemes_with_pauses)
    pause_positions = []
    
    for i, phoneme in enumerate(phonemes_with_pauses):
        if phoneme in PAUSE_SYMBOLS:
            pause_counts[phoneme] += 1
            pause_positions.append((i, phoneme))
    
    total_pauses = sum(pause_counts.values())
    
    return {
        "pause_counts": pause_counts,
        "total_pauses": total_pauses,
        "total_phonemes": total_phonemes,
        "pause_ratio": total_pauses / total_phonemes if total_phonemes > 0 else 0,
        "pause_positions": pause_positions
    }

def insert_automatic_pauses(phonemes, pause_interval=5, pause_type="SP"):
    """
    Автоматично вставляє паузи через певні інтервали.
    
    Args:
        phonemes: список фонем
        pause_interval: інтервал між паузами (кількість фонем)
        pause_type: тип паузи для вставки (SP, SP2, SIL)
    
    Returns:
        список фонем з автоматично вставленими паузами
    """
    result = []
    for i, phoneme in enumerate(phonemes):
        result.append(phoneme)
        # Додаємо паузу кожні pause_interval фонем (крім останньої)
        if (i + 1) % pause_interval == 0 and i < len(phonemes) - 1:
            result.append(pause_type)
    
    return result

def display_pauses_with_symbols(phonemes, style="brackets"):
    """
    Відображає паузи з символами в різних стилях.
    
    Args:
        phonemes: список фонем з паузами
        style: стиль відображення ("brackets", "unicode", "ascii", "verbose")
    
    Returns:
        рядок з відформатованими паузами
    """
    if style == "brackets":
        pause_map = {"SP": "[SP]", "SP2": "[SP2]", "SIL": "[SIL]"}
    elif style == "unicode":
        pause_map = {"SP": "‿", "SP2": "|", "SIL": "||"}
    elif style == "ascii":
        pause_map = {"SP": "_", "SP2": "-", "SIL": "="}
    elif style == "verbose":
        pause_map = {"SP": "<короткаПауза>", "SP2": "<середняПауза>", "SIL": "<довгаПауза>"}
    else:
        pause_map = {"SP": "‿", "SP2": "|", "SIL": "||"}
    
    result = []
    for phoneme in phonemes:
        if phoneme in pause_map:
            result.append(pause_map[phoneme])
        else:
            result.append(phoneme)
    
    return " ".join(result)

def analyze_pause_distribution(phonemes):
    """
    Аналізує розподіл пауз у списку фонем.
    
    Args:
        phonemes: список фонем з паузами
    
    Returns:
        dict з інформацією про розподіл пауз
    """
    pause_counts = {"SP": 0, "SP2": 0, "SIL": 0}
    total_phonemes = len(phonemes)
    
    for phoneme in phonemes:
        if phoneme in pause_counts:
            pause_counts[phoneme] += 1
    
    total_pauses = sum(pause_counts.values())
    
    return {
        "total_phonemes": total_phonemes,
        "total_pauses": total_pauses,
        "pause_distribution": {k: v for k, v in pause_counts.items() if v > 0},
        "pause_ratio": total_pauses / total_phonemes if total_phonemes > 0 else 0
    }

def convert_pauses_to_timing(phonemes, base_duration=0.1):
    """
    Перетворює фонеми з паузами у часові мітки.
    
    Args:
        phonemes: список фонем з паузами
        base_duration: базова тривалість фонеми в секундах
    
    Returns:
        список словників з часовими мітками
    """
    timing = []
    current_time = 0.0
    
    # Тривалості для різних типів звуків
    durations = {
        "SP": 0.15,
        "SP2": 0.30, 
        "SIL": 0.50,
        "WORD_BOUNDARY": 0.05,
    }
    
    for phoneme in phonemes:
        if phoneme in durations:
            duration = durations[phoneme]
            sound_type = "pause"
        else:
            duration = base_duration
            sound_type = "phoneme"
        
        timing.append({
            "phoneme": phoneme,
            "start": current_time,
            "end": current_time + duration,
            "duration": duration,
            "type": sound_type
        })
        
        current_time += duration
    
    return timing

# G2P функції
def normalize_arpabet(arpabet_list):
    """Нормалізує ARPAbet послідовність, видаляючи цифри наголосу"""
    import re
    cleaned = []
    for phone in arpabet_list:
        if phone in PAUSE_CODES:
            cleaned.append(phone)
        else:
            # Видаляємо цифри наголосу (0, 1, 2)
            clean_phone = re.sub(r'[012]$', '', phone)
            if clean_phone:  # Якщо залишилось щось після очищення
                cleaned.append(clean_phone)
    return cleaned

def g2p_arpabet(text):
    """Конвертує текст в ARPAbet фонеми"""
    from g2p_en import G2p
    g2p = G2p()
    raw = [p for p in g2p(text) if isinstance(p, str)]
    return normalize_arpabet(raw)

# Аналіз помилок (PER)
def compute_per(ref_phones, hyp_phones):
    """Обчислює Phone Error Rate між еталонними та гіпотезними фонемами"""
    from editdistance import eval as edit_distance
    dist = edit_distance(ref_phones, hyp_phones)
    if not ref_phones:
        return 0.0 if not hyp_phones else float('inf')
    return dist / len(ref_phones)

# Матриця плутанин
def save_confusion_matrix(ref_phones, hyp_phones, out_png="confusion.png", include_eps=False):
    """Зберігає матрицю плутанин фонем у каталог out"""
    from collections import defaultdict, Counter
    import matplotlib.pyplot as plt
    import numpy as np
    from editdistance import eval as edit_distance
    
    out_png = get_output_path(out_png)  # Зберігаємо у каталог out
    
    if not ref_phones or not hyp_phones:
        print("Немає пар для матриці плутанин.")
        return
    
    # Конвертуємо ARPAbet у IPA для відображення
    ref_ipa = [arpabet_token_to_ipa(phone) for phone in ref_phones]
    hyp_ipa = [arpabet_token_to_ipa(phone) for phone in hyp_phones]
    
    # Простий підхід: порівнюємо позиції один до одного
    pairs = []
    min_len = min(len(ref_ipa), len(hyp_ipa))
    
    for i in range(min_len):
        pairs.append((ref_ipa[i], hyp_ipa[i]))
    
    if not pairs:
        print("Немає пар для матриці плутанин.")
        return
    
    # Підрахунок частот
    conf_matrix = defaultdict(int)
    for ref, hyp in pairs:
        conf_matrix[(ref, hyp)] += 1
    
    # Унікальні фонеми в IPA
    all_phones = sorted(set(ref_ipa + hyp_ipa))
    if include_eps:
        all_phones = ["<eps>"] + all_phones
    
    n = len(all_phones)
    matrix = np.zeros((n, n))
    
    phone_to_idx = {phone: i for i, phone in enumerate(all_phones)}
    
    for (ref, hyp), count in conf_matrix.items():
        if ref in phone_to_idx and hyp in phone_to_idx:
            i, j = phone_to_idx[ref], phone_to_idx[hyp]
            matrix[i, j] = count
    
    # Візуалізація
    plt.figure(figsize=(12, 10))
    plt.imshow(matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix (REF → HYP) - IPA Format')
    plt.xlabel('Hypothesis (IPA)')
    plt.ylabel('Reference (IPA)')
    
    plt.xticks(range(n), all_phones, rotation=45)
    plt.yticks(range(n), all_phones)
    
    # Додаємо числа в клітинки
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0:
                plt.text(j, i, int(matrix[i, j]), ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Матрицю плутанин збережено у: {out_png}")

# CSV та Timeline функції
def save_hyp_phones_csv(phone_rows, csv_path):
    """Зберігає таймінги фонем у CSV файл в каталог out"""
    import csv
    
    csv_path = get_output_path(csv_path)  # Зберігаємо у каталог out
    
    if not phone_rows:
        print("Таймінги фонем збережено у", csv_path)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["phone_arpabet", "phone_ipa", "start", "end"])
        return
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["phone_arpabet", "phone_ipa", "start", "end"])
        for row_group in phone_rows:
            for row in row_group:
                if len(row) >= 3:
                    phone_arpabet = row[0]
                    phone_ipa = arpabet_token_to_ipa(phone_arpabet)
                    start = row[1]
                    end = row[2]
                    writer.writerow([phone_arpabet, phone_ipa, start, end])
    print("Таймінги фонем збережено у", csv_path)

def plot_timeline(phone_rows, png_path):
    """Створює графік таймлайну фонем у каталог out"""
    import matplotlib.pyplot as plt
    
    png_path = get_output_path(png_path)  # Зберігаємо у каталог out
    
    if not phone_rows:
        print("Немає даних для таймлайну.")
        return
    
    # Розгортаємо всі рядки
    all_phones = []
    for row_group in phone_rows:
        all_phones.extend(row_group)
    
    if not all_phones:
        print("Немає даних для таймлайну.")
        return
    
    phones_arpabet = [row[0] for row in all_phones]
    phones_ipa = [arpabet_token_to_ipa(phone) for phone in phones_arpabet]
    starts = [float(row[1]) for row in all_phones]
    ends = [float(row[2]) for row in all_phones]
    
    plt.figure(figsize=(14, 6))
    
    for i, (phone_ipa, start, end) in enumerate(zip(phones_ipa, starts, ends)):
        plt.barh(i, end - start, left=start, height=0.8, 
                alpha=0.7, edgecolor='black')
        plt.text((start + end) / 2, i, phone_ipa, 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('Час (с)')
    plt.ylabel('Фонеми (IPA)')
    plt.title('Таймлайн фонем - IPA Format')
    plt.yticks(range(len(phones_ipa)), phones_ipa)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Таймлайн збережено у: {png_path}")

# Приклад:
# arpabet = ["DH","AH0","B","ER1","CH","K","AH0","N","UW1","S","L","IH1","D","AA1","N",
#            "DH","AH0","S","M","UW1","DH","P","L","AE1","NG","K","S"]
# print(arpabet_seq_to_ipa(arpabet))
# → ð ə b ˈɝ tʃ k ə n ˈu s l ˈɪ d ˈɑ n ð ə s m ˈu ð p l ˈæ ŋ k s

# Приклад з паузами:
# arpabet_with_pauses = ["DH","AH0","B","ER1","CH","SP","K","AH0","N","UW1","S",
#                        "SP2","L","IH1","D","AA1","N","SIL"]
# print(phonemes_with_pauses_to_ipa(arpabet_with_pauses))
# → ð ə b ˈɝ tʃ ‿ k ə n ˈu s | l ˈɪ d ˈɑ n ||

if __name__ == "__main__":
    # Тестування з паузами
    arpabet = ["DH", "IH", "S", "SP", "IH", "Z", "SP2", "EY", "SP", "T", "EH", "S", "T", "SIL"]
    print("ARPAbet:", arpabet)
    print("IPA:", arpabet_seq_to_ipa(arpabet))
    
    # Демонстрація різних стилів відображення пауз
    print("\nРізні стили відображення пауз:")
    print("Unicode style:", display_pauses_with_symbols(arpabet, "unicode"))
    print("ASCII style:", display_pauses_with_symbols(arpabet, "ascii"))
    print("Brackets style:", display_pauses_with_symbols(arpabet, "brackets"))
    print("Verbose style:", display_pauses_with_symbols(arpabet, "verbose"))
    
    # Аналіз пауз
    stats = analyze_pause_distribution(arpabet)
    print("\nАналіз пауз:")
    print(f"Всього фонем: {stats['total_phonemes']}")
    print(f"Всього пауз: {stats['total_pauses']}")
    print(f"Співвідношення пауз до фонем: {stats['pause_ratio']:.2f}")
    print("Розподіл пауз:", stats['pause_distribution'])
    
    # Часові мітки
    timing = convert_pauses_to_timing(arpabet)
    print("\nЧасові мітки (перші 5):")
    for item in timing[:5]:
        print(f"{item['phoneme']}: {item['start']:.2f}s - {item['end']:.2f}s ({item['type']})")
    
    # Пошук файлу посилання
    ref_file = "output01.wav"
    ref_text = read_reference_text(ref_file)
    if ref_text:
        print(f"\nТекст посилання для {ref_file}: {ref_text}")
    else:
        print(f"\nНе знайдено текст посилання для {ref_file}")

def save_phoneme_comparison_json(data, output_file):
    """Зберігає дані порівняння фонем у JSON"""
    import json
    ensure_output_dir()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"📄 JSON дані збережено в: {output_file}")

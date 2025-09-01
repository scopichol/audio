#!/usr/bin/env python3
"""
Демонстрація детального порівняння фонем для аналізу помилок вимови
"""

import os
import torch
import whisperx
from g2p_en import G2p
from utilites import (
    g2p_arpabet, compute_per, save_confusion_matrix,
    save_hyp_phones_csv, plot_timeline
)
from test_phonem2 import (
    safe_load_whisperx_model, extract_hyp_arpabet_from_whisperx,
    detailed_phoneme_comparison, print_phoneme_alignment,
    plot_phoneme_comparison, analyze_error_patterns
)

# Налаштування
MODEL_NAME = "small.en"
LANG_CODE = "en"
AUDIO_FILE = "OSR_us_000_0010_8k.wav"  # Використаємо існуючий файл з мовленням
TARGET_TEXT = "The birch canoe slid on the smooth planks"  # Перше речення з файлу

def main():
    print("="*80)
    print("ДЕМОНСТРАЦІЯ АНАЛІЗУ ПОМИЛОК ВИМОВИ")
    print("="*80)
    print(f"Цільовий текст: '{TARGET_TEXT}'")
    print(f"Аудіо файл: {AUDIO_FILE}")
    print()

    # 1) Еталонні фонеми
    ref_phones = g2p_arpabet(TARGET_TEXT)
    print("ЕТАЛОННІ ФОНЕМИ:")
    print("", ref_phones)
    print()

    # 2) ASR + Align
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Завантаження WhisperX ({MODEL_NAME}) на {device}...")
    
    asr_model = safe_load_whisperx_model(MODEL_NAME, device)
    res = asr_model.transcribe(AUDIO_FILE)
    
    # Правильна обробка результату ASR
    asr_text = " ".join(seg["text"].strip() for seg in res["segments"])
    print(f"ASR розпізнав: '{asr_text}'")
    print()

    # Використаємо align-модель (англійська)
    align_model, metadata = whisperx.load_align_model(language_code=LANG_CODE, device=device)
    aligned = whisperx.align(res["segments"], align_model, metadata, AUDIO_FILE, device=device)

    # 3) Витяг гіпотезних фонем
    hyp_phones, phone_rows = extract_hyp_arpabet_from_whisperx(aligned.get("segments", []))
    print("РОЗПІЗНАНІ ФОНЕМИ:")
    print("", hyp_phones)
    print()

    # 4) PER
    per = compute_per(ref_phones, hyp_phones)
    print(f"PER (Phone Error Rate) = {per:.3f}")
    print("(0.0 = ідеально, чим менше - тим краще)")
    print()

    # 5) ДЕТАЛЬНИЙ АНАЛІЗ ПОМИЛОК
    print("="*80)
    print("ДЕТАЛЬНИЙ АНАЛІЗ ПОМИЛОК ВИМОВИ")
    print("="*80)
    
    # Детальне порівняння
    comparison_result = detailed_phoneme_comparison(ref_phones, hyp_phones)
    
    # Підрахунок статистики
    total_ops = len(comparison_result)
    matches = sum(1 for c in comparison_result if c["operation"] == "match")
    substitutions = sum(1 for c in comparison_result if c["operation"] == "replace")
    insertions = sum(1 for c in comparison_result if c["operation"] == "insert")
    deletions = sum(1 for c in comparison_result if c["operation"] == "delete")
    
    accuracy = (matches / total_ops * 100) if total_ops > 0 else 0
    
    # Візуальне вирівнювання
    print_phoneme_alignment(ref_phones, hyp_phones)
    
    # Аналіз паттернів помилок
    error_patterns = analyze_error_patterns(ref_phones, hyp_phones)
    
    # Графічна візуалізація
    plot_phoneme_comparison(ref_phones, hyp_phones, "demo_phoneme_comparison.png")
    print(f"\nГрафік порівняння збережено у: demo_phoneme_comparison.png")
    
    # Матриця плутанин
    save_confusion_matrix(ref_phones, hyp_phones, out_png="demo_confusion_matrix.png", include_eps=False)
    print(f"Матрицю плутанин збережено у: demo_confusion_matrix.png")
    
    print()
    print("="*80)
    print("РЕЗЮМЕ АНАЛІЗУ")
    print("="*80)
    print(f"Загальна точність: {accuracy:.1f}%")
    print(f"Правильних звуків: {matches}")
    print(f"Замін: {substitutions}")
    print(f"Вставок: {insertions}")
    print(f"Видалень: {deletions}")
    
    if error_patterns['substitutions']:
        print(f"\nНайчастіші помилки заміни:")
        for pattern, count in list(error_patterns['substitutions'].items())[:3]:
            print(f"  {pattern} (x{count})")

if __name__ == "__main__":
    main()

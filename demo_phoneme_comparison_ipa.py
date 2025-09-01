#!/usr/bin/env python3
"""
Демонстрація детального порівняння фонем для аналізу помилок вимови (IPA Format)
"""

import os
import torch
import whisperx
from utilites import (
    g2p_arpabet, compute_per, save_confusion_matrix,
    save_hyp_phones_csv, plot_timeline, get_output_path,
    arpabet_seq_to_ipa
)
from test_phonem2 import (
    safe_load_whisperx_model, extract_hyp_arpabet_from_whisperx,
    print_phoneme_alignment, plot_phoneme_comparison, analyze_error_patterns
)

# Налаштування
MODEL_NAME = "small.en"
LANG_CODE = "en"
AUDIO_FILE = "OSR_us_000_0010_8k.wav"
TARGET_TEXT = "The birch canoe slid on the smooth planks"

def main():
    print("="*80)
    print("ДЕМОНСТРАЦІЯ АНАЛІЗУ ПОМИЛОК ВИМОВИ (IPA FORMAT)")
    print("="*80)
    print(f"Цільовий текст: '{TARGET_TEXT}'")
    print(f"Аудіо файл: {AUDIO_FILE}")
    print()

    # 1) Еталонні фонеми
    ref_phones = g2p_arpabet(TARGET_TEXT)
    ref_ipa = arpabet_seq_to_ipa(ref_phones)
    print("ЕТАЛОННІ ФОНЕМИ:")
    print(f"ARPAbet: {ref_phones}")
    print(f"IPA: {ref_ipa}")
    print()

    # 2) ASR + Align
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Завантаження WhisperX ({MODEL_NAME}) на {device}...")
    
    asr_model = safe_load_whisperx_model(MODEL_NAME, device)
    res = asr_model.transcribe(AUDIO_FILE)
    
    asr_text = " ".join(seg["text"].strip() for seg in res["segments"])
    print(f"ASR розпізнав: '{asr_text}'")
    print()

    # Використаємо align-модель (англійська)
    align_model, metadata = whisperx.load_align_model(language_code=LANG_CODE, device=device)
    aligned = whisperx.align(res["segments"], align_model, metadata, AUDIO_FILE, device=device)

    # 3) Витяг гіпотезних фонем
    hyp_phones, phone_rows = extract_hyp_arpabet_from_whisperx(aligned.get("segments", []))
    hyp_ipa = arpabet_seq_to_ipa(hyp_phones)
    print("РОЗПІЗНАНІ ФОНЕМИ:")
    print(f"ARPAbet: {hyp_phones}")
    print(f"IPA: {hyp_ipa}")
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
    
    # Детальне порівняння з IPA відображенням
    comparison_result = print_phoneme_alignment(ref_phones, hyp_phones)
    
    # Аналіз паттернів помилок
    error_patterns = analyze_error_patterns(ref_phones, hyp_phones)
    
    # Графічна візуалізація (зберігаємо в каталог out)
    plot_phoneme_comparison(ref_phones, hyp_phones, "demo_phoneme_comparison.png")
    
    # Матриця плутанин (зберігаємо в каталог out)
    save_confusion_matrix(ref_phones, hyp_phones, out_png="demo_confusion_matrix.png", include_eps=False)
    
    print()
    print("="*80)
    print("РЕЗЮМЕ АНАЛІЗУ")
    print("="*80)
    print(f"Загальна точність: {comparison_result['accuracy']:.1f}%")
    print(f"Правильних звуків: {comparison_result['correct']}")
    print(f"Замін: {comparison_result['substitutions']}")
    print(f"Вставок: {comparison_result['insertions']}")
    print(f"Видалень: {comparison_result['deletions']}")
    
    if error_patterns['substitutions']:
        print(f"\nНайчастіші помилки заміни:")
        for pattern, count in list(error_patterns['substitutions'].items())[:3]:
            print(f"  {pattern} (x{count})")
    
    print(f"\n📁 Всі файли збережено в каталозі: {os.path.abspath('out')}")

if __name__ == "__main__":
    main()

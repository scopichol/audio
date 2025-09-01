#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест основної функціональності test_phonem2.py без запису аудіо
"""

from test_phonem2 import (
    g2p_arpabet, 
    safe_load_whisperx_model, 
    pick_compute_type,
    TARGET_TEXT,
    MODEL_NAME
)
import torch
import os

def test_basic_functionality():
    """Тестуємо базову функціональність без аудіо."""
    
    print("=== Тест базової функціональності ===")
    
    # 1) Тест g2p
    print("1. Тестування g2p_arpabet...")
    ref_phones = g2p_arpabet(TARGET_TEXT)
    print(f"Текст: {TARGET_TEXT}")
    print(f"REF phones: {ref_phones}")
    print()
    
    # 2) Тест визначення пристрою та compute_type
    print("2. Тестування визначення пристрою...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = pick_compute_type(device)
    print(f"Device: {device}")
    print(f"Compute type: {compute_type}")
    print()
    
    # 3) Тест завантаження моделі (якщо є інтернет)
    print("3. Тестування завантаження моделі WhisperX...")
    try:
        asr_model = safe_load_whisperx_model(MODEL_NAME, device)
        print("✅ Модель WhisperX завантажена успішно!")
        
        # Перевіримо, чи є тестовий аудіо файл
        test_audio = "output01.wav"  # або інший існуючий файл
        if os.path.exists(test_audio):
            print(f"4. Тестування транскрипції з файлом {test_audio}...")
            try:
                res = asr_model.transcribe(test_audio)
                print(f"ASR result: {res.get('text', '').strip()}")
                print("✅ Транскрипція працює!")
            except Exception as e:
                print(f"❌ Помилка транскрипції: {e}")
        else:
            print(f"4. Файл {test_audio} не знайдено, пропускаємо тест транскрипції")
            
    except Exception as e:
        print(f"❌ Помилка завантаження моделі: {e}")
        return False
    
    print("\n✅ Всі базові тести пройдені успішно!")
    return True

if __name__ == "__main__":
    test_basic_functionality()

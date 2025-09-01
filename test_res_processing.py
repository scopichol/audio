#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест оновленої обробки словника res у test_phonem2.py
"""

import json
from test_phonem2 import (
    g2p_arpabet, 
    extract_hyp_arpabet_from_whisperx,
    TARGET_TEXT
)

def test_res_processing():
    """Тестуємо правильну обробку результату ASR."""
    
    print("=== Тест обробки результату ASR ===")
    
    # Моделюємо результат WhisperX
    mock_res = {
        "segments": [
            {
                "text": " The quick brown",
                "words": [
                    {"word": "The", "start": 0.0, "end": 0.3},
                    {"word": "quick", "start": 0.4, "end": 0.7},
                    {"word": "brown", "start": 0.8, "end": 1.1}
                ]
            },
            {
                "text": " fox jumps over",
                "words": [
                    {"word": "fox", "start": 1.2, "end": 1.5},
                    {"word": "jumps", "start": 1.6, "end": 1.9},
                    {"word": "over", "start": 2.0, "end": 2.3}
                ]
            }
        ]
    }
    
    # Тестуємо правильну обробку тексту (як у test_phonem.py)
    asr_text = " ".join(seg["text"].strip() for seg in mock_res["segments"])
    print(f"ASR text (правильний спосіб): '{asr_text}'")
    
    # Тестуємо неправильний спосіб (старий)
    wrong_text = mock_res.get("text", "")
    print(f"ASR text (неправильний спосіб): '{wrong_text}'")
    
    # Тестуємо витяг фонем
    mock_aligned = {
        "segments": mock_res["segments"]
    }
    
    hyp_phones, phone_rows = extract_hyp_arpabet_from_whisperx(mock_aligned.get("segments", []))
    print(f"HYP phones: {hyp_phones}")
    print(f"Phone rows count: {len(phone_rows)}")
    
    # Тестуємо еталонні фонеми
    ref_phones = g2p_arpabet("The quick brown fox jumps over")
    print(f"REF phones: {ref_phones}")
    
    # Порівняння
    print(f"\nПорівняння:")
    print(f"REF довжина: {len(ref_phones)}")
    print(f"HYP довжина: {len(hyp_phones)}")
    
    # Моделюємо збереження у JSON
    test_data = {
        "ref": ref_phones,
        "hyp": hyp_phones, 
        "text": asr_text
    }
    
    print(f"\nТест JSON збереження:")
    print(json.dumps(test_data, ensure_ascii=False, indent=2))
    
    print("\n✅ Тест обробки результату завершено!")

if __name__ == "__main__":
    test_res_processing()

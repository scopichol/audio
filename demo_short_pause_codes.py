#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Демонстрація використання коротких кодів пауз (SP, SP2, SIL) 
для роботи з фонетичним аналізом, як у test_phonem.py
"""

from utilites import (
    arpabet_seq_to_ipa, 
    phonemes_with_pauses_to_ipa,
    display_pauses_with_symbols,
    analyze_pause_distribution,
    convert_pauses_to_timing
)

def demo_short_pause_codes():
    """Демонстрація роботи з короткими кодами пауз SP, SP2, SIL."""
    
    print("=== Демонстрація коротких кодів пауз ===")
    print("SP  - коротка пауза між словами: ‿")
    print("SP2 - середня пауза (кома): |") 
    print("SIL - довга пауза/тиша (крапка): ||")
    print()
    
    # Приклад з реального тексту, як у test_phonem.py
    target_text = "The quick brown fox jumps over the lazy dog."
    
    # Моделюємо результат g2p з паузами (як у test_phonem.py)
    arpabet_ref = [
        # "The"
        "DH", "AH0", "SP",
        # "quick" 
        "K", "W", "IH1", "K", "SP",
        # "brown"
        "B", "R", "AW1", "N", "SP", 
        # "fox"
        "F", "AA1", "K", "S", "SP",
        # "jumps"
        "JH", "AH1", "M", "P", "S", "SP",
        # "over"
        "OW1", "V", "ER0", "SP",
        # "the"
        "DH", "AH0", "SP",
        # "lazy"
        "L", "EY1", "Z", "IY0", "SP",
        # "dog"
        "D", "AO1", "G", "SIL"  # Кінець речення
    ]
    
    print("REF phones:", arpabet_ref)
    print("REF IPA:", arpabet_seq_to_ipa(arpabet_ref))
    print()
    
    # Різні стилі відображення пауз
    print("Різні стили відображення пауз:")
    print("Unicode style:", display_pauses_with_symbols(arpabet_ref, "unicode"))
    print("ASCII style:", display_pauses_with_symbols(arpabet_ref, "ascii"))
    print("Brackets style:", display_pauses_with_symbols(arpabet_ref, "brackets"))
    print("Verbose style:", display_pauses_with_symbols(arpabet_ref, "verbose"))
    print()
    
    # Аналіз пауз
    stats = analyze_pause_distribution(arpabet_ref)
    print("Аналіз пауз:")
    print(f"Всього фонем: {stats['total_phonemes']}")
    print(f"Всього пауз: {stats['total_pauses']}")
    print(f"Співвідношення пауз до фонем: {stats['pause_ratio']:.2f}")
    print("Розподіл пауз:", stats['pause_distribution'])
    print()
    
    # Часові мітки
    timing = convert_pauses_to_timing(arpabet_ref)
    print("Часові мітки (перші 10):")
    for item in timing[:10]:
        print(f"{item['phoneme']}: {item['start']:.2f}s - {item['end']:.2f}s ({item['type']})")
    print("...")
    total_duration = timing[-1]['end'] if timing else 0
    print(f"Загальна тривалість: {total_duration:.2f}s")
    print()

def demo_hypothesis_vs_reference():
    """Демонстрація порівняння гіпотези та еталону з паузами."""
    
    print("=== Порівняння REF та HYP з паузами ===")
    
    # Еталон (REF)
    arpabet_ref = [
        "HH", "EH1", "L", "OW0", "SP",      # Hello
        "W", "ER1", "L", "D", "SIL"         # world.
    ]
    
    # Гіпотеза (HYP) - трохи інша з іншими паузами
    arpabet_hyp = [
        "HH", "EH1", "L", "OW0", "SP2",     # Hello (середня пауза замість короткої)
        "W", "ER1", "L", "D"                # world (без паузи в кінці)
    ]
    
    print("REF phones:", arpabet_ref)
    print("REF IPA:", arpabet_seq_to_ipa(arpabet_ref))
    print("REF with pause symbols:", display_pauses_with_symbols(arpabet_ref, "brackets"))
    print()
    
    print("HYP phones:", arpabet_hyp)
    print("HYP IPA:", arpabet_seq_to_ipa(arpabet_hyp))
    print("HYP with pause symbols:", display_pauses_with_symbols(arpabet_hyp, "brackets"))
    print()
    
    # Аналіз пауз для обох
    ref_stats = analyze_pause_distribution(arpabet_ref)
    hyp_stats = analyze_pause_distribution(arpabet_hyp)
    
    print("Порівняння статистики пауз:")
    print(f"REF: {ref_stats['total_pauses']} пауз, розподіл: {ref_stats['pause_distribution']}")
    print(f"HYP: {hyp_stats['total_pauses']} пауз, розподіл: {hyp_stats['pause_distribution']}")
    print()

def demo_punctuation_to_pauses():
    """Демонстрація перетворення пунктуації у паузи."""
    
    print("=== Перетворення пунктуації у паузи ===")
    
    # Правила перетворення пунктуації у паузи (як у test_phonem.py)
    punctuation_to_pause = {
        ".": "SIL",     # крапка → довга пауза
        "!": "SIL",     # знак оклику → довга пауза
        "?": "SIL",     # знак питання → довга пауза
        ";": "SIL",     # крапка з комою → довга пауза
        ":": "SIL",     # двокрапка → довга пауза
        ",": "SP2",     # кома → середня пауза
        " ": "SP",      # пробіл → коротка пауза
    }
    
    print("Правила перетворення:")
    for punct, pause in punctuation_to_pause.items():
        symbol = {"SP": "‿", "SP2": "|", "SIL": "||"}[pause]
        print(f"'{punct}' → {pause} ({symbol})")
    print()
    
    # Приклад тексту з різною пунктуацією
    sample_texts = [
        "Hello world!",
        "How are you?",
        "I'm fine, thank you.",
        "Yes; no: maybe.",
    ]
    
    for text in sample_texts:
        print(f"Текст: '{text}'")
        
        # Моделюємо процес додавання пауз
        phonemes_with_pauses = []
        words = text.split()
        
        for i, word in enumerate(words):
            # Видаляємо пунктуацію з слова
            clean_word = ''.join(c for c in word if c.isalpha())
            punct = ''.join(c for c in word if not c.isalpha())
            
            # Додаємо фонеми слова (спрощено)
            if clean_word.lower() == "hello":
                phonemes_with_pauses.extend(["HH", "EH1", "L", "OW0"])
            elif clean_word.lower() == "world":
                phonemes_with_pauses.extend(["W", "ER1", "L", "D"])
            elif clean_word.lower() == "how":
                phonemes_with_pauses.extend(["HH", "AW1"])
            elif clean_word.lower() == "are":
                phonemes_with_pauses.extend(["AA1", "R"])
            elif clean_word.lower() == "you":
                phonemes_with_pauses.extend(["Y", "UW1"])
            else:
                # Заглушка для інших слів
                phonemes_with_pauses.extend(["W", "ER1", "D"])
            
            # Додаємо паузу за пунктуацією
            if punct:
                for p in punct:
                    if p in punctuation_to_pause:
                        phonemes_with_pauses.append(punctuation_to_pause[p])
            elif i < len(words) - 1:  # Не остання літера
                phonemes_with_pauses.append("SP")  # Пробіл між словами
        
        print(f"Фонеми: {phonemes_with_pauses}")
        print(f"IPA: {arpabet_seq_to_ipa(phonemes_with_pauses)}")
        print(f"З символами: {display_pauses_with_symbols(phonemes_with_pauses, 'brackets')}")
        print()

if __name__ == "__main__":
    demo_short_pause_codes()
    demo_hypothesis_vs_reference()
    demo_punctuation_to_pauses()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Приклади використання функцій для відображення пауз в IPA транскрипції.
"""

from utilites import (
    arpabet_seq_to_ipa, 
    phonemes_with_pauses_to_ipa,
    add_pause_to_phonemes,
    create_pause_sequence,
    format_ipa_with_pauses,
    analyze_pause_structure,
    insert_automatic_pauses,
    customize_pause_symbols,
    PAUSE_SYMBOLS
)

def demo_basic_pauses():
    """Демонстрація базової роботи з паузами."""
    print("=== Базова робота з паузами ===")
    
    # Звичайні фонеми
    phonemes = ["DH", "AH0", "B", "ER1", "CH", "K", "AH0", "N", "UW1", "S"]
    print(f"Звичайна транскрипція: {arpabet_seq_to_ipa(phonemes)}")
    
    # Додаємо паузи вручну
    phonemes_with_pauses = [
        "DH", "AH0", "B", "ER1", "CH", "SP",      # the birch
        "K", "AH0", "N", "UW1", "S", "SP2",       # canoe
        "L", "IH1", "D", "AA1", "N", "SIL"        # led on
    ]
    
    ipa_with_pauses = phonemes_with_pauses_to_ipa(phonemes_with_pauses)
    print(f"З паузами: {ipa_with_pauses}")
    
    print()

def demo_pause_types():
    """Демонстрація різних типів пауз."""
    print("=== Типи пауз ===")
    
    for pause_type, symbol in PAUSE_SYMBOLS.items():
        print(f"{pause_type}: '{symbol}'")
    
    # Приклад використання різних пауз
    text_with_different_pauses = [
        "HH", "EH1", "L", "OW0", "WORD_BOUNDARY",     # Hello
        "W", "ER1", "L", "D", "SP",                   # world (коротка пауза)
        "AY1", "M", "SP2",                            # I'm (середня пауза)
        "F", "AY1", "N", "SIL",                       # fine (довга пауза)
        "TH", "AE1", "NG", "K", "Y", "UW1"            # thank you
    ]
    
    result = phonemes_with_pauses_to_ipa(text_with_different_pauses)
    print(f"Змішаний приклад: {result}")
    print()

def demo_formatting_options():
    """Демонстрація різних опцій форматування."""
    print("=== Опції форматування ===")
    
    phonemes = ["HH", "EH1", "L", "OW0", "SP", "W", "ER1", "L", "D", "SP2"]
    ipa_text = phonemes_with_pauses_to_ipa(phonemes)
    
    print(f"Звичайний вигляд: {ipa_text}")
    print(f"Без пауз: {format_ipa_with_pauses(ipa_text, show_pauses=False)}")
    print(f"Компактний з паузами: {format_ipa_with_pauses(ipa_text, compact=True)}")
    print(f"Компактний без пауз: {format_ipa_with_pauses(ipa_text, show_pauses=False, compact=True)}")
    print()

def demo_pause_analysis():
    """Демонстрація аналізу структури пауз."""
    print("=== Аналіз структури пауз ===")
    
    phonemes = [
        "DH", "IH1", "S", "SP",
        "IH1", "Z", "WORD_BOUNDARY", "AH0", "SP2",
        "T", "EH1", "S", "T", "SIL",
        "S", "EH1", "N", "T", "AH0", "N", "S"
    ]
    
    analysis = analyze_pause_structure(phonemes)
    
    print(f"Загальна кількість фонем: {analysis['total_phonemes']}")
    print(f"Загальна кількість пауз: {analysis['total_pauses']}")
    print(f"Відношення пауз до фонем: {analysis['pause_ratio']:.2%}")
    print("Підрахунок по типам пауз:")
    for pause_type, count in analysis['pause_counts'].items():
        if count > 0:
            print(f"  {pause_type}: {count}")
    print()

def demo_automatic_pauses():
    """Демонстрація автоматичного додавання пауз."""
    print("=== Автоматичне додавання пауз ===")
    
    phonemes = ["HH", "EH1", "L", "OW0", "W", "ER1", "L", "D", "AY1", "M", "F", "AY1", "N"]
    
    print(f"Оригінал: {arpabet_seq_to_ipa(phonemes)}")
    
    # Автоматичні паузи кожні 3 фонеми
    with_auto_pauses = insert_automatic_pauses(phonemes, pause_interval=3, pause_type="SP")
    print(f"З автопаузами (кожні 3): {phonemes_with_pauses_to_ipa(with_auto_pauses)}")
    
    # Автоматичні паузи кожні 5 фонем із середніми паузами
    with_auto_pauses_medium = insert_automatic_pauses(phonemes, pause_interval=5, pause_type="SP2")
    print(f"З автопаузами (кожні 5): {phonemes_with_pauses_to_ipa(with_auto_pauses_medium)}")
    print()

def demo_custom_symbols():
    """Демонстрація налаштування власних символів пауз."""
    print("=== Налаштування власних символів ===")
    
    phonemes = ["T", "EH1", "S", "T", "SP", "W", "IH1", "TH", "SP2", "P", "AO1", "Z"]
    
    print("За замовчуванням:")
    print(f"  {phonemes_with_pauses_to_ipa(phonemes)}")
    
    # Змінюємо символи
    customize_pause_symbols(short="~", medium="--", long="===")
    print("З власними символами:")
    print(f"  {phonemes_with_pauses_to_ipa(phonemes)}")
    
    # Повертаємо стандартні символи
    customize_pause_symbols()
    print("Повернуто до стандартних:")
    print(f"  {phonemes_with_pauses_to_ipa(phonemes)}")
    print()

def demo_practical_example():
    """Практичний приклад використання."""
    print("=== Практичний приклад ===")
    
    # Моделюємо реальну ситуацію з розпізнаванням мовлення
    sentence_phonemes = [
        # "Hello"
        "HH", "EH1", "L", "OW0", "WORD_BOUNDARY",
        # "my" (коротка пауза)
        "M", "AY1", "SP",
        # "name" (середня пауза)
        "N", "EY1", "M", "SP2",
        # "is" (коротка пауза)
        "IH1", "Z", "SP",
        # "John" (довга пауза в кінці речення)
        "JH", "AA1", "N", "SIL"
    ]
    
    ipa_result = phonemes_with_pauses_to_ipa(sentence_phonemes)
    print(f"Результат: {ipa_result}")
    
    # Аналіз структури
    analysis = analyze_pause_structure(sentence_phonemes)
    print(f"Статистика: {analysis['total_pauses']} пауз з {analysis['total_phonemes']} елементів")
    
    # Різні формати відображення
    print("Формати відображення:")
    print(f"  Повний: {ipa_result}")
    print(f"  Без пауз: {format_ipa_with_pauses(ipa_result, show_pauses=False)}")
    print(f"  Компактний: {format_ipa_with_pauses(ipa_result, compact=True)}")

if __name__ == "__main__":
    demo_basic_pauses()
    demo_pause_types()
    demo_formatting_options()
    demo_pause_analysis()
    demo_automatic_pauses()
    demo_custom_symbols()
    demo_practical_example()

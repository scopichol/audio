#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Финальная демонстрация системы анализа произношения
- Предзаписной сигнал перед записью
- Отображение фонем в IPA формате
- Сохранение всех файлов в каталог 'out'
"""

from utilites import (
    ensure_output_dir, get_output_path, 
    arpabet_token_to_ipa, arpabet_seq_to_ipa,
    play_beep_signal, save_phoneme_comparison_json
)
import os

def demo_ipa_conversion():
    """Демонстрация конвертации ARPAbet в IPA"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ КОНВЕРТАЦИИ ARPAbet → IPA")
    print("=" * 60)
    
    # Примеры ARPAbet фонем
    arpabet_examples = [
        ['HH', 'EH1', 'L', 'OW0'],  # hello
        ['W', 'ER1', 'L', 'D'],     # world
        ['TH', 'IH1', 'S'],         # this
        ['IH1', 'Z'],               # is
        ['AH0'],                    # a
        ['T', 'EH1', 'S', 'T']      # test
    ]
    
    words = ['hello', 'world', 'this', 'is', 'a', 'test']
    
    for word, arpabet in zip(words, arpabet_examples):
        ipa = arpabet_seq_to_ipa(arpabet)
        print(f"{word:10} | ARPAbet: {' '.join(arpabet):15} | IPA: {ipa}")
    
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ ОБРАБОТКИ ПАУЗ")
    print("=" * 60)
    
    # Примеры с паузами
    pause_examples = [
        (['HH', 'EH1', 'L', 'OW0', 'SP', 'W', 'ER1', 'L', 'D'], 'hello [pause] world'),
        (['TH', 'IH1', 'S', 'SP2', 'IH1', 'Z'], 'this [short pause] is'),
        (['SIL', 'AH0', 'SP', 'T', 'EH1', 'S', 'T', 'SIL'], '[silence] a [pause] test [silence]')
    ]
    
    for arpabet, description in pause_examples:
        ipa = arpabet_seq_to_ipa(arpabet)
        print(f"{description:25} | ARPAbet: {' '.join(arpabet)}")
        print(f"{' ' * 25} | IPA:     {ipa}")
        print()

def demo_output_directory():
    """Демонстрация работы с каталогом вывода"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ С КАТАЛОГОМ 'OUT'")
    print("=" * 60)
    
    # Убеждаемся, что каталог существует
    ensure_output_dir()
    print(f"✅ Каталог 'out' подготовлен")
    
    # Создаем демонстрационные файлы
    demo_files = [
        'demo_recording.wav',
        'demo_analysis.json', 
        'demo_comparison.png',
        'demo_timeline.csv'
    ]
    
    for filename in demo_files:
        filepath = get_output_path(filename)
        print(f"📁 Путь к файлу: {filepath}")
    
    # Сохраняем демонстрационные данные
    demo_data = {
        "reference_phonemes": ["HH", "EH1", "L", "OW0"],
        "hypothesis_phonemes": ["HH", "AH1", "L", "OW0"],
        "reference_ipa": arpabet_seq_to_ipa(["HH", "EH1", "L", "OW0"]),
        "hypothesis_ipa": arpabet_seq_to_ipa(["HH", "AH1", "L", "OW0"]),
        "alignment": [
            {"ref": "HH", "hyp": "HH", "match": True},
            {"ref": "EH1", "hyp": "AH1", "match": False},
            {"ref": "L", "hyp": "L", "match": True},
            {"ref": "OW0", "hyp": "OW0", "match": True}
        ]
    }
    
    output_file = get_output_path('demo_phoneme_data.json')
    save_phoneme_comparison_json(demo_data, output_file)
    print(f"✅ Демонстрационные данные сохранены в: {output_file}")

def demo_audio_signal():
    """Демонстрация аудиосигнала"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПРЕДЗАПИСНОГО АУДИОСИГНАЛА")
    print("=" * 60)
    
    print("🔊 Сейчас прозвучит демонстрационный сигнал (3 гудка)...")
    print("   Обычно это звучит перед началом записи")
    
    try:
        play_beep_signal()
        print("✅ Аудиосигнал успешно воспроизведен!")
    except Exception as e:
        print(f"❌ Ошибка воспроизведения: {e}")
        print("   (Возможно, нет подключенных аудиоустройств)")

def main():
    """Главная демонстрация"""
    print("🎯 ФИНАЛЬНАЯ ДЕМОНСТРАЦИЯ СИСТЕМЫ АНАЛИЗА ПРОИЗНОШЕНИЯ")
    print("   Все новые возможности:")
    print("   ✅ Предзаписной аудиосигнал")
    print("   ✅ Отображение фонем в IPA формате") 
    print("   ✅ Сохранение всех файлов в каталог 'out'")
    print()
    
    # Демонстрация IPA конвертации
    demo_ipa_conversion()
    print()
    
    # Демонстрация каталога вывода
    demo_output_directory()
    print()
    
    # Демонстрация аудиосигнала
    demo_audio_signal()
    
    print("\n" + "=" * 60)
    print("🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
    print("   Система готова к использованию:")
    print("   - Запустите test_phonem2.py для анализа произношения")
    print("   - Все результаты будут в каталоге 'out'")
    print("   - Фонемы отображаются в удобном IPA формате")
    print("=" * 60)

if __name__ == '__main__':
    main()

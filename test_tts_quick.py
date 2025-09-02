"""
Швидкий тест TTS з текстом з sentence_12.txt
"""
import os
import sys
sys.path.append('.')

from test_tts_comparison import test_gtts, test_pyttsx3, test_coqui_tts, test_bark_tts, test_espeak, print_comparison_table, play_audio_file

def quick_tts_test():
    """Швидкий тест TTS з поточним текстом"""
    print("🚀 ШВИДКИЙ ТЕСТ TTS")
    print("="*30)
    
    # Читаємо текст з sentence_12.txt
    text_file = "sentences/sentence_13.txt"
    if os.path.exists(text_file):
        with open(text_file, 'r', encoding='utf-8') as f:
            test_text = f.read().strip()
        print(f"📝 Текст з {text_file}: '{test_text}'")
    else:
        test_text = "Hey, whatcha up to?"
        print(f"📝 Стандартний текст: '{test_text}'")
    
    # Створюємо директорію для результатів
    output_dir = "tts_quick_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Швидкий тест доступних систем
    print("\n🔄 Перевірка доступних TTS систем...")
    
    results = {}
    
    # Тест gTTS
    try:
        import gtts
        import pydub
        print("✅ gTTS доступний")
        result = test_gtts(test_text, f"{output_dir}/gtts_quick.wav")
        results["gTTS"] = result
    except ImportError:
        print("❌ gTTS недоступний")
        results["gTTS"] = {"success": False, "error": "Не встановлено", "duration": 0, "file_size": 0, "features": {}}
    
    # Тест pyttsx3
    try:
        import pyttsx3
        print("✅ pyttsx3 доступний")
        result = test_pyttsx3(test_text, f"{output_dir}/pyttsx3_quick.wav")
        results["pyttsx3"] = result
    except ImportError:
        print("❌ pyttsx3 недоступний")
        results["pyttsx3"] = {"success": False, "error": "Не встановлено", "duration": 0, "file_size": 0, "features": {}}
    
    # Тест Coqui TTS
    try:
        from TTS.api import TTS
        print("✅ Coqui TTS доступний")
        result = test_coqui_tts(test_text, f"{output_dir}/coqui_quick.wav")
        results["Coqui TTS"] = result
    except ImportError:
        print("❌ Coqui TTS недоступний")
        results["Coqui TTS"] = {"success": False, "error": "Не встановлено", "duration": 0, "file_size": 0, "features": {}}
    
    # Тест Bark TTS
    try:
        from bark import generate_audio
        print("✅ Bark TTS доступний")
        result = test_bark_tts(test_text, f"{output_dir}/bark_quick.wav")
        results["Bark TTS"] = result
    except ImportError:
        print("❌ Bark TTS недоступний")
        results["Bark TTS"] = {"success": False, "error": "Не встановлено", "duration": 0, "file_size": 0, "features": {}}
    
    # Тест espeak
    try:
        import subprocess
        subprocess.run(["espeak", "--version"], capture_output=True, check=True)
        print("✅ espeak доступний")
        result = test_espeak(test_text, f"{output_dir}/espeak_quick.wav")
        results["espeak"] = result
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ espeak недоступний")
        results["espeak"] = {"success": False, "error": "Не встановлено", "duration": 0, "file_size": 0, "features": {}}
    
    # Показуємо результати
    print_comparison_table(results)
    
    # Рекомендація для основної програми
    successful = [name for name, result in results.items() if result['success']]
    
    print(f"\n🎯 РЕКОМЕНДАЦІЇ ДЛЯ ВИКОРИСТАННЯ:")
    print("-" * 40)
    
    if successful:
        print(f"✅ Працюючі системи: {', '.join(successful)}")
        
        # Порядок пріоритету
        if "Bark TTS" in successful:
            print("🎖️ Рекомендовано: Bark TTS (найкращі емоції та інтонації)")
        elif "Coqui TTS" in successful:
            print("🥇 Рекомендовано: Coqui TTS (найкраща AI якість)")
        elif "gTTS" in successful:
            print("🥈 Рекомендовано: gTTS (висока якість)")
        elif "pyttsx3" in successful:
            print("🔄 Рекомендовано: pyttsx3 (офлайн)")
        elif "espeak" in successful:
            print("🔧 Рекомендовано: espeak (базовий)")
        
        # Можливість прослухати
        print(f"\n🎧 Прослухайте результати в папці: {output_dir}")
        
        # Автопрослуховування найкращого
        best_file = None
        if "Bark TTS" in successful:
            best_file = f"{output_dir}/bark_quick.wav"
        elif "Coqui TTS" in successful:
            best_file = f"{output_dir}/coqui_quick.wav"
        elif "gTTS" in successful:
            best_file = f"{output_dir}/gtts_quick.wav"
        elif "pyttsx3" in successful:
            best_file = f"{output_dir}/pyttsx3_quick.wav"
        elif "espeak" in successful:
            best_file = f"{output_dir}/espeak_quick.wav"
        
        if best_file and os.path.exists(best_file):
            try:
                input(f"📱 Натисніть Enter для прослуховування найкращого варіанту...")
                play_audio_file(best_file)
            except KeyboardInterrupt:
                print("⚠️ Прослуховування пропущено")
    else:
        print("❌ Жодна TTS система не працює!")
        print("💡 Встановіть:")
        print("   pip install TTS                    # Coqui TTS (найкраща якість)")
        print("   pip install gtts pydub             # Google TTS")
        print("   pip install pyttsx3                # Offline TTS")
    
    print("✅ Швидкий тест завершено!")

if __name__ == "__main__":
    quick_tts_test()

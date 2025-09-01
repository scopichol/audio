"""
Спрощений тест для діагностики системи
"""
import os
import sys

def test_basic_imports():
    """Тестує базові імпорти"""
    print("🔍 Тестування базових імпортів...")
    
    try:
        import numpy as np
        print("✅ NumPy:", np.__version__)
    except Exception as e:
        print("❌ NumPy:", e)
        return False
    
    try:
        import torch
        print("✅ PyTorch:", torch.__version__)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Пристрій: {device}")
    except Exception as e:
        print("❌ PyTorch:", e)
        return False
    
    try:
        import whisperx
        print("✅ WhisperX: OK")
    except Exception as e:
        print("❌ WhisperX:", e)
        return False
    
    try:
        from g2p_en import G2p
        print("✅ G2P English: OK")
    except Exception as e:
        print("❌ G2P English:", e)
        return False
    
    return True

def test_tts_systems():
    """Тестує TTS системи"""
    print("\n🔍 Тестування TTS систем...")
    
    available = []
    
    # gTTS
    try:
        import gtts
        import pydub
        print("✅ gTTS + pydub: OK")
        available.append("gTTS")
    except Exception as e:
        print("❌ gTTS:", e)
    
    # pyttsx3
    try:
        import pyttsx3
        print("✅ pyttsx3: OK")
        available.append("pyttsx3")
    except Exception as e:
        print("❌ pyttsx3:", e)
    
    print(f"📢 Доступні TTS системи: {available}")
    return available

def test_g2p():
    """Тестує G2P конвертацію"""
    print("\n🔍 Тестування G2P конвертації...")
    
    try:
        from g2p_en import G2p
        g2p = G2p()
        
        test_word = "hello"
        phones = g2p(test_word)
        print(f"✅ G2P тест '{test_word}' → {phones}")
        return True
    except Exception as e:
        print("❌ G2P тест:", e)
        return False

def test_file_structure():
    """Тестує структуру файлів"""
    print("\n🔍 Тестування структури файлів...")
    
    # Перевіряємо директорії
    dirs_to_check = ["sentences", "out"]
    for dir_name in dirs_to_check:
        if os.path.exists(dir_name):
            print(f"✅ Директорія {dir_name}: існує")
        else:
            print(f"⚠️ Директорія {dir_name}: відсутня, створюємо...")
            try:
                os.makedirs(dir_name, exist_ok=True)
                print(f"✅ Директорія {dir_name}: створена")
            except Exception as e:
                print(f"❌ Помилка створення {dir_name}: {e}")
    
    # Перевіряємо ключові файли
    files_to_check = [
        "sentences/sentence_1.txt",
        "sentences/sentence_1.wav"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ Файл {file_path}: існує")
        else:
            print(f"⚠️ Файл {file_path}: відсутній")

def test_simple_tts():
    """Простий тест TTS генерації"""
    print("\n🔍 Тестування простої TTS генерації...")
    
    test_text = "Hello world"
    test_file = "sentences/test_tts.wav"
    
    # Спробуємо gTTS
    try:
        from gtts import gTTS
        from pydub import AudioSegment
        import tempfile
        
        print("🔄 Тест gTTS...")
        tts = gTTS(text=test_text, lang="en", slow=False)
        
        # Зберігаємо як MP3, потім конвертуємо в WAV
        temp_mp3 = tempfile.mktemp(suffix='.mp3')
        tts.save(temp_mp3)
        
        # Конвертуємо MP3 в WAV
        audio = AudioSegment.from_mp3(temp_mp3)
        audio.export(test_file, format="wav")
        
        # Видаляємо тимчасовий файл
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
        
        print(f"✅ gTTS тест: {test_file} створено")
        return True
        
    except Exception as e:
        print(f"❌ gTTS тест: {e}")
    
    # Спробуємо pyttsx3
    try:
        import pyttsx3
        
        print("🔄 Тест pyttsx3...")
        engine = pyttsx3.init()
        engine.save_to_file(test_text, test_file)
        engine.runAndWait()
        
        print(f"✅ pyttsx3 тест: {test_file} створено")
        return True
        
    except Exception as e:
        print(f"❌ pyttsx3 тест: {e}")
    
    return False

def main():
    """Основна функція тестування"""
    print("🚀 Запуск спрощеного тесту системи")
    print("="*50)
    
    # Тестуємо імпорти
    if not test_basic_imports():
        print("\n❌ Критичні помилки з імпортами. Зупинка.")
        return
    
    # Тестуємо TTS
    available_tts = test_tts_systems()
    if not available_tts:
        print("\n⚠️ Жодна TTS система не доступна")
    
    # Тестуємо G2P
    if not test_g2p():
        print("\n❌ Проблеми з G2P")
        return
    
    # Тестуємо файлову структуру
    test_file_structure()
    
    # Тестуємо TTS генерацію
    if available_tts:
        test_simple_tts()
    
    print("\n✅ Спрощений тест завершено!")
    print("💡 Якщо всі тести пройшли, можна запускати основну програму")

if __name__ == "__main__":
    main()

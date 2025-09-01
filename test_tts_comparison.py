"""
Тест для порівняння різних TTS систем
Створює аудіо файли з одним текстом через різні TTS движки та дозволяє їх порівняти
"""
import os
import time
import tempfile
import subprocess
from pathlib import Path

def test_gtts(text, output_file, language="en"):
    """Тестує Google Text-to-Speech"""
    try:
        from gtts import gTTS
        from pydub import AudioSegment
        
        print("🔄 Тестування gTTS (Google Text-to-Speech)...")
        start_time = time.time()
        
        # Створюємо TTS об'єкт
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Зберігаємо як MP3, потім конвертуємо в WAV
        temp_mp3 = tempfile.mktemp(suffix='.mp3')
        tts.save(temp_mp3)
        
        # Конвертуємо MP3 в WAV
        audio = AudioSegment.from_mp3(temp_mp3)
        audio.export(output_file, format="wav")
        
        # Видаляємо тимчасовий файл
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
        
        duration = time.time() - start_time
        file_size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
        
        return {
            "success": True,
            "duration": duration,
            "file_size": file_size,
            "error": None,
            "features": {
                "quality": "Висока (Google Neural)",
                "internet_required": True,
                "languages": "Багато мов",
                "voice_options": "Обмежені",
                "speed_control": "Так (slow параметр)"
            }
        }
        
    except ImportError as e:
        return {
            "success": False,
            "duration": 0,
            "file_size": 0,
            "error": f"Не встановлено: {e}",
            "features": {}
        }
    except Exception as e:
        return {
            "success": False,
            "duration": 0,
            "file_size": 0,
            "error": str(e),
            "features": {}
        }

def test_pyttsx3(text, output_file, language="en"):
    """Тестує pyttsx3 (офлайн TTS)"""
    try:
        import pyttsx3
        
        print("🔄 Тестування pyttsx3 (Offline TTS)...")
        start_time = time.time()
        
        # Ініціалізуємо движок
        engine = pyttsx3.init()
        
        # Налаштування голосу
        voices = engine.getProperty('voices')
        english_voice = None
        
        for voice in voices:
            if language == "en" and ("english" in voice.name.lower() or "en" in voice.id.lower()):
                english_voice = voice
                break
        
        if english_voice:
            engine.setProperty('voice', english_voice.id)
        
        # Налаштування швидкості та гучності
        engine.setProperty('rate', 150)  # слів за хвилину
        engine.setProperty('volume', 0.9)  # гучність 0-1
        
        # Створюємо директорію якщо не існує
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Зберігаємо аудіо
        engine.save_to_file(text, output_file)
        engine.runAndWait()
        
        duration = time.time() - start_time
        file_size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
        
        # Отримуємо інформацію про доступні голоси
        voice_count = len(voices) if voices else 0
        available_voices = [v.name for v in voices[:3]] if voices else []  # Перші 3 голоси
        
        return {
            "success": True,
            "duration": duration,
            "file_size": file_size,
            "error": None,
            "features": {
                "quality": "Середня (системні голоси)",
                "internet_required": False,
                "languages": f"Системні ({voice_count} голосів)",
                "voice_options": f"Багато: {', '.join(available_voices)}...",
                "speed_control": "Так (rate, volume)"
            }
        }
        
    except ImportError as e:
        return {
            "success": False,
            "duration": 0,
            "file_size": 0,
            "error": f"Не встановлено: {e}",
            "features": {}
        }
    except Exception as e:
        return {
            "success": False,
            "duration": 0,
            "file_size": 0,
            "error": str(e),
            "features": {}
        }

def test_coqui_tts(text, output_file, language="en"):
    """Тестує Coqui TTS (AI-based TTS)"""
    try:
        from TTS.api import TTS
        import torch
        
        print("🔄 Тестування Coqui TTS (AI-based TTS)...")
        start_time = time.time()
        
        # Створюємо директорію якщо не існує
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Обираємо модель залежно від мови
        if language == "en":
            model_name = "tts_models/en/ljspeech/tacotron2-DDC"
        else:
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        
        # Ініціалізуємо TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS(model_name=model_name, progress_bar=False, gpu=(device == "cuda"))
        
        # Генеруємо аудіо
        tts.tts_to_file(text=text, file_path=output_file)
        
        duration = time.time() - start_time
        file_size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
        
        # Отримуємо інформацію про доступні моделі
        try:
            available_models = TTS.list_models()
            en_models = [m for m in available_models if "en/" in m][:3]
            model_count = len(available_models)
        except:
            en_models = ["ljspeech/tacotron2"]
            model_count = "невідомо"
        
        return {
            "success": True,
            "duration": duration,
            "file_size": file_size,
            "error": None,
            "features": {
                "quality": "Дуже висока (AI Neural)",
                "internet_required": False,  # Після завантаження моделі
                "languages": f"Багато ({model_count} моделей)",
                "voice_options": f"AI моделі: {', '.join(en_models)}...",
                "speed_control": "Так (через параметри моделі)"
            }
        }
        
    except ImportError as e:
        return {
            "success": False,
            "duration": 0,
            "file_size": 0,
            "error": f"Не встановлено: pip install TTS ({e})",
            "features": {}
        }
    except Exception as e:
        return {
            "success": False,
            "duration": 0,
            "file_size": 0,
            "error": str(e),
            "features": {}
        }

def test_espeak(text, output_file, language="en"):
    """Тестує espeak (системний TTS)"""
    try:
        print("🔄 Тестування espeak (System TTS)...")
        start_time = time.time()
        
        # Створюємо директорію якщо не існує
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Команда espeak
        cmd = [
            "espeak", 
            "-s", "150",  # швидкість (слова за хвилину)
            "-v", f"{language}",  # мова
            "-w", output_file,  # вихідний файл
            text
        ]
        
        # Виконуємо команду
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            file_size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
            
            # Отримуємо список доступних голосів
            try:
                voices_result = subprocess.run(["espeak", "--voices"], capture_output=True, text=True, timeout=10)
                voice_lines = voices_result.stdout.split('\n')[1:6]  # Перші 5 рядків (пропускаємо заголовок)
                available_voices = [line.split()[1] if len(line.split()) > 1 else "unknown" for line in voice_lines if line.strip()]
            except:
                available_voices = ["default"]
            
            return {
                "success": True,
                "duration": duration,
                "file_size": file_size,
                "error": None,
                "features": {
                    "quality": "Базова (синтетичний)",
                    "internet_required": False,
                    "languages": f"Багато: {', '.join(available_voices[:3])}...",
                    "voice_options": "Обмежені системні",
                    "speed_control": "Так (-s параметр)"
                }
            }
        else:
            return {
                "success": False,
                "duration": duration,
                "file_size": 0,
                "error": f"espeak помилка: {result.stderr}",
                "features": {}
            }
            
    except FileNotFoundError:
        return {
            "success": False,
            "duration": 0,
            "file_size": 0,
            "error": "espeak не встановлено (sudo apt-get install espeak)",
            "features": {}
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "duration": 0,
            "file_size": 0,
            "error": "espeak timeout (більше 30 секунд)",
            "features": {}
        }
    except Exception as e:
        return {
            "success": False,
            "duration": 0,
            "file_size": 0,
            "error": str(e),
            "features": {}
        }

def play_audio_file(file_path):
    """Відтворює аудіо файл для порівняння"""
    try:
        import sounddevice as sd
        import soundfile as sf
        
        if not os.path.exists(file_path):
            print(f"❌ Файл {file_path} не знайдено")
            return False
        
        # Читаємо аудіо файл
        data, samplerate = sf.read(file_path)
        
        print(f"🔊 Відтворюємо: {os.path.basename(file_path)}")
        
        # Відтворюємо аудіо
        sd.play(data, samplerate)
        sd.wait()  # Чекаємо до завершення відтворення
        
        return True
        
    except ImportError:
        print("❌ soundfile/sounddevice не встановлено для відтворення")
        return False
    except Exception as e:
        print(f"❌ Помилка відтворення {file_path}: {e}")
        return False

def format_file_size(size_bytes):
    """Форматує розмір файлу у зрозумілому вигляді"""
    if size_bytes == 0:
        return "0 B"
    elif size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def print_comparison_table(results):
    """Друкує таблицю порівняння результатів"""
    print("\n" + "="*80)
    print("ПОРІВНЯННЯ TTS СИСТЕМ")
    print("="*80)
    
    # Заголовок таблиці
    print(f"{'TTS Система':<15} {'Статус':<10} {'Час':<8} {'Розмір':<10} {'Якість':<20} {'Інтернет':<9}")
    print("-" * 80)
    
    # Дані по кожній системі
    for tts_name, result in results.items():
        if result['success']:
            status = "✅ OK"
            duration = f"{result['duration']:.1f}s"
            file_size = format_file_size(result['file_size'])
            quality = result['features'].get('quality', 'N/A')
            internet = "Так" if result['features'].get('internet_required', False) else "Ні"
        else:
            status = "❌ FAIL"
            duration = "N/A"
            file_size = "N/A"
            quality = "N/A"
            internet = "N/A"
        
        print(f"{tts_name:<15} {status:<10} {duration:<8} {file_size:<10} {quality:<20} {internet:<9}")
    
    print("-" * 80)
    
    # Детальна інформація
    print("\nДЕТАЛЬНА ІНФОРМАЦІЯ:")
    print("-" * 40)
    
    for tts_name, result in results.items():
        print(f"\n🔹 {tts_name}")
        if result['success']:
            features = result['features']
            print(f"  ✅ Статус: Працює")
            print(f"  ⏱️ Час генерації: {result['duration']:.2f} секунд")
            print(f"  💾 Розмір файлу: {format_file_size(result['file_size'])}")
            print(f"  🎵 Якість: {features.get('quality', 'N/A')}")
            print(f"  🌐 Потребує інтернет: {'Так' if features.get('internet_required', False) else 'Ні'}")
            print(f"  🗣️ Голоси: {features.get('voice_options', 'N/A')}")
            print(f"  ⚙️ Контроль: {features.get('speed_control', 'N/A')}")
        else:
            print(f"  ❌ Статус: Не працює")
            print(f"  🐛 Помилка: {result['error']}")

def main():
    """Основна функція тестування TTS систем"""
    print("🚀 ТЕСТ ПОРІВНЯННЯ TTS СИСТЕМ")
    print("="*50)
    
    # Тестовий текст
    test_text = "Hey, whatcha up to?"
    print(f"📝 Тестовий текст: '{test_text}'")
    
    # Створюємо директорію для результатів
    output_dir = "tts_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Тестуємо всі доступні TTS системи
    tts_systems = {
        "gTTS": lambda: test_gtts(test_text, f"{output_dir}/gtts_test.wav"),
        "pyttsx3": lambda: test_pyttsx3(test_text, f"{output_dir}/pyttsx3_test.wav"),
        "Coqui TTS": lambda: test_coqui_tts(test_text, f"{output_dir}/coqui_test.wav"),
        "espeak": lambda: test_espeak(test_text, f"{output_dir}/espeak_test.wav")
    }
    
    results = {}
    
    print(f"\n🔄 Тестування {len(tts_systems)} TTS систем...")
    print("-" * 50)
    
    for tts_name, test_func in tts_systems.items():
        print(f"\n📋 Тестування {tts_name}...")
        try:
            result = test_func()
            results[tts_name] = result
            
            if result['success']:
                print(f"✅ {tts_name}: Успішно ({result['duration']:.1f}s)")
            else:
                print(f"❌ {tts_name}: {result['error']}")
                
        except Exception as e:
            print(f"❌ {tts_name}: Критична помилка - {e}")
            results[tts_name] = {
                "success": False,
                "duration": 0,
                "file_size": 0,
                "error": f"Критична помилка: {e}",
                "features": {}
            }
    
    # Друкуємо таблицю порівняння
    print_comparison_table(results)
    
    # Рекомендації
    print("\n" + "="*80)
    print("РЕКОМЕНДАЦІЇ")
    print("="*80)
    
    successful_systems = [name for name, result in results.items() if result['success']]
    
    if successful_systems:
        print("🎯 Доступні системи:", ", ".join(successful_systems))
        
        # Рекомендація по якості
        if "Coqui TTS" in successful_systems:
            print("🏆 Найкраща якість: Coqui TTS (AI neural, офлайн)")
        if "gTTS" in successful_systems:
            print("🥇 Висока якість: gTTS (потребує інтернет)")
        if "pyttsx3" in successful_systems:
            print("🔄 Офлайн варіант: pyttsx3 (не потребує інтернет)")
        if "espeak" in successful_systems:
            print("⚡ Найшвидший: espeak (базова якість)")
    else:
        print("❌ Жодна TTS система не працює!")
        print("💡 Встановіть одну з бібліотек:")
        print("   pip install TTS                    # Coqui TTS (найкраща якість)")
        print("   pip install gtts pydub             # Google TTS")
        print("   pip install pyttsx3                # Offline TTS")
        print("   sudo apt-get install espeak        # espeak (Linux)")
    
    # Можливість прослухати результати
    if successful_systems:
        print(f"\n🎧 ПРОСЛУХОВУВАННЯ РЕЗУЛЬТАТІВ")
        print("-" * 40)
        
        for tts_name in successful_systems:
            file_path = f"{output_dir}/{tts_name.lower()}_test.wav"
            if os.path.exists(file_path):
                try:
                    input(f"📱 Натисніть Enter для прослуховування {tts_name}...")
                    play_audio_file(file_path)
                    time.sleep(1)  # Пауза між відтвореннями
                except KeyboardInterrupt:
                    print("\n⚠️ Прослуховування перервано")
                    break
    
    print(f"\n📁 Аудіо файли збережено в: {os.path.abspath(output_dir)}")
    print("✅ Тест TTS систем завершено!")

if __name__ == "__main__":
    main()

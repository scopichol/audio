# Встановлення та налаштування Coqui TTS

## Що таке Coqui TTS?

Coqui TTS - це продвинутий open-source TTS (Text-to-Speech) движок, що використовує нейронні мережі для генерації високоякісного мовлення. Розроблений командою Coqui AI як спадкоємець Mozilla TTS.

## Переваги Coqui TTS:
- 🎯 **Найвища якість звуку** - AI-генерований голос
- 🔄 **Працює офлайн** - після завантаження моделі
- 🌍 **Багато мов та голосів** - підтримка різних акцентів
- 🎨 **Клонування голосу** - можливість навчити власний голос
- ⚙️ **Гнучкі налаштування** - контроль швидкості, інтонації

## Встановлення

### Основне встановлення:
```bash
pip install TTS
```

### Повне встановлення з усіма залежностями:
```bash
pip install TTS[all]
```

### Для користувачів з GPU (рекомендовано):
```bash
pip install TTS torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Перевірка встановлення

### Перевірити доступні моделі:
```python
from TTS.api import TTS
print(TTS.list_models())
```

### Швидкий тест:
```python
from TTS.api import TTS

# Ініціалізація TTS
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

# Генерація аудіо
tts.tts_to_file(text="Hello, this is a test.", file_path="test_output.wav")
print("Аудіо згенеровано: test_output.wav")
```

## Рекомендовані моделі

### Для англійської мови:
1. **`tts_models/en/ljspeech/tacotron2-DDC`** - швидка, хороша якість
2. **`tts_models/en/vctk/vits`** - багато голосів
3. **`tts_models/en/ljspeech/glow-tts`** - природне звучання

### Мультилінгвальні моделі:
1. **`tts_models/multilingual/multi-dataset/xtts_v2`** - найновіша
2. **`tts_models/multilingual/multi-dataset/bark`** - емоційний голос

## Використання в проекті

### Базове використання:
```python
def generate_with_coqui(text, output_file, language="en"):
    try:
        from TTS.api import TTS
        import torch
        
        # Обираємо модель
        if language == "en":
            model_name = "tts_models/en/ljspeech/tacotron2-DDC"
        else:
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        
        # Ініціалізуємо TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS(model_name=model_name, progress_bar=False, gpu=(device == "cuda"))
        
        # Генеруємо аудіо
        tts.tts_to_file(text=text, file_path=output_file)
        
        return True
    except Exception as e:
        print(f"Помилка Coqui TTS: {e}")
        return False
```

### Розширене використання з вибором голосу:
```python
def generate_with_speaker(text, output_file, speaker_idx=0):
    try:
        from TTS.api import TTS
        
        # Модель з багатьма голосами
        tts = TTS("tts_models/en/vctk/vits")
        
        # Генеруємо з конкретним спікером
        tts.tts_to_file(text=text, file_path=output_file, speaker_idx=speaker_idx)
        
        return True
    except Exception as e:
        print(f"Помилка: {e}")
        return False
```

## Системні вимоги

### Мінімальні:
- **RAM**: 4 GB
- **Диск**: 2 GB вільного місця
- **Python**: 3.8+

### Рекомендовані:
- **RAM**: 8+ GB
- **GPU**: NVIDIA з CUDA підтримкою
- **Диск**: 5+ GB для кількох моделей

## Можливі проблеми та рішення

### 1. "Import TTS.api could not be resolved"
```bash
pip install --upgrade TTS
```

### 2. "CUDA out of memory"
```python
# Використовуйте CPU замість GPU
tts = TTS(model_name="...", gpu=False)
```

### 3. "Model download failed"
```bash
# Очистити кеш та повторити
rm -rf ~/.local/share/tts/
pip install --upgrade --force-reinstall TTS
```

### 4. Повільна генерація
```python
# Використовуйте швидшу модель
model_name = "tts_models/en/ljspeech/speedy-speech"
```

### 5. Проблеми з залежностями
```bash
pip install --upgrade torch torchvision torchaudio
pip install --upgrade TTS
```

## Порівняння моделей

| Модель | Швидкість | Якість | Розмір | Мови |
|--------|-----------|--------|--------|------|
| tacotron2-DDC | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ~100MB | EN |
| vits | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ~200MB | EN |
| xtts_v2 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ~500MB | Multi |
| bark | ⭐ | ⭐⭐⭐⭐⭐ | ~1GB | Multi |

## Інтеграція в основний проект

Після встановлення Coqui TTS, ваш `test_phonem2.py` автоматично буде використовувати його як найпріоритетніший TTS движок у функції `generate_reference_audio()`.

Послідовність спроб:
1. **Coqui TTS** (найкраща якість)
2. **gTTS** (швидкий, потребує інтернет)
3. **pyttsx3** (офлайн fallback)
4. **espeak** (базовий варіант)

## Корисні команди

### Список всіх моделей:
```bash
tts --list_models
```

### Генерація з командного рядка:
```bash
tts --text "Hello world" --model_name "tts_models/en/ljspeech/tacotron2-DDC" --out_path output.wav
```

### Інформація про модель:
```bash
tts --model_info "tts_models/en/ljspeech/tacotron2-DDC"
```

## Висновок

Coqui TTS значно покращить якість згенерованого референсного аудіо у вашому проекті аналізу вимови. Хоча встановлення може зайняти деякий час та простір, результат варто того для серйозного навчання вимови.

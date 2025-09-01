# Швидке налаштування автогенерації аудіо

## Встановлення залежностей

### Базове встановлення (рекомендовано):
```bash
pip install gtts pydub pyttsx3 soundfile sounddevice
```

### Або встановлення з requirements.txt:
```bash
pip install -r requirements.txt
```

### Для Linux додатково встановіть espeak:
```bash
sudo apt-get install espeak espeak-data
```

## Як це працює

1. **Автоматичне створення файлів**: Якщо `sentences/sentence_1.txt` або `sentences/sentence_1.wav` відсутні, система автоматично створить їх.

2. **Порядок спроб TTS**:
   - gTTS (Google) - найкраща якість
   - pyttsx3 (офлайн) - backup
   - espeak (системний) - Linux/macOS

3. **Перевірка доступності**: Система покаже які TTS движки доступні при запуску.

## Приклад виводу

```
🎯 Фраза для вимови: "The quick brown fox jumps over the lazy dog."
⚠️ Референсний аудіо файл sentences/sentence_1.wav не знайдено
🎤 Генеруємо референсний аудіо з тексту...
📢 Доступні TTS системи: gTTS (Google Text-to-Speech), pyttsx3 (Offline TTS)
🔄 Спроба генерації через gTTS...
✅ Аудіо згенеровано через gTTS: sentences/sentence_1.wav
```

## Помилки та рішення

### "Import gtts could not be resolved"
```bash
pip install gtts pydub
```

### "Import pyttsx3 could not be resolved"
```bash
pip install pyttsx3
```

### "espeak не встановлено" (Linux)
```bash
sudo apt-get install espeak
```

### Проблеми з аудіо на Linux
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

## Тестування

Запустіть скрипт і він автоматично:
1. Створить `sentences/sentence_1.txt` з стандартним текстом
2. Згенерує `sentences/sentence_1.wav` через доступний TTS
3. Відтворить згенерований аудіо
4. Почне процес запису та аналізу

```bash
python test_phonem2.py
```

## Налаштування якості

### Для gTTS (найкраща якість):
- Потребує інтернет
- Автоматично найкраща якість Google
- Зберігається як WAV 16kHz

### Для pyttsx3:
- Швидкість: 150 слів/хв
- Гучність: 90%
- Англійський голос (автовибір)

### Для espeak:
- Швидкість: 150 слів/хв
- Формат: WAV
- Англійська мова

## Структура файлів після першого запуску

```
sentences/
├── sentence_1.txt    # Створюється автоматично
└── sentence_1.wav    # Генерується автоматично

out/
├── sentence_1_record.wav              # Ваш запис
├── sentence_1_phones_confusion.png    # Матриця помилок
├── sentence_1_phones_timeline.png     # Часові мітки
├── sentence_1_phones_timing.csv       # Дані фонем
├── sentence_1_phoneme_comparison.png  # Порівняння фонем
└── sentence_1_phones_ref_hyp.json     # Результати аналізу
```

# PyTorch 2.6+ Compatibility Guide

## Проблема

У PyTorch 2.6 змінилися налаштування безпеки для завантаження моделей. За замовчуванням параметр `weights_only` встановлено в `True`, що може викликати помилки при завантаженні деяких моделей, зокрема Bark TTS.

### Типова помилка:
```
Weights only load failed. This file can still be loaded, to do so you have two options...
WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray.scalar was not an allowed global by default.
```

## Рішення

### 1. Автоматичне вирішення в коді

Наш код автоматично налаштовує безпечні globals для PyTorch:

```python
def setup_pytorch_safe_globals():
    """Налаштовує безпечні globals для PyTorch 2.6+"""
    try:
        import torch
        import numpy as np
        
        if hasattr(torch.serialization, 'add_safe_globals'):
            try:
                # Сучасний спосіб (numpy 1.20+)
                torch.serialization.add_safe_globals([np._core.multiarray.scalar])
            except AttributeError:
                # Застарілий спосіб
                torch.serialization.add_safe_globals([np.core.multiarray.scalar])
            return True
    except Exception as e:
        return False
```

### 2. Використання контекстних менеджерів

Для завантаження моделей використовуємо безпечні контексти:

```python
# Безпечне завантаження моделей
try:
    with torch.serialization.safe_globals([np._core.multiarray.scalar]):
        preload_models()
except (AttributeError, Exception):
    # Fallback до стандартного способу
    preload_models()
```

### 3. Альтернативні рішення

#### Варіант A: Зниження версії PyTorch
```bash
pip install torch==2.4.0
```

#### Варіант B: Використання небезпечного завантаження (не рекомендується)
```python
torch.load(model_path, weights_only=False)
```

## Протестовані версії

- ✅ PyTorch 2.6+ (з нашими виправленнями)
- ✅ PyTorch 2.4.0 (стабільна версія)
- ✅ Numpy 1.20+ (сучасні версії)
- ✅ Numpy < 1.20 (застарілі версії)

## Статус TTS систем

| TTS System | PyTorch 2.6+ | PyTorch 2.4 | Примітки |
|------------|--------------|-------------|----------|
| Bark TTS   | ✅ (з fix)   | ✅          | Потребує виправлення для 2.6+ |
| Coqui TTS  | ✅           | ✅          | Працює без проблем |
| gTTS       | ✅           | ✅          | Не залежить від PyTorch |
| pyttsx3    | ✅           | ✅          | Не залежить від PyTorch |
| espeak     | ✅           | ✅          | Не залежить від PyTorch |

## Встановлення та тестування

```bash
# Встановлення Bark TTS з залежностями
pip install bark scipy

# Альтернативно - стабільна версія PyTorch
pip install torch==2.4.0 bark scipy

# Тестування
python test_tts_comparison.py
```

## Діагностика

Якщо виникають проблеми, перевірте:

1. **Версію PyTorch:**
   ```python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   ```

2. **Версію Numpy:**
   ```python
   import numpy as np
   print(f"Numpy version: {np.__version__}")
   ```

3. **Доступність safe_globals:**
   ```python
   import torch
   print(f"Safe globals available: {hasattr(torch.serialization, 'add_safe_globals')}")
   ```

## Висновок

Наша реалізація автоматично обробляє проблеми сумісності з PyTorch 2.6+ і забезпечує роботу з усіма версіями PyTorch та Numpy.

"""
Основний скрипт з покроковим виконанням та детальною діагностикою
"""
import os
import sys
import time
sys.path.append('.')

from test_phonem2 import *

def main_step_by_step():
    """Основна функція з покроковим виконанням"""
    print("🚀 Запуск покрокового тесту з записом")
    print("="*50)
    
    try:
        # Крок 1: Діагностика системи
        print("📋 КРОК 1: Діагностика системи")
        print("-" * 30)
        check_system_status()
        input("📱 Натисніть Enter для продовження...")
        
        # Крок 2: Підготовка тексту
        print("\n📋 КРОК 2: Підготовка референсного тексту")
        print("-" * 30)
        target_text = read_reference_text(REFERENCE_TEXT_FILE)
        if not target_text:
            print(f"❌ Використовуємо стандартний текст замість файлу {REFERENCE_TEXT_FILE}")
            target_text = TARGET_TEXT
            
            # Створюємо референсний текстовий файл зі стандартним текстом
            print(f"📄 Створюємо референсний текстовий файл...")
            create_reference_text_file(REFERENCE_TEXT_FILE, target_text)
        else:
            print(f"✅ Референсний текст завантажено з {REFERENCE_TEXT_FILE}")
        
        print(f'🎯 Фраза для вимови: "{target_text}"')
        input("📱 Натисніть Enter для продовження...")
        
        # Крок 3: Підготовка або генерація аудіо
        print("\n📋 КРОК 3: Підготовка референсного аудіо")
        print("-" * 30)
        print("\n🔊 Спочатку послухайте референсний зразок:")
        
        # Перевіряємо чи існує референсний аудіо файл
        if not os.path.exists(REFERENCE_AUDIO):
            print(f"⚠️ Референсний аудіо файл {REFERENCE_AUDIO} не знайдено")
            print("🎤 Генеруємо референсний аудіо з тексту...")
            
            if generate_reference_audio(target_text, REFERENCE_AUDIO, language="en"):
                print("✅ Референсний аудіо успішно згенеровано")
            else:
                print("❌ Не вдалося згенерувати референсний аудіо")
                return False
        
        # Відтворюємо аудіо (згенерований або існуючий)
        if not play_reference_audio(REFERENCE_AUDIO):
            print("⚠️ Не вдалося відтворити референсний аудіо, продовжуємо без нього...")
        
        input("📱 Натисніть Enter для продовження до запису...")
        
        # Крок 4: Запис аудіо
        print("\n📋 КРОК 4: Запис вашого аудіо")
        print("-" * 30)
        
        # Пауза між референсним аудіо та записом
        print("\n⏳ Готуйтесь до запису через 3 секунди...")
        for i in range(3, 0, -1):
            print(f"⏰ {i}...")
            time.sleep(1)
        
        # Подаємо звуковий сигнал перед записом
        play_beep_signal(frequency=800, duration=0.3, count=3)
        
        print("🎤 Запис...")
        audio = sd.rec(int(SECONDS * SR), samplerate=SR, channels=1, dtype='int16')
        sd.wait()
        write(AUDIO_OUT, SR, audio)
        print(f"✅ Аудіо збережено: {AUDIO_OUT}")
        
        input("📱 Натисніть Enter для продовження до аналізу...")
        
        # Крок 5: Генерація еталонних фонем
        print("\n📋 КРОК 5: Генерація еталонних фонем")
        print("-" * 30)
        ref_phones = g2p_arpabet(target_text)
        ref_ipa = arpabet_seq_to_ipa(ref_phones)
        print(f"📝 Еталонні фонеми (ARPAbet): {ref_phones}")
        print(f"📝 Еталонні фонеми (IPA): {ref_ipa}")
        
        input("📱 Натисніть Enter для завантаження ASR моделі...")
        
        # Крок 6: Завантаження ASR моделі
        print("\n📋 КРОК 6: Завантаження ASR моделі")
        print("-" * 30)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Завантаження WhisperX ({MODEL_NAME}) на {device}...")
        
        asr_model = safe_load_whisperx_model(MODEL_NAME, device)
        print("✅ ASR модель завантажена успішно")
        
        input("📱 Натисніть Enter для транскрибування...")
        
        # Крок 7: Транскрибування
        print("\n📋 КРОК 7: Транскрибування аудіо")
        print("-" * 30)
        print("🔄 Транскрибуємо ваш запис...")
        res = asr_model.transcribe(AUDIO_OUT)
        
        # Правильна обробка результату ASR
        asr_text = " ".join(seg["text"].strip() for seg in res["segments"])
        print(f"🗣️ ASR розпізнав: '{asr_text}'")
        
        input("📱 Натисніть Enter для вирівнювання...")
        
        # Крок 8: Вирівнювання
        print("\n📋 КРОК 8: Вирівнювання слів та фонем")
        print("-" * 30)
        # Використаємо align-модель (англійська)
        print("🔄 Завантаження align моделі...")
        align_model, metadata = whisperx.load_align_model(language_code=LANG_CODE, device=device)
        print("✅ Align модель завантажена")
        
        print("🔄 Вирівнювання слів...")
        aligned = whisperx.align(res["segments"], align_model, metadata, AUDIO_OUT, device=device)
        print("✅ Вирівнювання завершено")
        
        input("📱 Натисніть Enter для аналізу фонем...")
        
        # Крок 9: Аналіз фонем
        print("\n📋 КРОК 9: Витяг та аналіз фонем")
        print("-" * 30)
        # Витяг гіпотезних фонем + CSV
        hyp_phones, phone_rows = extract_hyp_arpabet_from_whisperx(aligned.get("segments", []))
        hyp_ipa = arpabet_seq_to_ipa(hyp_phones)
        print(f"🎯 Розпізнані фонеми (ARPAbet): {hyp_phones}")
        print(f"🎯 Розпізнані фонеми (IPA): {hyp_ipa}")
        
        input("📱 Натисніть Enter для обчислення метрик...")
        
        # Крок 10: Обчислення метрик
        print("\n📋 КРОК 10: Обчислення метрик точності")
        print("-" * 30)
        
        # PER + детальний аналіз
        per = compute_per(ref_phones, hyp_phones)
        print(f"\n📊 PER (Phone Error Rate) = {per:.3f} (0.0 = ідеально)")

        # WER (Word Error Rate) + детальний аналіз слів
        wer = compute_wer(target_text, asr_text)
        print(f"📊 WER (Word Error Rate) = {wer:.3f} (0.0 = ідеально)")
        
        input("📱 Натисніть Enter для детального аналізу...")
        
        # Крок 11: Детальний аналіз
        print("\n📋 КРОК 11: Детальний аналіз помилок")
        print("-" * 30)
        
        # Детальне порівняння слів
        word_comparison_results = print_word_alignment(target_text, asr_text)

        # Детальне порівняння фонем з IPA відображенням
        comparison_results = print_phoneme_alignment(ref_phones, hyp_phones)
        analyze_error_patterns(ref_phones, hyp_phones)
        
        input("📱 Натисніть Enter для збереження результатів...")
        
        # Крок 12: Збереження результатів
        print("\n📋 КРОК 12: Збереження результатів")
        print("-" * 30)
        
        save_hyp_phones_csv(phone_rows, "phones_timing.csv", prefix=REFERENCE_NAME)
        plot_timeline(phone_rows, "phones_timeline.png", prefix=REFERENCE_NAME)
        plot_phoneme_comparison(ref_phones, hyp_phones, "phoneme_comparison.png", prefix=REFERENCE_NAME, 
                              ref_text=target_text, hyp_text=asr_text)

        # Матриця плутанин
        save_confusion_matrix(ref_phones, hyp_phones, out_png="phones_confusion.png", include_eps=False, prefix=REFERENCE_NAME)

        # Збережемо результати у JSON
        json_path = get_output_path(f"{REFERENCE_NAME}_phones_ref_hyp.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "reference_file": REFERENCE_NAME,
                "target_text": target_text,
                "asr_text": asr_text,
                "ref_phones_arpabet": ref_phones,
                "ref_phones_ipa": ref_ipa,
                "hyp_phones_arpabet": hyp_phones,
                "hyp_phones_ipa": hyp_ipa,
                "per": per,
                "wer": wer,
                "phoneme_accuracy": comparison_results["accuracy"],
                "word_accuracy": (1 - word_comparison_results["wer"]) * 100,
                "phoneme_errors": {
                    "substitutions": comparison_results["substitutions"],
                    "insertions": comparison_results["insertions"],
                    "deletions": comparison_results["deletions"]
                },
                "word_errors": {
                    "correct": word_comparison_results["correct"],
                    "substitutions": word_comparison_results["substitutions"],
                    "insertions": word_comparison_results["insertions"],
                    "deletions": word_comparison_results["deletions"]
                }
            }, f, ensure_ascii=False, indent=2)
        print(f"💾 Результати збережено у: {json_path}")
        
        print(f"\n📁 Всі файли збережено в каталозі: {os.path.abspath('out')}")
        print("\n🎉 Покроковий тест завершено успішно!")
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Тест перерваний користувачем")
        return False
    except Exception as e:
        print(f"\n❌ Помилка в покроковому тесті: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main_step_by_step()

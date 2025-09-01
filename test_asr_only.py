"""
Тест основної функціональності без запису звуку
Використовує існуючий референсний аудіо файл для тестування ASR
"""
import os
import sys
sys.path.append('.')

from test_phonem2 import (
    check_system_status, read_reference_text, g2p_arpabet,
    safe_load_whisperx_model, extract_hyp_arpabet_from_whisperx,
    compute_per, compute_wer, print_word_alignment, print_phoneme_alignment,
    save_confusion_matrix, save_hyp_phones_csv, plot_timeline,
    plot_phoneme_comparison, get_output_path
)
from utilites import arpabet_seq_to_ipa
import torch
import whisperx
import json

# Налаштування
REFERENCE_NAME = "sentence_1"
REFERENCE_AUDIO = f"sentences/{REFERENCE_NAME}.wav"
REFERENCE_TEXT_FILE = f"sentences/{REFERENCE_NAME}.txt"
MODEL_NAME = "small.en"
LANG_CODE = "en"

def test_asr_pipeline():
    """Тестує повний ASR пайплайн без запису"""
    print("🚀 Тестування ASR пайплайну...")
    print("="*60)
    
    # Перевіряємо систему
    check_system_status()
    
    # Читаємо референсний текст
    target_text = read_reference_text(REFERENCE_TEXT_FILE)
    if not target_text:
        print("❌ Не вдалося прочитати референсний текст")
        return False
    
    print(f'🎯 Референсний текст: "{target_text}"')
    
    # Перевіряємо що аудіо файл існує
    if not os.path.exists(REFERENCE_AUDIO):
        print(f"❌ Аудіо файл {REFERENCE_AUDIO} не знайдено")
        return False
    
    print(f"✅ Використовуємо аудіо: {REFERENCE_AUDIO}")
    
    try:
        # 1) Еталонні фонеми
        print("\n🔄 Генерація еталонних фонем...")
        ref_phones = g2p_arpabet(target_text)
        ref_ipa = arpabet_seq_to_ipa(ref_phones)
        print(f"📝 Еталонні фонеми (ARPAbet): {ref_phones}")
        print(f"📝 Еталонні фонеми (IPA): {ref_ipa}")

        # 2) ASR + Align
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n🔄 Завантаження WhisperX ({MODEL_NAME}) на {device}...")
        
        asr_model = safe_load_whisperx_model(MODEL_NAME, device)
        print("✅ ASR модель завантажена")
        
        print("🔄 Транскрибуємо аудіо...")
        res = asr_model.transcribe(REFERENCE_AUDIO)
        
        # Правильна обробка результату ASR
        asr_text = " ".join(seg["text"].strip() for seg in res["segments"])
        print(f"🗣️ ASR розпізнав: '{asr_text}'")

        # Використаємо align-модель (англійська)
        print("🔄 Завантаження align моделі...")
        align_model, metadata = whisperx.load_align_model(language_code=LANG_CODE, device=device)
        print("✅ Align модель завантажена")
        
        print("🔄 Вирівнювання слів...")
        aligned = whisperx.align(res["segments"], align_model, metadata, REFERENCE_AUDIO, device=device)
        print("✅ Вирівнювання завершено")

        # 3) Витяг гіпотезних фонем + CSV
        print("🔄 Витяг фонем з результатів...")
        hyp_phones, phone_rows = extract_hyp_arpabet_from_whisperx(aligned.get("segments", []))
        hyp_ipa = arpabet_seq_to_ipa(hyp_phones)
        print(f"🎯 Розпізнані фонеми (ARPAbet): {hyp_phones}")
        print(f"🎯 Розпізнані фонеми (IPA): {hyp_ipa}")

        # 4) Зберігаємо результати
        print("🔄 Збереження результатів...")
        save_hyp_phones_csv(phone_rows, "phones_timing.csv", prefix=REFERENCE_NAME)
        plot_timeline(phone_rows, "phones_timeline.png", prefix=REFERENCE_NAME)

        # 5) PER + детальний аналіз
        per = compute_per(ref_phones, hyp_phones)
        print(f"\n📊 PER (Phone Error Rate) = {per:.3f} (0.0 = ідеально)")

        # 6) WER (Word Error Rate) + детальний аналіз слів
        wer = compute_wer(target_text, asr_text)
        print(f"📊 WER (Word Error Rate) = {wer:.3f} (0.0 = ідеально)")
        
        # Детальне порівняння слів
        word_comparison_results = print_word_alignment(target_text, asr_text)

        # Детальне порівняння фонем з IPA відображенням
        comparison_results = print_phoneme_alignment(ref_phones, hyp_phones)
        plot_phoneme_comparison(ref_phones, hyp_phones, "phoneme_comparison.png", prefix=REFERENCE_NAME, 
                              ref_text=target_text, hyp_text=asr_text)

        # Матриця плутанин
        save_confusion_matrix(ref_phones, hyp_phones, out_png="phones_confusion.png", include_eps=False, prefix=REFERENCE_NAME)

        # 7) Збережемо результати у JSON
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
        print("✅ Тест ASR пайплайну завершено успішно!")
        return True
        
    except Exception as e:
        print(f"❌ Помилка в ASR пайплайні: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Основна функція тестування"""
    print("🚀 Запуск тесту ASR пайплайну")
    print("="*50)
    
    success = test_asr_pipeline()
    
    if success:
        print("\n🎉 Все працює! Можна запускати повну версію з записом.")
    else:
        print("\n❌ Є проблеми, потрібно їх вирішити.")

if __name__ == "__main__":
    main()

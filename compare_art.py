import librosa
import numpy as np
from scipy.spatial.distance import cdist
from librosa import effects
import glob
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_LOCAL_DIR_USE_SYMLINKS"] = "0" 

Ref_file = "sentences/sentence_1.wav"
output_pattern = "output*.wav"

def compare_pronunciation_MFCC(ref_path, test_path, sr=16000, n_mfcc=13):
    # 1) Завантаження
    y1, _ = librosa.load(ref_path, sr=sr, mono=True)
    y2, _ = librosa.load(test_path, sr=sr, mono=True)

    # 2) Прибрати тишу (щоб DTW не «вирівнював» паузи)
    y1, _ = effects.trim(y1, top_db=25)  # підкоригуй top_db за потреби
    y2, _ = effects.trim(y2, top_db=25)

    # 3) MFCC + дельти
    hop_length = 256
    n_fft = 1024
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Додаємо дельта-ознаки (1-го та 2-го порядку)
    d1 = librosa.feature.delta(mfcc1)
    d2 = librosa.feature.delta(mfcc1, order=2)
    X = np.vstack([mfcc1, d1, d2])  # shape: (3*n_mfcc, T1)

    d1b = librosa.feature.delta(mfcc2)
    d2b = librosa.feature.delta(mfcc2, order=2)
    Y = np.vstack([mfcc2, d1b, d2b])  # shape: (3*n_mfcc, T2)

    # 4) DTW по косинусній «відстані» між векторами ознак у кожному кадрі
    D, wp = librosa.sequence.dtw(X, Y, metric="cosine")

    # 5) Середня вартість шляху (нормалізація на довжину)
    dist_avg = D[-1, -1] / len(wp)

    # 6) Перетворення у «схожість» 0–100 (монотонна функція; підлаштуй alpha)
    alpha = 3.0
    similarity = float(100.0 * np.exp(-alpha * dist_avg))
    return dist_avg, similarity

def compare_pronunciation_embedding(ref_path, test_path, model_name="speechbrain/spkrec-ecapa-voxceleb"):
    """
    Порівнює два аудіо-файли через ембедінги голосу (cosine similarity).
    Потрібно встановити speechbrain: pip install speechbrain
    """
    from speechbrain.inference import SpeakerRecognition
    from huggingface_hub import snapshot_download

    savedir = r"D:\projects\RobotDreams\GenAI\audio\tmp_spkrec"

    # Качаємо всю репу моделі прямо у savedir БЕЗ symlink-ів
    snapshot_download(
        repo_id=model_name,
        local_dir=str(savedir),
        local_dir_use_symlinks=False,   # ключове
    )

    # Тепер ініціалізуємо з локальної папки: SpeechBrain нічого не лінкує/переміщує
    recognizer = SpeakerRecognition.from_hparams(
        source=str(savedir),
        savedir=str(savedir),
    )   
    
    # Отримати косинусну схожість (1.0 - ідеально)
    score, prediction = recognizer.verify_files(ref_path, test_path)
    return float(score), bool(prediction)

def compare_pronunciation_embedding2(ref_path, test_path, model_name="speechbrain/spkrec-ecapa-voxceleb"):
    from pathlib import Path
    from speechbrain.utils.fetching import fetch, LocalStrategy  # правильні імена
    from speechbrain.inference import SpeakerRecognition

    model_id = "speechbrain/spkrec-ecapa-voxceleb"
    savedir = Path(r"D:\projects\RobotDreams\GenAI\audio\tmp_spkrec")
    savedir.mkdir(parents=True, exist_ok=True)

    # Скачати hyperparams.yaml прямо в savedir, оминаючи symlink
    print('FETCHING...')
    fetch(
        filename="hyperparams.yaml",
        source=model_id,
        savedir=str(savedir),
        local_strategy=LocalStrategy.NO_LINK #LocalStrategy.COPY_SKIP_CACHE,  # 
    )

    # Далі ініціалізація з локальної теки (файли вже лежать там)
    recognizer = SpeakerRecognition.from_hparams(
        source=str(savedir),
        savedir=str(savedir),
    )
    # Отримати косинусну схожість (1.0 - ідеально)
    score, prediction = recognizer.verify_files(ref_path, test_path)
    return float(score), bool(prediction)

if __name__ == "__main__":
    output_files = sorted(glob.glob(output_pattern))
    print(f"Reference: {Ref_file}")
    for idx, test_file in enumerate(output_files, 1):
        dist, score = compare_pronunciation_MFCC(Ref_file, test_file)
        print(f"[{idx}] {os.path.basename(test_file)}: DTW avg distance: {dist:.4f}, Similarity score: {score:.2f}%")
        # Порівняння через ембедінги голосу
        try:
            emb_score, emb_pred = compare_pronunciation_embedding2(Ref_file, test_file)
            print(f"    Embedding cosine similarity: {emb_score:.4f}, Same speaker: {emb_pred}")
        except Exception as e:
            print(f"    Embedding comparison error: {e}")

# # Завантаження двох аудіо
# y1, sr1 = librosa.load( Ref_file, sr=16000)
# y2, sr2 = librosa.load( Test_file, sr=16000)

# # MFCC ознаки
# mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
# mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)
# print(mfcc1.shape, mfcc2.shape)
# # DTW порівняння
# dist, wp = librosa.sequence.dtw(mfcc1.T, mfcc2.T, metric="cosine")
# print("Схожість (DTW відстань):", dist[-1, -1])

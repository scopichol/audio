import librosa
import numpy as np
from scipy.spatial.distance import cdist
from librosa import effects
import glob
import os

Ref_file = "sentences/sentence_1.wav"
output_pattern = "output*.wav"

def compare_pronunciation(ref_path, test_path, sr=16000, n_mfcc=13):
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

if __name__ == "__main__":
    output_files = sorted(glob.glob(output_pattern))
    print(f"Reference: {Ref_file}")
    for idx, test_file in enumerate(output_files, 1):
        dist, score = compare_pronunciation(Ref_file, test_file)
        print(f"[{idx}] {os.path.basename(test_file)}: DTW avg distance: {dist:.4f}, Similarity score: {score:.2f}%")

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

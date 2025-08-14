from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

input_file = "Harvard list 01.wav"
output_dir = "sentences"
os.makedirs(output_dir, exist_ok=True)

audio = AudioSegment.from_wav(input_file)

# Split on silence: adjust min_silence_len and silence_thresh as needed
chunks = split_on_silence(
    audio,
    min_silence_len=500,      # minimum length of silence (ms) to be a split
    silence_thresh=audio.dBFS-16,  # silence threshold (dBFS)
    keep_silence=200          # keep a bit of silence at edges
)

for i, chunk in enumerate(chunks):
    out_path = os.path.join(output_dir, f"sentence_{i+1}.wav")
    chunk.export(out_path, format="wav")
    print(f"Saved: {out_path}")


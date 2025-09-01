import os
import glob
from test_phonem2 import (
    g2p_arpabet, 
    pick_compute_type,
    # g2p_word_arpabet
)
from g2p_en import G2p
g2p = G2p()
    
# def get_words_timestamps(audio_path):
#     from whisper_timestamped import load_model, transcribe
#     model = load_model("small")
#     result = transcribe(model, audio_path, language="en")
#     words = []
#     for segment in result["segments"]:
#         for word_info in segment["words"]:
#             word = word_info["text"].strip()
#             words.append(word)
#     return words

# os.makedirs("out", exist_ok=True)
# for wav in glob.glob("output*.wav"):
#     words = get_words_timestamps(wav)
#     text = " ".join(words)
#     out_path = os.path.join("out", os.path.splitext(os.path.basename(wav))[0] + ".txt")
#     with open(out_path, "w", encoding="utf-8") as f:
#         f.write(text)
        
# from huggingface_hub import snapshot_download

# snapshot_download("speechbrain/spkrec-ecapa-voxceleb")

print(g2p_arpabet("The birch canoe slid on the smooth planks"))
print(g2p("birch"))
print(g2p("The birch canoe"))
print(g2p("Glue the sheet to the dark blue background"))
print(g2p("shit"))
print(g2p("Sheet"))
print(g2p("important"))
print(g2p("interesting"))
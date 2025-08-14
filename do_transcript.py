import os
import re

input_file = "H1 Harvard Sentences.txt"
output_dir = "sentences"
os.makedirs(output_dir, exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]
for sentence in sentences:
    match = re.match(r"(\d+)\.\s+(.*)", sentence)
    if match:
        num = match.group(1)
        text = match.group(2)
        out_path = os.path.join(output_dir, f"sentence_{num}.txt")
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(text)
import os
import re

folder = "sentences"
files = []
for filename in os.listdir(folder):
    match = re.match(r"sentence_(\d+)\.wav", filename)
    if match:
        num = int(match.group(1))
        files.append((num, filename))

files.sort()  # сортування по номеру

for num, filename in files:
    new_num = num - 1
    if new_num >= 0:
        new_name = f"sentence_{new_num}.wav"
        os.rename(
            os.path.join(folder, filename),
            os.path.join(folder, new_name)
        )
        print(f"{filename} -> {new_name}")
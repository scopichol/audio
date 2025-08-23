import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

def record_audio(filename="output.wav", duration=5, fs=16000, silence_thresh=500, min_speech=0.5):
    print(f"Recording for up to {duration} seconds...")
    frames = []
    blocksize = 1024
    max_blocks = int(duration * fs / blocksize)
    silent_blocks = 0
    speech_started = False

    def callback(indata, frames_count, time, status):
        nonlocal silent_blocks, speech_started
        volume_norm = np.linalg.norm(indata)
        frames.append(indata.copy())
        if volume_norm > silence_thresh:
            speech_started = True
            silent_blocks = 0
        elif speech_started:
            silent_blocks += 1

    stream = sd.InputStream(samplerate=fs, channels=1, dtype='int16', blocksize=blocksize, callback=callback)
    with stream:
        for i in range(max_blocks):
            sd.sleep(int(blocksize / fs * 1000))
            # Якщо після початку мовлення тиша > min_speech секунд, зупинити запис
            if speech_started and silent_blocks * blocksize / fs > min_speech:
                break

    audio = np.concatenate(frames, axis=0)
    write(filename, fs, audio)
    print(f"Audio saved to {filename}")

if __name__ == "__main__":
    record_audio()

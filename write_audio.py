import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename="output.wav", duration=5, fs=44100):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, audio)
    print(f"Audio saved to {filename}")

if __name__ == "__main__":
    record_audio()

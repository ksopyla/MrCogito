import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def create_tutorial_plots():
    # 1. Load an audio file
    # librosa.ex('trumpet') downloads a sample audio file
    print("Loading audio...")
    try:
        filename = librosa.ex('trumpet')
        y, sr = librosa.load(filename)
    except Exception as e:
        print(f"Could not download example file: {e}")
        # Fallback: generate a synthetic signal (sine sweep)
        sr = 22050
        T = 5.0
        t = np.linspace(0, T, int(T*sr), endpoint=False)
        # Chirp signal: frequency increases from 220Hz to 880Hz
        y = 0.5 * np.sin(2 * np.pi * 220 * t * (1 + t/T))
        print("Generated synthetic chirp signal instead.")

    print(f"Audio duration: {len(y)/sr:.2f} seconds")
    print(f"Sampling rate: {sr} Hz")

    # Set up the figure for plotting
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)

    # 2. Waveform (Time Domain)
    # This shows amplitude over time.
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set(title='1. Waveform (Time Domain)', xlabel='Time (s)', ylabel='Amplitude')

    # 3. Linear Spectrogram (STFT)
    # We convert time-domain signal to frequency-domain using Short-Time Fourier Transform (STFT).
    # n_fft: Window size for FFT (typically 2048 for speech/music)
    # hop_length: Number of samples between successive frames (typically 512)
    n_fft = 2048
    hop_length = 512
    
    # D is a complex matrix containing magnitude and phase
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    # We care about magnitude (amplitude) for the spectrogram
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    img1 = librosa.display.specshow(S_db, y_axis='linear', x_axis='time', 
                                   sr=sr, n_fft=n_fft, hop_length=hop_length, ax=ax[1])
    ax[1].set(title='2. Linear Frequency Spectrogram (STFT)', ylabel='Frequency (Hz)')
    fig.colorbar(img1, ax=ax[1], format="%+2.0f dB")

    # 4. Mel Spectrogram
    # Humans don't perceive frequency linearly. We are more sensitive to low frequencies.
    # The Mel scale transforms frequency Hz to Mels to mimic human hearing.
    # n_mels: Number of Mel bands (filters) - typically 80 or 128 for LLMs like Qwen/Whisper.
    n_mels = 80
    
    # Option A: Create Mel Spectrogram directly from audio (convenience function)
    S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    
    # Convert to log scale (dB). Audio power is logarithmic (like our hearing volume).
    # This is effectively "Log-Mel Spectrogram"
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

    img2 = librosa.display.specshow(S_mel_db, x_axis='time', y_axis='mel', 
                                   sr=sr, fmax=8000, ax=ax[2])
    ax[2].set(title=f'3. Mel Spectrogram ({n_mels} bands)', ylabel='Frequency (Mel)')
    fig.colorbar(img2, ax=ax[2], format="%+2.0f dB")

    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), 'mel_spectrogram_tutorial.png')
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    create_tutorial_plots()



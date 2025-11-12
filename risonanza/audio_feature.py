import librosa, os, requests
import numpy as np 
from urllib.parse import urlparse

#Initialize parameters
SAMPLE_RATE = 22050  # Standard sample rate for audio processing
HOP_LENGTH = 512     # Number of samples between successive frames
N_MELS = 128        # Number of mel frequency bins
N_FFT = 2048        # Number of FFT components
WINDOW_SIZE = 1024  # Window size for STFT

FEATURE_NAMES = ['MFCC_Mean', 'MFCC_Delta', 'MFCC_Delta+Delta', 'Pitch_FFT', 'Spectral_Centroid', 'Spectral_Rolloff', 'Spectral_Bandwidth', 'RMS']

#testing frontend 
def select_audio_file():
    """open file dialog to select audio file """
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[
            ("Audio files", "*.wav *.mp3 *.flac *.m4a *.ogg"),
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("All files", "*.*")
        ]
    )
    return file_path
def load_audio(file_path):
    """Load audio file and resample to standard rate"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    print(f"Loaded audio: {os.path.basename(file_path)}")
    print(f"Duration: {len(y)/SAMPLE_RATE:.2f} seconds")
    print(f"Sample rate: {sr} Hz")
    return y, sr

def download_audio_from_url(url, save_path=None):
    """Download audio file from URL"""
    try:
        # If no save path provided, create one from URL
        if save_path is None:
            filename = os.path.basename(urlparse(url).path)
            if not filename or '.' not in filename:
                filename = "downloaded_audio.wav"
            save_path = filename
        
        print(f"Downloading audio from: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Audio downloaded successfully to: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

def load_audio_from_url(url):
    """Load audio directly from URL (downloads temporarily)"""
    try:
        # Download to temporary file
        temp_file = download_audio_from_url(url)
        if temp_file:
            # Load the audio
            audio, sr = load_audio(temp_file)
            # Clean up temporary file
            os.remove(temp_file)
            return audio, sr
        return None, None
    except Exception as e:
        print(f"Error loading audio from URL: {e}")
        return None, None

#audio feature extraction
def extract_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=13):
    """Extract MFCC features from audio"""
    mfcc = librosa.feature.mfcc(
        y=audio, 
        sr=sr, 
        n_mfcc=n_mfcc
    )
    return mfcc

def extract_spectral_features(audio, sr=SAMPLE_RATE):
    """Extract various spectral features"""
    # Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(
        y=audio, sr=sr
    )
    
    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr
    )
    
    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr
    )
    
    return np.mean(spectral_centroids), np.mean(spectral_rolloff), np.mean(spectral_bandwidth)

def extract_delta_features(mfcc):
    """Extract delta and delta-delta features from MFCC"""
    # Delta (1st derivative)
    mfcc_delta = librosa.feature.delta(mfcc)
    
    # Delta-Delta (2nd derivative)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    return np.mean(mfcc_delta.T, axis=0), np.mean(mfcc_delta2.T, axis=0)

def extract_pitch_fft(audio, sr=SAMPLE_RATE):
    pitches, _ = librosa.piptrack(y=audio, sr=sr)
    return np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

# audio augmentation functions

def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def pitch_shift(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def time_stretch(y, rate=0.9):
    return librosa.effects.time_stretch(y, rate=rate)

def change_volume(y, factor=1.5):
    return y * factor

def time_shift(y, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)
    
#feature Extraction function 

def extract_features_from_signal(y, sr, n_mfcc=40):
    # MFCCs
    mfccs = extract_mfcc(y, sr, n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Energy
    rms = np.mean(librosa.feature.rms(y=y))

    return np.hstack([
        mfccs_mean,
        *extract_delta_features(mfccs),
        extract_pitch_fft(y, sr),
        *extract_spectral_features(y, sr),
        rms
    ])

#example usage
if __name__ == "__main__":
    from tkinter import filedialog, Tk
    print("Librosa audio processing module initialized")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Hop length: {HOP_LENGTH} samples")
    print(f"Mel bins: {N_MELS}")
    print(f"FFT size: {N_FFT}")
    
    # Method 1: Select audio file interactively
    print("\n=== Loading Audio File ===")
    audio_file = select_audio_file()
    if audio_file:
        # Load the audio
        audio, sr = load_audio(audio_file)
        
        # Extract features
        print("\n=== Extracting Features ===")
        mfcc_features = extract_mfcc(audio, sr)
        spectral_features = extract_spectral_features(audio, sr)
        
        print(f"MFCC shape: {mfcc_features.shape}")
    else:
        print("No file selected")     


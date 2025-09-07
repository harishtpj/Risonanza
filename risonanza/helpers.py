# Helper functions for the risonanza module
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa

from . import audio_feature as af

EMOTION_MAP = dict(enumerate(
    ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise'], 
    start=1
))

def load_dataset(dataset_path):
    rootdir = Path(dataset_path)
    file_emotion = []
    file_path = []

    for actor_dir in rootdir.iterdir():
        for file in actor_dir.iterdir():
            spec = file.stem.split('-')
            file_emotion.append(EMOTION_MAP[int(spec[2])])
            file_path.append(file.absolute())

    dataset = pd.DataFrame({
        "Emotions": file_emotion,
        "Path": file_path
    })

    return dataset

def build_features(dataset, augment=True):
    X, y = [], []

    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Extracting Features"):
        y_audio, sr = librosa.load(row["Path"], duration=3, offset=.5)

        signals = [y_audio]
        if augment:
            signals += [
                    af.add_noise(y_audio),
                    af.pitch_shift(y_audio, sr, n_steps=2),
                    af.time_stretch(y_audio, 0.9),
                    af.change_volume(y_audio, 1.2),
                    af.time_shift(y_audio)
            ]

        for sig in signals:
            features = af.extract_features_from_signal(sig, sr, n_mfcc=40)
            X.append(features)
            y.append(row["Emotions"])

    return np.array(X), np.array(y)


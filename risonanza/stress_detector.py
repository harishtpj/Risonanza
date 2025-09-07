# The Voice Emotion and Stress detector ML model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib, librosa

from . import helpers
from . import audio_feature as af

class StressDetectorModel:
    def __init__(self, n_mfcc=40, verbose=0):
        self.n_mfcc = n_mfcc
        self.classifier = RandomForestClassifier(
                n_estimators=800,
                random_state=42,
                max_depth=20,
                class_weight='balanced',
                verbose=verbose
        )
        self.label_encoder = None
        self.scaler = None
        self.model = None

    def train(self, X, y, test_size=.2):
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=test_size, random_state=42
        )

        print("Training model...")
        self.model = self.classifier.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        print("Trained model!")

        train_score = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {train_score*100:.2f}%")

        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        return train_score

    def predict(self, fpath):
        if not self.model or not self.label_encoder or not self.scaler:
            raise ValueError("Model not Trained yet. Have you loaded any models?")

        y, sr = librosa.load(fpath, duration=3, offset=0.5)

        features = af.extract_features_from_signal(y, sr, n_mfcc=self.n_mfcc).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        pred = self.model.predict(features_scaled)
        return self.label_encoder.inverse_transform(pred)[0]

    def save(self, model_path="voice_emotion_model.pkl"):
        joblib.dump({
            "model": self.model,
            "label_encoder": self.label_encoder,
            "scaler": self.scaler,
            "n_mfcc": self.n_mfcc
        }, model_path)
        print(f"Model successfully saved to: {model_path}")

    def load(self, model_path="voice_emotion_model.pkl"):
        data = joblib.load(model_path)
        self.model = data["model"]
        self.label_encoder = data["label_encoder"]
        self.scaler = data["scaler"]
        self.n_mfcc = data["n_mfcc"]
        print(f"Model loaded from {model_path}")



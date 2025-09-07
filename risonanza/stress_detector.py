# The Voice Emotion and Stress detector ML model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

from . import helpers

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


# Helper script to download the sample dataset for training the ML model.
# The following dataset is used: RAVDESS
import os, shutil, zipfile, requests

URL = "https://www.kaggle.com/api/v1/datasets/download/uwrfkaggler/ravdess-emotional-speech-audio"
FINAL_DIR = "ravdess_dataset"
CHUNK_SIZE = 8192

print("Downloading RAVDESS dataset...")
resp = requests.get(URL, stream=True)
resp.raise_for_status()

with open("ravdess.zip", 'wb') as f:
    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
        f.write(chunk)

print("Extracting dataset...")
with zipfile.ZipFile("ravdess.zip", 'r') as zf:
    zf.extractall(FINAL_DIR)

# Delete the inner folder(duplicate for RAVDESS)
dup_folder = os.path.join(FINAL_DIR, "audio_speech_actors_01-24")
if os.path.exists(dup_folder):
    print("Cleaning duplicate folder")
    shutil.rmtree(dup_folder)

os.remove("ravdess.zip")
print("RAVDESS dataset downloaded successfully")



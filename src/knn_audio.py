"""knn_features.py"""

import os
import requests
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier

from src.include.get_features import get_features

URL = "https://ocw.cs.pub.ro/courses/_media/ps/data.mat"
DATA_DIR = "../data/"
DATA_PATH = os.path.join(DATA_DIR, "data.mat")


def download_data():
    # Ensure the data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download the file
    response = requests.get(URL)
    if response.status_code == 200:
        with open(DATA_PATH, "wb") as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved to {DATA_PATH}")
    else:
        print(f"Failed to download the file. HTTP status code: {response.status_code}")
        return False

    return True


def main():
    if not os.path.exists(DATA_PATH):
        print("Downloading dataset...")
        if not download_data():
            return
    else:
        print(f"Dataset already exists at {DATA_PATH}")

    try:
        data = loadmat(DATA_PATH)
        audio_train = data['audio_train'].T
        audio_test = data['audio_test'].T
        labels_train = data['labels_train']
        labels_test = data['labels_test']
        fs = data['fs'][0, 0]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Use a fraction of the audio data for training and testing
    alpha = 1.0  # Fraction of the audio used
    start1 = audio_train.shape[1] // 2 - int(alpha * audio_train.shape[1] // 2) + 1
    end1 = audio_train.shape[1] // 2 + int(alpha * audio_train.shape[1] // 2)
    audio_train_small = audio_train[:, start1:end1]

    start2 = audio_test.shape[1] // 2 - int(alpha * audio_test.shape[1] // 2) + 1
    end2 = audio_test.shape[1] // 2 + int(alpha * audio_test.shape[1] // 2)
    audio_test_small = audio_test[:, start2:end2]

    # Extract features from the audio data
    print("Extracting features...")
    feat_train = get_features(audio_train_small, fs)
    feat_test = get_features(audio_test_small, fs)

    # Flatten the label arrays for compatibility
    labels_train = labels_train[:, 0]
    labels_test = labels_test[:, 0]

    # Train a K-Nearest Neighbors classifier
    print("Training KNN classifier...")
    clf = KNeighborsClassifier()
    clf.fit(feat_train, labels_train)

    # Make predictions
    print("Making predictions...")
    pred_train = clf.predict(feat_train)
    pred_test = clf.predict(feat_test)

    # Calculate and print accuracy
    acc_train = np.mean(pred_train == labels_train)
    acc_test = np.mean(pred_test == labels_test)
    print(f'Accuracy on train: {acc_train:.2f}')
    print(f'Accuracy on test:  {acc_test:.2f}')


if __name__ == "__main__":
    main()

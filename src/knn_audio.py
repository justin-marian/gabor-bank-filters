"""knn_features.py"""

import scipy
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from src.include.get_features import get_features


def main():
    data = scipy.io.loadmat('../data/data.mat')
    audio_train, audio_test = data['audio_train'].T, data['audio_test'].T
    labels_train, labels_test = data['labels_train'], data['labels_test']
    fs = data['fs'][0, 0]

    alpha = 1.0  # Fraction of the audio used
    start1 = audio_train.shape[1] // 2 - int(alpha * audio_train.shape[1] // 2) + 1
    end1 = audio_train.shape[1] // 2 + int(alpha * audio_train.shape[1] // 2)
    audio_train_small = audio_train[:, start1:end1]

    start2 = audio_test.shape[1] // 2 - int(alpha * audio_test.shape[1] // 2) + 1
    end2 = audio_test.shape[1] // 2 + int(alpha * audio_test.shape[1] // 2)
    audio_test_small = audio_test[:, start2:end2]

    # The dimensions of the data should be:
    # audio_train_small: [D1, N]
    # audio_test_small: [D2, N]
    # labels_train: [D1, 1]
    # labels_test: [D2, 1]

    feat_train = get_features(audio_train_small, fs)
    feat_test = get_features(audio_test_small, fs)

    # Flatten the label arrays
    labels_train = labels_train[:, 0]
    labels_test = labels_test[:, 0]

    clf = KNeighborsClassifier()
    clf.fit(feat_train, labels_train)

    pred_train = clf.predict(feat_train)
    pred_test = clf.predict(feat_test)

    acc_train = np.mean(pred_train == labels_train)
    acc_test = np.mean(pred_test == labels_test)
    print(f'Accuracy on train: {acc_train:.2f}')
    print(f'Accuracy on test:  {acc_test:.2f}')


if __name__ == "__main__":
    main()

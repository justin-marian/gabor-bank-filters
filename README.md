# Sound Classifier

<div align="center">
  <img src="./images/Gabor%20Spectrum.svg" alt="GABOR_BANK_FILTERS" width="80%">
</div>

1. **Perceptual Frequency Analysis**:
   - Utilizes the Mel scale for filter design, emphasizing frequencies relevant to human auditory perception.
   - Gabor filters provide time-frequency localization, critical for capturing audio features effectively.

2. **Efficient Feature Extraction**:
   - Audio signals are segmented into overlapping windows.
   - Each window is processed with a Gabor filter bank to extract mean and standard deviation responses.
   - The resulting feature vector captures rich time-frequency information.

<div align="center" style="display: flex; justify-content: center; gap: 20px;">
  <img src="./images/Gabor%Cos.svg" alt="GABOR_COS" width="45%">
  <img src="./images/Gabor%Sine.svg" alt="GABOR_SINE" width="45%">
</div>

1. **Simple yet Effective Classifier**:
   - A KNN classifier is trained using the extracted features.
   - Designed for easy evaluation and modification.

<div align="center">
  <img src="./images/Mel%20to%20Hz%20Mapping.svg" alt="MEL_TO_HZ_MAPPING" width="60%">
</div>

## Metrics

| Metric                   | Value           |
|--------------------------|-----------------|
| **Training Accuracy**    | 77%             |
| **Testing Accuracy**     | 67%             |

Performance aligns with the expected range of **55% - 68%** for the test set.

## Structure

- `data` (*Download the [data.mat](https://ocw.cs.pub.ro/courses/_media/ps/data.mat) file and add it in data folder*)
- `images` (*Folder in which the images are saved*)

- `include`
  - **[1] `gabor_filters.py`** (*Generates Gabor filters based on the Mel scale*):
    - `gabor_filter`: Generates a Gabor filter (cosine and sine components).
    - `gabor_filter_bank`: Creates a bank of Gabor filters across a range of Mel-scale frequencies.
    - `gaussian_filter`: Creates Gaussian filters for comparison.
    - `from_hz_to_mel`, `from_mel_to_hz`: Converts between Hertz and Mel scale.

  - **[2] `get_features.py`** (*Core features extractor*):
    - Segments audio into overlapping windows.
    - Applies the Gabor filter bank to extract features (mean and standard deviation for each filter response).
    - Outputs a feature vector representing the signal.

  - **[3] `plots.py`** (*Plotting tools*):
    - Spectrum for Gaussian and Gabor filters.
    - Mel scale versus normal frequency mapping.
    - Time-frequency representations.

- `knn_audio.py`
  - Loads audio data.
  - Extracts features using `get_features`.
  - Trains and evaluates a KNN classifier.

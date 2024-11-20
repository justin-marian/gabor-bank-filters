# Sound classifier

## Key Features

1. **Perceptual Frequency Analysis**:
   - Utilizes the Mel scale for filter design, emphasizing frequencies relevant to human auditory perception.
   - Gabor filters provide time-frequency localization, critical for capturing audio features effectively.

2. **Efficient Feature Extraction**:
   - Audio signals are segmented into overlapping windows.
   - Each window is processed with a Gabor filter bank to extract mean and standard deviation responses.
   - The resulting feature vector captures rich time-frequency information.

3. **Simple yet Effective Classifier**:
   - A KNN classifier is trained using the extracted features.
   - Designed for easy evaluation and modification.

## Structure

- `data` (*Download the [data.mat](https://ocw.cs.pub.ro/courses/_media/ps/data.mat) file and add it in data folder*)

- `include`
  - **[1] `gabor_filters.py`** (*Generates Gabor filters based on the Mel scale*)
    - `gabor_filter`: Generates a Gabor filter (cosine and sine components).
    - `gabor_filter_bank`: Creates a bank of Gabor filters across a range of Mel-scale frequencies.
    - `gaussian_filter`: Creates Gaussian filters for comparison.
    - `from_hz_to_mel`, `from_mel_to_hz`: Converts between Hertz and Mel scale.

  - **[2] `get_features.py`** (*Core features extracter*)
    - Segments audio into overlapping windows.
    - Applies the Gabor filter bank to extract features (mean and standard deviation for each filter response).
    - Outputs a feature vector representing the signal.

  - **[3] `plots.py`** (*Ploting tools*)
    - Spectrum for Gaussian and Gabor filters.
    - Mel scale versus normal frequency mapping.
    - Time-frequency representations.

- **[4] `knn_audio.py`** (*Main Script*)
  - Loads audio data.
  - Extracts features using `get_features`.
  - Trains and evaluates a KNN classifier.

### Gabor Filter Bank

[GABOR_BANK_FILTERS](./images/Gabor%20Spectrum.png)

- Filters are centered on frequencies distributed across the Mel scale.
  1. Divide the Mel scale into equal segments between a minimum (`A`) and maximum (`B`) frequency.
  2. Convert Mel values to Hertz for the filter's center frequency and bandwidth.
  3. Generate Gabor filters with cosine (real) and sine (imaginary) components for time-frequency representation.

### Feature Extraction

- Audio signals are segmented into overlapping windows of size `K` with a step of **12 ms**.
- Each window is convolved with the Gabor filter bank.
- Feature vector construction:
  - Compute **mean** and **standard deviation** of filter responses for each window.
  - Concatenate these statistics across all filters, resulting in a fixed-size vector (2M features per window).

### Classification

- A **KNN classifier** is trained on the extracted features.
- Evaluates performance on training and testing datasets.

## Metrics

| Metric                   | Value           |
|--------------------------|-----------------|
| **Training Accuracy**    | 77%             |
| **Testing Accuracy**     | 67%             |

Performance aligns with the expected range of **55% - 68%** for the test set.

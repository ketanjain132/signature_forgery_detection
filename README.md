
# Signature Forgery Detection

This project aims to detect forged signatures using image processing and machine learning techniques. The code preprocesses images, extracts features, and uses a neural network to classify signatures as genuine or forged.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Feature Extraction](#feature-extraction)
- [Model Training and Testing](#model-training-and-testing)
- [Evaluation](#evaluation)
- [References](#references)

## Introduction

Signature verification is a crucial task in many financial and legal transactions. This project implements a system to automatically detect forged signatures using image preprocessing, feature extraction, and a neural network classifier.

## Dataset

The dataset consists of genuine and forged signature images stored in separate directories. Each image is preprocessed and features are extracted to train and test the neural network model.

## Installation

To run this project, you'll need Python and the following libraries:

- NumPy
- SciPy
- scikit-image
- TensorFlow
- Keras
- Pandas
- Matplotlib

You can install these dependencies using `pip`:

```bash
pip install numpy scipy scikit-image tensorflow keras pandas matplotlib
```

## Usage

1. **Preprocess Images**: Convert images to grayscale and binary, and crop to the signature region.
2. **Extract Features**: Compute features such as ratio, centroid, eccentricity, solidity, skewness, and kurtosis.
3. **Train and Test Model**: Use the extracted features to train and test a neural network model.

### Example

To preprocess an image and extract features:

```python
from preprocess import preproc, getCSVFeatures

# Path to a signature image
path = "path/to/signature.png"

# Preprocess image and extract features
features = getCSVFeatures(path)
print(features)
```

To train and test the model:

```python
from model import trainAndTest

# Train and test the model
train_avg, test_avg, time_taken = trainAndTest(rate=0.001, epochs=1000, neurons=7, display=True)
print(f"Training Average: {train_avg}")
print(f"Testing Average: {test_avg}")
print(f"Time Taken: {time_taken}")
```

## Feature Extraction

The feature extraction process includes:

1. **Ratio**: The ratio of the number of white pixels to the total number of pixels.
2. **Centroid**: The center of mass of the signature.
3. **Eccentricity and Solidity**: Calculated using region properties.
4. **Skewness and Kurtosis**: Measures of the asymmetry and peakedness of the pixel distribution.

## Model Training and Testing

The neural network is trained using the extracted features. The architecture consists of three hidden layers with varying numbers of neurons. The model is optimized using the Adam optimizer and evaluated using accuracy metrics.

## Evaluation

The model's performance is evaluated based on its accuracy on both training and testing datasets. The accuracy is calculated by comparing the predicted labels with the true labels.

## References

For more details, refer to the research paper:

- [Signature Forgery Detection: A Comprehensive Approach](path/to/paper)

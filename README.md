# Grammar Scoring from Audio using Rich Acoustic Features

This project is part of the SHL Intern Hiring Assessment on Kaggle. The objective is to predict grammar scores (Likert scale from 0 to 5) for spoken audio samples using machine learning techniques. The evaluation metric is **Pearson Correlation Coefficient** between predicted and actual scores.

## 📁 Dataset
- **train.csv**: Contains audio filenames and corresponding grammar labels.
- **test.csv**: Contains only audio filenames for prediction.
- **Audio files** are stored in `train_audio_folder/` and `test_audio_folder/`.

## 🚀 Pipeline Overview

### ✅ Step 1: Feature Extraction
We extract rich audio features using `librosa` for every `.wav` file:
- Trim silence
- Apply pre-emphasis filtering
- Extract the following features:
  - **MFCC** (mean & std)
  - **Chroma**
  - **Spectral Contrast**
  - **Tonnetz**
  - **Zero Crossing Rate**
  - **RMS Energy**

### ✅ Step 2: Normalize Features
Standardization is performed using `StandardScaler` to normalize training and testing data.

### ✅ Step 3: Model Training
We use **Random Forest Regressor** from `sklearn` and perform:
- 5-Fold Cross Validation
- Pearson Correlation computation for validation scores

Average CV Pearson score obtained: **0.623**

### ✅ Step 4: Inference and Submission
- Final model is used to generate predictions on the test set.
- Results are saved to `submission_new.csv`.

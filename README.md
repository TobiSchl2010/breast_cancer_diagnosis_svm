# Breast Cancer SVM Classifier
*A Support Vector Machine model for classifying tumors as malignant or benign using the Breast Cancer Wisconsin dataset.*

---

## 📘 Overview
This project demonstrates a complete machine learning pipeline using an **SVM classifier** on the **Breast Cancer Wisconsin dataset** from scikit-learn.

It includes data loading, preprocessing, model training, evaluation, and prediction, following a clean modular structure suitable for GitHub and reproducible research.

---

## 🗂️ Project Structure
breast_cancer_svm/
├── data/ # raw and processed data
├── notebooks/ # exploratory notebooks
├── src/ # core Python modules
│ ├── data/ # data loading
│ ├── models/ # training, evaluation, prediction
│ └── utils/ # helper functions
├── models/ # saved trained models
└── tests/ # test scripts

---

## ⚙️ Installation

### Using uv
```bash
uv init breast_cancer_svm
cd breast_cancer_svm
uv add numpy pandas scikit-learn matplotlib seaborn joblib
uv sync

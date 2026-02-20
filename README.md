# <Breast Cancer Classifier>

## Badges

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-completed-brightgreen)

## Description

This project explores a complete **machine learning pipeline** using an **SVM classifier** on the **Breast Cancer Wisconsin dataset** from scikit-learn. 

The main goal is to investigate how different models, preprocessing steps, and hyperparameter tuning affect predictive performance, while producing reproducible results and clear visualizations.

It covers **data loading, preprocessing, feature analysis, model training, evaluation, and prediction**, all organized in a modular, GitHub-ready structure.

The final model, a well-tuned **LinearSVC**, predicts whether tumors are malignant or benign, demonstrating how machine learning can support early detection and medical decision-making.

Throughout this project, I refined my skills in **Python, scikit-learn, and project structuring**, and learned to produce professional exploration notebooks with detailed markdown documentation.

## Features

This dataset contains **30 numerical features** extracted from fine needle aspirate (FNA) images of breast cell nuclei. The features are grouped into three categories: **mean**, **standard error (se)**, and **worst** (largest value). Each category has 10 measurements:

1. **radius** – Average distance from the nucleus center to the perimeter.
2. **texture** – Variation in gray-scale values within the nucleus (surface texture).
3. **perimeter** – The perimeter of the nucleus.
4. **area** – The area of the nucleus.
5. **smoothness** – Smoothness of the nucleus boundary (local variation in radius length).
6. **compactness** – Compactness of the nucleus, calculated as (perimeter² / area – 1).
7. **concavity** – Severity of concave portions of the nucleus contour.
8. **concave points** – Number of concave points in the nucleus contour.
9. **symmetry** – Symmetry of the nucleus.
10. **fractal dimension** – “Coastline” complexity of the nucleus contour.

Each feature is recorded in three forms:
- **_mean** – mean value across all cells in the image  
- **_se** – standard error of the measurement  
- **_worst** – largest value observed (“worst case”)

This results in a total of **30 features** (10 features × 3 types).

These features are used to classify tumors as **benign** or **malignant**.

## Project Structure

breast_cancer_diagnosis_project/
├── README.md
├── pyproject.toml
├── src/
│ ├── data/
│ ├── models/
│ └── utils/
├── notebooks/
├── outputs/
│ └── plots/
└── models/

## Installation & Usage


#1. Clone the repository
```bash
git clone https://github.com/TobiSchl2010/breast_cancer_diagnosis_svm.git
cd breast_cancer_diagnosis_svm ```

#2. Install dependencies (from pyproject.toml)
poetry install 

# 3. Activate the virtual environment (optional)
poetry shell 

# 4. Verify installation (optional)
python -c "import sklearn, numpy, matplotlib, joblib"
```

--- Usage via Notebook ---

Open the exploration notebook in Jupyter
```bash
jupyter notebook notebooks/exploration.ipynb```
```

Or in VSCode, open notebooks/exploration.ipynb

Notebook workflow:
  - Data Loading & inspection
  - Exploratory Data Analysis: stats, outliers, missing values, feature plots, correlation,    mean values by class
  - Train-Test Split
  - Baseline Model Comparison: 5-fold CV with F1-macro
  - Learning Curves for top models
  - Hyperparameter Tuning via GridSearchCV
  - Final Model Selection & Hyperparameters
  - Decision Threshold Adjustment: optimize malignant recall vs precision
  - Evaluation: classification report, confusion matrix, Precision–Recall curve
  - Feature Importance Analysis

All plots are saved to outputs/plots/

--- Usage via Scripts ---

Train a model
```bash
python src/models/train_svm.py
```
Trains the LinearSVC with preprocessing and threshold adjustments
Saves the model to models/

Evaluate a model
```bash
python src/models/evaluate_svm.py
```
Loads the trained model, evaluates on test set, displays confusion matrix

# Predict a new sample
```bash
python src/models/predict_svm.py

Enter sample features (comma-separated) []
Output: Benign or Malignant
```

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

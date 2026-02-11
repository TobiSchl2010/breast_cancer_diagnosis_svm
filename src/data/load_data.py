from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
1️⃣ Load Data
- Retrieve raw data from files, databases, APIs, or sklearn datasets.
- Optionally decide whether to load the data as a pandas DataFrame or a numpy array.
- Goal: make the data immediately available for the next steps.

2️⃣ Feature and Target Separation
- Separate input features (X) from the target variable (y).
- Clarifies the data structure: what is input vs. what is output.

3️⃣ Optional: Initial Cleaning / Transformation
- Check for missing values, inconsistent data types, and outliers.
- Perform basic encoding (e.g., converting categorical data to numeric) or prepare for it.
- Goal: ensure the data is consistent and ready for machine learning models.

4️⃣ Data Splitting
- Split the data into training and test sets, optionally including a validation set.
- Use parameters like test_size and random_state to make the split reproducible.
- Goal: provide a clean foundation for model training and evaluation.

5️⃣ Optional: Prepare Feature Scaling / Transformations
- Prepare standardization, normalization, or other transformations.
- Usually, no final scaling is applied here; just fit transformers for pipelines if needed.

6️⃣ Return
- Return the split and prepared data, typically as X_train, X_test, y_train, y_test.
- Goal: the main code can immediately proceed with training, model selection, and evaluation without worrying about preprocessing.
"""

# def load_and_prepare_data(test_size = 0.2, random_state = 42):

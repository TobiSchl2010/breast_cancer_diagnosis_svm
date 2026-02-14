from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import os

from .project_paths import get_project_root


def learning_curve_plot(
    model,
    model_name,
    X_train,
    y_train,
    n_splits=5,
    train_sizes=np.linspace(0.05, 1.0, 40),
    scoring="f1_macro",
):
    """
    Plots the learning curve for a given machine learning model on the training dataset.

    This function evaluates the performance of a model as the training set size increases.
    It uses stratified k-fold cross-validation to estimate both training and validation scores
    for multiple subsets of the training data. The resulting plot provides insights into:
    - How the model improves as it sees more data.
    - Potential underfitting or overfitting based on the gap between training and validation scores.

    Parameters
    ----------
    model : estimator object
        A scikit-learn compatible model (e.g., LinearSVC, RandomForestClassifier)
        that implements the fit and predict methods.

    X_train : array-like of shape (n_samples, n_features)
        The feature matrix used for training the model.

    y_train : array-like of shape (n_samples,)
        The target vector corresponding to `X_train`.

    n_splits : int, default=5
        Number of folds to use in Stratified K-Fold cross-validation.
        Stratification ensures that each fold maintains the same class proportion as the original dataset.

    train_sizes : array-like, default=np.linspace(0.05, 1.0, 40)
        Relative or absolute numbers of training examples to use for generating the learning curve.
        Each value specifies the fraction or number of samples from `X_train` to include in training subsets.

    scoring : str, default="f1_macro"
        Metric used to evaluate the model performance. Can be any metric supported by scikit-learn
        (e.g., 'accuracy', 'f1_macro', 'roc_auc'). "f1_macro" calculates F1-score per class and averages them.

    Returns
    -------
    None
        This function does not return any object. Instead, it:
        - Generates a learning curve plot showing the mean training and validation scores
          as a function of training set size.
        - Saves the plot as "outputs/plots/learning_curve_svm_rbf.png" with 300 dpi resolution.
        - Displays the plot using matplotlib.

    Notes
    -----
    - The function computes the mean score across all cross-validation folds for both training
      and validation sets at each training size.
    - The training curve (red line) shows how well the model fits the subsets of training data.
    - The validation curve (blue line) shows the generalization performance on unseen data.
    - A large gap between training and validation curves may indicate overfitting,
      while low scores on both curves may indicate underfitting.
    - The y-axis is currently fixed to the range 0.7–1.05 for visualization purposes.

    Example
    -------
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.datasets import load_breast_cancer
    >>> data = load_breast_cancer()
    >>> X, y = data.data, data.target
    >>> model = LinearSVC(max_iter=5000)
    >>> learning_curve_plot(model, X, y)
    """

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_sizes, train_scores, valid_scores = learning_curve(
        model, X_train, y_train, train_sizes=train_sizes, cv=cv, scoring=scoring
    )

    train_mean_scores = train_scores.mean(axis=1)
    valid_mean_scores = valid_scores.mean(axis=1)

    plt.plot(train_sizes, train_mean_scores, "r-+", linewidth=2, label="Training score")
    plt.plot(
        train_sizes, valid_mean_scores, "b-", linewidth=3, label="Validation score"
    )

    plt.xlabel("Training set size")
    plt.ylabel("Score")
    plt.grid()
    plt.legend(loc="upper right")
    plt.ylim(0.7, 1.05)
    plt.title(f"Learning Curve ({model_name} on Breast Cancer Dataset)")
    project_path = get_project_root()
    os.chdir(project_path)
    plt.savefig(f"outputs/plots/learning_curve_{model_name}.png", dpi=300)

    plt.show()

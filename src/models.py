import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def get_models():
    return {
        "kNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 9, 11, 15, 21]},
        ),
        "GaussianNB": (
            GaussianNB(),
            {"var_smoothing": np.logspace(-11, -7, 5)},
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=5000, random_state=42),
            {"C": [0.01, 0.1, 1, 10, 100]},
        ),
        "LDA": (
            LinearDiscriminantAnalysis(),
            {"solver": ["svd", "lsqr", "eigen"]},
        ),
        "QDA": (
            QuadraticDiscriminantAnalysis(),
            {"reg_param": [0.0, 0.1, 0.3, 0.5, 0.7]},
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [3, 5, 7, 10, 15, None]},
        ),
        "SVM": (
            SVC(probability=True, random_state=42),
            {"C": [0.01, 0.1, 1, 10, 100]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {"max_depth": [5, 10, 15, None], "n_estimators": [50, 100, 200]},
        ),
    }

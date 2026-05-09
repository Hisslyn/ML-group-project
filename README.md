# Heart Disease Binary Classification

Binary classification project comparing 8 scikit-learn classifiers on the UCI Heart Disease dataset to predict the presence or absence of heart disease.

## Dataset

**File:** `heart_disease_uci.csv`

- **Shape:** 920 rows × 16 columns
- **Target column:** `num` — binarized to 0 (no disease) vs 1 (disease, original values 1–4)
- **Class balance:** 411 (44.67%) no disease, 509 (55.33%) disease

**Features (15, excluding target):**

| Column | Type | Notes |
|---|---|---|
| age | numeric | |
| sex | categorical | |
| dataset | categorical | source clinic (Cleveland, Hungary, Switzerland, VA Long Beach) |
| cp | categorical | chest pain type |
| trestbps | numeric | resting blood pressure |
| chol | numeric | serum cholesterol |
| fbs | categorical | fasting blood sugar |
| restecg | categorical | resting ECG results |
| thalch | numeric | max heart rate achieved |
| exang | categorical | exercise-induced angina |
| oldpeak | numeric | ST depression |
| slope | categorical | slope of peak exercise ST segment |
| ca | numeric | number of major vessels |
| thal | categorical | thalassemia |

**Missing values (columns with >0% missing):**

| Column | Missing count | % |
|---|---|---|
| trestbps | 59 | 6.41% |
| chol | 30 | 3.26% |
| fbs | 90 | 9.78% |
| restecg | 2 | 0.22% |
| thalch | 55 | 5.98% |
| exang | 55 | 5.98% |
| oldpeak | 62 | 6.74% |
| slope | 309 | 33.59% |
| ca | 611 | 66.41% |
| thal | 486 | 52.83% |

## Methodology

### Preprocessing

Applied uniformly to all 8 models via a single fitted `ColumnTransformer`:

- **Numeric features:** `SimpleImputer(strategy='median')` → `StandardScaler()`
- **Categorical features:** `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')`
- **Train/test split:** 80/20 stratified, `random_state=42`
- **Preprocessor fit on training data only** — no leakage from test set

Result: 29 features after one-hot encoding (6 numeric + 23 from 8 categorical columns).

### Models

All implementations from `sklearn`:

| Key | sklearn class |
|---|---|
| kNN | `KNeighborsClassifier` |
| GaussianNB | `GaussianNB` |
| LogisticRegression | `LogisticRegression(max_iter=5000, random_state=42)` |
| LDA | `LinearDiscriminantAnalysis` |
| QDA | `QuadraticDiscriminantAnalysis` |
| DecisionTree | `DecisionTreeClassifier(random_state=42)` |
| SVM | `SVC(probability=True, random_state=42)` |
| RandomForest | `RandomForestClassifier(random_state=42)` |

### Hyperparameter tuning

`GridSearchCV(cv=5, scoring='roc_auc', n_jobs=-1, refit=True)` on training set only.

| Model | Tuned parameter(s) | Grid |
|---|---|---|
| kNN | `n_neighbors` | [3, 5, 7, 9, 11, 15, 21] |
| GaussianNB | `var_smoothing` | `np.logspace(-11, -7, 5)` |
| LogisticRegression | `C` | [0.01, 0.1, 1, 10, 100] |
| LDA | `solver` | ['svd', 'lsqr', 'eigen'] |
| QDA | `reg_param` | [0.0, 0.1, 0.3, 0.5, 0.7] |
| DecisionTree | `max_depth` | [3, 5, 7, 10, 15, None] |
| SVM | `C` | [0.01, 0.1, 1, 10, 100] |
| RandomForest | `max_depth` × `n_estimators` | [5, 10, 15, None] × [50, 100, 200] |

### Evaluation metrics

Computed on the held-out test set (never seen during training or CV):

- Accuracy
- Precision (binary, `pos_label=1`)
- Recall (binary, `pos_label=1`)
- F1 score (binary, `pos_label=1`)
- Confusion matrix
- ROC curve (via `predict_proba[:, 1]` on the positive class)
- ROC-AUC

## Project structure

```
ML group project/
├── heart_disease_uci.csv
├── requirements.txt
├── README.md
└── src/
│   ├── data_loader.py       # load_data, run_eda
│   ├── preprocessing.py     # binarize_target, build_preprocessor, prepare_data
│   ├── models.py            # get_models — all 8 estimators + grids
│   ├── train.py             # train_all_models (GridSearchCV + joblib save)
│   ├── evaluate.py          # evaluate_all_models, confusion matrix + ROC plots
│   └── main.py              # end-to-end orchestration (single entry point)
└── results/
    ├── metrics/
    │   ├── eda_summary.csv          # column dtypes, missing values, target dist
    │   ├── cv_results.csv           # best params + best CV ROC-AUC per model
    │   └── test_metrics.csv         # accuracy, precision, recall, F1, ROC-AUC per model
    ├── models/
    │   └── *.joblib                 # 8 best-estimator files, one per model
    └── plots/
        ├── cm_*.png                 # 8 confusion matrix heatmaps
        ├── roc_*.png                # 8 individual ROC curve plots
        └── roc_combined.png         # all 8 ROC curves on one figure
```

## How to reproduce

```bash
pip install -r requirements.txt
python src/main.py
```

Runs the full pipeline from scratch: EDA → preprocessing → GridSearchCV tuning → test-set evaluation → all CSV and PNG outputs. Deleting `results/` and rerunning produces identical outputs (fixed `random_state=42` throughout).

## Decisions log

- **Binary classification:** target column `num` mapped to 0 (no disease) vs 1 (disease, original values 1–4) — chosen for cleaner ROC-AUC interpretation with `predict_proba` on the positive class
- **Train/test split:** 80/20 stratified, `random_state=42`
- **Stratified split:** preserves the ~45/55 class balance in both train and test partitions
- **Median imputation for numeric features:** robust to outliers relative to mean imputation
- **Most-frequent (mode) imputation for categorical features:** preserves the dominant category; avoids introducing a spurious new level
- **One-hot encoding for categorical features:** `handle_unknown='ignore'` so unseen categories at test time produce an all-zero row rather than an error
- **StandardScaler on all numeric features:** required by SVM, LR, kNN, LDA, QDA; applied uniformly so the same pipeline serves all 8 models without branching
- **Preprocessor fit on training data only:** prevents test-set leakage; `transform` applied to test set using training statistics
- **GridSearchCV:** `cv=5`, `scoring='roc_auc'`, `n_jobs=-1`, `refit=True` so the best estimator is immediately usable on the test set
- **`random_state=42`** set on all stochastic estimators: LogisticRegression, DecisionTree, SVC, RandomForestClassifier
- **`SVC(probability=True)`:** required to enable `predict_proba` for ROC-AUC computation
- **`LogisticRegression(max_iter=5000)`:** increased from default 100 to ensure convergence across all regularisation strengths in the grid

## Outputs reference

| File | Description |
|---|---|
| `results/metrics/eda_summary.csv` | Column dtypes, missing value counts/%, and target distribution before and after binarization |
| `results/metrics/cv_results.csv` | Best hyperparameters and best 5-fold CV ROC-AUC for each of the 8 models |
| `results/metrics/test_metrics.csv` | Test-set accuracy, precision, recall, F1, ROC-AUC for each of the 8 models, sorted by ROC-AUC |
| `results/models/{name}.joblib` | Fitted best estimator for each model (8 files), loadable with `joblib.load` |
| `results/plots/cm_{name}.png` | Confusion matrix heatmap for each model on the test set (8 files) |
| `results/plots/roc_{name}.png` | Individual ROC curve with AUC in legend for each model (8 files) |
| `results/plots/roc_combined.png` | All 8 ROC curves overlaid on one figure with distinct colors and AUC legend |

## Checklist

- [x] 8 algorithms implemented: kNN, GaussianNB, LogisticRegression, LDA, QDA, DecisionTree, SVM (soft-margin, `C` tuned), RandomForest
- [x] Missing values handled: median imputation (numeric), mode imputation (categorical)
- [x] Feature scaling applied: `StandardScaler` on all numeric features
- [x] Same preprocessing pipeline and same train/test split used for all 8 models
- [x] At least one hyperparameter tuned per algorithm via `GridSearchCV`
- [x] Cross-validation used: 5-fold
- [x] Test metrics saved: accuracy, precision, recall, F1, confusion matrix (`results/metrics/test_metrics.csv`, `results/plots/cm_*.png`)
- [x] ROC curves and ROC-AUC saved: individual PNGs + combined comparison plot (`results/plots/roc_*.png`, `results/plots/roc_combined.png`)
- [x] All metrics in CSV, all plots in PNG

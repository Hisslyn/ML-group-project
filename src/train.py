import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "results", "models")
METRICS_DIR = os.path.join(PROJECT_ROOT, "results", "metrics")


def train_all_models(X_train, y_train, models_dict):
    rows = []
    for name, (estimator, param_grid) in models_dict.items():
        print(f"Tuning {name}...", flush=True)
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X_train, y_train)

        model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(gs.best_estimator_, model_path)

        grid_df = pd.DataFrame({
            "params": [str(p) for p in gs.cv_results_["params"]],
            "mean_test_score": gs.cv_results_["mean_test_score"],
            "std_test_score": gs.cv_results_["std_test_score"],
        }).sort_values("mean_test_score", ascending=False)
        grid_df.to_csv(os.path.join(METRICS_DIR, f"grid_scores_{name}.csv"), index=False)

        rows.append({
            "model": name,
            "best_params": str(gs.best_params_),
            "best_cv_roc_auc": round(gs.best_score_, 6),
        })
        print(f"  best_params={gs.best_params_}  cv_roc_auc={gs.best_score_:.6f}")

    cv_df = pd.DataFrame(rows, columns=["model", "best_params", "best_cv_roc_auc"])
    out_path = os.path.join(METRICS_DIR, "cv_results.csv")
    cv_df.to_csv(out_path, index=False)
    print(f"\nCV results saved to {out_path}")
    return cv_df

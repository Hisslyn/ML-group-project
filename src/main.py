import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data, run_eda
from preprocessing import prepare_data
from models import get_models
from train import train_all_models
import joblib
from evaluate import (
    evaluate_all_models, plot_roc_curve_single, plot_roc_curves_combined,
    MODELS_DIR, METRICS_DIR, PLOTS_DIR,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    for d in [MODELS_DIR, METRICS_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)

    df = load_data(os.path.join(PROJECT_ROOT, "heart_disease_uci.csv"))

    print("\n--- EDA ---")
    summary = run_eda(df)

    eda_rows = []
    for col in df.columns:
        eda_rows.append({
            "column": col,
            "dtype": summary["dtypes"].get(col, ""),
            "missing_count": summary["missing_count"].get(col, 0),
            "missing_pct": summary["missing_pct"].get(col, 0.0),
        })
    eda_rows.append({"column": "--- target before binarization ---", "dtype": "", "missing_count": "", "missing_pct": ""})
    for val, count in summary["target_dist_before"].items():
        eda_rows.append({"column": f"num={val}", "dtype": "count", "missing_count": count, "missing_pct": ""})
    eda_rows.append({"column": "--- target after binarization ---", "dtype": "", "missing_count": "", "missing_pct": ""})
    for val, count in summary["target_dist_after"].items():
        label = "no disease" if val == 0 else "disease"
        eda_rows.append({"column": f"{val} ({label})", "dtype": "count", "missing_count": count, "missing_pct": ""})
    import pandas as _pd
    _pd.DataFrame(eda_rows).to_csv(os.path.join(METRICS_DIR, "eda_summary.csv"), index=False)

    print("\n--- PREPROCESSING ---")
    X_train_t, X_test_t, y_train, y_test, preprocessor, feature_names = prepare_data(df)
    print(f"X_train_t: {X_train_t.shape}  X_test_t: {X_test_t.shape}")
    print(f"Train class dist: {dict(y_train.value_counts().sort_index())}")
    print(f"Test  class dist: {dict(y_test.value_counts().sort_index())}")

    print("\n--- TRAINING ---")
    models_dict = get_models()
    cv_df = train_all_models(X_train_t, y_train, models_dict)

    print("\n--- CV RESULTS SUMMARY ---")
    print(cv_df.to_string(index=False))

    print("\n--- EVALUATION ON TEST SET ---")
    test_df = evaluate_all_models(MODELS_DIR, X_test_t, y_test)
    test_df_sorted = test_df.sort_values("roc_auc", ascending=False)

    out_path = os.path.join(METRICS_DIR, "test_metrics.csv")
    test_df_sorted.to_csv(out_path, index=False)
    print(test_df_sorted.to_string(index=False))
    print(f"\nTest metrics saved to {out_path}")

    print("\n--- ROC CURVES ---")
    best_models = {
        name: joblib.load(os.path.join(MODELS_DIR, f"{name}.joblib"))
        for name in get_models()
    }
    for model_name, model in best_models.items():
        single_path = os.path.join(PLOTS_DIR, f"roc_{model_name}.png")
        plot_roc_curve_single(model, X_test_t, y_test, model_name, single_path)
        print(f"  Saved {single_path}")

    combined_path = os.path.join(PLOTS_DIR, "roc_combined.png")
    plot_roc_curves_combined(best_models, X_test_t, y_test, combined_path)
    print(f"  Saved {combined_path}")


if __name__ == "__main__":
    main()

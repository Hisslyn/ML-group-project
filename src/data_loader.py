import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df


def run_eda(df):
    print("=" * 60)
    print("SHAPE")
    print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\nDTYPES")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")

    print("\nMISSING VALUES")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    has_missing = False
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]} ({missing_pct[col]}%)")
            has_missing = True
    if not has_missing:
        print("  None")

    print("\nTARGET DISTRIBUTION (before binarization)")
    target_dist = df["num"].value_counts().sort_index()
    for val, count in target_dist.items():
        print(f"  num={val}: {count} ({count / len(df) * 100:.2f}%)")

    binary_series = (df["num"] > 0).astype(int)

    print("\nTARGET DISTRIBUTION (after binarization: 0=no disease, 1=disease)")
    binary_dist = binary_series.value_counts().sort_index()
    for val, count in binary_dist.items():
        label = "no disease" if val == 0 else "disease"
        print(f"  {val} ({label}): {count} ({count / len(df) * 100:.2f}%)")

    summary = {
        "shape_rows": df.shape[0],
        "shape_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_count": missing.to_dict(),
        "missing_pct": missing_pct.to_dict(),
        "target_dist_before": target_dist.to_dict(),
        "target_dist_after": binary_dist.sort_index().to_dict(),
    }

    print("=" * 60)
    return summary


if __name__ == "__main__":
    import os
    import json

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = load_data(os.path.join(project_root, "heart_disease_uci.csv"))
    summary = run_eda(df)

    rows = []
    for col in df.columns:
        rows.append({
            "column": col,
            "dtype": summary["dtypes"].get(col, ""),
            "missing_count": summary["missing_count"].get(col, 0),
            "missing_pct": summary["missing_pct"].get(col, 0.0),
        })

    target_before = summary["target_dist_before"]
    target_after = summary["target_dist_after"]

    summary_rows = rows + [
        {"column": "--- target before binarization ---", "dtype": "", "missing_count": "", "missing_pct": ""},
    ]
    for val, count in target_before.items():
        summary_rows.append({"column": f"num={val}", "dtype": "count", "missing_count": count, "missing_pct": ""})

    summary_rows.append(
        {"column": "--- target after binarization ---", "dtype": "", "missing_count": "", "missing_pct": ""}
    )
    for val, count in target_after.items():
        label = "no disease" if val == 0 else "disease"
        summary_rows.append({"column": f"{val} ({label})", "dtype": "count", "missing_count": count, "missing_pct": ""})

    import pandas as pd
    out_path = os.path.join(project_root, "results", "metrics", "eda_summary.csv")
    pd.DataFrame(summary_rows).to_csv(out_path, index=False)
    print(f"\nEDA summary saved to {out_path}")

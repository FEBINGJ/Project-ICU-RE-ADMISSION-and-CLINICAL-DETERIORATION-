"""Phase 4, part A: build the deterioration feature table.

Run from the project root (after `python -m ml.data.build_labels`):
    python -m ml.features.build_features

Output: data/processed/deterioration_features.parquet
"""
import pandas as pd

from ml.config import DATA_PROCESSED
from ml.data.mimic_demo import find_root
from ml.features.clean import load_labs, load_vitals
from ml.features.windows import build_deterioration_features, feature_groups


def main() -> None:
    path = DATA_PROCESSED / "deterioration_windows.parquet"
    if not path.exists():
        raise SystemExit("Missing data/processed/deterioration_windows.parquet. Run: python -m ml.data.build_labels")
    windows = pd.read_parquet(path)

    root = find_root()
    vitals = load_vitals(root, windows["stay_id"].unique())
    labs = load_labs(root, windows["hadm_id"].unique())
    feats = build_deterioration_features(windows, vitals, labs)
    feats.to_parquet(DATA_PROCESSED / "deterioration_features.parquet", index=False)

    print(f"Feature table: {len(feats)} time points x {feats.shape[1]} columns, "
          f"{int(feats.y.sum())} positives from {feats[feats.y == 1].subject_id.nunique()} patients")
    for name, cols in feature_groups(feats.columns).items():
        print(f"  {name}: {len(cols)} columns")
    print(f"NEWS2 available for {feats.news2.notna().mean():.1%} of time points, "
          f"MEWS for {feats.mews.notna().mean():.1%}")
    miss = feats.drop(columns=["first_careunit"]).isna().mean().sort_values(ascending=False)
    print("Most-missing columns (share of time points):")
    print(miss.head(8).round(3).to_string())
    print("Saved to", DATA_PROCESSED / "deterioration_features.parquet")


if __name__ == "__main__":
    main()

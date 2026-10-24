"""Phase 4: build every feature table and the LSTM arrays.

Run from the project root (after `python -m ml.data.build_labels`):
    python -m ml.features.build_features

Outputs in data/processed/
    deterioration_features.parquet   tabular features, one row per 6-hourly time point
    readmission_features.parquet     tabular features, one row per ICU stay (at ICU discharge)
    deterioration_sequences.npz      LSTM arrays (24 hourly steps) for the same time points
    readmission_sequences.npz        LSTM arrays for the same ICU stays
"""
import numpy as np
import pandas as pd

from ml.config import DATA_PROCESSED
from ml.data.mimic_demo import find_root, load_core
from ml.features.clean import load_labs, load_vitals
from ml.features.readmission import build_readmission_features
from ml.features.sequences import build_sequences
from ml.features.windows import build_deterioration_features, feature_groups


def _report(name: str, feats: pd.DataFrame, label: str, unit: str) -> None:
    pos = feats[feats[label] == 1]
    print(f"\n{name}: {len(feats)} rows x {feats.shape[1]} columns, "
          f"{len(pos)} positive {unit} from {pos.subject_id.nunique()} patients")
    for group, cols in feature_groups(feats.columns).items():
        if cols:
            print(f"  {group}: {len(cols)} columns")
    print(f"  NEWS2 available for {feats.news2.notna().mean():.1%}, MEWS for {feats.mews.notna().mean():.1%}")


def main() -> None:
    det_path = DATA_PROCESSED / "deterioration_windows.parquet"
    rea_path = DATA_PROCESSED / "readmission_index.parquet"
    if not (det_path.exists() and rea_path.exists()):
        raise SystemExit("Missing label tables in data/processed/. Run first: python -m ml.data.build_labels")
    windows, index = pd.read_parquet(det_path), pd.read_parquet(rea_path)

    root = find_root()
    core = load_core(root)
    stays = np.union1d(windows["stay_id"].unique(), index["stay_id"].unique())
    hadms = np.union1d(windows["hadm_id"].unique(), index["hadm_id"].unique())
    vitals, labs = load_vitals(root, stays), load_labs(root, hadms)

    det = build_deterioration_features(windows, vitals, labs)
    det.to_parquet(DATA_PROCESSED / "deterioration_features.parquet", index=False)
    _report("Deterioration features", det, "y", "time points")

    rea = build_readmission_features(index, core["icu"], core["adm"], core["proc"], vitals, labs)
    rea.to_parquet(DATA_PROCESSED / "readmission_features.parquet", index=False)
    _report("Readmission features", rea, "readmit_30d", "stays")

    seq = build_sequences(windows, vitals, labs, time_col="t")
    np.savez_compressed(DATA_PROCESSED / "deterioration_sequences.npz", **seq, y=windows["y"].to_numpy(),
                        subject_id=windows["subject_id"].to_numpy(), stay_id=windows["stay_id"].to_numpy())
    seq_r = build_sequences(index, vitals, labs, time_col="index_time")
    np.savez_compressed(DATA_PROCESSED / "readmission_sequences.npz", **seq_r, y=index["readmit_30d"].to_numpy(),
                        subject_id=index["subject_id"].to_numpy(), stay_id=index["stay_id"].to_numpy())
    print(f"\nSequences: deterioration {seq['X'].shape}, readmission {seq_r['X'].shape} "
          f"(samples, hourly steps, channels); channels = {', '.join(seq['var_names'])}")
    print("Saved to", DATA_PROCESSED)


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from brain_completion.configs import DatasetConfig, ModelConfig
from brain_completion.data import ROICSVLoader, AAL_CUSTOM_LABELS, compute_global_normalization_params
from brain_completion.tester import BrainCompletionTester

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phenotypic_csv", required=True)
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--test_root", required=True)
    ap.add_argument("--atlas_type", default="aal", choices=["aal","schaefer200"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--id_col", default="image_id")
    ap.add_argument("--age_col", default="entry_age")
    ap.add_argument("--sex_col", default="PTGENDER")
    args = ap.parse_args()

    data_cfg = DatasetConfig(
        phenotypic_csv=Path(args.phenotypic_csv),
        roots={"train": Path(args.train_root), "test": Path(args.test_root)},
        atlas_type=args.atlas_type,
        model_dir=Path("models"),
        id_col=args.id_col, age_col=args.age_col, sex_col=args.sex_col,
    )
    model_cfg = ModelConfig()
    loader = ROICSVLoader(data_cfg)
    train_ds, _ = loader.load_dataset("train")
    mean, std = compute_global_normalization_params(train_ds, num_rois=len(loader.custom_labels))

    tester = BrainCompletionTester(model_cfg, data_cfg)
    tester.load_norm(mean, std)
    tester.load_model(Path(args.model_path))
    completed = tester.impute_real_missing()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    col_names = AAL_CUSTOM_LABELS if args.atlas_type == "aal" else [str(x) for x in loader.custom_labels]

    for sid, mat in completed.items():
        pd.DataFrame(mat.T, columns=col_names).to_csv(out_dir / f"{sid}.csv", index=False)

if __name__ == "__main__":
    main()

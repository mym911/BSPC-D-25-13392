from __future__ import annotations
import argparse
from pathlib import Path

from brain_completion.configs import DatasetConfig, ModelConfig
from brain_completion.trainer import BrainCompletionTrainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phenotypic_csv", required=True)
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--test_root", required=True)
    ap.add_argument("--atlas_type", default="aal", choices=["aal","schaefer200"])
    ap.add_argument("--model_dir", default="models")

    ap.add_argument("--id_col", default="image_id")
    ap.add_argument("--age_col", default="entry_age")
    ap.add_argument("--sex_col", default="PTGENDER")

    ap.add_argument("--epochs_pretrain", type=int, default=100)
    ap.add_argument("--epochs_gan", type=int, default=200)
    ap.add_argument("--mask_rate", type=float, default=0.1)
    args = ap.parse_args()

    data_cfg = DatasetConfig(
        phenotypic_csv=Path(args.phenotypic_csv),
        roots={"train": Path(args.train_root), "test": Path(args.test_root)},
        atlas_type=args.atlas_type,
        model_dir=Path(args.model_dir),
        id_col=args.id_col, age_col=args.age_col, sex_col=args.sex_col,
    )
    model_cfg = ModelConfig(epochs_pretrain=args.epochs_pretrain, epochs_gan=args.epochs_gan, mask_rate=args.mask_rate)
    trainer = BrainCompletionTrainer(model_cfg, data_cfg)
    trainer.run_kfold_with_stress()

if __name__ == "__main__":
    main()

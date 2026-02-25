from __future__ import annotations
import argparse
from pathlib import Path

from dac_fedgraph.configs import PathConfig, TrainConfig
from dac_fedgraph.utils import setup_logging
from dac_fedgraph.experiment import run_experiment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phenotypic_csv", required=True)
    ap.add_argument("--ts_dir_aal", required=True)
    ap.add_argument("--ts_dir_sch", required=True)
    ap.add_argument("--out_csv", default="run_summaries.csv")

    ap.add_argument("--pca_components", type=int, default=120)
    ap.add_argument("--init_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--num_rounds", type=int, default=200)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--num_clients", type=int, default=20)
    ap.add_argument("--n_splits", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])

    args = ap.parse_args()
    setup_logging()

    paths = PathConfig(
        phenotypic_csv=Path(args.phenotypic_csv),
        ts_dir_aal=Path(args.ts_dir_aal),
        ts_dir_sch=Path(args.ts_dir_sch),
    )
    cfg = TrainConfig(
        pca_components=args.pca_components,
        init_lr=args.init_lr,
        weight_decay=args.weight_decay,
        k=args.k,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        num_clients=args.num_clients,
        n_splits=args.n_splits,
        seeds=args.seeds,
    )

    df = run_experiment(paths, cfg)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")

if __name__ == "__main__":
    main()

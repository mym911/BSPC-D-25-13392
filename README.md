# brain_completion_refactor

Refactor of the monolithic cGAN imputation script into modules.

## Modules
- `configs.py`: ModelConfig / DatasetConfig (phenotype column mapping supports ABIDE/ADHD)
- `data.py`: CSV loader, dataset, collate, normalization
- `masks.py`: MCAR + structured missingness (mcar/roi_weighted/cluster/block)
- `models.py`: Generator / Discriminator
- `metrics.py`: R² (global denominator, paper-compatible)
- `trainer.py`: training + kfold + stress tests, writes `ab_stress_fold*.csv` and `ab_stress_summary.csv`
- `tester.py`: real-missing imputation using `roi_completion_best.pth`

## Run
Train (kfold + stress tests):
```bash
python scripts/run_train_cv.py --phenotypic_csv ... --train_root ... --test_root ... --atlas_type aal --model_dir models
```

Impute real-missing test set:
```bash
python scripts/run_impute_real.py --phenotypic_csv ... --train_root ... --test_root ... --atlas_type aal --model_path models/roi_completion_best.pth --out_dir out_csv
```

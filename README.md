# DAC-FedGraph (refactored for review)

This package reorganizes the main federated learning script into importable modules for easier review and maintenance.

## Layout
- `dac_fedgraph/configs.py`: path + training hyper-parameters
- `dac_fedgraph/utils.py`: seeding, logging, device helpers
- `dac_fedgraph/metrics.py`: evaluation utilities (AUC/F1 helpers, calibration metrics)
- `dac_fedgraph/models.py`: GATv2-based graph classifier and auxiliary modules
- `dac_fedgraph/data_handler.py`: feature preprocessing and graph construction
- `dac_fedgraph/federated.py`: Client/Server logic and aggregation
- `dac_fedgraph/experiment.py`: experiment runner (cross-validation over seeds)
- `scripts/run_fedgraph.py`: CLI entry point

## Run (example)
```bash
python scripts/run_fedgraph.py \
  --phenotypic_csv /path/to/phenotypic.csv \
  --ts_dir_aal /path/to/AAL_csv \
  --ts_dir_sch /path/to/SCH_csv \
  --out_csv run_summaries.csv
```

Adjust CLI arguments (rounds/clients/splits) as needed.

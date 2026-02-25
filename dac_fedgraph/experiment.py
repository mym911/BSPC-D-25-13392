from __future__ import annotations
import copy
import random
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

from .configs import PathConfig, TrainConfig
from .federated import Client, Server, fedavg
from .metrics import brier_score_binary, ece_score_binary
from .utils import get_device, set_global_seed

def run_experiment(paths: PathConfig, cfg: TrainConfig):
    device = get_device()

    # NOTE: Keep the original main() logic structure for review.
    pheno = pd.read_csv(paths.phenotypic_csv)
    # The original script uses multiple possible id/label column conventions.
    # Please ensure the columns used below match your dataset CSV.
    if "SUB_ID" in pheno.columns and "DX_GROUP" in pheno.columns:
        all_ids = pheno["SUB_ID"].astype(str).tolist()
        raw_labels = np.where(pheno["DX_GROUP"].values == 2, 1, 0)
        pheno["SUB_ID"] = pheno["SUB_ID"].astype(str)
        sub2site = dict(zip(pheno["SUB_ID"], pheno.get("SITE_ID", pd.Series(["NA"]*len(pheno)))))
        id_col = "SUB_ID"
    else:
        # fallback: ScanDir ID / DX
        all_ids = pheno["ScanDir ID"].astype(str).tolist()
        raw_labels = np.where(pheno["DX"].values == 1, 0, 1)  # mirror original preprocessing
        sub2site = dict(zip(pheno["ScanDir ID"].astype(str), pheno.get("SITE_ID", pd.Series(["NA"]*len(pheno)))))
        id_col = "ScanDir ID"

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=42)

    run_summaries = []
    for run_seed in cfg.seeds:
        seed_here = int(run_seed * 100)
        set_global_seed(seed_here, deterministic=False)

        all_y_true, all_y_prob, all_sub_ids = [], [], []

        for fold, (train_idx_global, test_idx_global) in enumerate(skf.split(all_ids, raw_labels), start=1):
            train_ids = [all_ids[i] for i in train_idx_global]
            test_ids_global = [all_ids[i] for i in test_idx_global]

            random.shuffle(train_ids)
            client_subsets = np.array_split(train_ids, cfg.num_clients)

            clients = []
            for cid, sub_ids in enumerate(client_subsets):
                clients.append(
                    Client(
                        client_id=cid,
                        raw_ids=list(sub_ids),
                        test_ids_global=test_ids_global,
                        device=device,
                        k=cfg.k,
                        lr=cfg.init_lr,
                    )
                )

            model_instance = copy.deepcopy(clients[0].model)
            server = Server(model_instance, device)
            global_state = copy.deepcopy(model_instance.state_dict())

            best_val = 0.0
            best_state = copy.deepcopy(global_state)

            for rnd in range(1, cfg.num_rounds + 1):
                for c in clients:
                    c.load_global(global_state)

                phase1_states, sizes, val_accs = [], [], []
                for c in clients:
                    mu_init, mu_max = 0.0, 0.1
                    mu = mu_init + (mu_max - mu_init) * (rnd / cfg.num_rounds)
                    w1, n, v_acc, v_loss = c.local_train(cfg.local_epochs, global_state=global_state, mu=mu)
                    phase1_states.append(w1); sizes.append(n); val_accs.append(v_acc)

                cluster_models, labels, best_k, best_score = server.cluster_aggregate(
                    global_state, phase1_states, sizes, cfg.num_clients, max_k=20, seed=seed_here
                )

                for idx, c in enumerate(clients):
                    c.load_global(cluster_models[labels[idx]])

                phase2_states, val_accs2 = [], []
                for c in clients:
                    w2, n2, v_acc2, v_loss2 = c.local_train(cfg.local_epochs)
                    phase2_states.append(w2); val_accs2.append(v_acc2)

                weighted_val2 = sum(v * s for v, s in zip(val_accs2, sizes)) / sum(sizes)
                if weighted_val2 > best_val:
                    best_val = weighted_val2
                    best_state = copy.deepcopy(fedavg(phase2_states, sizes))

                global_state = server.fedavg(phase2_states, sizes)
                server.model.load_state_dict(global_state)

            # Load best
            server.model.load_state_dict(best_state)
            for c in clients:
                c.load_global(best_state)

            pred_bank: Dict[str, Any] = {}
            for c in clients:
                _, _, sub_ids, y_true, y_prob = c.evaluate_local(return_arrays=True)
                for sid, yt, yp in zip(sub_ids, y_true, y_prob):
                    if sid not in pred_bank:
                        pred_bank[sid] = (int(yt), [float(yp)])
                    else:
                        pred_bank[sid][1].append(float(yp))

            sub_ids_fold = list(pred_bank.keys())
            y_true_fold = np.array([pred_bank[sid][0] for sid in sub_ids_fold], dtype=int)
            y_prob_fold = np.array([np.mean(pred_bank[sid][1]) for sid in sub_ids_fold], dtype=float)
            y_pred_fold = (y_prob_fold >= 0.5).astype(int)

            acc = float(np.mean(y_pred_fold == y_true_fold))
            f1 = float(f1_score(y_true_fold, y_pred_fold, average='macro'))
            try:
                auc = float(roc_auc_score(y_true_fold, y_prob_fold))
            except ValueError:
                auc = float('nan')

            brier = brier_score_binary(y_true_fold, y_prob_fold)
            ece = ece_score_binary(y_true_fold, y_prob_fold, n_bins=10)

            all_y_true.append(y_true_fold)
            all_y_prob.append(y_prob_fold)
            all_sub_ids.append(np.array(sub_ids_fold, dtype=str))

        all_sub_ids_arr = np.concatenate(all_sub_ids, axis=0).astype(str)
        all_y_true_arr = np.concatenate(all_y_true, axis=0).astype(int)
        all_y_prob_arr = np.concatenate(all_y_prob, axis=0).astype(float)
        all_y_pred_arr = (all_y_prob_arr >= 0.5).astype(int)

        acc_run = float(np.mean(all_y_pred_arr == all_y_true_arr))
        auc_run = float(roc_auc_score(all_y_true_arr, all_y_prob_arr))
        brier_run = brier_score_binary(all_y_true_arr, all_y_prob_arr)
        ece_run = ece_score_binary(all_y_true_arr, all_y_prob_arr, n_bins=10)

        run_summaries.append({
            "seed": run_seed,
            "acc": acc_run,
            "auc": auc_run,
            "brier": brier_run,
            "ece": ece_run,
        })

    return pd.DataFrame(run_summaries)

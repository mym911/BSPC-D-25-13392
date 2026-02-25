from __future__ import annotations
import copy
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.metrics import f1_score, roc_auc_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering

from .configs import PathConfig, TrainConfig
from .data_handler import DataHandler
from .models import SingleGraphModel, FocalLoss
from .metrics import brier_score_binary, ece_score_binary, sens_spec_from_binary
from .utils import get_device

device = get_device()

def flatten_state_dict(state: Dict[str, torch.Tensor]) -> np.ndarray:
    """把一个模型 state_dict 的所有参数按 key 排序后 flatten 成 1D 向量"""
    parts = []
    for k in sorted(state.keys()):
        parts.append(state[k].cpu().numpy().ravel())
    return np.concatenate(parts, axis=0)



class Client:
    def __init__(
        self,
        client_id: int,
        raw_ids: List[str],
        test_ids_global: List[str],
        device: torch.device,
        k: int = K,
        lr: float = INIT_LR,
        batch_size: int = 64,
    ):
        self.id = client_id
        self.raw_ids = raw_ids
        self.test_ids_global = test_ids_global
        self.device = device
        self.k = k
        self.lr = lr
        self.batch_size = batch_size
        # **本地预处理**：加载 CSV、生成 feat/labels/phenos 并过滤不匹配
        self._run_local_preprocess()
        #    （可选）把 raw_ids 更新成只剩下 common 样本，避免后面 split_ids 用到无效 ID
        self.raw_ids = list(self.labels.keys())
        dh = DataHandler(
            ts_dir=TS_DIR_AAL,
            atlas_type='aal',
            k=k,
            pca_components=PCA_COMPONENTS,
            device=device
        )
        self.graph = dh.build_group_graph(
            feat_aal = self.feat_aal,
            feat_sch = self.feat_sch,
            phenos = self.phenos,
            labels = self.labels,
            k = self.k
        )
        self.data = self.graph.x
        self.y_all = self.graph.y
        self.subject_ids = self.graph.subject_ids
        self._make_local_split()
        train_index_cpu = self.train_idx_local.cpu()
        self.loader = DataLoader(
            train_index_cpu,
            batch_size=self.batch_size,
            shuffle=True
        )

        # local model
        in_dim = self.data.size(1)
        self.model = SingleGraphModel(
            in_channels=in_dim,
            hidden_dim=128,
            num_classes=2
        ).to(self.device)
        self.criterion = FocalLoss(gamma=2).to(self.device)
        #  self.criterion = torch.nn.CrossEntropyLoss()


    def _run_local_preprocess(self):
        # ---- ① 预处理 ----
        dh_aal = DataHandler(TS_DIR_AAL,  'aal',
                             k=K, pca_components=PCA_COMPONENTS,
                             device=self.device)
        dh_sch = DataHandler(TS_DIR_SCH, 'schaefer200',
                             k=K, pca_components=PCA_COMPONENTS,
                             device=self.device)

        self.feat_aal, self.phenos, self.labels, _ = dh_aal.preprocess_abide_data(
            PHENOTYPIC_CSV_PATH, TS_DIR_AAL,
            atlas='aal', selected_subjects=self.raw_ids+self.test_ids_global
        )
        self.feat_sch, _, _, _ = dh_sch.preprocess_abide_data(
            PHENOTYPIC_CSV_PATH, TS_DIR_SCH,
            atlas='schaefer200', selected_subjects=self.raw_ids+self.test_ids_global
        )

        # ---- ② 过滤：两套图谱都必须有 ----
        common = set(self.feat_aal) & set(self.feat_sch)
        # 只保留 common 的键
        self.feat_aal = {sid: self.feat_aal[sid] for sid in common}
        self.feat_sch = {sid: self.feat_sch[sid] for sid in common}
        self.phenos = {sid: self.phenos[sid] for sid in common}
        self.labels = {sid: self.labels[sid] for sid in common}

        if len(common) == 0:
            raise RuntimeError(f"[Client{self.id}] 没有可用样本，直接退出。")

    def _make_local_split(self):
        all_ids = self.subject_ids
        all_local_ids = [
            sid for sid in self.subject_ids
            if sid not in self.test_ids_global
        ]
        self.all_local_ids = all_local_ids
        y_full = [self.labels[sid] for sid in all_local_ids]

        tr_ids, val_ids = train_test_split(
            all_local_ids, test_size=0.2, random_state=42,
            stratify=y_full)
        self.train_idx_local = torch.tensor([i for i, sid in enumerate(all_ids) if sid in tr_ids], dtype=torch.long, device=self.device)
        self.val_idx_local = torch.tensor([i for i, sid in enumerate(all_ids) if sid in val_ids], dtype=torch.long, device=self.device)
        # test_idx_glob 只给全局评估用
        self.test_idx_global = torch.tensor([i for i, sid in enumerate(all_ids) if sid in self.test_ids_global], dtype=torch.long, device=self.device)

    def local_train(self, epochs=1, global_state=None, mu=0.01):
        """FedProx local update (paper): minimize F_k(w) + (mu/2)||w - w_global||^2.

        Notes:
        - Use *name-aligned* proximal term to avoid mismatch between named_parameters() and state_dict values.
        - Pass global_state as a state_dict (OrderedDict / dict of tensors).
        """
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for _ in range(epochs):
            for batch_idx in self.loader:
                batch_idx = batch_idx.to(self.device)
                h, logits = self.model(self.data, self.graph.edge_index, self.graph.edge_weight)
                loss = self.criterion(logits[batch_idx], self.y_all[batch_idx])

                # --- FedProx proximal term: (mu/2) * ||w - w_global||^2 ---
                if global_state is not None and mu is not None and mu > 0:
                    prox = 0.0
                    for name, p_local in self.model.named_parameters():
                        p_global = global_state[name].to(self.device)
                        prox = prox + (p_local - p_global).pow(2).sum()
                    loss = loss + (float(mu) / 2.0) * prox

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        state_dict_cpu = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        with torch.no_grad():
            _, logits = self.model(self.data, self.graph.edge_index, self.graph.edge_weight)
            val_loss = self.criterion(logits[self.val_idx_local], self.y_all[self.val_idx_local])
            val_acc = (logits[self.val_idx_local].argmax(1) == self.y_all[self.val_idx_local]).float().mean().item()

        return state_dict_cpu, self.graph.num_nodes, val_acc, val_loss.item()

    def evaluate_local(self, return_arrays: bool = False):
        if self.test_idx_global.numel() == 0:
            if return_arrays:
                return {'acc': 0, 'f1': 0, 'auc': 0}, 0, [], np.array([]), np.array([])
            return {'acc': 0, 'f1': 0, 'auc': 0}, 0

        self.model.eval()
        with torch.no_grad():
            _, logits = self.model(self.graph.x, self.graph.edge_index, self.graph.edge_weight)
            probs = F.softmax(logits[self.test_idx_global], dim=1)
            preds = probs.argmax(1)
            y = self.graph.y[self.test_idx_global]

        acc = (preds == y).float().mean().item()
        f1 = f1_score(y.cpu(), preds.cpu(), average='macro')
        try:
            auc = roc_auc_score(y.cpu(), probs[:, 1].cpu())
        except ValueError:
            auc = float('nan')

        if not return_arrays:
            return {'acc': acc, 'f1': f1, 'auc': auc}, len(self.test_idx_global)

        # ✅ 逐样本输出（顺序与 test_idx_global 一致）
        idx_list = self.test_idx_global.detach().cpu().tolist()
        sub_ids_test = [self.subject_ids[i] for i in idx_list]  # 你这里已经有 self.subject_ids = self.graph.subject_ids
        y_true = y.detach().cpu().numpy().astype(int)
        y_prob = probs[:, 1].detach().cpu().numpy().astype(float)  # 这就是 y_prob (P(y=1))

        return {'acc': acc, 'f1': f1, 'auc': auc}, len(self.test_idx_global), sub_ids_test, y_true, y_prob

    def load_global(self, state_dict):
        self.model.load_state_dict(state_dict)

    def update_graph(self, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        # 注意 x, y, subject_ids 不变
        self.graph.edge_index  = edge_index.to(self.device)
        self.graph.edge_weight = edge_weight.to(self.device)



class Server:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.best_val = 0.0
        self.best_state = copy.deepcopy(model.state_dict())

    @staticmethod
    def flatten_state_dict(state):
        parts = []
        for k in sorted(state.keys()):
            parts.append(state[k].cpu().numpy().ravel())
        return np.concatenate(parts, axis=0)

    @staticmethod
    def fedavg(states, sizes):
        total = sum(sizes)
        agg = copy.deepcopy(states[0])
        for k in agg.keys():
            agg[k] = sum(states[i][k] * (sizes[i] / total) for i in range(len(states)))
        return agg

    def cluster_aggregate(self, global_state, client_states, sizes, num_clients, max_k=20, seed: int = 42):
        # 1. 计算Δw相似度矩阵
        global_vec = self.flatten_state_dict(global_state)
        delta_mat = []
        for w in client_states:
            vec = self.flatten_state_dict(w)
            delta_mat.append(vec - global_vec)
        delta_mat = np.vstack(delta_mat)
        sim = cosine_similarity(delta_mat)
        sim = np.clip(sim, 0, None)
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0)

        # 2. 自动选择最佳簇数k
        best_k, best_score = 2, -1.0
        max_k = min(max_k, num_clients - 1)
        for k in range(2, max_k + 1):
            labels_k = SpectralClustering(
                n_clusters=k,
                affinity='precomputed',
                assign_labels='kmeans',
                random_state=seed
            ).fit_predict(sim)
            score = silhouette_score(dist, labels_k, metric='precomputed')
            if score > best_score:
                best_k, best_score = k, score

        print(f"Auto-selected n_clusters = {best_k} (silhouette = {best_score:.3f})")

        # 3. 用最佳k重新分簇
        clustering = SpectralClustering(
            n_clusters=best_k,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=seed
        )
        labels = clustering.fit_predict(sim)

        # 4. 按簇聚合模型参数
        cluster_states = defaultdict(list)
        cluster_sizes = defaultdict(list)
        for idx, state in enumerate(client_states):
            cid = labels[idx]
            cluster_states[cid].append(state)
            cluster_sizes[cid].append(sizes[idx])

        cluster_models = {
            cid: self.fedavg(cluster_states[cid], cluster_sizes[cid])
            for cid in cluster_states
        }

        # 5. 合并成新的全局模型权重
        new_global_state = self.fedavg(
            list(cluster_models.values()),
            [sum(cluster_sizes[cid]) for cid in cluster_models]
        )
        self.model.load_state_dict(new_global_state)

        return cluster_models, labels, best_k, best_score

    def update_best(self, val, state):
        if val > self.best_val:
            self.best_val = val
            self.best_state = copy.deepcopy(state)

    def load_best_state(self):
        self.model.load_state_dict(self.best_state)



def fedavg(states, sizes):
        total = sum(sizes)
        agg = copy.deepcopy(states[0])
        for k in agg.keys():
            agg[k] = sum(states[i][k] * (sizes[i] / total) for i in range(len(states)))
        return agg

    def cluster_aggregate(self, global_state, client_states, sizes, num_clients, max_k=20, seed: int = 42):
        # 1. 计算Δw相似度矩阵
        global_vec = self.flatten_state_dict(global_state)
        delta_mat = []
        for w in client_states:
            vec = self.flatten_state_dict(w)
            delta_mat.append(vec - global_vec)
        delta_mat = np.vstack(delta_mat)
        sim = cosine_similarity(delta_mat)
        sim = np.clip(sim, 0, None)
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0)

        # 2. 自动选择最佳簇数k
        best_k, best_score = 2, -1.0
        max_k = min(max_k, num_clients - 1)
        for k in range(2, max_k + 1):
            labels_k = SpectralClustering(
                n_clusters=k,
                affinity='precomputed',
                assign_labels='kmeans',
                random_state=seed
            ).fit_predict(sim)
            score = silhouette_score(dist, labels_k, metric='precomputed')
            if score > best_score:
                best_k, best_score = k, score

        print(f"Auto-selected n_clusters = {best_k} (silhouette = {best_score:.3f})")

        # 3. 用最佳k重新分簇
        clustering = SpectralClustering(
            n_clusters=best_k,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=seed
        )
        labels = clustering.fit_predict(sim)

        # 4. 按簇聚合模型参数
        cluster_states = defaultdict(list)
        cluster_sizes = defaultdict(list)
        for idx, state in enumerate(client_states):
            cid = labels[idx]
            cluster_states[cid].append(state)
            cluster_sizes[cid].append(sizes[idx])

        cluster_models = {
            cid: self.fedavg(cluster_states[cid], cluster_sizes[cid])
            for cid in cluster_states
        }

        # 5. 合并成新的全局模型权重
        new_global_state = self.fedavg(
            list(cluster_models.values()),
            [sum(cluster_sizes[cid]) for cid in cluster_models]
        )
        self.model.load_state_dict(new_global_state)

        return cluster_models, labels, best_k, best_score

    def update_best(self, val, state):
        if val > self.best_val:
            self.best_val = val
            self.best_state = copy.deepcopy(state)

    def load_best_state(self):
        self.model.load_state_dict(self.best_state)

def fedavg(states, sizes):
    total = sum(sizes)
    agg = OrderedDict()
    for k in states[0]:
        agg[k] = sum(states[i][k]*(sizes[i]/total) for i in range(len(states)))
    return agg



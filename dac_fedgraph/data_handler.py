from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Data

from .utils import get_device

logger = logging.getLogger(__name__)
device = get_device()

def knn_edges(sim: np.ndarray, k: int = 10):
    """固定 k 条最近邻，不再用 μ+2σ."""
    idx = np.argsort(-sim, axis=1)[:, 1:k + 1]  # 第 0 是自己
    send, recv = [], []
    for i, row in enumerate(idx):
        for j in row:
            send.append(i);
            recv.append(j)
    edge_index = torch.tensor([send, recv], dtype=torch.long)
    edge_weight = torch.tensor(sim[send, recv], dtype=torch.float32)
    return edge_index, edge_weight


class DataHandler:
    def __init__(self,
                 ts_dir: Path,
                 atlas_type: str,
                 k: int = 10,
                 sigma: float = 1,
                 device: torch.device = device,
                 pca_components: int = PCA_COMPONENTS,
                 num_heads: int = 4
                 ):
        self.atlas_type = atlas_type
        self.k = k
        self.ts_dir = ts_dir
        self.sigma = sigma
        self.device = device
        self.num_heads = num_heads
        self.token_projection = None
        self.scaler = None
        self.pca = None
        self.pca_components = pca_components
        self.scaler_fitted = False
        self.pca_fitted = False
        self.window_fusion_transformer = None
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def fit_and_save_scaler_pca(self, features_matrix: np.ndarray, scaler_path: Path, pca_path: Path):
        self.scaler = StandardScaler().fit(features_matrix)
        features_scaled = self.scaler.transform(features_matrix)
        self.pca = PCA(n_components=self.pca_components).fit(features_scaled)
        features_pca = self.pca.transform(features_scaled)
        logger.debug(f"PCA转换后数据示例: {features_pca[:2]}")
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.pca, pca_path)
        self.scaler_fitted = True
        self.pca_fitted = True
        logger.info(f"Scaler/PCA 模型已保存到 {scaler_path} 和 {pca_path}")

    def load_scaler_pca(self, scaler_path: Path, pca_path: Path):
        self.scaler = joblib.load(scaler_path)
        self.pca = joblib.load(pca_path)
        self.scaler_fitted = True
        self.pca_fitted = True
        logger.info(f"成功从 {scaler_path} 和 {pca_path} 加载 Scaler/PCA。")

    def preprocess_abide_data(self,
                              phenotypic_csv_path: Path,
                              ts_dir: Path,
                              pca_components: int = PCA_COMPONENTS,
                              num_subjects: Optional[int] = None,
                              atlas: str = 'aal',
                              selected_subjects: Optional[List[Any]] = None
                              ):
        logger = logging.getLogger(__name__)
        try:
            phenotypic_data = pd.read_csv(phenotypic_csv_path)
        except Exception as e:
            logger.error(f"加载表型数据时出错: {e}")
            return {}, {}, {}, []

        required_columns = ['ScanDir ID', 'DX', 'Age', 'Gender']
        phenotypic_data.columns = phenotypic_data.columns.str.strip()
        for col in required_columns:
            if col not in phenotypic_data.columns:
                logger.error(f"表型数据中缺少 '{col}' 列。请确保 CSV 文件包含该列。")
                return {}, {}, {}, []

        if selected_subjects:
            subjects = selected_subjects[:num_subjects] if num_subjects else selected_subjects
        else:
            all_ids = phenotypic_data['ScanDir ID'].astype(str).tolist()
            subjects = all_ids[:num_subjects] if num_subjects else all_ids

        logger.info(f"选择了 {len(subjects)} 名未处理的受试者进行处理。")

        features_dict_raw, phenotypes_dict, labels_dict = {}, {}, {}
        skipped_subjects = []

        w1, s1 = 30, 5
        w2, s2 = 40, 15

        for subject in subjects:
            csv_path = ts_dir / f"{subject}.csv"
            if not csv_path.exists():
                logger.warning(f"找不到 {csv_path}，已跳过")
                continue
            ts = pd.read_csv(csv_path, header=0).values  # shape (T, n_roi)
            if ts.shape[0] < max(w1, w2):
                logger.warning(f"受试者 {subject} 的时间序列太短，跳过")
                continue

            windows1 = [ts[i:i + w1] for i in range(0, ts.shape[0] - w1 + 1, s1)]
            windows2 = [ts[i:i + w2] for i in range(0, ts.shape[0] - w2 + 1, s2)]

            def fcn_to_mat(window: np.ndarray) -> np.ndarray:
                with np.errstate(divide='ignore', invalid='ignore'):
                    fc = np.corrcoef(window, rowvar=False)
                    fc = np.clip(fc, -0.999999, 0.999999)
                    fc = np.arctanh(fc)
                    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
                return fc

            mats1 = [fcn_to_mat(w) for w in windows1]
            mats2 = [fcn_to_mat(w) for w in windows2]

            grp_feat1 = self.dynamic_feature_aggregation(mats1, method='seq_transformer')
            grp_feat2 = self.dynamic_feature_aggregation(mats2, method='seq_transformer')

            f1 = grp_feat1.squeeze(0)
            f2 = grp_feat2.squeeze(0)
            fused_vec = np.concatenate([f1, f2], axis=0)
            features_dict_raw[subject] = fused_vec

            try:
                phenotype = phenotypic_data[phenotypic_data['ScanDir ID'].astype(str) == subject]
                if phenotype.empty:
                    logger.warning(f"受试者 {subject} 的表型数据不存在。")
                    skipped_subjects.append(subject)
                    continue
                phenotype = phenotype.iloc[0]
                age = phenotype['Age']
                sex = phenotype['Gender']
                phenotypes_dict[subject] = [age, sex]

                if 'DX' in phenotype:
                    raw_label = phenotype['DX']
                    if raw_label == 1:
                        label_processed = 0
                    else:
                        label_processed = 1
                    labels_dict[subject] = label_processed
            except Exception as e:
                logger.error(f"处理受试者 {subject} 时出错: {e}")
                skipped_subjects.append(subject)
        logger.info(f"共成功处理了 {len(subjects) - len(skipped_subjects)} 名受试者，跳过了 {len(skipped_subjects)} 名受试者。")
        return features_dict_raw, phenotypes_dict, labels_dict, list(features_dict_raw.keys())


    def construct_subject_similarity(self, features_matrix: np.ndarray) -> np.ndarray:
        assert features_matrix.ndim == 2, f"特征矩阵维度错误: {features_matrix.shape}"
        pairwise_dist = pairwise_distances(features_matrix, metric='euclidean')
        sigma = np.percentile(pairwise_dist[pairwise_dist > 0], 25)
        similarity_matrix = np.exp(-pairwise_dist ** 2 / (2 * (sigma + 1e-9) ** 2))
        np.fill_diagonal(similarity_matrix, 1.0)
        return similarity_matrix

    def construct_phenotype_similarity(self, phenotypes_array: np.ndarray) -> np.ndarray:
        num_subjects = phenotypes_array.shape[0]
        if num_subjects == 0:
            return np.array([[]], dtype=np.float32)

        ages = phenotypes_array[:, 0].astype(float)
        sexes = phenotypes_array[:, 1]

        # 1) 年龄距离
        min_age, max_age = ages.min(), ages.max()
        age_range = max_age - min_age if max_age > min_age else 1.0
        age_dist = np.zeros((num_subjects, num_subjects), dtype=np.float32)
        for i in range(num_subjects):
            for j in range(i + 1, num_subjects):
                dist_ij = abs(ages[i] - ages[j]) / age_range
                age_dist[i, j] = dist_ij
                age_dist[j, i] = dist_ij

        # 2) 性别距离
        sex_dist = np.zeros((num_subjects, num_subjects), dtype=np.float32)
        for i in range(num_subjects):
            for j in range(i + 1, num_subjects):
                dist_ij = 0.0 if sexes[i] == sexes[j] else 1.0
                sex_dist[i, j] = dist_ij
                sex_dist[j, i] = dist_ij


        gower_dist = (age_dist + sex_dist) / 2.0

        use_rbf = False
        if not use_rbf:
            S_phi = 1.0 - gower_dist
        else:
            S_phi = np.exp(-np.power(gower_dist, 2) / (2 * self.sigma ** 2))
        return S_phi

    def build_group_graph(
        self,
        feat_aal: Dict[str, np.ndarray],
        feat_sch: Dict[str, np.ndarray],
        phenos:  Dict[str, List[Any]],
        labels:  Dict[str, int],
        k:       Optional[int] = None
    ) -> Data:
        if k is None:
            k = self.k

        # 1) 排序所有 subject IDs
        subject_ids = sorted(feat_aal.keys())
        N = len(subject_ids)

        # 2) 堆叠成矩阵
        mat_aal = np.vstack([feat_aal[sid] for sid in subject_ids])  # (N, aal_dim)
        mat_sch = np.vstack([feat_sch[sid] for sid in subject_ids])  # (N, sch_dim)
        mat_ph  = np.array([phenos[sid]     for sid in subject_ids])  # (N, 2) 或更多

        # 3) 分别计算相似度
        S_aal = self.construct_subject_similarity(mat_aal)
        S_sch = self.construct_subject_similarity(mat_sch)
        S_phi = self.construct_phenotype_similarity(mat_ph)

        # 4) 三者融合
        fused_sim = S_aal * S_sch * S_phi

        # 5) 构造 KNN 边
        edge_index, edge_weight = self._build_knn_edges(fused_sim, k)

        # 6) 拼接节点特征输入
        feat_concat = np.hstack([mat_aal, mat_sch])  # (N, aal_dim+sch_dim)
        x = torch.tensor(feat_concat,
                         dtype=torch.float32,
                         device=self.device)
        y = torch.tensor([labels[sid] for sid in subject_ids],
                         dtype=torch.long,
                         device=self.device)

        # 7) 返回 PyG Data
        data = Data(
            x=x,
            edge_index=edge_index.to(self.device),
            edge_weight=edge_weight.to(self.device),
            y=y
        )
        data.subject_ids = subject_ids
        return data

    def _build_knn_edges(self, similarity_matrix, k=None):
        num_nodes = similarity_matrix.shape[0]
        if k is None:
            k = min(30, int(np.sqrt(num_nodes)))
        # 归一化相似度
        sim = (similarity_matrix - similarity_matrix.min()) \
              / (similarity_matrix.max() - similarity_matrix.min() + 1e-8)
        # 取每行前 k+1（含自己），然后丢掉自己
        idx = np.argsort(-sim, axis=1)[:, 1:k+1]
        send, recv = [], []
        for i, nbrs in enumerate(idx):
            for j in nbrs:
                send.append(i)
                recv.append(j)
        edge_index  = torch.tensor([send, recv], dtype=torch.long)
        edge_weight = torch.tensor(sim[send, recv], dtype=torch.float32)
        return edge_index, edge_weight

'''


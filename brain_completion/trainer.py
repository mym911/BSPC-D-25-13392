from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from .configs import ModelConfig, DatasetConfig
from .data import ROICSVLoader, ROISequenceDataset, collate_fn_with_dynamic_padding
from .masks import DynamicMaskGenerator, StructuredMaskGenerator
from .metrics import r2_global
from .models import ROICompletionGenerator, ROICompletionDiscriminator
from .utils import get_device, get_logger

logger = get_logger(__name__)
device = get_device()

def compute_roi_missing_weights(dataset: Dict[str, Any], eps: float = 1e-3) -> Optional[np.ndarray]:
    masks = []
    for d in dataset.values():
        masks.append(d["session_1"]["missing_mask"].astype(np.float32))
    if not masks:
        return None
    masks = np.stack(masks, axis=0)
    missing_freq = 1.0 - masks.mean(axis=0)
    w = missing_freq + eps
    return w / w.sum()

class BrainCompletionTrainer:
    def __init__(self, model_cfg: ModelConfig, data_cfg: DatasetConfig):
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.loader = ROICSVLoader(data_cfg)
        self.num_rois = len(self.loader.custom_labels)

        self.gen = ROICompletionGenerator(model_cfg.cond_dim, self.num_rois).to(device)
        self.disc = ROICompletionDiscriminator(self.num_rois).to(device)

        self.opt_g = optim.AdamW(self.gen.parameters(), lr=model_cfg.lr)
        self.opt_d = optim.AdamW(self.disc.parameters(), lr=model_cfg.lr * 0.5)

        self.recon_loss = nn.L1Loss(reduction="none")
        self.adv_loss = nn.BCELoss()

        self.mcar_masker = DynamicMaskGenerator(mask_rate=model_cfg.mask_rate)

    def train_pre_recon(self, train_loader: DataLoader):
        self.gen.train()
        for epoch in range(self.model_cfg.epochs_pretrain):
            total = 0.0
            for feats, roi_avail_mask, cond in train_loader:
                feats = feats.to(device); roi_avail_mask = roi_avail_mask.to(device); cond = cond.to(device)
                sim_mask = self.mcar_masker.generate_mask(feats.shape).to(device)
                masked = feats.clone()
                masked[sim_mask] = 0.0
                valid_mask = roi_avail_mask & (~sim_mask)

                pred = self.gen(masked, cond=cond, mask=valid_mask)

                time_padding_mask = (feats != 0).any(dim=1, keepdim=True)
                overall_valid_mask = valid_mask & time_padding_mask

                loss = self.recon_loss(pred[overall_valid_mask], feats[overall_valid_mask]).mean()
                self.opt_g.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.gen.parameters(), 1.0)
                self.opt_g.step()
                total += float(loss.item())

            if (epoch + 1) % 10 == 0:
                logger.info(f"[pre-recon] epoch {epoch+1}/{self.model_cfg.epochs_pretrain} loss={total/max(1,len(train_loader)):.4f}")

    def _prepare_roi_data(self, dataset: Dict[str, Any]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        out = []
        for sid, subj in dataset.items():
            feats = subj["session_1"]["rest_features"]
            roi_avail = subj["session_1"]["missing_mask"]
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)  # (1,R,T)
            T = feats.shape[-1]
            m2d = (np.tile(roi_avail[:, None], (1, T)) == 1)
            m = torch.tensor(m2d, dtype=torch.bool).unsqueeze(0).to(device)
            cond_np = self.loader.phenotypes.get(sid, np.zeros(self.model_cfg.cond_dim, dtype=np.float32))
            c = torch.tensor(cond_np, dtype=torch.float32).to(device)
            out.append((x, m, c))
        return out

    def pretrain_gan_onepass(self, roi_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        # D one pass
        self.gen.train(); self.disc.train()
        for feats, roi_avail_mask, cond in roi_data:
            cond_b = cond.unsqueeze(0)
            sim_mask = self.mcar_masker.generate_mask(feats.shape).to(device)
            masked = feats * (~sim_mask).float()
            valid_mask = roi_avail_mask & (~sim_mask)

            with torch.no_grad():
                fake = self.gen(masked, cond=cond_b, mask=valid_mask)

            real_pred = self.disc(feats, mask=roi_avail_mask)
            fake_pred = self.disc(fake, mask=roi_avail_mask)
            d_loss = 0.5 * (self.adv_loss(real_pred, torch.ones_like(real_pred)) +
                            self.adv_loss(fake_pred, torch.zeros_like(fake_pred)))
            self.opt_d.zero_grad()
            d_loss.backward()
            nn.utils.clip_grad_norm_(self.disc.parameters(), 1.0)
            self.opt_d.step()

        # G one pass
        self.gen.train(); self.disc.eval()
        for feats, roi_avail_mask, cond in roi_data:
            cond_b = cond.unsqueeze(0)
            sim_mask = self.mcar_masker.generate_mask(feats.shape).to(device)
            masked = feats * (~sim_mask).float()
            valid_mask = roi_avail_mask & (~sim_mask)

            fake = self.gen(masked, cond=cond_b, mask=valid_mask)
            gen_pred = self.disc(fake, mask=roi_avail_mask)
            g_loss_adv = self.adv_loss(gen_pred, torch.ones_like(gen_pred))

            time_padding_mask = (feats != 0).any(dim=1, keepdim=True)
            final_valid_mask = valid_mask & time_padding_mask
            g_loss_rec = self.recon_loss(fake[final_valid_mask], feats[final_valid_mask]).mean()
            g_loss = 0.1 * g_loss_adv + 0.9 * g_loss_rec

            self.opt_g.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(self.gen.parameters(), 1.0)
            self.opt_g.step()

    def adversarial_train(self, roi_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        grad_clip = nn.utils.clip_grad_norm_
        for epoch in range(self.model_cfg.epochs_gan):
            d_total = 0.0; g_total = 0.0
            for feats, roi_avail_mask, cond in roi_data:
                batch_size = feats.size(0)

                # D
                self.disc.train(); self.gen.eval()
                sim_mask = self.mcar_masker.generate_mask(feats.shape).to(device)
                masked = feats.clone()
                masked[sim_mask] = 0.0
                valid_mask = roi_avail_mask & (~sim_mask)
                cond_b = cond.unsqueeze(0)

                with torch.no_grad():
                    fake = self.gen(masked, cond=cond_b, mask=valid_mask)

                real_pred = self.disc(feats, mask=valid_mask)
                fake_pred = self.disc(fake.detach(), mask=valid_mask)

                real_labels = torch.ones(batch_size, 1, device=device).uniform_(0.9, 1.0)
                fake_labels = torch.zeros(batch_size, 1, device=device).uniform_(0.0, 0.1)

                d_loss = 0.5 * (self.adv_loss(real_pred, real_labels) + self.adv_loss(fake_pred, fake_labels))
                self.opt_d.zero_grad()
                d_loss.backward()
                grad_clip(self.disc.parameters(), 1.0)
                self.opt_d.step()

                # G
                self.disc.eval(); self.gen.train()
                fake = self.gen(masked, cond=cond_b, mask=valid_mask)
                gen_pred = self.disc(fake, mask=valid_mask)
                g_loss_adv = self.adv_loss(gen_pred, real_labels)

                time_padding_mask = (feats != 0).any(dim=1, keepdim=True)
                final_valid_mask = valid_mask & time_padding_mask
                g_loss_rec = self.recon_loss(fake[final_valid_mask], feats[final_valid_mask]).mean()
                g_loss = 0.05 * g_loss_adv + 0.95 * g_loss_rec

                self.opt_g.zero_grad()
                g_loss.backward()
                grad_clip(self.gen.parameters(), 1.0)
                self.opt_g.step()

                d_total += float(d_loss.item()); g_total += float(g_loss.item())

            logger.info(f"[GAN] epoch {epoch+1}/{self.model_cfg.epochs_gan} D={d_total/max(1,len(roi_data)):.4f} G={g_total/max(1,len(roi_data)):.4f}")

    def evaluate_avg_r2(self, val_roi_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
        r2_list = []
        for feats, roi_mask, cond in val_roi_data:
            sim_mask = self.mcar_masker.generate_mask(feats.shape).to(device)
            masked = feats.clone()
            masked[sim_mask] = 0.0
            with torch.no_grad():
                pred = self.gen(masked, cond=cond.unsqueeze(0), mask=(roi_mask & (~sim_mask)))
            r2_list.append(r2_global(feats.squeeze(0).cpu().numpy(),
                                    pred.squeeze(0).cpu().numpy(),
                                    sim_mask.squeeze(0).cpu().numpy()))
        return float(np.nanmean(r2_list))

    def stress_test(self, val_roi_data, roi_w, patterns, rates, fold_idx, seed=42):
        masker_struct = StructuredMaskGenerator(roi_weights=roi_w, seed=seed+fold_idx)
        rec = []
        for pat in patterns:
            for rate in rates:
                rr = []
                for feats, roi_mask, cond in val_roi_data:
                    sim_mask = masker_struct.generate_mask(feats, roi_mask, rate=rate, pattern=pat)
                    masked = feats.clone()
                    masked[sim_mask] = 0.0
                    with torch.no_grad():
                        pred = self.gen(masked, cond=cond.unsqueeze(0), mask=(roi_mask & (~sim_mask)))
                    rr.append(r2_global(feats.squeeze(0).cpu().numpy(),
                                       pred.squeeze(0).cpu().numpy(),
                                       sim_mask.squeeze(0).cpu().numpy()))
                rec.append({"fold": fold_idx, "pattern": pat, "rate": rate,
                            "r2_mean": float(np.nanmean(rr)), "r2_std": float(np.nanstd(rr))})
        return pd.DataFrame(rec)

    def run_kfold_with_stress(self, n_splits=5, seed=42,
                             patterns=("mcar","roi_weighted","cluster","block"),
                             rates=(0.1,0.2,0.3,0.4,0.5,0.6)):
        train_ds, _ = self.loader.load_dataset("train")
        test_ds, _ = self.loader.load_dataset("test")
        roi_w = compute_roi_missing_weights(test_ds)
        subject_ids = list(train_ds.keys())

        self.data_cfg.model_dir.mkdir(parents=True, exist_ok=True)
        best_r2 = -np.inf
        best_model_path = self.data_cfg.model_dir / "roi_completion_best.pth"
        fold_r2 = []

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, va_idx) in enumerate(kfold.split(subject_ids), start=1):
            tr_ids = [subject_ids[i] for i in tr_idx]
            va_ids = [subject_ids[i] for i in va_idx]
            tr_fold = {sid: train_ds[sid] for sid in tr_ids}
            va_fold = {sid: train_ds[sid] for sid in va_ids}

            dl = DataLoader(
                ROISequenceDataset(tr_fold, self.loader.phenotypes),
                batch_size=self.model_cfg.batch_size,
                collate_fn=collate_fn_with_dynamic_padding,
                shuffle=True,
            )

            # fresh model each fold (mirrors your script)
            self.__init__(self.model_cfg, self.data_cfg)

            self.train_pre_recon(dl)
            tr_roi = self._prepare_roi_data(tr_fold)
            self.pretrain_gan_onepass(tr_roi)
            self.adversarial_train(tr_roi)

            # save fold
            torch.save(self.gen.state_dict(), self.data_cfg.model_dir / f"roi_completion_fold{fold_idx}.pth")

            # validate
            va_roi = self._prepare_roi_data(va_fold)
            avg_r2 = self.evaluate_avg_r2(va_roi)
            fold_r2.append(avg_r2)
            logger.info(f"[fold {fold_idx}] avg_r2={avg_r2:.4f}")

            # stress csv
            df_ab = self.stress_test(va_roi, roi_w, patterns, rates, fold_idx, seed=seed)
            df_ab.to_csv(self.data_cfg.model_dir / f"ab_stress_fold{fold_idx}.csv", index=False)

            # best by avg_r2
            if np.isfinite(avg_r2) and avg_r2 > best_r2:
                best_r2 = avg_r2
                torch.save(self.gen.state_dict(), best_model_path)

        # summary
        all_csv = list(self.data_cfg.model_dir.glob("ab_stress_fold*.csv"))
        if all_csv:
            df_all = pd.concat([pd.read_csv(p) for p in all_csv], ignore_index=True)
            df_sum = df_all.groupby(["pattern","rate"]).agg(
                r2_mean=("r2_mean","mean"),
                r2_std=("r2_mean","std"),
            ).reset_index()
            df_sum.to_csv(self.data_cfg.model_dir / "ab_stress_summary.csv", index=False)

        return fold_r2

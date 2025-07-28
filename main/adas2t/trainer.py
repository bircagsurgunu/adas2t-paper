# adas2t/trainer.py
import logging
import os
from collections import defaultdict
from typing import Tuple, List
import itertools

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from jiwer import wer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..data_handler import load_and_prepare_dataset
from .feature_extractor import AudioFeatureExtractor
from .runtime import S2TAlgorithmPool
from .models import MetaLearnerMLP  # Import the new MLP model

logger = logging.getLogger(__name__)


def _algo_one_hot(idx: int, k: int) -> np.ndarray:
    v = np.zeros(k, dtype=np.float32)
    v[idx] = 1.0
    return v


class ADAS2TTrainer:
    def __init__(self, config: dict):
        self.cfg = config
        self.device = config["device"]
        self.learner_type = config["learner_type"]
        self.cache_dir = config["cache_dir"]

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        self.s2t_pool = S2TAlgorithmPool(
            algorithm_names=config["algorithm_pool_models"],
            device=self.device,
            torch_dtype=config["torch_dtype"],
        )
        self.algorithm_names: List[str] = self.s2t_pool.algorithm_names
        self.num_algorithms: int = len(self.algorithm_names)

        self.feature_extractor = AudioFeatureExtractor(sample_rate=16_000, device=self.device)
        self.scaler = StandardScaler()
        self.model = None  # Will hold either xgb.Booster or nn.Module

        # Expected acoustic-feature dimensionality (will be set on first sample)
        self.expected_feat_dim: int | None = None

    # --------------------------------------------------------------------- #
    # Helper to guarantee constant feature length
    # --------------------------------------------------------------------- #
    def _ensure_feature_dim(self, feats: np.ndarray) -> np.ndarray:
        """
        Make sure the feature vector has the same length for every sample.

        • On first call, records the length as self.expected_feat_dim.
        • On subsequent calls, pads with zeros or truncates if necessary.
        """
        if self.expected_feat_dim is None:
            self.expected_feat_dim = len(feats)
            logger.info(f"[ADAS2T] Feature dimension set to {self.expected_feat_dim}")
            return feats

        if len(feats) == self.expected_feat_dim:
            return feats

        logger.warning(
            f"[ADAS2T] Feature length mismatch "
            f"({len(feats)} vs {self.expected_feat_dim}). Padding / truncating."
        )
        if len(feats) < self.expected_feat_dim:
            return np.pad(feats, (0, self.expected_feat_dim - len(feats)))
        else:
            return feats[: self.expected_feat_dim]

    # --------------------------------------------------------------------- #
    # Main data-preparation routine
    # --------------------------------------------------------------------- #
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        """
        Prepares training data by generating or loading from cache.
        For each audio sample, it caches the acoustic features and transcriptions from all algorithms.
        """
        logger.info("Preparing meta-learner training data (with caching)...")
        all_samples_data = []
        sample_counter = 0

        for ds_name, ds_cfg in self.cfg["training_datasets"]:
            ds_display = f"{ds_name.replace('/', '_')}-{ds_cfg.get('config', ds_cfg.get('lang_config', ''))}-{ds_cfg['split']}"
            cache_path = os.path.join(self.cache_dir, f"{ds_display}.pkl")

            cached_samples = self._load_valid_cache(cache_path)

            num_required = self.cfg["num_training_samples"]
            num_to_generate = num_required - len(cached_samples)

            if num_to_generate <= 0:
                logger.info(f"  Using {num_required} samples from cache for {ds_display}.")
                all_samples_data.extend(cached_samples[:num_required])
                continue

            logger.info(f"  Need to generate {num_to_generate} new samples for {ds_display}.")
            dataset_stream = load_and_prepare_dataset(ds_name, ds_cfg, num_samples=None)

            # We need to process samples starting from where the cache ends, up to the required number
            dataset_iterator = itertools.islice(dataset_stream, len(cached_samples), num_required)

            newly_generated_samples = []
            pbar_desc = f"   generating ({ds_display[:25]})"
            for ex in tqdm(dataset_iterator, desc=pbar_desc, total=num_to_generate):
                audio, ref = ex["audio"]["array"], ex[ds_cfg["field"]].strip().lower()
                if not ref or len(audio) == 0:
                    continue

                # 1. extract acoustic features
                base_feats_raw = self.feature_extractor.extract_all_features(audio)
                base_feats = self._ensure_feature_dim(base_feats_raw)

                # 2. run every ASR system
                hypos = {
                    algo_name: self.s2t_pool.transcribe_with_algorithm(audio, algo_name)
                    for algo_name in self.algorithm_names
                }

                newly_generated_samples.append(
                    {
                        "ref_text": ref,
                        "base_feats": base_feats,
                        "hypos": hypos,
                    }
                )

            # Combine old and new data, then save back to cache
            full_dataset_samples = cached_samples + newly_generated_samples
            self._save_to_cache(cache_path, full_dataset_samples)
            all_samples_data.extend(full_dataset_samples)

        # ----------------------------------------------------------------- #
        # Build X / y matrices
        # ----------------------------------------------------------------- #
        logger.info(f"Building training matrices from {len(all_samples_data)} total samples.")
        X_rows, y_rows, sample_ids, algo_ids = [], [], [], []
        for sample_idx, sample_data in enumerate(tqdm(all_samples_data, desc="Building matrices")):
            base_feats = self._ensure_feature_dim(sample_data["base_feats"])
            ref = sample_data["ref_text"]
            for algo_idx, algo_name in enumerate(self.algorithm_names):
                hypo = sample_data["hypos"].get(algo_name, "")
                wer_score = wer(ref, hypo) if hypo and ref else 1.0

                feats_plus = np.concatenate(
                    [base_feats, _algo_one_hot(algo_idx, self.num_algorithms)]
                )
                X_rows.append(feats_plus)
                y_rows.append(wer_score)
                sample_ids.append(sample_idx)
                algo_ids.append(algo_idx)

        X = np.asarray(X_rows, dtype=np.float32)
        y = np.asarray(y_rows, dtype=np.float32)
        logger.info(
            f"✓ Collected {X.shape[0]} rows "
            f"({len(all_samples_data)} utterances, {self.num_algorithms} algorithms each)."
        )
        return X, y, sample_ids, algo_ids

    # --------------------------------------------------------------------- #
    # Cache helpers
    # --------------------------------------------------------------------- #
    def _load_valid_cache(self, cache_path: str) -> List[dict]:
        """Loads cache if it exists and is valid for the current algorithm pool."""
        if not os.path.exists(cache_path):
            return []

        try:
            logger.info(f"  → Found cache file: {os.path.basename(cache_path)}. Verifying...")
            cache_content = joblib.load(cache_path)

            # Check for new format with metadata
            if isinstance(cache_content, dict) and 'metadata' in cache_content and 'data' in cache_content:
                metadata = cache_content['metadata']
                if set(metadata.get('algorithms', [])) == set(self.algorithm_names):
                    logger.info(
                        f"  ✓ Cache is valid. Loaded {len(cache_content['data'])} pre-processed samples."
                    )
                    return cache_content['data']
                else:
                    logger.warning("  ! Cache is stale (algorithm pool changed). Ignoring and regenerating.")
                    return []
            else:  # Old format without metadata
                logger.warning("  ! Old cache format detected. Ignoring and regenerating.")
                return []
        except Exception as e:
            logger.error(f"  ✗ Error loading cache file {cache_path}: {e}. Ignoring and regenerating.")
            return []

    def _save_to_cache(self, cache_path: str, data: List[dict]):
        """Saves data and metadata to a cache file."""
        metadata = {'algorithms': self.algorithm_names}
        cache_content_to_save = {'metadata': metadata, 'data': data}
        try:
            joblib.dump(cache_content_to_save, cache_path)
            logger.info(f"  Saved {len(data)} samples to cache at {os.path.basename(cache_path)}")
        except Exception as e:
            logger.error(f"  Could not save cache to {cache_path}: {e}")

    # --------------------------------------------------------------------- #
    # Training back-ends
    # --------------------------------------------------------------------- #
    def _train_xgb(self, X_train_s, y_train, X_test_s, y_test) -> xgb.Booster:
        dtrain = xgb.DMatrix(X_train_s, label=y_train)
        dvalid = xgb.DMatrix(X_test_s, label=y_test)
        logger.info("Training XGBoost …")
        params = dict(
            objective="reg:squarederror",
            eval_metric="rmse",
            max_depth=self.cfg["xgb_params"]["max_depth"],
            eta=self.cfg["xgb_params"]["eta"],
            subsample=0.9,
            colsample_bytree=0.8,
            tree_method="gpu_hist" if "cuda" in self.device else "hist",
            seed=42,
        )
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.cfg["xgb_params"]["num_boost"],
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=100,
            verbose_eval=100,
        )
        return model

    def _train_mlp(self, X_train_s, y_train, X_test_s, y_test) -> MetaLearnerMLP:
        input_dim = X_train_s.shape[1]
        p = self.cfg["mlp_params"]
        model = MetaLearnerMLP(
            input_dim,
            hidden_dim=p["hidden_dim"],
            num_layers=p["num_layers"],
            dropout=p["dropout"],
            ).to(self.device)

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train_s).float(), torch.from_numpy(y_train).float()
            ),
            batch_size=p["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_test_s).float(), torch.from_numpy(y_test).float()
            ),
            batch_size=p["batch_size"],
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=p["lr"],
            weight_decay=p.get("weight_decay", 0.0),
        )
        criterion = nn.MSELoss()
        best_val_loss, patience_counter = float("inf"), 0

        patience_limit = p.get("early_stop_patience", 10)
        logger.info(
            f"Training MLP for up to {p['epochs']} epochs "
            f"(early stop patience = {patience_limit})..."
        )
        for epoch in range(p["epochs"]):
            model.train()
            train_loss = sum(
                self._train_one_epoch(model, train_loader, criterion, optimizer)
            ) / len(train_loader)

            model.eval()
            val_loss = sum(self._validate_one_epoch(model, val_loader, criterion)) / len(
                val_loader
            )

            logger.info(
                f"Epoch {epoch+1}/{p['epochs']}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "temp_best_mlp.pth")
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    logger.info("Early stopping triggered.")
                    break

        model.load_state_dict(torch.load("temp_best_mlp.pth"))
        os.remove("temp_best_mlp.pth")
        return model

    def _train_one_epoch(self, model, loader, criterion, optimizer):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            yield loss.item()

    def _validate_one_epoch(self, model, loader, criterion):
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                yield criterion(outputs, batch_y).item()

    # --------------------------------------------------------------------- #
    # Evaluation helper
    # --------------------------------------------------------------------- #
    def _evaluate_performance(self, X_test_s, y_test, sample_test, algo_test):
        if self.learner_type == "xgboost":
            preds = self.model.predict(xgb.DMatrix(X_test_s))
        else:  # MLP
            self.model.eval()
            with torch.no_grad():
                test_tensor = torch.from_numpy(X_test_s).float().to(self.device)
                preds = self.model(test_tensor).cpu().numpy()

        act_wers, prd_wers = defaultdict(dict), defaultdict(dict)
        for sid, aid, a_w, p_w in zip(sample_test, algo_test, y_test, preds):
            act_wers[sid][aid], prd_wers[sid][aid] = a_w, p_w

        picked, oracle = [], []
        static = {i: [] for i in range(self.num_algorithms)}
        for sid in act_wers:
            best_algo = min(prd_wers[sid], key=prd_wers[sid].get)
            picked.append(act_wers[sid][best_algo])
            oracle.append(min(act_wers[sid].values()))
            for i in range(self.num_algorithms):
                static[i].append(act_wers[sid][i])

        print("\n--- Meta-Learner Evaluation ---------------------------")
        print(f"ADAS2T Picked WER : {np.mean(picked):.4f}")
        print(f"Oracle  (best)    : {np.mean(oracle):.4f}")
        for i, name in enumerate(self.algorithm_names):
            print(f"Static '{name[:30]:30}' : {np.mean(static[i]):.4f}")
        print("-------------------------------------------------------\n")

    # --------------------------------------------------------------------- #
    # Public interface
    # --------------------------------------------------------------------- #
    def train_and_evaluate(self, X, y, sample_ids, algo_ids):
        gss = GroupShuffleSplit(test_size=0.20, random_state=42)
        train_idx, test_idx = next(gss.split(X, groups=sample_ids))

        X_train_s = self.scaler.fit_transform(X[train_idx])
        X_test_s = self.scaler.transform(X[test_idx])

        y_train, y_test = y[train_idx], y[test_idx]

        if self.learner_type == "xgboost":
            self.model = self._train_xgb(X_train_s, y_train, X_test_s, y_test)
        elif self.learner_type == "mlp":
            self.model = self._train_mlp(X_train_s, y_train, X_test_s, y_test)

        self._evaluate_performance(
            X_test_s,
            y_test,
            [sample_ids[i] for i in test_idx],
            [algo_ids[i] for i in test_idx],
        )

    def save_model(self):
        assert self.model is not None, "Model not trained yet."
        if self.learner_type == "xgboost":
            self.model.save_model(self.cfg["model_path"])
        elif self.learner_type == "mlp":
            p = self.cfg["mlp_params"]
            model_payload = {
                'state_dict': self.model.state_dict(),
                'hyperparams': {
                    'hidden_dim': p['hidden_dim'],
                    'num_layers': p['num_layers'],
                    'dropout': p['dropout'],
                }
            }
            torch.save(model_payload, self.cfg["model_path"])

        joblib.dump(self.scaler, self.cfg["scaler_path"])
        joblib.dump(self.algorithm_names, self.cfg["algorithms_path"])
        logger.info(f"✓ Model   : {self.cfg['model_path']}")
        logger.info(f"✓ Scaler  : {self.cfg['scaler_path']}")
        logger.info(f"✓ Alg list: {self.cfg['algorithms_path']}")

    def run_full_training_pipeline(self):
        X, y, sample_ids, algo_ids = self.prepare_training_data()
        self.train_and_evaluate(X, y, sample_ids, algo_ids)
        self.save_model()
        logger.info("✅ ADAS2T training finished.")
# adas2t/trainer.py
import logging
import os
from collections import defaultdict
from typing import Tuple, List, Dict, Union
import itertools

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from jiwer import wer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..data_handler import load_and_prepare_dataset
from .feature_extractor import AudioFeatureExtractor
from .runtime import S2TAlgorithmPool
from .models import MetaLearnerMLP, MetaLearnerTransformer, MetaLearnerAST # Added MetaLearnerAST

logger = logging.getLogger(__name__)


class ADAS2TTrainer:
    def __init__(self, config: dict):
        self.cfg = config
        self.device = config["device"]
        self.learner_type = config["learner_type"]
        self.cache_dir = config["cache_dir"]
        os.makedirs(self.cache_dir, exist_ok=True)

        self.s2t_pool = S2TAlgorithmPool(
            algorithm_names=config["algorithm_pool_models"],
            device=self.device,
            torch_dtype=config["torch_dtype"],
        )
        self.algorithm_names: List[str] = self.s2t_pool.algorithm_names
        self.num_algorithms: int = len(self.algorithm_names)
        
        # The acoustic feature extractor is only needed for non-AST models
        if self.learner_type != "ast":
            self.feature_extractor = AudioFeatureExtractor(sample_rate=16_000, device=self.device)
        
        # Scalers are only used for tabular models (e.g., Transformer)
        self.scalers: Dict[str, StandardScaler] = {}
        self.model = None

        self.feature_keys: List[str] | None = None
        self.feature_dims: Dict[str, int] | None = None

    def _get_cache_path_and_type(self, ds_name, ds_cfg) -> Tuple[str, str]:
        """Determines cache path and type based on the learner model."""
        ds_display = f"{ds_name.replace('/', '_')}-{ds_cfg.get('config', ds_cfg.get('lang_config', ''))}-{ds_cfg['split']}"
        if self.learner_type == "ast":
            return os.path.join(self.cache_dir, f"raw_audio_{ds_display}.pkl"), "audio"
        else:
            # All other models use structured acoustic features
            return os.path.join(self.cache_dir, f"structured_{ds_display}.pkl"), "features"

    def prepare_training_data(self) -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], np.ndarray, List[int], List[int]]:
        """
        Prepares training data by generating or loading from cache.
        - For 'ast' learner: returns raw audio waveforms.
        - For other learners: returns a dictionary of structured feature matrices.
        """
        logger.info(f"Preparing meta-learner training data for '{self.learner_type}' model...")
        all_samples_data = []

        for ds_name, ds_cfg in self.cfg["training_datasets"]:
            cache_path, cache_type = self._get_cache_path_and_type(ds_name, ds_cfg)
            cached_samples = self._load_valid_cache(cache_path)
            num_required = self.cfg["num_training_samples"]
            num_to_generate = num_required - len(cached_samples)

            if num_to_generate <= 0:
                logger.info(f"  Using {num_required} samples from cache for {ds_name}.")
                all_samples_data.extend(cached_samples[:num_required])
                continue

            logger.info(f"  Need to generate {num_to_generate} new samples for {ds_name}.")
            dataset_stream = load_and_prepare_dataset(ds_name, ds_cfg, num_samples=None)
            dataset_iterator = itertools.islice(dataset_stream, len(cached_samples), num_required)
            newly_generated_samples = []
            
            pbar_desc = f"   generating ({ds_name[:25]})"
            for ex in tqdm(dataset_iterator, desc=pbar_desc, total=num_to_generate):
                audio, ref = ex["audio"]["array"], ex[ds_cfg["field"]].strip().lower()
                if not ref or len(audio) < 160: continue # Skip empty/tiny audio

                # Generate transcriptions from the S2T pool
                hypos = {
                    algo_name: self.s2t_pool.transcribe_with_algorithm(audio, algo_name)
                    for algo_name in self.algorithm_names
                }
                
                # --- DATA FORK: Store raw audio for AST, features for others ---
                if cache_type == "audio":
                    # For AST, we only need the raw audio and its WERs
                    sample = {"ref_text": ref, "waveform": audio, "hypos": hypos}
                else:
                    # For tabular models, extract acoustic features
                    base_feats_dict = self.feature_extractor.extract_all_features(audio)
                    sample = {"ref_text": ref, "base_feats": base_feats_dict, "hypos": hypos}
                
                newly_generated_samples.append(sample)
            
            full_dataset_samples = cached_samples + newly_generated_samples
            self._save_to_cache(cache_path, full_dataset_samples)
            all_samples_data.extend(full_dataset_samples)

        if not all_samples_data:
            raise ValueError("No data could be generated or loaded from cache.")
        
        logger.info(f"Building training matrices from {len(all_samples_data)} total samples.")
        
        # --- Build final matrices based on learner type ---
        y_rows, sample_ids, algo_ids = [], [], []
        
        if self.learner_type == "ast":
            waveforms = []
            max_len = max(len(s["waveform"]) for s in all_samples_data)
            logger.info(f"Padding all audio samples to max length: {max_len / 16000:.2f}s")
            
            for sample_idx, sample_data in enumerate(tqdm(all_samples_data, desc="Building matrices")):
                padded_wav = np.pad(sample_data["waveform"], (0, max_len - len(sample_data["waveform"])))
                ref = sample_data["ref_text"]
                for algo_idx, algo_name in enumerate(self.algorithm_names):
                    waveforms.append(padded_wav) # Append one waveform per algorithm
                    hypo = sample_data["hypos"].get(algo_name, "")
                    y_rows.append(wer(ref, hypo) if hypo and ref else 1.0)
                    sample_ids.append(sample_idx)
                    algo_ids.append(algo_idx)

            X_data = np.asarray(waveforms, dtype=np.float32)
            y = np.asarray(y_rows, dtype=np.float32)
            return X_data, y, sample_ids, algo_ids

        else: # For tabular models ('transformer', 'mlp')
            self.feature_keys = sorted(all_samples_data[0]['base_feats'].keys())
            logger.info(f"Detected feature groups: {self.feature_keys}")
            X_dict = defaultdict(list)

            for sample_idx, sample_data in enumerate(tqdm(all_samples_data, desc="Building matrices")):
                base_feats_dict = sample_data["base_feats"]
                ref = sample_data["ref_text"]
                for algo_idx, algo_name in enumerate(self.algorithm_names):
                    for key in self.feature_keys:
                        X_dict[key].append(base_feats_dict[key])
                    hypo = sample_data["hypos"].get(algo_name, "")
                    y_rows.append(wer(ref, hypo) if hypo and ref else 1.0)
                    sample_ids.append(sample_idx)
                    algo_ids.append(algo_idx)
            
            X_dict_np = {key: np.asarray(val, dtype=np.float32) for key, val in X_dict.items()}
            X_dict_np['algo_id'] = np.asarray(algo_ids, dtype=np.int64)
            y = np.asarray(y_rows, dtype=np.float32)
            logger.info(f"✓ Collected {y.shape[0]} rows across {len(self.feature_keys)} feature groups.")
            return X_dict_np, y, sample_ids, algo_ids

    def _load_valid_cache(self, cache_path: str) -> List[dict]:
        if not os.path.exists(cache_path): return []
        try:
            logger.info(f"  → Found cache file: {os.path.basename(cache_path)}. Verifying...")
            cache_content = joblib.load(cache_path)
            if isinstance(cache_content, dict) and 'metadata' in cache_content and 'data' in cache_content:
                metadata = cache_content['metadata']
                if set(metadata.get('algorithms', [])) == set(self.algorithm_names):
                    logger.info(f"  ✓ Cache is valid. Loaded {len(cache_content['data'])} pre-processed samples.")
                    return cache_content['data']
            logger.warning("  ! Cache is stale or invalid. Ignoring and regenerating.")
            return []
        except Exception as e:
            logger.error(f"  ✗ Error loading cache file {cache_path}: {e}. Ignoring.")
            return []

    def _save_to_cache(self, cache_path: str, data: List[dict]):
        metadata = {'algorithms': self.algorithm_names}
        try:
            joblib.dump({'metadata': metadata, 'data': data}, cache_path)
            logger.info(f"  Saved {len(data)} samples to cache at {os.path.basename(cache_path)}")
        except Exception as e:
            logger.error(f"  Could not save cache to {cache_path}: {e}")

    def _train_pytorch_model(self, X_train_dict, y_train, X_val_dict, y_val, algo_ids_train, algo_ids_val) -> nn.Module:
        """Trains tabular models like the FT-Transformer."""
        p = self.cfg["transformer_params"]
        model = MetaLearnerTransformer(
            feature_dims=self.feature_dims, num_algorithms=self.num_algorithms,
            d_model=p["d_model"], nhead=p["nhead"], num_encoder_layers=p["num_encoder_layers"],
            dropout=self.cfg.get('transformer_dropout', 0.1)
        ).to(self.device)
        
        train_tensors = [torch.from_numpy(X_train_dict[key]) for key in self.feature_keys]
        train_tensors.append(torch.from_numpy(algo_ids_train).long())
        train_tensors.append(torch.from_numpy(y_train).float())
        
        val_tensors = [torch.from_numpy(X_val_dict[key]) for key in self.feature_keys]
        val_tensors.append(torch.from_numpy(algo_ids_val).long())
        val_tensors.append(torch.from_numpy(y_val).float())

        train_ds = TensorDataset(*train_tensors)
        val_ds = TensorDataset(*val_tensors)
        train_dl = DataLoader(train_ds, batch_size=p["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=p["batch_size"] * 2, num_workers=4, pin_memory=True)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p.get("weight_decay", 0.0))
        criterion = nn.MSELoss()
        best_val_loss, patience_counter = float("inf"), 0
        patience_limit = p.get("early_stop_patience", 10)

        logger.info(f"Training {self.learner_type.upper()} for up to {p['epochs']} epochs...")
        for epoch in range(p["epochs"]):
            model.train()
            train_losses = []
            for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{p['epochs']} Train", leave=False):
                y_batch = batch[-1].to(self.device)
                x_dict = {key: batch[i].to(self.device) for i, key in enumerate(self.feature_keys)}
                x_dict['algo_id'] = batch[len(self.feature_keys)].to(self.device)
                optimizer.zero_grad()
                outputs = model(x_dict)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in tqdm(val_dl, desc=f"Epoch {epoch+1}/{p['epochs']} Val", leave=False):
                    y_batch = batch[-1].to(self.device)
                    x_dict = {key: batch[i].to(self.device) for i, key in enumerate(self.feature_keys)}
                    x_dict['algo_id'] = batch[len(self.feature_keys)].to(self.device)
                    outputs = model(x_dict)
                    val_losses.append(criterion(outputs, y_batch).item())
            
            avg_train_loss, avg_val_loss = np.mean(train_losses), np.mean(val_losses)
            logger.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "temp_best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    logger.info("Early stopping triggered.")
                    break
        
        model.load_state_dict(torch.load("temp_best_model.pth"))
        os.remove("temp_best_model.pth")
        return model

    def _train_ast_model(self, wav_train, y_train, wav_val, y_val, algo_ids_train, algo_ids_val) -> nn.Module:
        """Trains the AST-based transfer learning model."""
        p = self.cfg["ast_params"]
        model = MetaLearnerAST(
            num_algorithms=self.num_algorithms,
            pretrained_name=p["pretrained_name"],
            train_backbone=not p["freeze_backbone"],
        ).to(self.device)

        train_ds = TensorDataset(torch.from_numpy(wav_train), torch.from_numpy(algo_ids_train).long(), torch.from_numpy(y_train).float())
        val_ds = TensorDataset(torch.from_numpy(wav_val), torch.from_numpy(algo_ids_val).long(), torch.from_numpy(y_val).float())
        train_dl = DataLoader(train_ds, batch_size=p["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=p["batch_size"] * 2, num_workers=4, pin_memory=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"])
        criterion = nn.MSELoss()
        best_val_loss, patience_counter = float("inf"), 0
        patience_limit = self.cfg.get("early_stop_patience", 5) # Can use a shared param

        logger.info(f"Training {self.learner_type.upper()} for up to {p['epochs']} epochs...")
        for epoch in range(p["epochs"]):
            model.train()
            train_losses = []
            for wav_batch, algo_batch, y_batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{p['epochs']} Train", leave=False):
                wav_batch, algo_batch, y_batch = wav_batch.to(self.device), algo_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(wav_batch, algo_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            val_losses = []
            with torch.no_grad():
                for wav_batch, algo_batch, y_batch in tqdm(val_dl, desc=f"Epoch {epoch+1}/{p['epochs']} Val", leave=False):
                    wav_batch, algo_batch, y_batch = wav_batch.to(self.device), algo_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(wav_batch, algo_batch)
                    val_losses.append(criterion(outputs, y_batch).item())

            avg_train_loss, avg_val_loss = np.mean(train_losses), np.mean(val_losses)
            logger.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "temp_best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    logger.info("Early stopping triggered.")
                    break
        
        model.load_state_dict(torch.load("temp_best_model.pth"))
        os.remove("temp_best_model.pth")
        return model

    def _evaluate_performance(self, X_test, y_test, sample_test, algo_test):
        self.model.eval()
        preds = []
        if self.learner_type == "ast":
            test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(algo_test).long())
            test_dl = DataLoader(test_ds, batch_size=self.cfg["ast_params"]["batch_size"] * 2)
            with torch.no_grad():
                for wav_batch, algo_batch in tqdm(test_dl, desc="Evaluating"):
                    wav_batch, algo_batch = wav_batch.to(self.device), algo_batch.to(self.device)
                    preds.append(self.model(wav_batch, algo_batch).cpu().numpy())
        
        elif self.learner_type == "transformer":
            test_tensors = [torch.from_numpy(X_test[key]) for key in self.feature_keys]
            test_tensors.append(torch.from_numpy(algo_test).long())
            test_ds = TensorDataset(*test_tensors)
            test_dl = DataLoader(test_ds, batch_size=self.cfg["transformer_params"]["batch_size"] * 2)
            with torch.no_grad():
                for batch in tqdm(test_dl, desc="Evaluating"):
                    x_dict = {key: batch[i].to(self.device) for i, key in enumerate(self.feature_keys)}
                    x_dict['algo_id'] = batch[len(self.feature_keys)].to(self.device)
                    preds.append(self.model(x_dict).cpu().numpy())
        else:
            raise ValueError(f"Unsupported learner type for evaluation: {self.learner_type}")
        
        preds = np.concatenate(preds)
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

    def train_and_evaluate(self, X_data, y, sample_ids, algo_ids):
        # Split data into train/validation sets, ensuring all data points
        # for a given audio sample stay in the same set.
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        idx_placeholder = X_data if isinstance(X_data, np.ndarray) else X_data[next(iter(X_data))]
        train_idx, val_idx = next(gss.split(idx_placeholder, groups=sample_ids))
        
        y_train, y_val = y[train_idx], y[val_idx]
        algo_ids_train, algo_ids_val = np.array(algo_ids)[train_idx], np.array(algo_ids)[val_idx]
        sample_val = [sample_ids[i] for i in val_idx]

        if self.learner_type == "ast":
            wav_train, wav_val = X_data[train_idx], X_data[val_idx]
            self.model = self._train_ast_model(wav_train, y_train, wav_val, y_val, algo_ids_train, algo_ids_val)
            self._evaluate_performance(wav_val, y_val, sample_val, algo_ids_val)
        
        elif self.learner_type == "transformer":
            X_train_dict, X_val_dict = {}, {}
            self.feature_dims = {}
            for key in self.feature_keys:
                self.scalers[key] = StandardScaler()
                X_train_dict[key] = self.scalers[key].fit_transform(X_data[key][train_idx])
                X_val_dict[key] = self.scalers[key].transform(X_data[key][val_idx])
                self.feature_dims[key] = X_train_dict[key].shape[1]
            
            self.model = self._train_pytorch_model(X_train_dict, y_train, X_val_dict, y_val, algo_ids_train, algo_ids_val)
            self._evaluate_performance(X_val_dict, y_val, sample_val, algo_ids_val)

        else:
            raise NotImplementedError(f"Training pipeline for learner '{self.learner_type}' is not implemented.")

    def save_model(self):
        """Saves the trained model and all necessary metadata for inference."""
        assert self.model is not None, "Model not trained yet."
        
        hyperparams = {'learner_type': self.learner_type}
        
        if self.learner_type == 'transformer':
            p = self.cfg["transformer_params"]
            hyperparams.update({
                'feature_dims': self.feature_dims, 'num_algorithms': self.num_algorithms,
                'd_model': p['d_model'], 'nhead': p['nhead'],
                'num_encoder_layers': p['num_encoder_layers'],
                'dropout': self.cfg.get('transformer_dropout', 0.1)
            })
        elif self.learner_type == 'ast':
            p = self.cfg["ast_params"]
            hyperparams.update({
                "num_algorithms": self.num_algorithms,
                "pretrained_name": p["pretrained_name"],
            })
        
        torch.save({
            'state_dict': self.model.state_dict(),
            'hyperparams': hyperparams
        }, self.cfg["model_path"])

        # For AST, scalers is an empty dict, which is fine.
        # This ensures the file is always created, simplifying the ADAS2THandler logic.
        joblib.dump(self.scalers, self.cfg["scaler_path"])
        joblib.dump(self.algorithm_names, self.cfg["algorithms_path"])

        logger.info(f"✓ Model       : {self.cfg['model_path']}")
        logger.info(f"✓ Scalers     : {self.cfg['scaler_path']}")
        logger.info(f"✓ Algorithms  : {self.cfg['algorithms_path']}")

    def run_full_training_pipeline(self):
        X_data, y, sample_ids, algo_ids = self.prepare_training_data()
        self.train_and_evaluate(X_data, y, sample_ids, algo_ids)
        self.save_model()
        logger.info("✅ ADAS2T training finished.")
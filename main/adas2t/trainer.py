# adas2t/trainer.py
#
# Train the ADAS2T XGBoost meta-learner *with an explicit algorithm
# one-hot*, so the model can tell which ASR system a row belongs to.

import logging
from collections import defaultdict
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from jiwer import wer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..data_handler import load_and_prepare_dataset
from .feature_extractor import AudioFeatureExtractor
from .runtime import S2TAlgorithmPool

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------
def _algo_one_hot(idx: int, k: int) -> np.ndarray:
    """
    Make a 1-of-K vector for algorithm `idx` out of `k` algorithms.
    """
    v = np.zeros(k, dtype=np.float32)
    v[idx] = 1.0
    return v


# ------------------------------------------------------------------
# Main trainer
# ------------------------------------------------------------------
class ADAS2TTrainer:
    """
    Trainer for the ADAS2T XGBoost meta-learner *with algorithm ID*.
    The learner now receives (acoustic features + algorithm one-hot)
    and predicts the expected WER for that •audio, algo• pair.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.device = config["device"]

        # Pool of candidate ASR systems
        self.s2t_pool = S2TAlgorithmPool(
            algorithm_names=config["algorithm_pool_models"],
            device=self.device,
            torch_dtype=config["torch_dtype"],
        )
        self.algorithm_names: List[str] = self.s2t_pool.algorithm_names
        self.num_algorithms: int = len(self.algorithm_names)

        # Feature extractor + scaler
        self.feature_extractor = AudioFeatureExtractor(sample_rate=16_000, device=self.device)
        self.scaler = StandardScaler()

        self.model: xgb.Booster | None = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        """
        Returns
        -------
        X : ndarray, shape (N·K , D+K)
            Acoustic features concatenated with algorithm one-hot.
        y : ndarray, shape (N·K ,)
            Corresponding WER for that audio/algorithm pair.
        sample_ids : list[int]
            Same length as `X`, id of the *original* utterance
            (used for grouped train / test split and evaluation).
        algo_ids : list[int]
            Algorithm index (0 … K-1) for each row in `X`.
        """
        logger.info("Preparing meta-learner training data …")
        X_rows, y_rows, sample_ids, algo_ids = [], [], [], []

        datasets_to_use = self.cfg["training_datasets"]
        sample_counter = 0  # unique id per utterance

        for ds_name, ds_cfg in datasets_to_use:
            ds_display = f"{ds_name}-{ds_cfg.get('config')}" if "config" in ds_cfg else ds_name
            logger.info(f" • Dataset: {ds_display}")

            dataset = load_and_prepare_dataset(
                ds_name, ds_cfg, num_samples=self.cfg["num_training_samples"]
            )

            for ex in tqdm(dataset, desc=f"   extracting ({ds_name[:18]})"):
                audio = ex["audio"]["array"]
                ref = ex[ds_cfg["field"]].strip().lower()

                if not ref or len(audio) == 0:
                    continue

                base_feats = self.feature_extractor.extract_all_features(audio)

                for algo_idx, algo_name in enumerate(self.algorithm_names):
                    hypo = self.s2t_pool.transcribe_with_algorithm(audio, algo_name)
                    wer_score = wer(ref, hypo) if hypo else 1.0  # 1.0 => max error

                    feats_plus = np.concatenate([base_feats, _algo_one_hot(algo_idx, self.num_algorithms)])
                    X_rows.append(feats_plus)
                    y_rows.append(wer_score)
                    sample_ids.append(sample_counter)
                    algo_ids.append(algo_idx)

                sample_counter += 1

        X = np.asarray(X_rows, dtype=np.float32)
        y = np.asarray(y_rows, dtype=np.float32)

        logger.info(f"✓ Collected {X.shape[0]} rows "
                    f"({sample_counter} utterances, {self.num_algorithms} algorithms each).")
        return X, y, sample_ids, algo_ids

    # ------------------------------------------------------------------
    # Train & evaluate
    # ------------------------------------------------------------------
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: List[int],
        algo_ids: List[int],
    ) -> xgb.Booster:
        """
        Train the XGB regressor and print a small evaluation summary.
        """
        # Grouped split => no utterance appears in both train & test
        gss = GroupShuffleSplit(test_size=0.20, random_state=42)
        train_idx, test_idx = next(gss.split(X, groups=sample_ids))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        algo_test = [algo_ids[i] for i in test_idx]
        sample_test = [sample_ids[i] for i in test_idx]

        # Scale
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # Train
        dtrain = xgb.DMatrix(X_train_s, label=y_train)
        dvalid = xgb.DMatrix(X_test_s, label=y_test)

        logger.info("Training XGBoost …")
        params = dict(
            objective="reg:squarederror",
            eval_metric="rmse",
            max_depth=6,
            eta=0.1,
            subsample=0.9,
            colsample_bytree=0.8,
            seed=42,
        )
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=600,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=40,
            verbose_eval=50,
        )

        # --------------------------------------------------------------
        # Evaluation: how good is the *selection*?
        # --------------------------------------------------------------
        preds = self.model.predict(dvalid)

        # collect per-utterance info
        act_wers = defaultdict(dict)   # {sample_id: {algo: actual}}
        prd_wers = defaultdict(dict)   # {sample_id: {algo: pred}}

        for sid, aid, a_w, p_w in zip(sample_test, algo_test, y_test, preds):
            act_wers[sid][aid] = a_w
            prd_wers[sid][aid] = p_w

        picked, oracle = [], []
        static = {i: [] for i in range(self.num_algorithms)}

        for sid in act_wers:
            # meta-learner pick
            best_algo = min(prd_wers[sid], key=prd_wers[sid].get)
            picked.append(act_wers[sid][best_algo])

            # oracle lowest possible
            oracle.append(min(act_wers[sid].values()))

            # static baselines
            for i in range(self.num_algorithms):
                static[i].append(act_wers[sid][i])

        print("\n--- Meta-Learner Evaluation ---------------------------")
        print(f"ADAS2T Picked WER : {np.mean(picked):.4f}")
        print(f"Oracle  (best)    : {np.mean(oracle):.4f}")
        for i, name in enumerate(self.algorithm_names):
            print(f"Static '{name[:30]:30}' : {np.mean(static[i]):.4f}")
        print("-------------------------------------------------------\n")

        return self.model

    # ------------------------------------------------------------------
    # Saving utilities
    # ------------------------------------------------------------------
    def save_model(self):
        """
        Persist Booster, scaler and algorithm list to disk.
        """
        assert self.model is not None, "Model not trained yet."
        self.model.save_model(self.cfg["model_path"])
        joblib.dump(self.scaler, self.cfg["scaler_path"])
        joblib.dump(self.algorithm_names, self.cfg["algorithms_path"])
        logger.info(f"✓ Model   : {self.cfg['model_path']}")
        logger.info(f"✓ Scaler  : {self.cfg['scaler_path']}")
        logger.info(f"✓ Alg list: {self.cfg['algorithms_path']}")

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def run_full_training_pipeline(self):
        """
        End-to-end: extract data, train, evaluate, save.
        """
        X, y, sample_ids, algo_ids = self.prepare_training_data()
        self.train_and_evaluate(X, y, sample_ids, algo_ids)
        self.save_model()
        logger.info("✅ ADAS2T training finished.")
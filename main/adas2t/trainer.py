# adas2t/trainer.py

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from jiwer import wer
from tqdm import tqdm
import joblib
import logging
from typing import Tuple

from .feature_extractor import AudioFeatureExtractor
from .runtime import S2TAlgorithmPool
from ..data_handler import load_and_prepare_dataset

logger = logging.getLogger(__name__)

class CustomXGBoostObjective:
    """Custom XGBoost objective function for WER minimization."""
    def __init__(self, wer_matrix: np.ndarray, epsilon: float = 1e-6):
        self.wer_matrix = wer_matrix
        self.epsilon = epsilon
    
    def __call__(self, y_pred: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        N, K = self.wer_matrix.shape
        logits = y_pred.reshape(N, K)
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        
        expected_losses = np.sum(probs * self.wer_matrix, axis=1)
        gradients = probs * (self.wer_matrix - expected_losses[:, np.newaxis])
        hessians = np.maximum(gradients * (1 - 2 * probs), self.epsilon)
        
        return gradients.flatten(), hessians.flatten()

class ADAS2TTrainer:
    """Main trainer class for ADAS2T XGBoost meta-learner."""
    
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.s2t_pool = S2TAlgorithmPool(
            config['algorithm_pool_models'],
            self.device,
            config['torch_dtype']
        )
        self.feature_extractor = AudioFeatureExtractor(sample_rate=16000, device=self.device)
        self.scaler = StandardScaler()
        self.algorithm_names = self.s2t_pool.algorithm_names
    
    def prepare_training_data(self):
        """Prepares training data by extracting features and computing the WER matrix."""
        logger.info("Preparing training data for ADAS2T...")
        features_list, wer_matrix_list = [], []
        
        # Use a single, representative dataset for training the meta-learner
        ds_name, ds_cfg = self.config['training_dataset']
        dataset = load_and_prepare_dataset(ds_name, ds_cfg, self.config['num_training_samples'])

        for sample in tqdm(dataset, desc="Generating WER Matrix"):
            audio = sample['audio']['array']
            reference_text = sample[ds_cfg["field"]].strip().lower()

            if not reference_text: continue

            features = self.feature_extractor.extract_all_features(audio)
            
            wer_scores = []
            for algo_name in self.algorithm_names:
                transcription = self.s2t_pool.transcribe_with_algorithm(audio, algo_name)
                score = wer(reference_text, transcription) if transcription else 1.0
                wer_scores.append(score)
            
            features_list.append(features)
            wer_matrix_list.append(wer_scores)

        features_array = np.array(features_list)
        wer_matrix = np.array(wer_matrix_list)
        
        logger.info(f"Prepared training data: Features shape {features_array.shape}, WER matrix shape {wer_matrix.shape}")
        df = pd.DataFrame(wer_matrix, columns=self.algorithm_names)
        logger.info(f"Average WER per algorithm:\n{df.mean().to_string()}")
        
        return features_array, wer_matrix

    def train_and_evaluate(self, features: np.ndarray, wer_matrix: np.ndarray):
        """Trains the XGBoost meta-learner and evaluates its performance."""
        logger.info("Training XGBoost meta-learner...")
        X_train, X_test, W_train, W_test = train_test_split(features, wer_matrix, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        labels = W_train.flatten()
        dtrain = xgb.DMatrix(np.repeat(X_train_scaled, W_train.shape[1], axis=0), label=labels)        
        custom_obj = CustomXGBoostObjective(W_train)
        
        model = xgb.train(
            params={'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'seed': 42},
            dtrain=dtrain,
            obj=custom_obj,
            num_boost_round=150,
            early_stopping_rounds=15,
            evals=[(dtrain, 'train')],
            verbose_eval=25
        )
        
        logger.info("Evaluating meta-learner...")
        preds = model.predict(xgb.DMatrix(X_test_scaled)).reshape(X_test.shape[0], -1)
        best_algo_indices = np.argmin(preds, axis=1)
        
        selected_wers = W_test[np.arange(len(W_test)), best_algo_indices]
        oracle_wers = np.min(W_test, axis=1)
        
        print("\n--- Meta-Learner Evaluation ---")
        print(f"ADAS2T Predicted WER: {np.mean(selected_wers):.4f}")
        print(f"Oracle (Best Possible) WER: {np.mean(oracle_wers):.4f}")
        for i, name in enumerate(self.algorithm_names):
            print(f"Static '{name}' WER: {np.mean(W_test[:, i]):.4f}")
        print("-----------------------------\n")

        return model

    def save_model(self, model):
        """Saves the trained model, scaler, and algorithm names."""
        model.save_model(self.config['model_path'])
        joblib.dump(self.scaler, self.config['scaler_path'])
        joblib.dump(self.algorithm_names, self.config['algorithms_path'])
        logger.info(f"ADAS2T model saved to {self.config['model_path']}")
        logger.info(f"Scaler saved to {self.config['scaler_path']}")
        logger.info(f"Algorithm list saved to {self.config['algorithms_path']}")

    def run_full_training_pipeline(self):
        """Runs the complete training pipeline."""
        features, wer_matrix = self.prepare_training_data()
        model = self.train_and_evaluate(features, wer_matrix)
        self.save_model(model)
        logger.info("✅ ADAS2T training pipeline completed successfully!")
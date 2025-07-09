#!/usr/bin/env python3
"""
ADAS2T: Adaptive S2T Algorithm Selection via Meta-Learning
XGBoost-based Meta-Learner Training Script

This script implements the complete training pipeline for the XGBoost meta-learner
as described in the ADAS2T paper, including all proposed audio feature extraction
and the custom WER-based objective function.
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# HuggingFace and Speech Recognition
from datasets import load_dataset, Dataset
import torch
import transformers
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    HubertModel, HubertConfig
)
import speech_recognition as sr
from jiwer import wer

# Signal processing
import scipy.signal
from scipy.stats import entropy
import webrtcvad

# Audio feature extraction
import python_speech_features as psf
from python_speech_features import mfcc, delta
import parselmouth
from parselmouth.praat import call

# Progress tracking
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioFeatureExtractor:
    """
    Comprehensive audio feature extractor implementing all features described in ADAS2T paper:
    - Spectral-Phonetic Features (MFCC)
    - Prosodic Features (Pitch, VAD)
    - Neural Speech Embeddings (Whisper, Wav2Vec2, HuBERT)
    - Signal and Temporal Complexity Features
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3
        
        # Load pre-trained models for embeddings
        self._load_embedding_models()
        
    def _load_embedding_models(self):
        """Load pre-trained models for neural embeddings"""
        try:
            # Whisper
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            
            # Wav2Vec2
            self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            
            # HuBERT
            self.hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
            
            logger.info("Successfully loaded neural embedding models")
            
        except Exception as e:
            logger.warning(f"Failed to load some embedding models: {e}")
            self.whisper_processor = None
            self.whisper_model = None
            self.wav2vec2_processor = None
            self.wav2vec2_model = None
            self.hubert_model = None
    
    def extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features with derivatives as described in paper
        Returns 39 features: 13 static + 13 delta + 13 delta-delta
        """
        # Extract 13 MFCC coefficients (0-12)
        mfcc_features = mfcc(audio, self.sample_rate, numcep=13, nfilt=26, nfft=512)
        
        # Calculate first and second derivatives
        delta_features = delta(mfcc_features, N=2)
        delta_delta_features = delta(delta_features, N=2)
        
        # Concatenate all features
        all_features = np.concatenate([mfcc_features, delta_features, delta_delta_features], axis=1)
        
        # Return mean and std statistics
        feature_stats = np.concatenate([
            np.mean(all_features, axis=0),
            np.std(all_features, axis=0)
        ])
        
        return feature_stats
    
    def extract_prosodic_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract prosodic features including pitch and VAD-based features"""
        features = []
        
        try:
            # Convert to Parselmouth Sound object for pitch extraction
            sound = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
            
            # Extract pitch
            pitch = sound.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced frames
            
            if len(pitch_values) > 0:
                features.extend([
                    np.mean(pitch_values),  # Mean pitch
                    np.std(pitch_values),   # Pitch variability
                    np.max(pitch_values) - np.min(pitch_values),  # Pitch range
                    np.median(pitch_values)  # Median pitch
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # VAD-based features
        vad_features = self._extract_vad_features(audio)
        features.extend(vad_features)
        
        return np.array(features)
    
    def _extract_vad_features(self, audio: np.ndarray) -> List[float]:
        """Extract Voice Activity Detection based features"""
        # Convert to 16-bit PCM
        audio_16bit = (audio * 32767).astype(np.int16)
        
        # Frame-based VAD
        frame_length = int(0.030 * self.sample_rate)  # 30ms frames
        frame_step = int(0.010 * self.sample_rate)    # 10ms step
        
        voiced_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio_16bit) - frame_length, frame_step):
            frame = audio_16bit[i:i + frame_length]
            
            # Ensure frame is correct length and convert to bytes
            if len(frame) == frame_length:
                frame_bytes = frame.tobytes()
                try:
                    is_voiced = self.vad.is_speech(frame_bytes, self.sample_rate)
                    if is_voiced:
                        voiced_frames += 1
                    total_frames += 1
                except:
                    pass
        
        # Calculate VAD-based features
        silence_ratio = 1.0 - (voiced_frames / max(total_frames, 1))
        voice_activity_ratio = voiced_frames / max(total_frames, 1)
        
        return [silence_ratio, voice_activity_ratio]
    
    def extract_neural_embeddings(self, audio: np.ndarray) -> np.ndarray:
        """Extract neural speech embeddings from pre-trained models"""
        embeddings = []
        
        # Whisper embeddings
        if self.whisper_model is not None:
            try:
                inputs = self.whisper_processor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
                with torch.no_grad():
                    encoder_outputs = self.whisper_model.model.encoder(**inputs)
                    whisper_embedding = encoder_outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                    embeddings.append(whisper_embedding)
            except Exception as e:
                logger.warning(f"Whisper embedding extraction failed: {e}")
                embeddings.append(np.zeros(512))  # Default embedding size
        
        # Wav2Vec2 embeddings
        if self.wav2vec2_model is not None:
            try:
                inputs = self.wav2vec2_processor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.wav2vec2_model.wav2vec2(**inputs)
                    wav2vec2_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                    embeddings.append(wav2vec2_embedding)
            except Exception as e:
                logger.warning(f"Wav2Vec2 embedding extraction failed: {e}")
                embeddings.append(np.zeros(768))  # Default embedding size
        
        # HuBERT embeddings
        if self.hubert_model is not None:
            try:
                inputs = torch.tensor(audio).unsqueeze(0)
                with torch.no_grad():
                    outputs = self.hubert_model(inputs)
                    hubert_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                    embeddings.append(hubert_embedding)
            except Exception as e:
                logger.warning(f"HuBERT embedding extraction failed: {e}")
                embeddings.append(np.zeros(768))  # Default embedding size
        
        if embeddings:
            return np.concatenate(embeddings)
        else:
            return np.zeros(2048)  # Default combined embedding size
    
    def extract_signal_complexity_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract signal and temporal complexity features"""
        features = []
        
        # Signal-to-Noise Ratio (SNR)
        try:
            # Estimate noise as the quietest 10% of frames
            frame_energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            noise_level = np.percentile(frame_energy, 10)
            signal_level = np.mean(frame_energy)
            snr = 10 * np.log10(signal_level / max(noise_level, 1e-10))
            features.append(snr)
        except:
            features.append(0.0)
        
        # Zero Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # Spectral features
        try:
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
            
            # Spectral entropy
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            power_spectrum = magnitude ** 2
            
            spectral_entropy_values = []
            for frame in power_spectrum.T:
                if np.sum(frame) > 0:
                    normalized = frame / np.sum(frame)
                    spectral_entropy_values.append(entropy(normalized + 1e-10))
                else:
                    spectral_entropy_values.append(0.0)
            
            features.extend([np.mean(spectral_entropy_values), np.std(spectral_entropy_values)])
            
        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {e}")
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Temporal features
        duration = len(audio) / self.sample_rate
        features.append(duration)
        
        # Speech rate (approximate)
        try:
            # Estimate speech rate as voiced frames per second
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            frame_step = int(0.010 * self.sample_rate)    # 10ms step
            
            voiced_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio) - frame_length, frame_step):
                frame = audio[i:i + frame_length]
                # Simple energy-based voicing detection
                energy = np.sum(frame ** 2)
                if energy > np.mean(audio ** 2) * 0.1:  # Above 10% of average energy
                    voiced_frames += 1
                total_frames += 1
            
            speech_rate = voiced_frames / max(duration, 1.0)
            features.append(speech_rate)
            
            # Pause ratio
            pause_ratio = 1.0 - (voiced_frames / max(total_frames, 1))
            features.append(pause_ratio)
            
        except:
            features.extend([0.0, 0.0])
        
        return np.array(features)
    
    def extract_all_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract all features described in the paper"""
        all_features = []
        
        # MFCC features (78 features: 39 stats * 2)
        mfcc_features = self.extract_mfcc_features(audio)
        all_features.append(mfcc_features)
        
        # Prosodic features (6 features)
        prosodic_features = self.extract_prosodic_features(audio)
        all_features.append(prosodic_features)
        
        # Neural embeddings (2048 features combined)
        neural_features = self.extract_neural_embeddings(audio)
        all_features.append(neural_features)
        
        # Signal complexity features (10 features)
        signal_features = self.extract_signal_complexity_features(audio)
        all_features.append(signal_features)
        
        # Concatenate all features
        final_features = np.concatenate(all_features)
        
        # Handle any NaN or infinite values
        final_features = np.nan_to_num(final_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return final_features

class S2TAlgorithmPool:
    """Pool of Speech-to-Text algorithms"""
    
    def __init__(self):
        self.algorithms = {}
        self._initialize_algorithms()
    
    def _initialize_algorithms(self):
        """Initialize different S2T algorithms"""
        # Whisper
        try:
            self.algorithms['whisper'] = {
                'processor': WhisperProcessor.from_pretrained("openai/whisper-base"),
                'model': WhisperForConditionalGeneration.from_pretrained("openai/whisper-base"),
                'type': 'whisper'
            }
        except:
            logger.warning("Failed to load Whisper model")
        
        # Wav2Vec2
        try:
            self.algorithms['wav2vec2'] = {
                'processor': Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h"),
                'model': Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h"),
                'type': 'wav2vec2'
            }
        except:
            logger.warning("Failed to load Wav2Vec2 model")
        
        # Google Speech Recognition (as a baseline)
        try:
            self.algorithms['google'] = {
                'recognizer': sr.Recognizer(),
                'type': 'google'
            }
        except:
            logger.warning("Failed to initialize Google Speech Recognition")
    
    def transcribe_with_algorithm(self, audio: np.ndarray, algorithm_name: str, sample_rate: int = 16000) -> str:
        """Transcribe audio using specified algorithm"""
        if algorithm_name not in self.algorithms:
            return ""
        
        algo = self.algorithms[algorithm_name]
        
        try:
            if algo['type'] == 'whisper':
                inputs = algo['processor'](audio, sampling_rate=sample_rate, return_tensors="pt")
                with torch.no_grad():
                    predicted_ids = algo['model'].generate(**inputs)
                    transcription = algo['processor'].batch_decode(predicted_ids, skip_special_tokens=True)[0]
                return transcription
            
            elif algo['type'] == 'wav2vec2':
                inputs = algo['processor'](audio, sampling_rate=sample_rate, return_tensors="pt")
                with torch.no_grad():
                    logits = algo['model'](**inputs).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = algo['processor'].batch_decode(predicted_ids)[0]
                return transcription
            
            elif algo['type'] == 'google':
                # Convert to audio format for Google API
                audio_data = sr.AudioData((audio * 32767).astype(np.int16).tobytes(), sample_rate, 2)
                try:
                    transcription = algo['recognizer'].recognize_google(audio_data)
                    return transcription
                except sr.UnknownValueError:
                    return ""
                except sr.RequestError:
                    return ""
            
        except Exception as e:
            logger.warning(f"Transcription failed for {algorithm_name}: {e}")
            return ""
        
        return ""

class CustomXGBoostObjective:
    """Custom XGBoost objective function for WER minimization"""
    
    def __init__(self, wer_matrix: np.ndarray, epsilon: float = 1e-6):
        self.wer_matrix = wer_matrix  # Shape: (N, K)
        self.epsilon = epsilon
    
    def __call__(self, y_pred: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        Custom objective function that returns gradients and hessians
        y_pred: predictions from XGBoost (shape: N*K)
        dtrain: XGBoost DMatrix (not used but required by interface)
        """
        N, K = self.wer_matrix.shape
        
        # Reshape predictions to (N, K)
        logits = y_pred.reshape(N, K)
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Calculate expected loss for each sample
        expected_losses = np.sum(probs * self.wer_matrix, axis=1)
        
        # Calculate gradients (Equation 5 from paper)
        gradients = np.zeros_like(logits)
        for n in range(N):
            for m in range(K):
                gradients[n, m] = probs[n, m] * (self.wer_matrix[n, m] - expected_losses[n])
        
        # Calculate Hessians (Equation 6 from paper)
        hessians = np.zeros_like(logits)
        for n in range(N):
            for m in range(K):
                hessians[n, m] = probs[n, m] * (1 - 2 * probs[n, m]) * (self.wer_matrix[n, m] - expected_losses[n])
        
        # Clip negative hessians to small positive value for XGBoost stability
        hessians = np.maximum(hessians, self.epsilon)
        
        return gradients.flatten(), hessians.flatten()

class ADAS2TTrainer:
    """Main trainer class for ADAS2T XGBoost meta-learner"""
    
    def __init__(self, dataset_name: str = "librispeech_asr", dataset_config: str = "clean", 
                 sample_rate: int = 16000, max_samples: int = 1000):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        
        # Initialize components
        self.feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)
        self.s2t_pool = S2TAlgorithmPool()
        self.scaler = StandardScaler()
        
        # Training data
        self.features = None
        self.wer_matrix = None
        self.algorithm_names = list(self.s2t_pool.algorithms.keys())
        self.K = len(self.algorithm_names)
        
        logger.info(f"Initialized ADAS2T trainer with {self.K} S2T algorithms: {self.algorithm_names}")
    
    def load_dataset(self) -> Dataset:
        """Load dataset from HuggingFace"""
        logger.info(f"Loading dataset: {self.dataset_name}:{self.dataset_config}")
        
        try:
            # Load the dataset
            dataset = load_dataset(self.dataset_name, self.dataset_config, split="train")
            
            # Limit the number of samples
            if len(dataset) > self.max_samples:
                dataset = dataset.select(range(self.max_samples))
            
            logger.info(f"Loaded {len(dataset)} samples from dataset")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def prepare_training_data(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data by extracting features and computing WER matrix
        Following Algorithm 2 from the paper
        """
        logger.info("Preparing training data...")
        
        features_list = []
        wer_matrix_list = []
        
        for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            try:
                # Get audio and reference text
                audio = sample['audio']['array']
                reference_text = sample['text']
                
                # Resample if necessary
                if sample['audio']['sampling_rate'] != self.sample_rate:
                    audio = librosa.resample(audio, 
                                           orig_sr=sample['audio']['sampling_rate'], 
                                           target_sr=self.sample_rate)
                
                # Extract features
                features = self.feature_extractor.extract_all_features(audio)
                features_list.append(features)
                
                # Get transcriptions from all algorithms and compute WER
                wer_scores = []
                for algo_name in self.algorithm_names:
                    transcription = self.s2t_pool.transcribe_with_algorithm(audio, algo_name, self.sample_rate)
                    
                    # Compute WER
                    if transcription.strip() == "":
                        wer_score = 1.0  # Maximum error for empty transcription
                    else:
                        wer_score = wer(reference_text, transcription)
                    
                    wer_scores.append(wer_score)
                
                wer_matrix_list.append(wer_scores)
                
            except Exception as e:
                logger.warning(f"Failed to process sample {idx}: {e}")
                continue
        
        # Convert to numpy arrays
        features_array = np.array(features_list)
        wer_matrix = np.array(wer_matrix_list)
        
        logger.info(f"Prepared training data: {features_array.shape[0]} samples, {features_array.shape[1]} features")
        logger.info(f"WER matrix shape: {wer_matrix.shape}")
        logger.info(f"Average WER per algorithm: {np.mean(wer_matrix, axis=0)}")
        
        return features_array, wer_matrix
    
    def train_xgboost_meta_learner(self, features: np.ndarray, wer_matrix: np.ndarray, 
                                   test_size: float = 0.2) -> xgb.XGBRegressor:
        """
        Train XGBoost meta-learner with custom WER objective
        Following Algorithm 2 from the paper
        """
        logger.info("Training XGBoost meta-learner...")
        
        # Split data into train and validation sets
        X_train, X_val, W_train, W_val = train_test_split(
            features, wer_matrix, test_size=test_size, random_state=42
        )
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Prepare data for XGBoost
        N_train, K = W_train.shape
        N_val = W_val.shape[0]
        
        # Create labels (dummy, since we use custom objective)
        y_train = np.zeros(N_train * K)
        y_val = np.zeros(N_val * K)
        
        # Repeat features for each algorithm
        X_train_repeated = np.repeat(X_train_scaled, K, axis=0)
        X_val_repeated = np.repeat(X_val_scaled, K, axis=0)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train_repeated, label=y_train)
        dval = xgb.DMatrix(X_val_repeated, label=y_val)
        
        # Custom objective function
        custom_obj = CustomXGBoostObjective(W_train)
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',  # Will be overridden by custom objective
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbosity': 1
        }
        
        # Train model
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            obj=custom_obj,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=10,
            verbose_eval=10
        )
        
        logger.info("XGBoost meta-learner training completed")
        return model
    
    def evaluate_model(self, model: xgb.Booster, X_test: np.ndarray, W_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the trained model"""
        logger.info("Evaluating model...")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get predictions
        dtest = xgb.DMatrix(X_test_scaled)
        predictions = model.predict(dtest)
        
        # Reshape predictions to (N, K)
        N_test = X_test.shape[0]
        logits = predictions.reshape(N_test, self.K)
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Calculate metrics
        # 1. Expected WER
        expected_wer = np.mean(np.sum(probs * W_test, axis=1))
        
        # 2. Best possible WER (oracle)
        oracle_wer = np.mean(np.min(W_test, axis=1))
        
        # 3. Algorithm selection accuracy (how often we pick the best algorithm)
        predicted_best = np.argmax(probs, axis=1)
        actual_best = np.argmin(W_test, axis=1)
        selection_accuracy = np.mean(predicted_best == actual_best)
        
        # 4. Random baseline
        random_wer = np.mean(W_test)
        
        results = {
            'expected_wer': expected_wer,
            'oracle_wer': oracle_wer,
            'random_wer': random_wer,
            'selection_accuracy': selection_accuracy,
            'improvement_over_random': (random_wer - expected_wer) / random_wer * 100
        }
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def save_model(self, model: xgb.Booster, model_path: str = "adas2t_xgboost_model.json"):
        """Save trained model and scaler"""
        logger.info(f"Saving model to {model_path}")
        
        # Save XGBoost model
        model.save_model(model_path)
        
        # Save scaler
        scaler_path = model_path.replace('.json', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # Save algorithm names
        algo_path = model_path.replace('.json', '_algorithms.pkl')
        joblib.dump(self.algorithm_names, algo_path)
        
        logger.info("Model saved successfully")
    
    def run_full_training_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("Starting ADAS2T training pipeline...")
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Prepare training data
        features, wer_matrix = self.prepare_training_data(dataset)
        
        # Train model
        model = self.train_xgboost_meta_learner(features, wer_matrix)
        
        # Evaluate model
        X_train, X_test, W_train, W_test = train_test_split(
            features, wer_matrix, test_size=0.2, random_state=42
        )
        results = self.evaluate_model(model, X_test, W_test)
        
        # Save model
        self.save_model(model)
        
        logger.info("ADAS2T training pipeline completed successfully!")
        return model, results

def main():
    """Main training function"""
    # Configuration
    config = {
        'dataset_name': 'librispeech_asr',
        'dataset_config': 'clean',
        'sample_rate': 16000,
        'max_samples': 500,  # Adjust based on computational resources
    }
    
    # Initialize trainer
    trainer = ADAS2TTrainer(**config)
    
    # Run training
    try:
        model, results = trainer.run_full_training_pipeline()
        
        print("\n" + "="*50)
        print("ADAS2T Training Results:")
        print("="*50)
        print(f"Expected WER: {results['expected_wer']:.4f}")
        print(f"Oracle WER (Best Possible): {results['oracle_wer']:.4f}")
        print(f"Random Baseline WER: {results['random_wer']:.4f}")
        print(f"Algorithm Selection Accuracy: {results['selection_accuracy']:.4f}")
        print(f"Improvement over Random: {results['improvement_over_random']:.2f}%")
        print("="*50)
        
        # Additional analysis
        print("\nTraining Summary:")
        print(f"- Dataset: {config['dataset_name']} ({config['dataset_config']})")
        print(f"- Number of samples: {config['max_samples']}")
        print(f"- Sample rate: {config['sample_rate']} Hz")
        print(f"- Number of S2T algorithms: {len(trainer.algorithm_names)}")
        print(f"- Algorithms used: {', '.join(trainer.algorithm_names)}")
        print(f"- Feature vector size: {trainer.features.shape[1] if trainer.features is not None else 'N/A'}")
        
        # Performance interpretation
        print("\nPerformance Analysis:")
        if results['improvement_over_random'] > 10:
            print("✓ Excellent performance - Model significantly outperforms random selection")
        elif results['improvement_over_random'] > 5:
            print("✓ Good performance - Model shows meaningful improvement")
        elif results['improvement_over_random'] > 0:
            print("~ Moderate performance - Model shows some improvement")
        else:
            print("✗ Poor performance - Model does not improve over random selection")
        
        if results['selection_accuracy'] > 0.7:
            print("✓ High algorithm selection accuracy")
        elif results['selection_accuracy'] > 0.5:
            print("~ Moderate algorithm selection accuracy")
        else:
            print("✗ Low algorithm selection accuracy")
        
        print("\nModel files saved:")
        print("- adas2t_xgboost_model.json (XGBoost model)")
        print("- adas2t_xgboost_model_scaler.pkl (Feature scaler)")
        print("- adas2t_xgboost_model_algorithms.pkl (Algorithm names)")
        
        print("\nTraining completed successfully!")
        
        return model, results
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        print(f"\nTraining failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check internet connection for downloading models and datasets")
        print("2. Ensure sufficient memory (>8GB recommended)")
        print("3. Verify all required packages are installed")
        print("4. Try reducing max_samples if running out of memory")
        print("5. Check that audio files are accessible and in correct format")
        
        # Print detailed error information for debugging
        import traceback
        print("\nDetailed error information:")
        traceback.print_exc()
        
        return None, None

if __name__ == "__main__":
    # Set up environment
    print("ADAS2T: Adaptive S2T Algorithm Selection via Meta-Learning")
    print("XGBoost-based Meta-Learner Training Script")
    print("="*60)
    
    # Check system requirements
    print("Checking system requirements...")
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available - using CPU")
    except ImportError:
        print("✗ PyTorch not installed")
    
    # Check XGBoost
    try:
        import xgboost as xgb
        print(f"✓ XGBoost version: {xgb.__version__}")
    except ImportError:
        print("✗ XGBoost not installed")
    
    # Check key dependencies
    required_packages = ['librosa', 'transformers', 'datasets', 'jiwer', 'webrtcvad', 'parselmouth']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            print(f"✗ {package} not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install missing packages before running training.")
        exit(1)
    
    print("\nAll requirements satisfied. Starting training...\n")
    
    # Run main training
    model, results = main()
    
    if model is not None and results is not None:
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("The trained ADAS2T model is ready for deployment.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Training failed. Please check the error messages above.")
        print("="*60)
        exit(1)
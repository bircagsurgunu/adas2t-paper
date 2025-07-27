# adas2t/feature_extractor.py

import numpy as np
import librosa
import torch
import warnings
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    HubertModel
)
from scipy.stats import entropy
import webrtcvad
from python_speech_features import mfcc, delta
import parselmouth
from parselmouth.praat import call
import logging

# Suppress warnings from libraries
warnings.filterwarnings('ignore')
# Set transformers logging to error to avoid spam
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

class AudioFeatureExtractor:
    """
    Comprehensive audio feature extractor implementing all features described in ADAS2T paper.
    """
    
    def __init__(self, sample_rate: int = 16000, device="cpu"):
        self.sample_rate = sample_rate
        self.device = device
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3
        self._load_embedding_models()
        
    def _load_embedding_models(self):
        """Load pre-trained models for neural embeddings"""
        logger.info("ADAS2T: Loading neural embedding models for feature extraction...")
        try:
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(self.device)
            self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
            self.hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(self.device)
            logger.info("ADAS2T: Successfully loaded neural embedding models.")
        except Exception as e:
            logger.warning(f"ADAS2T: Could not load all embedding models: {e}. Features will be incomplete.")
            self.whisper_processor, self.whisper_model = None, None
            self.wav2vec2_processor, self.wav2vec2_model = None, None
            self.hubert_model = None
    
    def extract_all_features(self, audio: np.ndarray) -> np.ndarray:
        """Extracts and concatenates all feature types."""
        # MFCC features (78 features)
        mfcc_features = self._extract_mfcc_features(audio)
        # Prosodic features (6 features)
        prosodic_features = self._extract_prosodic_features(audio)
        # Neural embeddings (512 + 768 + 768 = 2048 features)
        neural_features = self._extract_neural_embeddings(audio)
        # Signal complexity features (~10 features)
        signal_features = self._extract_signal_complexity_features(audio)
        
        final_features = np.concatenate([
            mfcc_features,
            prosodic_features,
            neural_features,
            signal_features
        ])
        return np.nan_to_num(final_features, nan=0.0, posinf=0.0, neginf=0.0)

    def _extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        mfcc_feat = mfcc(audio, self.sample_rate, numcep=13, nfilt=26, nfft=512)
        delta_feat = delta(mfcc_feat, N=2)
        delta_delta_feat = delta(delta_feat, N=2)
        all_features = np.concatenate([mfcc_feat, delta_feat, delta_delta_feat], axis=1)
        return np.concatenate([np.mean(all_features, axis=0), np.std(all_features, axis=0)])

    def _extract_prosodic_features(self, audio: np.ndarray) -> np.ndarray:
        features = []
        try:
            sound = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
            pitch = sound.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values != 0]
            if len(pitch_values) > 0:
                features.extend([np.mean(pitch_values), np.std(pitch_values), np.max(pitch_values) - np.min(pitch_values), np.median(pitch_values)])
            else:
                features.extend([0.0] * 4)
        except Exception:
            features.extend([0.0] * 4)
        
        audio_16bit = (audio * 32767).astype(np.int16)
        frame_length = int(0.030 * self.sample_rate)
        voiced_frames, total_frames = 0, 0
        for i in range(0, len(audio_16bit) - frame_length, int(0.010 * self.sample_rate)):
            frame = audio_16bit[i:i+frame_length]
            if len(frame) == frame_length:
                try:
                    if self.vad.is_speech(frame.tobytes(), self.sample_rate):
                        voiced_frames += 1
                    total_frames += 1
                except Exception: pass
        
        silence_ratio = 1.0 - (voiced_frames / max(total_frames, 1))
        features.extend([silence_ratio, 1.0 - silence_ratio])
        return np.array(features)

    def _extract_neural_embeddings(self, audio: np.ndarray) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            if self.whisper_model:
                inputs = self.whisper_processor(audio, sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)
                encoder_outputs = self.whisper_model.model.encoder(**inputs)
                embeddings.append(encoder_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
            else: embeddings.append(np.zeros(512))

            if self.wav2vec2_model:
                inputs = self.wav2vec2_processor(audio, sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)
                outputs = self.wav2vec2_model.wav2vec2(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
            else: embeddings.append(np.zeros(768))

            if self.hubert_model:
                inputs = torch.tensor(audio, dtype=torch.float32, device=self.device).unsqueeze(0)
                outputs = self.hubert_model(inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
            else: embeddings.append(np.zeros(768))

        return np.concatenate(embeddings) if embeddings else np.zeros(2048)

    def _extract_signal_complexity_features(self, audio: np.ndarray) -> np.ndarray:
        features = []
        try:
            frame_energy = librosa.feature.rms(y=audio)[0]
            snr = 10 * np.log10(np.mean(frame_energy) /
                                max(np.percentile(frame_energy, 10), 1e-10))
            features.append(snr)
        except Exception:
            features.append(0.0)

        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features.extend([np.mean(zcr), np.std(zcr)])

        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])

            spectral_bw = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            features.extend([np.mean(spectral_bw), np.std(spectral_bw)])

            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            features.append(np.mean(rolloff))
            
            stft = np.abs(librosa.stft(audio)) ** 2
            spec_entropy = [entropy(frame / np.sum(frame))
                            for frame in stft.T if np.sum(frame) > 0]
            features.extend([np.mean(spec_entropy), np.std(spec_entropy)]
                            if spec_entropy else [0.0, 0.0])
        except Exception:
            features.extend([0.0] * 7)

        features.append(len(audio) / self.sample_rate)

        return np.array(features)
# model_handler.py

import torch
import nemo.collections.asr as nemo_asr
from transformers import (
    pipeline,
    AutoProcessor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    HubertForCTC,
    AutoModelForSpeechSeq2Seq
)
from abc import ABC, abstractmethod
import os
import joblib
import xgboost as xgb
import numpy as np
import logging

# Import ADAS2T components
from .adas2t.feature_extractor import AudioFeatureExtractor
from .adas2t.runtime import S2TAlgorithmPool
from . import config

logger = logging.getLogger(__name__)

class ASRModelHandler(ABC):
    """Abstract base class for ASR model handlers."""
    def __init__(self, model_name, device, torch_dtype):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self):
        """Loads the model and processor into memory."""
        pass

    @abstractmethod
    def transcribe(self, audio_array):
        """Transcribes a given audio array."""
        pass

class WhisperHandler(ASRModelHandler):
    def load_model(self):
        print(f"Loading Whisper pipeline for {self.model_name}...")
        self.model = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        print("Whisper model loaded.")

    def transcribe(self, audio_array):
        # Whisper pipeline handles everything
        result = self.model(audio_array, generate_kwargs={"language": "english"})
        return result["text"]

class CTCHandler(ASRModelHandler):
    def load_model(self):
        print(f"Loading CTC model {self.model_name}...")
        ModelClass = HubertForCTC if "hubert" in self.model_name else Wav2Vec2ForCTC
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = ModelClass.from_pretrained(self.model_name).to(self.device)
        print("CTC model loaded.")

    def transcribe(self, audio_array):
        inputs = self.processor(audio_array, return_tensors="pt", padding="longest").input_values
        inputs = inputs.to(self.device)
        with torch.no_grad():
            logits = self.model(inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)[0]

class NemoParakeetHandler(ASRModelHandler):
    def load_model(self):
        print(f"Loading NeMo model {self.model_name}...")
        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name, map_location=self.device)
        print("NeMo model loaded.")
        
    def transcribe(self, audio_array):
        # NeMo models have a built-in transcribe method
        return self.model.transcribe([audio_array])[0].text

class GraniteHandler(ASRModelHandler):
    def load_model(self):
        print(f"Loading Granite model {self.model_name}...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name, 
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        print("Granite model loaded.")

    def transcribe(self, audio_array):
        chat = [
            {"role": "user", "content": "<|audio|>Transcribe the audio."},
        ]
        text_inputs = self.processor.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        
        model_inputs = self.processor(
            text=text_inputs,
            audio=audio_array,
            return_tensors="pt",
        ).to(self.device)

        model_outputs = self.model.generate(**model_inputs, max_new_tokens=256)
        output_text = self.processor.tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
        return output_text[0]


class ADAS2THandler(ASRModelHandler):
    """
    Handler for the ADAS2T meta-learner. It doesn't load one model, but a system
    for selecting the best model from a pool for each audio input.
    """
    def __init__(self, model_name, device, torch_dtype):
        super().__init__(model_name, device, torch_dtype)
        self.scaler = None
        self.algorithm_names = None
        self.feature_extractor = None
        self.s2t_pool = None

    def _algo_one_hot(self, idx: int) -> np.ndarray:
        v = np.zeros(self.num_algorithms, dtype=np.float32)
        v[idx] = 1.0
        return v

    def load_model(self):
        """Loads the XGBoost model, scaler, algorithm list, and the algorithm pool."""
        print(f"Loading ADAS2T meta-learning system...")

        # 1. Check if model files exist
        if not all(os.path.exists(p) for p in [config.ADAS2T_MODEL_PATH, config.ADAS2T_SCALER_PATH, config.ADAS2T_ALGORITHMS_PATH]):
            raise FileNotFoundError(
                "ADAS2T model files not found. Please run 'python train_metaleaner.py' first."
            )

        # 2. Load the XGBoost model, scaler, and algorithm names
        self.model = xgb.Booster()
        self.model.load_model(config.ADAS2T_MODEL_PATH)
        self.scaler = joblib.load(config.ADAS2T_SCALER_PATH)
        self.algorithm_names = joblib.load(config.ADAS2T_ALGORITHMS_PATH)
        self.num_algorithms = len(self.algorithm_names)
        print("  - XGBoost model, scaler, and algorithm list loaded.")

        # 3. Initialize the feature extractor
        self.feature_extractor = AudioFeatureExtractor(sample_rate=16000, device=self.device)
        print("  - Audio feature extractor initialized.")

        # 4. Initialize the pool of ASR models that can be selected
        self.s2t_pool = S2TAlgorithmPool(config.ADAS2T_ALGORITHM_POOL, self.device, self.torch_dtype)
        print("ADAS2T meta-learning system loaded successfully.")

    def transcribe(self, audio_array):
        # 1. acoustic features
        base_feats = self.feature_extractor.extract_all_features(audio_array)

        # 2. build one row per algorithm  →  shape = (K , D+K)
        feature_rows = [
            np.concatenate([base_feats, self._algo_one_hot(k)])
            for k in range(self.num_algorithms)
        ]
        feature_rows = np.asarray(feature_rows, dtype=np.float32)

        # 3. scale exactly like during training
        feature_rows = self.scaler.transform(feature_rows)

        # 4. predict expected WER for every algorithm
        dmat   = xgb.DMatrix(feature_rows)
        preds  = self.model.predict(dmat)          # length K
        best_k = int(np.argmin(preds))
        best_algo_name = self.algorithm_names[best_k]

        # 5. run the chosen ASR system
        return self.s2t_pool.transcribe_with_algorithm(audio_array,
                                                       best_algo_name)

# Factory function to get the correct handler
def get_model_handler(model_name: str, device: str, torch_dtype) -> ASRModelHandler:
    """Returns the appropriate model handler based on the model name."""
    if model_name == "adas2t-meta-learner":
        return ADAS2THandler(model_name, device, torch_dtype)
    if "whisper" in model_name:
        return WhisperHandler(model_name, device, torch_dtype)
    if "parakeet" in model_name:
        return NemoParakeetHandler(model_name, device, torch_dtype)
    if "granite" in model_name:
        return GraniteHandler(model_name, device, torch_dtype)
    if "wav2vec2" in model_name or "hubert" in model_name:
        return CTCHandler(model_name, device, torch_dtype)
    
    raise ValueError(f"No handler available for model: {model_name}")
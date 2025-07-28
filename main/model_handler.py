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
# --- MODIFIED: Import both models for runtime loading ---
from .adas2t.models import MetaLearnerMLP, MetaLearnerTransformer, MetaLearnerAST
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
    for selecting the best model from a pool for each audio input. Can handle
    both XGBoost and MLP-based meta-learners.
    """
    def __init__(self, model_name, device, torch_dtype):
        super().__init__(model_name, device, torch_dtype)
        self.scaler = None
        self.algorithm_names = None
        self.feature_extractor = None
        self.s2t_pool = None
        self.learner_type = None # Will be 'xgboost' or 'mlp' or 'transformer'

    def _algo_one_hot(self, idx: int) -> np.ndarray:
        v = np.zeros(self.num_algorithms, dtype=np.float32)
        v[idx] = 1.0
        return v

    def load_model(self):
        """Loads the meta-learner model (XGBoost, MLP, or Transformer), scaler, algorithm list, and the algorithm pool."""
        print(f"Loading ADAS2T meta-learning system...")

        # 1. Check if model files exist
        model_path = config.ADAS2T_MODEL_PATH
        if not all(os.path.exists(p) for p in [model_path, config.ADAS2T_SCALER_PATH, config.ADAS2T_ALGORITHMS_PATH]):
            raise FileNotFoundError(
                "ADAS2T model files not found. Please run 'python -m adas2t.train_metalearner' first."
            )

        # 2. Load scaler and algorithm list
        self.scaler = joblib.load(config.ADAS2T_SCALER_PATH)
        self.algorithm_names = joblib.load(config.ADAS2T_ALGORITHMS_PATH)
        self.num_algorithms = len(self.algorithm_names)
        print("  - Scaler and algorithm list loaded.")
        
        # 3. Load the meta-learner model based on file extension and determine type
        if model_path.endswith(".json"):
            print("  - Loading XGBoost model...")
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            self.learner_type = "xgboost"
        elif model_path.endswith((".pth", ".pt")):
            print("  - Loading PyTorch model...")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            hyperparams = checkpoint['hyperparams']
            state_dict = checkpoint['state_dict']
            
            self.learner_type = hyperparams.get('learner_type', 'mlp')
            print(f"    - Detected learner type: {self.learner_type.upper()}")

            if self.learner_type == 'transformer':
                self.model = MetaLearnerTransformer(
                    feature_dims=hyperparams['feature_dims'],
                    num_algorithms=hyperparams['num_algorithms'],
                    d_model=hyperparams['d_model'],
                    nhead=hyperparams['nhead'],
                    num_encoder_layers=hyperparams['num_encoder_layers'],
                    dropout=hyperparams.get('dropout', 0.1)
                ).to(self.device)
            elif self.learner_type == 'mlp':
                dummy_feats_dict = self.feature_extractor.extract_all_features(np.zeros(16000, dtype=np.float32))
                flat_dummy_feats = np.concatenate([v for k,v in sorted(dummy_feats_dict.items())])
                input_dim = len(flat_dummy_feats) + self.num_algorithms
                
                self.model = MetaLearnerMLP(
                    input_dim=input_dim,
                    hidden_dim=hyperparams['hidden_dim'],
                    num_layers=hyperparams['num_layers'],
                    dropout=hyperparams['dropout']
                ).to(self.device)
            elif self.learner_type == 'ast':
                p = checkpoint["hyperparams"]
                self.model = MetaLearnerAST(
                    num_algorithms = p["num_algorithms"],
                    pretrained_name= p.get("pretrained_name",
                                        "MIT/ast-finetuned-audioset-10-10-0.4593"),
                    train_backbone = False,
                ).to(self.device)
            else:
                raise ValueError(f"Unknown PyTorch learner_type in checkpoint: '{self.learner_type}'")
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
        else:
            raise ValueError(f"Unknown ADAS2T model type: {model_path}. Must be .json or .pth/.pt")

        # 4. Conditionally initialize the feature extractor (AST model does not need it)
        if self.learner_type != 'ast':
            self.feature_extractor = AudioFeatureExtractor(sample_rate=16000, device=self.device)
            print("  - Acoustic feature extractor initialized.")
        else:
            print("  - AST model selected. Skipping acoustic feature extractor initialization.")

        # 5. Initialize the pool of ASR models that can be selected
        self.s2t_pool = S2TAlgorithmPool(config.ADAS2T_ALGORITHM_POOL, self.device, self.torch_dtype)
        print("ADAS2T meta-learning system loaded successfully.")

    def transcribe(self, audio_array):
        """
        Generates a transcription by first predicting the best ASR model for the
        given audio, then using that model to perform the transcription. This method
        is updated to handle both new Transformer models and legacy flat-feature models.
        """
        preds = None
        
        # --- MODIFIED: Handle AST model separately as it uses raw audio ---
        if self.learner_type == 'ast':
            # Create a batch: repeat the same audio for each algorithm
            waveform_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).repeat(self.num_algorithms, 1).to(self.device)
            # Create a tensor of algorithm IDs [0, 1, 2, ...]
            algo_ids_tensor = torch.arange(self.num_algorithms, device=self.device).long()
            with torch.no_grad():
                preds = self.model(waveform_tensor, algo_ids_tensor).cpu().numpy().flatten()
        
        # --- This block handles all other models that require feature extraction ---
        else:
            # 1. Extract acoustic features (returns a dictionary)
            base_feats_dict = self.feature_extractor.extract_all_features(audio_array)

            # 2. Prepare features and predict based on the loaded learner type
            if self.learner_type == 'transformer':
                model_input_dict = {}
                for key, vector in base_feats_dict.items():
                    scaler = self.scaler[key]
                    scaled_vector = scaler.transform(vector.reshape(1, -1))
                    tiled_vector = np.tile(scaled_vector, (self.num_algorithms, 1))
                    model_input_dict[key] = torch.from_numpy(tiled_vector).float().to(self.device)

                algo_ids = np.arange(self.num_algorithms)
                model_input_dict['algo_id'] = torch.from_numpy(algo_ids).long().to(self.device)
                
                with torch.no_grad():
                    preds = self.model(model_input_dict).cpu().numpy().flatten()
            
            elif self.learner_type in ['mlp', 'xgboost']:
                if not isinstance(self.scaler, dict):
                     raise TypeError(f"Loaded a {self.learner_type} model but the scaler is not a dictionary. The codebase is inconsistent.")

                scaled_flat_feats_list = []
                for key in sorted(self.scaler.keys()):
                     vector = base_feats_dict[key].reshape(1, -1)
                     scaled_vector = self.scaler[key].transform(vector).flatten()
                     scaled_flat_feats_list.append(scaled_vector)
                scaled_base_feats = np.concatenate(scaled_flat_feats_list)

                feature_rows = np.asarray([
                    np.concatenate([scaled_base_feats, self._algo_one_hot(k)])
                    for k in range(self.num_algorithms)
                ], dtype=np.float32)

                if self.learner_type == "xgboost":
                    dmat = xgb.DMatrix(feature_rows)
                    preds = self.model.predict(dmat)
                else: # MLP
                    features_tensor = torch.from_numpy(feature_rows).float().to(self.device)
                    with torch.no_grad():
                        preds = self.model(features_tensor).cpu().numpy().flatten()
            else:
                raise RuntimeError(f"Unknown or unsupported feature-based learner type: '{self.learner_type}'")

        if preds is None:
            raise RuntimeError(f"Prediction failed for learner type '{self.learner_type}'.")

        # 3. Find the algorithm with the lowest predicted WER
        best_k = int(np.argmin(preds))
        best_algo_name = self.algorithm_names[best_k]

        # 4. Run transcription with the chosen ASR system
        return self.s2t_pool.transcribe_with_algorithm(audio_array, best_algo_name)

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
# adas2t/runtime.py

import torch
import logging
# The circular import from model_handler is removed from the top level

logger = logging.getLogger(__name__)

class S2TAlgorithmPool:
    """
    Pool of Speech-to-Text algorithms that the meta-learner chooses from.
    This implementation leverages the existing model_handler architecture.
    """
    
    def __init__(self, algorithm_names: list, device, torch_dtype):
        self.device = device
        self.torch_dtype = torch_dtype
        self.algorithm_names = algorithm_names
        self.algorithms = {}
        self._initialize_algorithms()
    
    def _initialize_algorithms(self):
        """Initialize different S2T algorithms using the main model_handler."""
        from ..model_handler import get_model_handler 

        logger.info(f"ADAS2T: Initializing S2T Algorithm Pool with: {self.algorithm_names}")
        for name in self.algorithm_names:
            try:
                # Use the factory from the main model_handler to get a model handler
                handler = get_model_handler(name, self.device, self.torch_dtype)
                handler.load_model()
                self.algorithms[name] = handler
                logger.info(f"ADAS2T: Successfully loaded '{name}' into the pool.")
            except Exception as e:
                logger.warning(f"ADAS2T: Failed to load model '{name}' for the algorithm pool. It will be unavailable. Error: {e}")
    
    def transcribe_with_algorithm(self, audio: list, algorithm_name: str) -> str:
        """Transcribe audio using a specified algorithm from the pool."""
        if algorithm_name not in self.algorithms:
            logger.error(f"ADAS2T: Algorithm '{algorithm_name}' not found in the pool.")
            return ""
        
        handler = self.algorithms[algorithm_name]
        try:
            # Use the handler's transcribe method
            transcription = handler.transcribe(audio).strip().lower()
            return transcription
        except Exception as e:
            logger.warning(f"ADAS2T: Transcription failed for '{algorithm_name}': {e}")
            return ""
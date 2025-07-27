# train_metaleaner.py

import argparse
import torch
import logging

from main.adas2t.trainer import ADAS2TTrainer

# --- Configuration for Training ---

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to run the ADAS2T meta-learner training pipeline.
    """
    parser = argparse.ArgumentParser(description="Train the ADAS2T XGBoost Meta-Learner.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples from the dataset to use for training."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="librispeech_asr",
        help="Hugging Face dataset name to use for training (e.g., 'librispeech_asr')."
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="clean",
        help="Dataset configuration (e.g., 'clean' for librispeech)."
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train.clean.100",
        help="Dataset split to use (e.g., 'train')."
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="The name of the field in the dataset containing the reference text."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="adas2t_model.json",
        help="Path to save the trained XGBoost model."
    )
    parser.add_argument(
        "--scaler_path",
        type=str,
        default="adas2t_scaler.pkl",
        help="Path to save the feature scaler."
    )
    parser.add_argument(
        "--algorithms_path",
        type=str,
        default="adas2t_algorithms.pkl",
        help="Path to save the list of algorithm names."
    )
    args = parser.parse_args()

    print("--- Starting ADAS2T Meta-Learner Training ---")

    # Dynamic configuration based on argparse and system capabilities
    training_config = {
        'device': "cuda:0" if torch.cuda.is_available() else "cpu",
        'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
        'num_training_samples': args.num_samples,
        'training_dataset': (args.dataset, {"name": args.dataset_config, "split": args.dataset_split, "field": args.text_field}),
        
        # The pool of models the meta-learner will be trained to choose from.
        # This should match the pool used at inference time.
        'algorithm_pool_models': [
            "openai/whisper-large-v3",
            "facebook/wav2vec2-base-960h",
            "nvidia/parakeet-tdt-0.6b-v2",
        ],

        # Output paths for the trained artifacts
        'model_path': args.model_path,
        'scaler_path': args.scaler_path,
        'algorithms_path': args.algorithms_path,
    }

    print(f"Using device: {training_config['device']}")
    print(f"Training with {training_config['num_training_samples']} samples from {args.dataset}/{args.dataset_config}")
    
    trainer = ADAS2TTrainer(training_config)
    trainer.run_full_training_pipeline()

    print("\n--- Training Complete ---")
    print(f"Model saved to: {args.model_path}")
    print(f"Scaler saved to: {args.scaler_path}")
    print(f"Algorithms list saved to: {args.algorithms_path}")
    print("You can now run the main benchmark with 'adas2t-meta-learner' in MODELS_TO_BENCHMARK.")

if __name__ == "__main__":
    main()
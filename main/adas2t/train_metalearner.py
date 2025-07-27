# train_metaleaner.py

import argparse
import torch
import logging

from main.adas2t.trainer import ADAS2TTrainer

# --- Configuration for Training ---

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Training Dataset Configuration ---
# List of datasets to use for training the meta-learner.
# This should ideally cover a diverse range of audio conditions corresponding to the benchmark datasets.
# Each entry is a tuple: (dataset_name, configuration_dictionary)
TRAINING_DATASETS_INFO = [
    ("mozilla-foundation/common_voice_16_1", {"split": "train", "field": "sentence", "lang_config": "en"}),
    ("openslr/librispeech_asr", {"split": "train.clean.100", "field": "text"}),
    ("edinburghcstr/ami", {"split": "train", "config": "ihm", "field": "text"}),
    # To train on more data, uncomment the following lines.
    # ("distil-whisper/earnings22", {"split": "train", "field": "transcription", "lang_config": "chunked"}),
    # ("google/fleurs", {"split": "train", "field": "transcription", "lang_config": "en_us"}),
]


def main():
    """
    Main function to run the ADAS2T meta-learner training pipeline.
    """
    parser = argparse.ArgumentParser(description="Train the ADAS2T XGBoost Meta-Learner.")
    parser.add_argument(
        "--num_samples_per_dataset",
        type=int,
        default=100,
        help="Number of samples from EACH dataset to use for training."
    )
    parser.add_argument(
        '--use_all_datasets',
        action='store_true',
        help="Flag to use all datasets defined in TRAINING_DATASETS_INFO. If not set, only the first dataset is used for a quick test run."
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
    
    # Determine which datasets to use for this training run
    datasets_to_use = TRAINING_DATASETS_INFO if args.use_all_datasets else [TRAINING_DATASETS_INFO[0]]

    # Dynamic configuration based on argparse and system capabilities
    training_config = {
        'device': "cuda:0" if torch.cuda.is_available() else "cpu",
        'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
        'num_training_samples': args.num_samples_per_dataset,
        'training_datasets': datasets_to_use,  # Pass the list of datasets
        
        # The pool of models the meta-learner will be trained to choose from.
        # This should match the pool used at inference time.
        'algorithm_pool_models': [
            "facebook/hubert-large-ls960-ft",
            "facebook/wav2vec2-base-960h",
            "nvidia/parakeet-tdt-0.6b-v2",
        ],

        # Output paths for the trained artifacts
        'model_path': args.model_path,
        'scaler_path': args.scaler_path,
        'algorithms_path': args.algorithms_path,
    }

    print(f"Using device: {training_config['device']}")
    print(f"Training with {training_config['num_training_samples']} samples from {len(datasets_to_use)} dataset(s).")
    if args.use_all_datasets:
        print("Training on all configured datasets.")
    else:
        print(f"Training on a single dataset for quick run: {datasets_to_use[0][0]}")

    
    trainer = ADAS2TTrainer(training_config)
    trainer.run_full_training_pipeline()

    print("\n--- Training Complete ---")
    print(f"Model saved to: {args.model_path}")
    print(f"Scaler saved to: {args.scaler_path}")
    print(f"Algorithms list saved to: {args.algorithms_path}")
    print("You can now run the main benchmark with 'adas2t-meta-learner' in MODELS_TO_BENCHMARK.")

if __name__ == "__main__":
    main()
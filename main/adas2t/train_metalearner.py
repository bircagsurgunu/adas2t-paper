# train_metaleaner.py

import argparse
import torch
import logging
import os
import shutil  # Import shutil for directory operations

from main.adas2t.trainer import ADAS2TTrainer
from main.adas2t.models import MetaLearnerMLP, MetaLearnerTransformer

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
    #("distil-whisper/earnings22", {"split": "train", "field": "transcription", "lang_config": "chunked"}),
    #("google/fleurs", {"split": "train", "field": "transcription", "lang_config": "en_us"}),
]


def main():
    """
    Main function to run the ADAS2T meta-learner training pipeline.
    """
    parser = argparse.ArgumentParser(description="Train the ADAS2T Meta-Learner.")
    parser.add_argument(
        "--learner_type",
        type=str,
        default="xgboost",
        choices=["xgboost", "mlp", "transformer", "ast"],
        help="Type of meta-learner model to train."
    )
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
        default=None,
        help="Path to save the trained model. Defaults to 'adas2t_model_xgboost.json' or 'adas2t_model_mlp.pth'."
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

    # --- Caching Arguments ---
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".adas2t_cache",
        help="Directory to store cached features and transcriptions to speed up subsequent runs."
    )
    parser.add_argument(
        '--clear_cache',
        action='store_true',
        help="If set, deletes the cache directory before starting, forcing data regeneration."
    )
    parser.add_argument("--transformer_d_model", type=int, default=128, help="Embedding dimension for Transformer.")
    parser.add_argument("--transformer_nhead", type=int, default=8, help="Number of attention heads for Transformer.")
    parser.add_argument("--transformer_num_layers", type=int, default=4, help="Number of layers for Transformer Encoder.")
    parser.add_argument("--transformer_lr", type=float, default=1e-4, help="Learning rate for Transformer optimizer.")

    # --- XGBoost Specific Arguments ---
    parser.add_argument("--xgb_max_depth", type=int, default=10, help="Max depth for XGBoost trees.")
    parser.add_argument("--xgb_num_trees", type=int, default=2000, help="Number of boosting rounds for XGBoost.")
    parser.add_argument("--xgb_eta", type=float, default=0.05, help="Learning rate (eta) for XGBoost.")

    # --- MLP Specific Arguments ---
    parser.add_argument("--mlp_epochs", type=int, default=50, help="Number of training epochs for MLP.")
    parser.add_argument("--mlp_lr", type=float, default=1e-4, help="Learning rate for MLP optimizer.")
    parser.add_argument("--mlp_hidden_dim", type=int, default=256, help="Hidden layer dimension for MLP.")
    parser.add_argument("--mlp_num_layers", type=int, default=3, help="Number of hidden layers for MLP.")
    parser.add_argument("--mlp_batch_size", type=int, default=256, help="Batch size for MLP training.")
    parser.add_argument("--mlp_dropout", type=float, default=0.20, help="Drop-out probability for each MLP layer.")
    parser.add_argument(
    "--early_stop_patience",
    type=int,
    default=10,
    help="How many epochs to wait for an improvement in validation loss before early stopping.",
    )
    parser.add_argument(
        "--mlp_weight_decay",
        type=float,
         default=0.0,
         help="L2 weight-decay value (passed to the Adam optimizer).",)

    parser.add_argument("--ast_epochs",      type=int,   default=5)
    parser.add_argument("--ast_lr",          type=float, default=1e-4)
    parser.add_argument("--ast_batch_size",  type=int,   default=8)
    parser.add_argument("--ast_freeze_backbone", action="store_true",
                        help="If passed, the AST backbone stays frozen.")
    parser.add_argument("--ast_pretrained_name",
                        default="MIT/ast-finetuned-audioset-10-10-0.4593")
    args = parser.parse_args()
    # Handle cache clearing
    if args.clear_cache and os.path.exists(args.cache_dir):
        logging.warning(f"Clearing cache directory: {args.cache_dir}")
        shutil.rmtree(args.cache_dir)

    # Set default model path if not provided by the user
    if args.model_path is None:
        file_extension = "json" if args.learner_type == "xgboost" else "pth"
        args.model_path = f"adas2t_model_{args.learner_type}.{file_extension}"

    print(f"--- Starting ADAS2T Meta-Learner Training (Learner: {args.learner_type.upper()}) ---")
    
    datasets_to_use = TRAINING_DATASETS_INFO if args.use_all_datasets else [TRAINING_DATASETS_INFO[0]]

    training_config = {
        'learner_type': args.learner_type,
        'device': "cuda:0" if torch.cuda.is_available() else "cpu",
        'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
        'num_training_samples': args.num_samples_per_dataset,
        'training_datasets': datasets_to_use,
        
        'algorithm_pool_models': [
            "facebook/hubert-large-ls960-ft",
            "facebook/wav2vec2-base-960h",
            "nvidia/parakeet-tdt-0.6b-v2",
        ],

        'model_path': args.model_path,
        'scaler_path': args.scaler_path,
        'algorithms_path': args.algorithms_path,
        'cache_dir': args.cache_dir, # Add cache directory to config

        "xgb_params": {
            "max_depth" : args.xgb_max_depth,
            "num_boost" : args.xgb_num_trees,
            "eta"       : args.xgb_eta,
        },
        "mlp_params": {
            "epochs": args.mlp_epochs,
            "lr": args.mlp_lr,
            "hidden_dim": args.mlp_hidden_dim,
            "num_layers": args.mlp_num_layers,
            "batch_size": args.mlp_batch_size,
            "dropout": args.mlp_dropout,
            "early_stop_patience": args.early_stop_patience,
            "weight_decay": args.mlp_weight_decay,
        },
        "transformer_params": {
            "d_model": args.transformer_d_model,
            "nhead": args.transformer_nhead,
            "num_encoder_layers": args.transformer_num_layers,
            "lr": args.transformer_lr,
            # Re-use some mlp params for consistency
            "epochs": args.mlp_epochs,
            "batch_size": args.mlp_batch_size,
            "early_stop_patience": args.early_stop_patience,
            "weight_decay": args.mlp_weight_decay,
            "transformer_dropout": args.mlp_dropout,
        },
        "ast_params": {
        "epochs":           args.ast_epochs,
        "lr":               args.ast_lr,
        "batch_size":       args.ast_batch_size,
        "freeze_backbone":  args.ast_freeze_backbone,
        "pretrained_name":  args.ast_pretrained_name,
    },

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
    print(f"Ensure 'ADAS2T_MODEL_PATH' in config.py is set to '{os.path.basename(args.model_path)}'.")

if __name__ == "__main__":
    main()
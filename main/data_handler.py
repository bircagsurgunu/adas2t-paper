# data_handler.py

from datasets import load_dataset, Audio
from itertools import islice

def load_and_prepare_dataset(ds_name, cfg, num_samples=None):
    """
    Loads, configures, and prepares a streaming dataset.

    Args:
        ds_name (str): The name of the dataset.
        cfg (dict): Configuration dictionary for the dataset.
        num_samples (int, optional): The number of samples to limit to. Defaults to None.

    Returns:
        An iterable of dataset samples.
    """
    print(f"→ Loading dataset: {ds_name} with config: {cfg}")
    
    # Handle specific loading arguments
    load_args = {
        "path": ds_name,
        "split": cfg["split"],
        "streaming": True,
        "token": True,  # Use HF token for gated datasets like Common Voice & VoxPopuli
    }
    
    # Add optional language or config name
    if "lang_config" in cfg:
        load_args["name"] = cfg["lang_config"]
    elif "config" in cfg:
        load_args["name"] = cfg["config"]

    # For datasets with old-style .py loading scripts (like AMI, VoxPopuli, Fleurs),
    # we must explicitly enable `trust_remote_code` due to a recent change in the `datasets` library.
    if "ami" in ds_name or "voxpopuli" in ds_name or "fleurs" in ds_name:
        load_args["trust_remote_code"] = True

    # VoxPopuli is not a streaming dataset
    if "voxpopuli" in ds_name:
        load_args["streaming"] = False

    ds = load_dataset(**load_args)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    # Apply specific filters
    if "voxpopuli" in ds_name:
        print("  Filtering VoxPopuli to keep entries <= 30 seconds.")
        ds = ds.filter(lambda ex: len(ex["audio"]["array"]) / 16_000 <= 30.0)

    # Limit number of examples for testing
    if num_samples:
        print(f"  Limiting to the first {num_samples} samples.")
        return list(islice(ds, num_samples))
        
    return ds
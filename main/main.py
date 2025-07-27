# main.py

import pandas as pd
import evaluate
from tqdm import tqdm

from . import config
from .model_handler import get_model_handler
from .data_handler import load_and_prepare_dataset

def run_benchmark():
    """
    Runs the full ASR benchmarking process.
    """
    print("Starting ASR Benchmark...")
    print(f"Using device: {config.DEVICE}")

    # 1. Load metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    results = []

    # 2. Loop over models
    for model_name in config.MODELS_TO_BENCHMARK:
        try:
            handler = get_model_handler(model_name, config.DEVICE, config.TORCH_DTYPE)
            handler.load_model()
        except Exception as e:
            print(f"Failed to load model {model_name}. Skipping. Error: {e}")
            continue

        # 3. Loop over datasets for the current model
        for ds_name, ds_cfg in config.DATASETS_INFO:
            try:
                dataset = load_and_prepare_dataset(ds_name, ds_cfg, config.NUM_SAMPLES)
                
                refs, hypos = [], []
                
                print(f"  Processing {len(dataset)} samples...")
                for example in tqdm(dataset, desc=f"  {ds_name[:25]}..."):
                    audio_array = example["audio"]["array"]
                    reference_text = example[ds_cfg["field"]].strip().lower()

                    try:
                        hypothesis_text = handler.transcribe(audio_array).strip().lower()
                    except Exception as e:
                        print(f"    Error during transcription: {e}. Skipping sample.")
                        hypothesis_text = "" # Assign empty string on error

                    refs.append(reference_text)
                    hypos.append(hypothesis_text)
                
                # Calculate metrics
                wer = wer_metric.compute(references=refs, predictions=hypos)
                cer = cer_metric.compute(references=refs, predictions=hypos)
                print(f"    WER: {wer:.4f}, CER: {cer:.4f}")
                
                dataset_display_name = ds_name
                if "config" in ds_cfg:
                    dataset_display_name = f"{ds_name}-{ds_cfg['config']}"

                results.append({
                    "model": model_name,
                    "dataset": dataset_display_name,
                    "split": ds_cfg["split"],
                    "WER": wer,
                    "CER": cer
                })

            except Exception as e:
                print(f"Failed to process dataset {ds_name}. Skipping. Error: {e}")
                continue
    
    # 4. Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(config.RESULTS_CSV_PATH, index=False)
        print(f"\n✅ Benchmark complete. Results saved to {config.RESULTS_CSV_PATH}")
    else:
        print("\nNo results were generated.")

if __name__ == "__main__":
    run_benchmark()
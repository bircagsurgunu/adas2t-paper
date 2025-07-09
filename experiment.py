import pandas as pd
import evaluate
from datasets import load_dataset, Audio
from transformers import pipeline, AutoProcessor, AutoModelForPreTraining, Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC, AutoModelForSpeechSeq2Seq
import torch
from itertools import islice
import nemo.collections.asr as nemo_asr

# 1. Define models and datasets
models = [
    # "facebook/wav2vec2-base-960h",
    # "facebook/hubert-large-ls960-ft",
    "openai/whisper-large-v3",
    # "ibm-granite/granite-speech-3.3-8b",
    # "nvidia/parakeet-tdt-0.6b-v2",
]
datasets_info = [
    ("mozilla-foundation/common_voice_16_1", {"split": "test", "field": "sentence"}),
    ("openslr/librispeech_asr",      {"split": "test.clean", "field": "text"}),
    # ("espnet/yodas2",                {"field": "text"}),
    # ("edinburghcstr/ami",            {"split": "test", "config": "ihm", "field": "text"}),
    # ("edinburghcstr/ami",            {"split": "test", "config": "sdm", "field": "text"}),
    # ("distil-whisper/earnings22",    {"split": "test",       "field": "transcription"}),
    # ("facebook/voxpopuli",       {"split": "test[:100]",       "field": "raw_text"}),
    # ("google/fleurs",            {"split": "test",       "field": "transcription"}),
]

# 2. Load metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# 3. Loop over models & datasets
results = []
device = 0 if torch.cuda.is_available() else -1

for model_name in models:
    print(f"\nLoading pipeline for {model_name}…")
    if "whisper" in model_name:
        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        asr = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,
            device=device,
        )
    elif "960h" in model_name:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        asr = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    elif "hubert" in model_name:
        processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        asr = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
    elif "parakeet" in model_name:
        asr = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    elif "ibm" in model_name:
        processor = AutoProcessor.from_pretrained("ibm-granite/granite-speech-3.3-8b")
        tokenizer = processor.tokenizer
        asr = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b").to(device)
        chat = [
            {
                "role": "system",
                "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant",
            },
            {
                "role": "user",
                "content": "<|audio|>can you transcribe the speech into a written format?",
            }
        ]
        text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
    for ds_name, cfg in datasets_info:
        print(f"→ Dataset: {ds_name} ")
        if "mozilla" in ds_name:
            ds = load_dataset(ds_name, "en", split=cfg["split"], streaming = True, trust_remote_code=True)
        elif "espnet" in ds_name:
            ds = load_dataset(ds_name, 'en000', streaming=True, trust_remote_code=True)
        elif "edinburghcstr/ami" in ds_name:
            config = cfg.get("config", "ihm")  # Default to "ihm" if no config specified
            ds = load_dataset(ds_name, config, split=cfg["split"], streaming=True, trust_remote_code=True)
        elif "distil-whisper/earnings22" in ds_name:
            ds = load_dataset(ds_name, "chunked", split=cfg["split"], streaming=True, trust_remote_code=True)
        elif "voxpopuli" in ds_name:
            ds = load_dataset(ds_name, "en_accented", split=cfg["split"])
        elif "google/fleurs" in ds_name:
            ds = load_dataset(ds_name, "en_us", split=cfg["split"], streaming = True, trust_remote_code=True)
        else:
            ds = load_dataset(ds_name, split=cfg["split"], streaming = True, trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000))   
        # Optionally limit number of examples (e.g., first 100) for speed:
        if "voxpopuli" not in ds_name:
            ds = list(islice(ds, 100)) # Remove or adjust as needed
        
        refs, hypos = [], []
        for ex in ds:
            audio = ex["audio"]
            # pipeline can take path or array+sampling_rate
            if "parakeet" in model_name:
                pred = asr.transcribe(audio["array"])[0].text
            elif "960h" in model_name or "hubert" in model_name:
                input_values = processor(audio["array"], return_tensors="pt", padding="longest").input_values  # Batch size 1
 
                # retrieve logits
                logits = asr(input_values).logits
                
                # take argmax and decode
                predicted_ids = torch.argmax(logits, dim=-1)
                pred = processor.batch_decode(predicted_ids)[0]
            elif "ibm" in model_name:
                model_inputs = processor(
                    text,
                    audio["array"],
                    device=device, # Computation device; returned tensors are put on CPU
                    return_tensors="pt",
                ).to(device)
                model_outputs = asr.generate(
                    **model_inputs,
                    max_new_tokens=200,
                    num_beams=4,
                    do_sample=False,
                    min_length=1,
                    top_p=1.0,
                    repetition_penalty=1.0,
                    length_penalty=1.0,
                    temperature=1.0,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                num_input_tokens = model_inputs["input_ids"].shape[-1]
                new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)

                output_text = tokenizer.batch_decode(
                    new_tokens, add_special_tokens=False, skip_special_tokens=True
                )
                pred = output_text[0]
                # print(f"Pred: {pred}")
            else:
                pred = asr(audio["array"], generate_kwargs={"language": "english"})["text"]
            hypos.append(pred.strip().lower())
            refs.append(ex[cfg["field"]].strip().lower())
        
        wer = wer_metric.compute(references=refs, predictions=hypos)
        cer = cer_metric.compute(references=refs, predictions=hypos)
        print(f"    WER: {wer:.3f},  CER: {cer:.3f}")
        
        dataset_name = ds_name
        if "edinburghcstr/ami" in ds_name:
            dataset_name = f"{ds_name}-{cfg.get('config', 'ihm')}"
        
        results.append({
            "model": model_name,
            "dataset": dataset_name,
            "split": cfg["split"],
            "WER": wer,
            "CER": cer
        })

# 4. Save to CSV
df = pd.DataFrame(results)
df.to_csv("asr_comparison_results.csv", index=False)
print("\nResults saved to asr_comparison_results.csv")

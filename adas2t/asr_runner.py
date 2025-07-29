# asr_runner.py
"""
Yield a flat stream of
    (dataset_name, model_name, reference_text, hypothesis_text, wav_path)
so downstream code can compute per‑clip WER, extract features, etc.

Only three models are handled here:
    • openai/whisper-large-v3
    • mistralai/Voxtral-Mini-3B-2507   (via vLLM‑style OpenAI endpoint)
    • nvidia/canary-qwen-2.5b          (SALM)

Add more loaders if necessary.
"""
import io, os, base64, uuid, tempfile
from itertools import islice

import soundfile as sf
import torch
from datasets import load_dataset, Audio
from transformers import (
    pipeline,
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
)
from nemo.collections.speechlm2.models import SALM
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────  CONFIG  ───────────────────────────────────────
MODELS = [
    "openai/whisper-large-v3",
    "mistralai/Voxtral-Mini-3B-2507",
    "nvidia/canary-qwen-2.5b",
]

DATASETS_INFO = [
    ("mozilla-foundation/common_voice_16_1", {"split": "test",       "field": "sentence"}),
    # ("openslr/librispeech_asr",              {"split": "test.clean", "field": "text"}),
    # ("openslr/librispeech_asr",              {"split": "test.other", "field": "text"}),
    ("edinburghcstr/ami",                    {"split": "test", "config": "ihm", "field": "text"}),
    ("edinburghcstr/ami",                    {"split": "test", "config": "sdm", "field": "text"}),
    ("distil-whisper/earnings22",            {"split": "test",       "field": "transcription"}),
    ("facebook/voxpopuli",                   {"split": "test",       "field": "raw_text"}),
    ("google/fleurs",                        {"split": "test",       "field": "transcription"}),
    ("MLCommons/peoples_speech",             {"split": "test[:400]", "field": "text"}),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CLIPS_PER_DATASET = 400        # speed cap; adjust as you like
SAMPLE_RATE = 16_000

# ────────────────────────  MODEL LOADING HELPERS  ─────────────────────────────
def _load_whisper() -> callable:
    """Return callable: float32 waveform -> transcript str."""
    model_id = "openai/whisper-large-v3"
    proc  = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEVICE)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=proc.tokenizer,
        feature_extractor=proc.feature_extractor,
        torch_dtype=torch.float16,
        device=DEVICE,
        generate_kwargs={"language": "english"},
    )

    def transcribe(arr):
        return pipe(arr)["text"]          # ← extract only the string
    return transcribe

def _load_mistral() -> tuple[callable, object]:
    """
    Assumes you expose Voxtral through a vLLM / OpenAI‑compatible endpoint
    at http://localhost:8000/v1 with API_KEY="EMPTY".
    """
    from mistral_common.protocol.transcription.request import TranscriptionRequest
    from mistral_common.protocol.instruct.messages import RawAudio
    from openai import OpenAI

    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )
    model_name = client.models.list().data[0].id   # e.g. "Voxtral-Mini-3B-2507"

    def transcribe(arr):
        # 1) write bytes to memory buffer
        buf = io.BytesIO()
        sf.write(buf, arr, SAMPLE_RATE, format="WAV")
        raw = RawAudio(
            data=base64.b64encode(buf.getvalue()).decode("utf-8"),
            format="wav",
        )
        req = TranscriptionRequest(
            model=model_name,
            audio=raw,
            language="en",
            temperature=0.0,
        ).to_openai(exclude=("top_p", "seed"))
        return client.audio.transcriptions.create(**req).text
    return transcribe

def _load_canary() -> callable:
    salm = SALM.from_pretrained("nvidia/canary-qwen-2.5b").to(DEVICE)

    def transcribe(arr):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, arr, SAMPLE_RATE)
            wav_path = tmp.name
        ids = salm.generate(
            prompts=[
                [{"role":"user",
                  "content":f"Transcribe the following: {salm.audio_locator_tag}",
                  "audio": wav_path}]
            ],
            max_new_tokens=128,
        )
        return salm.tokenizer.ids_to_text(ids[0].cpu())
    return transcribe

# Dispatcher — load once then reuse
MODEL_BANK = {
    "openai/whisper-large-v3"        : _load_whisper(),
    "mistralai/Voxtral-Mini-3B-2507" : _load_mistral(),
    "nvidia/canary-qwen-2.5b"        : _load_canary(),
}

# ──────────────────────────  DATASET LOADER  ──────────────────────────────────
def _load_dataset_split(ds_name: str, cfg: dict):
    if "mozilla" in ds_name:
        ds = load_dataset(ds_name, "en", split=cfg["split"], streaming=True, trust_remote_code=True)
    elif "edinburghcstr/ami" in ds_name:
        ds = load_dataset(ds_name, cfg["config"], split=cfg["split"], streaming=True, trust_remote_code=True)
    elif "distil-whisper/earnings22" in ds_name:
        ds = load_dataset(ds_name, "chunked", split=cfg["split"], streaming=True, trust_remote_code=True)
    elif "facebook/voxpopuli" in ds_name:
        ds = load_dataset(ds_name, "en_accented", split=cfg["split"])  # not streaming
    elif "google/fleurs" in ds_name:
        ds = load_dataset(ds_name, "en_us", split=cfg["split"], streaming=True, trust_remote_code=True)
    elif "MLCommons/peoples_speech" in ds_name:
        ds = load_dataset(ds_name, "test", split=cfg["split"], trust_remote_code=True)
    else:
        ds = load_dataset(ds_name, split=cfg["split"], streaming=True, trust_remote_code=True)

    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    if "voxpopuli" in ds_name or "MLCommons/peoples_speech" in ds_name:
        def _keep_short(ex):
            arr = ex["audio"]["array"]
            sr = ex["audio"]["sampling_rate"]
            if sr == 0 or arr is None:
                return False
            length = len(arr) / sr
            return length <= 30.0
        ds = ds.filter(_keep_short)
    return ds

# ──────────────────────────  MAIN GENERATOR  ──────────────────────────────────
def iter_datasets_and_models():
    """
    Yields tuples
        (dataset_name, model_name, reference_text, hypothesis_text, wav_path)
    """
    for ds_name, cfg in DATASETS_INFO:
        print(f"\n▸ Loading dataset {ds_name} ({cfg['split']}) …")
        ds_iter = _load_dataset_split(ds_name, cfg)
        ds_iter = islice(ds_iter, MAX_CLIPS_PER_DATASET)

        for ex in tqdm(ds_iter, desc=ds_name, unit="clip"):
            # write once to temp .wav for models that need a path
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, ex["audio"]["array"], SAMPLE_RATE)
                wav_path = tmp.name

            reference = ex[cfg["field"]].strip().lower()

            for model_name in MODELS:
                transcribe_fn = MODEL_BANK[model_name]
                hypothesis = transcribe_fn(ex["audio"]["array"]).strip().lower()

                yield ds_name, model_name, reference, hypothesis, wav_path

            # Run all models in parallel
            # with ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
            #     # submit one future per model
            #     futures = {
            #         executor.submit(MODEL_BANK[m], ex["audio"]["array"]): m
            #         for m in MODELS
            #     }
            #     # as each finishes, yield its result
            #     for fut in as_completed(futures):
            #         model_name = futures[fut]
            #         try:
            #             hypothesis = fut.result().strip().lower()
            #         except Exception as e:
            #             hypothesis = f"<error: {e}>"
            #         yield ds_name, model_name, reference, hypothesis, wav_path

# ──────────────────────────  TEST CLI  (optional)  ────────────────────────────
if __name__ == "__main__":
    from jiwer import wer
    import json

    out_path = "quick_check.jsonl"
    with open(out_path, "w") as fout:
        for ds, mdl, ref, hyp, wav in islice(iter_datasets_and_models(), 20):
            rec = {
                "uid": str(uuid.uuid4()),
                "dataset": ds,
                "model": mdl,
                "wer": wer(ref, hyp),
            }
            fout.write(json.dumps(rec) + "\n")
            print(rec)
    print(f"\nWrote first 20 records to {out_path}")

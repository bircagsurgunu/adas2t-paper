import sys, tempfile, soundfile as sf
from feature_extractor import extract_features
import xgboost as xgb, numpy as np
MODELS = [
    "openai/whisper-large-v3",
    "mistralai/Voxtral-Mini-3B-2507",
    "nvidia/canary-qwen-2.5b"
]

bst = xgb.Booster();  bst.load_model("adas2t_xgb.json")

def pick(wav_path):
    feats = extract_features(wav_path)[None,:]
    wer_hat = bst.predict(xgb.DMatrix(feats))[0]
    idx = int(np.argmin(wer_hat))
    return MODELS[idx], wer_hat[idx]

if __name__ == "__main__":
    wav = sys.argv[1]
    model, exp_wer = pick(wav)
    print(f"Choose → {model}  (expected WER≈{exp_wer:.2%})")

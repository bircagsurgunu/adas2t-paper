# feature_extractor.py
import warnings, librosa, numpy as np, torch, torchaudio
from transformers import AutoModel, AutoProcessor
from scipy.stats import entropy 

_SR = 16_000                      # paper uses 16 kHz throughout
_MFCC_BANDS = 40                  # 40 MFCCs → mean+std = 80 dims

# --------------------------- 1. Self‑supervised embedding -----------------------
# Load once, reuse
_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
_ssl_model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h").eval().to("cuda")

@torch.inference_mode()
def _wav2vec2_embed(wav: np.ndarray) -> np.ndarray:
    inputs = _processor(wav, sampling_rate=_SR, return_tensors="pt").to("cuda")
    hidden = _ssl_model(**inputs).last_hidden_state
    # paper: mean‑pool over time (768‑D)
    return hidden.mean(dim=1).squeeze().cpu().float().numpy()   # (768,)

# --------------------------- 2. Utility helpers --------------------------------
def _snr_db(y):
    sig = np.mean(y ** 2)
    noise = np.mean((y - y.mean()) ** 2)
    return 10 * np.log10(sig / (noise + 1e-12))

def _speaking_rate(y):
    # rough: phones per second ≈ zero‑crossings‑per‑sec / 10  (paper’s heuristic)
    zc = np.mean(librosa.feature.zero_crossing_rate(y))
    return zc * _SR / 10

def _percent_silence(y, thr_db=-35):
    rms  = librosa.feature.rms(y=y).flatten()
    db  = librosa.amplitude_to_db(rms, ref=np.max)
    return np.mean(db < thr_db)

# --------------------------- 3. Main public API --------------------------------
def extract_features(wav_path: str) -> np.ndarray:
    """
    Returns a single 130‑D vector exactly matching paper:
      0‑79   : MFCC mean+std (40×2)
      80‑85  : ZCR mean+std, RMS mean+std, SNR, spectral_entropy
      86‑88  : speaking_rate, percent_silence, pitch_mean
      89‑129 : Wav2Vec2 mean pooled embedding (768)  -> **we PCA‑reduce to 41 dims**
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    y, sr = librosa.load(wav_path, sr=_SR, mono=True)
    # ---- 1. MFCCs --------------------------------------------------------------
    mfcc = librosa.feature.mfcc(y=y, sr=_SR, n_mfcc=_MFCC_BANDS)
    mfcc_stats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])      # 80
    # ---- 2. Signal complexity -------------------------------------------------
    zcr  = librosa.feature.zero_crossing_rate(y=y).flatten()
    rms  = librosa.feature.rms(y=y).flatten()
    S = np.abs(librosa.stft(y=y, n_fft=1024, hop_length=512))**2    # power
    psd = S.mean(axis=1)                                            # mean over frames
    psd_norm = psd / (psd.sum() + 1e-12)
    spec_ent = entropy(psd_norm, base=None)                         # natural log
    sig_feats = np.array([
        zcr.mean(), zcr.std(),
        rms.mean(), rms.std(),
        _snr_db(y),
        spec_ent
    ])                                                                    # 6
    # ---- 3. Prosody -----------------------------------------------------------
    pitch, _, _ = librosa.pyin(y=y, fmin=65, fmax=400, sr=_SR)
    if np.isnan(pitch).all():
        pitch_mean = 0.0            # or np.nan, or -1.0 → pick one convention
    else:
        pitch_mean = np.nanmean(pitch)

    prosody = np.array([
        _speaking_rate(y),
        _percent_silence(y),
        pitch_mean
    ])                                                                 # 3
    # ---- 4. SSL embedding -----------------------------------------------------
    ssl = _wav2vec2_embed(y)                                                # 768
    # Reduce to 41 dims via fixed PCA matrix shipped with the repo
    pca = np.load("pca_w2v2_768x41.npy")                                    # (768,41)
    ssl_reduced = ssl @ pca                                                 # 41

    return np.concatenate([mfcc_stats, sig_feats, prosody, ssl_reduced])    # 130‑D

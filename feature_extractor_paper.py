import numpy as np
import librosa
import torch
import warnings
import logging
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    HubertModel
)
from scipy.stats import entropy
from scipy.signal import hilbert
import webrtcvad
from python_speech_features import mfcc, delta
import parselmouth
from parselmouth.praat import call

# Suppress warnings from libraries
warnings.filterwarnings('ignore')
# Set transformers logging to error to avoid spam
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

_EPS = 1e-10
_SR = 16000 


class AudioFeatureExtractor:
    """
    Comprehensive audio feature extractor implementing all features described in ADAS2T,
    extended with regime-specific descriptors:
      - Channel/room proxies (spectral tilt, HNR, reverberation/decay statistics, clarity-like ratios)
      - Bandwise (mel) SNR statistics
      - Modulation spectrum bands (2–20 Hz)
      - Rich prosody: pitch stats + jitter/shimmer; VAD segment statistics/quantiles
      - Signal health: clipping ratio, dynamic range, DC offset
      - Neural embeddings with richer temporal pooling statistics
    Output is a dict of named feature groups; values are float32 vectors.
    """

    def __init__(self, sample_rate: int = 16000, device: str = "cpu"):
        self.sample_rate = sample_rate
        self.device = device
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3
        self._load_embedding_models()

    def _load_embedding_models(self):
        """Load pre-trained models for neural embeddings"""
        logger.info("ADAS2T: Loading neural embedding models for feature extraction...")
        try:
            # self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            # self.whisper_model = (
            #     WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            #     .to(self.device)
            #     .eval()
            # )
            self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.wav2vec2_model = (
                Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
                .to(self.device)
                .eval()
            )
            self.hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(self.device).eval()
            logger.info("ADAS2T: Successfully loaded neural embedding models.")
        except Exception as e:
            logger.warning(
                f"ADAS2T: Could not load all embedding models: {e}. Neural embedding features will be partial."
            )
            # self.whisper_processor = self.whisper_model = None
            self.wav2vec2_processor = self.wav2vec2_model = None
            self.hubert_model = None

    # -------------------------
    # Public API
    # -------------------------
    def extract_all_features(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        """
        Extract and return all feature groups as a dict.
        Each value is a 1-D float32 vector (NaNs/Infs sanitized to 0.0).
        """
        features: dict[str, np.ndarray] = {}

        # MFCC (+Δ, ΔΔ) with mean/std pooling
        features["mfcc"] = self._extract_mfcc_features(audio)

        # Prosodic: pitch stats + jitter/shimmer + voiced/silence segment statistics
        features["prosodic"] = self._extract_prosodic_features(audio)

        # VAD/activity profile (quantiles, counts, overlap proxy not included)
        features["activity"] = self._extract_activity_features(audio)

        # Channel/room & signal-health proxies
        features["channel"] = self._extract_channel_room_features(audio)

        # Noise & modulation spectrum (2–20 Hz energy bands, spectral flatness)
        features["noise_modulation"] = self._extract_noise_modulation_features(audio)

        # Classical signal complexity (SNR proxy, ZCR, spectral stats, duration)
        features["signal"] = self._extract_signal_complexity_features(audio)

        # Neural embeddings (per model mean vector, as before)
        neural = self._extract_neural_embeddings(audio)
        features.update(neural)

        # Rich temporal statistics of embedding sequences (model-agnostic difficulty/dispersion)
        emb_stats = self._extract_embedding_stats(audio)
        features.update(emb_stats)

        # Metadata / simple flags (here: the sampling rate)
        features["metadata"] = np.array([float(self.sample_rate)], dtype=np.float32)

        # Sanitize
        for k, v in features.items():
            features[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return features

    # -------------------------
    # MFCCs
    # -------------------------
    def _extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        mfcc_feat = mfcc(audio, self.sample_rate, numcep=13, nfilt=26, nfft=512)
        delta_feat = delta(mfcc_feat, N=2)
        delta_delta_feat = delta(delta_feat, N=2)
        all_features = np.concatenate([mfcc_feat, delta_feat, delta_delta_feat], axis=1)
        return np.concatenate([np.mean(all_features, axis=0), np.std(all_features, axis=0)])

    # -------------------------
    # Prosody (pitch + jitter/shimmer)
    # -------------------------
    def _extract_prosodic_features(self, audio: np.ndarray) -> np.ndarray:
        out = []
        try:
            snd = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
            # Pitch
            pitch = snd.to_pitch()
            f0 = pitch.selected_array["frequency"]
            f0 = f0[f0 > 0]
            if f0.size > 0:
                out.extend([
                    float(np.mean(f0)), float(np.std(f0)),
                    float(np.max(f0) - np.min(f0)), float(np.median(f0))
                ])
            else:
                out.extend([0.0] * 4)

            # Jitter/Shimmer
            pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
            try:
                jitter_local = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            except Exception:
                jitter_local = 0.0
            try:
                shimmer_local = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            except Exception:
                shimmer_local = 0.0
            out.extend([float(jitter_local), float(shimmer_local)])
        except Exception:
            out.extend([0.0] * 6)
        return np.array(out, dtype=np.float32)

    # -------------------------
    # VAD-derived activity statistics
    # -------------------------
    def _extract_activity_features(self, audio: np.ndarray) -> np.ndarray:
        sr = self.sample_rate
        frame_ms, hop_ms = 30, 10
        frame_len = int(frame_ms * sr / 1000)
        hop_len = int(hop_ms * sr / 1000)

        x16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        flags = []
        for i in range(0, len(x16) - frame_len + 1, hop_len):
            frame = x16[i: i + frame_len]
            try:
                flags.append(int(self.vad.is_speech(frame.tobytes(), sr)))
            except Exception:
                flags.append(0)
        flags = np.array(flags, dtype=np.int32)
        if flags.size == 0:
            return np.zeros(10, dtype=np.float32)

        voiced_ratio = float(np.mean(flags))
        silence_ratio = 1.0 - voiced_ratio

        def _segments(binary_seq):
            segs, run_val, run_len = [], None, 0
            for v in binary_seq:
                if run_val is None or v != run_val:
                    if run_val is not None:
                        segs.append((run_val, run_len))
                    run_val, run_len = v, 1
                else:
                    run_len += 1
            if run_val is not None:
                segs.append((run_val, run_len))
            return segs

        segs = _segments(flags)
        step_s = hop_len / sr
        voiced_durs = [l * step_s for val, l in segs if val == 1]
        sil_durs = [l * step_s for val, l in segs if val == 0]
        T = len(audio) / sr + _EPS
        vps = len(voiced_durs) / T
        sps = len(sil_durs) / T

        def _stats(vals):
            return [float(np.mean(vals)), float(np.median(vals)), float(np.std(vals))] if vals else [0.0, 0.0, 0.0]

        feats = [
            voiced_ratio, silence_ratio,
            float(vps), *_stats(voiced_durs),
            float(sps), *_stats(sil_durs),
        ]
        return np.array(feats, dtype=np.float32)

    # -------------------------
    # Channel/room and signal-health proxies
    # -------------------------
    def _extract_channel_room_features(self, audio: np.ndarray) -> np.ndarray:
        sr = self.sample_rate
        out = []

        # Signal health
        dc = float(np.mean(audio))
        clipping = float(np.mean(np.abs(audio) >= 0.999))
        dyn_range = float(20 * np.log10((np.max(np.abs(audio)) + _EPS) /
                                        (np.percentile(np.abs(audio), 5) + _EPS)))
        out.extend([dc, clipping, dyn_range])

        # STFT & spectral tilt
        S = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256)) + _EPS
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        mean_spec = np.mean(S, axis=1)
        xf, yf = np.log(freqs[1:]), np.log(mean_spec[1:])
        try:
            A = np.vstack([xf, np.ones_like(xf)]).T
            slope, _ = np.linalg.lstsq(A, yf, rcond=None)[0]
        except Exception:
            slope = 0.0
        out.append(float(slope))

        # HNR
        try:
            snd = parselmouth.Sound(audio, sampling_frequency=sr)
            harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=75)
            hnr = float(call(harm, "Get mean", 0, 0))
        except Exception:
            hnr = 0.0
        out.append(hnr)

        # Spectral decay proxy percentiles
        try:
            logS_t = np.log(S + _EPS)
            t = np.arange(logS_t.shape[1], dtype=np.float32)
            t = (t - t.mean()) / (t.std() + _EPS)
            num = (logS_t * t).mean(axis=1) - logS_t.mean(axis=1) * t.mean()
            den = t.var() + _EPS
            slopes = num / den
            p25, p50, p75 = np.percentile(slopes, [25, 50, 75])
        except Exception:
            p25 = p50 = p75 = 0.0
        out.extend([float(p25), float(p50), float(p75)])

        # Clarity-like ratio
        try:
            hop = int(0.01 * sr)
            rms = librosa.feature.rms(y=audio, frame_length=2 * hop, hop_length=hop)[0]
            w = int(0.05 / (hop / sr) + 1)
            early = float(np.sum(rms[:w]))
            late = float(np.sum(rms[w:]) + _EPS)
            clarity = early / late
        except Exception:
            clarity = 0.0
        out.append(clarity)

        return np.array(out, dtype=np.float32)

    # -------------------------
    # Noise & modulation spectrum features
    # -------------------------
    def _extract_noise_modulation_features(self, audio: np.ndarray) -> np.ndarray:
        sr = self.sample_rate
        feats = []

        # Bandwise mel-SNR
        try:
            S = librosa.feature.melspectrogram(y=audio, sr=sr,
                                               n_fft=1024, hop_length=256,
                                               n_mels=40, power=2.0)
            band_mean = np.mean(S, axis=1)
            band_floor = np.percentile(S, 10, axis=1) + _EPS
            mel_snr = 10.0 * np.log10((band_mean + _EPS) / band_floor)
            feats.extend([
                float(np.mean(mel_snr)), float(np.std(mel_snr)),
                float(np.percentile(mel_snr, 10)),
                float(np.percentile(mel_snr, 90))
            ])
        except Exception:
            feats.extend([0.0, 0.0, 0.0, 0.0])

        # Spectral flatness
        try:
            flat = librosa.feature.spectral_flatness(y=audio)[0]
            feats.extend([float(np.mean(flat)), float(np.std(flat))])
        except Exception:
            feats.extend([0.0, 0.0])

        # Modulation bands
        try:
            hop = int(0.01 * sr)
            rms = librosa.feature.rms(y=audio, frame_length=2 * hop, hop_length=hop)[0]
            rms = (rms - rms.mean()) / (rms.std() + _EPS)
            fs_env = sr / hop
            F = np.fft.rfft(rms)
            freqs = np.fft.rfftfreq(len(rms), d=1.0 / fs_env)
            power = (np.abs(F) ** 2)

            def band_energy(flo, fhi):
                idx = np.where((freqs >= flo) & (freqs < fhi))[0]
                return float(power[idx].sum() / (power.sum() + _EPS)) if idx.size else 0.0

            feats.extend([
                band_energy(0.0, 2.0), band_energy(2.0, 4.0),
                band_energy(4.0, 10.0), band_energy(10.0, 20.0)
            ])
        except Exception:
            feats.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(feats, dtype=np.float32)

    # -------------------------
    # Classic signal/complexity features
    # -------------------------
    def _extract_signal_complexity_features(self, audio: np.ndarray) -> np.ndarray:
        feats = []
        try:
            frame_energy = librosa.feature.rms(y=audio)[0]
            snr = 10 * np.log10(
                (np.mean(frame_energy) + _EPS) /
                max(np.percentile(frame_energy, 10), _EPS)
            )
            feats.append(float(snr))
        except Exception:
            feats.append(0.0)

        try:
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            feats.extend([float(np.mean(zcr)), float(np.std(zcr))])

            spectral_centroid = librosa.feature.spectral_centroid(y=audio,
                                                                   sr=self.sample_rate)[0]
            feats.extend([float(np.mean(spectral_centroid)),
                          float(np.std(spectral_centroid))])

            spectral_bw = librosa.feature.spectral_bandwidth(y=audio,
                                                             sr=self.sample_rate)[0]
            feats.extend([float(np.mean(spectral_bw)),
                          float(np.std(spectral_bw))])

            rolloff = librosa.feature.spectral_rolloff(y=audio,
                                                       sr=self.sample_rate)[0]
            feats.append(float(np.mean(rolloff)))

            stft = np.abs(librosa.stft(audio)) ** 2
            spec_entropy = [
                entropy(frame / (np.sum(frame) + _EPS)) for frame in stft.T
                if np.sum(frame) > 0
            ]
            feats.extend(
                [float(np.mean(spec_entropy)), float(np.std(spec_entropy))]
                if spec_entropy else [0.0, 0.0]
            )
        except Exception:
            feats.extend([0.0] * 9)

        feats.append(float(len(audio) / self.sample_rate))
        return np.array(feats, dtype=np.float32)

    # -------------------------
    # Neural embeddings (mean vector per model)
    # -------------------------
    def _extract_neural_embeddings(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        embeddings = {}
        with torch.no_grad():
            # Whisper
            # if self.whisper_model:
            #     try:
            #         inputs = self.whisper_processor(
            #             audio, sampling_rate=self.sample_rate,
            #             return_tensors="pt"
            #         ).to(self.device)
            #         enc = self.whisper_model.model.encoder(**inputs)
            #         embeddings["neural_whisper"] = enc.last_hidden_state.mean(
            #             dim=1).squeeze().cpu().numpy()
            #     except Exception:
            #         embeddings["neural_whisper"] = np.zeros(512, dtype=np.float32)
            # else:
            #     embeddings["neural_whisper"] = np.zeros(512, dtype=np.float32)

            # Wav2Vec2
            if self.wav2vec2_model:
                try:
                    inputs = self.wav2vec2_processor(
                        audio, sampling_rate=self.sample_rate,
                        return_tensors="pt"
                    ).to(self.device)
                    out = self.wav2vec2_model.wav2vec2(**inputs)
                    embeddings["neural_w2v2"] = out.last_hidden_state.mean(
                        dim=1).squeeze().cpu().numpy()
                except Exception:
                    embeddings["neural_w2v2"] = np.zeros(768, dtype=np.float32)
            else:
                embeddings["neural_w2v2"] = np.zeros(768, dtype=np.float32)

            # HuBERT
            if self.hubert_model:
                try:
                    inp = torch.tensor(audio, dtype=torch.float32,
                                       device=self.device).unsqueeze(0)
                    out = self.hubert_model(inp)
                    embeddings["neural_hubert"] = out.last_hidden_state.mean(
                        dim=1).squeeze().cpu().numpy()
                except Exception:
                    embeddings["neural_hubert"] = np.zeros(768, dtype=np.float32)
            else:
                embeddings["neural_hubert"] = np.zeros(768, dtype=np.float32)

        # ensure float32
        for k in embeddings:
            embeddings[k] = embeddings[k].astype(np.float32, copy=False)
        return embeddings

    # -------------------------
    # Rich temporal statistics over embedding sequences
    # -------------------------
    def _extract_embedding_stats(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        stats = {
            # "embedding_stats_whisper": np.zeros(6, dtype=np.float32),
            "embedding_stats_w2v2":   np.zeros(6, dtype=np.float32),
            "embedding_stats_hubert": np.zeros(6, dtype=np.float32),
        }

        def _seq_stats(H: torch.Tensor) -> np.ndarray:
            if H is None or H.numel() == 0:
                return np.zeros(6, dtype=np.float32)
            x = H.detach().cpu().numpy()
            norms = np.linalg.norm(x, axis=1)
            var_t = np.var(x, axis=0)
            if x.shape[0] >= 2:
                a, b = x[:-1], x[1:]
                num = (a * b).sum(axis=1)
                den = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + _EPS)
                cos = num / den
            else:
                cos = np.array([0.0])
            return np.array([
                float(np.mean(norms)), float(np.std(norms)),
                float(np.mean(var_t)), float(np.std(var_t)),
                float(np.mean(cos)),  float(np.std(cos)),
            ], dtype=np.float32)

        with torch.no_grad():
            # Whisper
            # if self.whisper_model:
            #     try:
            #         inp = self.whisper_processor(
            #             audio, sampling_rate=self.sample_rate,
            #             return_tensors="pt"
            #         ).to(self.device)
            #         enc = self.whisper_model.model.encoder(**inp)
            #         H = enc.last_hidden_state.squeeze(0)
            #         stats["embedding_stats_whisper"] = _seq_stats(H)
            #     except Exception:
            #         pass

            # Wav2Vec2
            if self.wav2vec2_model:
                try:
                    inp = self.wav2vec2_processor(
                        audio, sampling_rate=self.sample_rate,
                        return_tensors="pt"
                    ).to(self.device)
                    out = self.wav2vec2_model.wav2vec2(**inp)
                    H = out.last_hidden_state.squeeze(0)
                    stats["embedding_stats_w2v2"] = _seq_stats(H)
                except Exception:
                    pass

            # HuBERT
            if self.hubert_model:
                try:
                    inp = torch.tensor(audio, dtype=torch.float32,
                                       device=self.device).unsqueeze(0)
                    out = self.hubert_model(inp).last_hidden_state.squeeze(0)
                    stats["embedding_stats_hubert"] = _seq_stats(out)
                except Exception:
                    pass

        return stats


import numpy as np
import librosa

# instantiate once (device choice is up to you)
_global_extractor = AudioFeatureExtractor(sample_rate=16000, device="cuda")

def extract_features(path_or_array) -> np.ndarray:
    """
    Legacy wrapper to behave like your old function:
      - input: path or raw waveform
      - output: 1D np.ndarray of all features concatenated in a fixed order
    """
    if isinstance(path_or_array, str):
        audio, _ = librosa.load(path_or_array,
                                sr=_global_extractor.sample_rate,
                                mono=True)
    else:
        audio = path_or_array.astype(np.float32)

    feats = _global_extractor.extract_all_features(audio)

    key_order = [
        "mfcc",
        "prosodic",
        "activity",
        "channel",
        "noise_modulation",
        "signal",
        # "neural_whisper",
        "neural_w2v2",
        "neural_hubert",
        # "embedding_stats_whisper",
        "embedding_stats_w2v2",
        "embedding_stats_hubert",
        "metadata",
    ]

    return np.concatenate([feats[k] for k in key_order]).astype(np.float32)

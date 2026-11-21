"""Speech-to-text using faster-whisper (runs locally).

Audio from the browser is WAV, so we decode it ourselves with the standard
library + numpy and hand faster-whisper a ready float32 array. This avoids the
PyAV (`av`) decoding path, which can break on some package version combos.
"""
import io
import os
import tempfile
import wave
from functools import lru_cache

import numpy as np

TARGET_SR = 16000

# Hint so Whisper spells technical terms correctly (override with WHISPER_PROMPT in .env)
DEFAULT_PROMPT = (
    "Technical software interview. Terms: Node.js, Express, Redis, PostgreSQL, MongoDB, "
    "Docker, BullMQ, LRU cache, L1, L2, TTL, pub/sub, backend, two-tier cache, dead-letter queue, "
    "REST API, JWT, Pandas, NumPy, scikit-learn, XGBoost, Isolation Forest, GeeksforGeeks, ShortLink."
)


def _opts() -> dict:
    return dict(beam_size=5, vad_filter=True,
                initial_prompt=os.getenv("WHISPER_PROMPT", DEFAULT_PROMPT))


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel

    return WhisperModel(os.getenv("WHISPER_MODEL", "base"), device="cpu", compute_type="int8")


def _wav_to_float32(audio_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(audio_bytes), "rb") as w:
        channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())

    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {width} bytes")

    if channels > 1:  # mix down to mono
        data = data[: len(data) - len(data) % channels].reshape(-1, channels).mean(axis=1)

    if rate != TARGET_SR:  # simple linear resample to 16 kHz
        n = int(len(data) * TARGET_SR / rate)
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data)

    return data.astype(np.float32)


def transcribe(audio_bytes: bytes) -> str:
    try:
        audio = _wav_to_float32(audio_bytes)
        segments, _ = _model().transcribe(audio, **_opts())
    except (wave.Error, ValueError, EOFError):
        # Not a plain WAV: fall back to letting faster-whisper decode a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            path = f.name
        try:
            segments, _ = _model().transcribe(path, **_opts())
            return " ".join(s.text.strip() for s in segments).strip()
        finally:
            os.remove(path)
    return " ".join(s.text.strip() for s in segments).strip()

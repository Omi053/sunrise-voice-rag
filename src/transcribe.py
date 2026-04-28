import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, MutableMapping
import argparse

from faster_whisper import WhisperModel

from .utils import Timer, ensure_dir, get_logger

logger = get_logger(__name__)

LOW_CONFIDENCE_THRESHOLD = 0.5

class EmptyAudioError(ValueError):
    """Raised when the audio file exists but is empty / unreadable."""

@dataclass
class TranscriptionConfig:
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"   
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = True
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD


@dataclass
class TranscriptionResult:

    transcript: str
    language: str
    duration_sec: float
    avg_confidence: float
    words: list[dict[str, Any]] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)
    warning: str | None = None
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcript": self.transcript,
            "language": self.language,
            "duration_sec": round(self.duration_sec, 3),
            "avg_confidence": round(self.avg_confidence, 4),
            "words": self.words,
            "segments": self.segments,
            "warning": self.warning,
            "model": self.model,
        }

_model_cache: dict[str, WhisperModel] = {}

def _get_model(cfg: TranscriptionConfig) -> WhisperModel:
    
    key = f"{cfg.model_size}|{cfg.device}|{cfg.compute_type}"
    
    if key not in _model_cache:
        logger.info(
            "Loading Faster-Whisper '%s' on %s (compute_type=%s)",
            cfg.model_size, cfg.device, cfg.compute_type,
        )
        _model_cache[key] = WhisperModel(
            cfg.model_size, device=cfg.device, compute_type=cfg.compute_type
        )
    
    return _model_cache[key]

def save_transcript(result: TranscriptionResult, output_path: str | Path) -> Path:

    path = Path(output_path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    logger.info("Wrote transcript → %s", path)
    return path

def transcribe(
    audio_path: str | Path,
    cfg: TranscriptionConfig | None = None,
    bench: MutableMapping[str, float] | None = None,
) -> TranscriptionResult:
    """
        Transcribe an audio file and return a structured result.
    """
    
    cfg = cfg or TranscriptionConfig()
    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.stat().st_size == 0:
        raise EmptyAudioError(f"Audio file is empty: {path}")

    model = _get_model(cfg)

    with Timer("transcribe", sink=bench, logger=logger):
        segments_iter, info = model.transcribe(
            str(path),
            language=cfg.language,
            beam_size=cfg.beam_size,
            word_timestamps=True,
            vad_filter=cfg.vad_filter,
        )
        segments = list(segments_iter)

    words: list[dict[str, Any]] = []
    seg_dicts: list[dict[str, Any]] = []
    
    for seg in segments:
        seg_dicts.append({
            "id": seg.id,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "avg_logprob": round(seg.avg_logprob, 4),
        })
        for w in (seg.words or []):
            words.append({
                "text": w.word.strip(),
                "start": round(w.start, 3),
                "end": round(w.end, 3),
                "confidence": round(float(w.probability), 4),
            })

    transcript = " ".join(s["text"] for s in seg_dicts).strip()
    avg_conf = (
        sum(w["confidence"] for w in words) / len(words) if words else 0.0
    )

    warning: str | None = None
    
    if not words and not transcript:
        warning = "no_speech_detected"
        logger.warning("Whisper returned no segments - likely silence/noise")
    
    elif avg_conf < cfg.low_confidence_threshold:
        warning = "low_confidence"
        logger.warning("Average word confidence %.2f below threshold %.2f", avg_conf, cfg.low_confidence_threshold)

    return TranscriptionResult(
        transcript=transcript,
        language=info.language,
        duration_sec=float(info.duration),
        avg_confidence=avg_conf,
        words=words,
        segments=seg_dicts,
        warning=warning,
        model=cfg.model_size,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Transcribe an audio file.")
    parser.add_argument("audio", type=Path, help="Path to audio file.")
    parser.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size. Default: base.",
    )
    args = parser.parse_args()

    cfg = TranscriptionConfig(model_size=args.model)
    bench: dict[str, float] = {}
    
    try:
        result = transcribe(args.audio, cfg=cfg, bench=bench)
    except (FileNotFoundError, EmptyAudioError) as exc:
        logging.error("Transcription failed: %s", exc)
        raise SystemExit(2)

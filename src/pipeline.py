import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ingest import CHROMA_DIR, build_index
from .rag import RAGConfig, answer_query
from .transcribe import EmptyAudioError, TranscriptionConfig, save_transcript, transcribe
from .utils import Timer, ensure_dir, get_logger, project_root

logger = get_logger(__name__)

@dataclass
class PipelineConfig:
    audio_path: Path
    pdf_path: Path
    output_dir: Path = field(default_factory=lambda: project_root() / "output")
    chroma_dir: Path = field(default_factory=lambda: CHROMA_DIR)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    rebuild_index: bool = False

def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    
    ensure_dir(cfg.output_dir)
    bench: dict[str, float] = {}

    logger.info("Stage 1/3: Indexing...")
    collection = build_index(cfg.pdf_path, persist_dir=cfg.chroma_dir, rebuild=cfg.rebuild_index, bench=bench)
    bench.setdefault("ingest", 0.0)

    logger.info("Stage 2/3: Transcribing...")
    try:
        transcription = transcribe(cfg.audio_path, cfg=cfg.transcription, bench=bench)
    except (FileNotFoundError, EmptyAudioError) as exc:
        logger.error("Transcription failed: %s", exc)
        return {"ok": False, "stage": "transcribe", "error": str(exc), "latency_sec": bench}
    save_transcript(transcription, cfg.output_dir / "transcript.json")

    if not transcription.transcript:
        logger.warning("Empty transcript — skipping RAG.")
        result = {
            "ok": False,
            "stage": "transcribe",
            "error": transcription.warning or "no_speech_detected",
            "transcript": transcription.to_dict(),
            "latency_sec": bench,
        }
        (cfg.output_dir / "answer.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    logger.info("Stage 3/3: Generating RAG answer...")
    with Timer("rag_total", bench):
        rag_result = answer_query(transcription.transcript, cfg=cfg.rag, collection=collection, bench=bench)

    final: dict[str, Any] = {
        "ok": rag_result.error is None,
        "transcript": transcription.to_dict(),
        "rag": rag_result.to_dict(),
        "latency_sec": bench,
        "config": {
            "whisper_model": cfg.transcription.model_size,
            "llm_model": cfg.rag.llm_model,
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": cfg.rag.top_k,
        },
    }

    out_path = cfg.output_dir / "answer.json"
    out_path.write_text(json.dumps(final, indent=2, ensure_ascii=False))
    logger.info("Wrote answer : %s", out_path)

    print(f"Investor Question : {transcription.transcript}")
    print(f"Assistant : {rag_result.answer}")
    print(f"Question Sources : {', '.join(rag_result.sources)}")
    print(f"Latency(s) : {json.dumps(bench)}")

    return final
#!/usr/bin/env python3
"""
Single command-line entrypoint for the Sunrise voice-RAG pipeline.

Examples
--------
Run the full pipeline against the bundled sample::

    python run.py

Custom audio + alternate LLM::

    python run.py --audio path/to/clip.mp3 --llm mistral:7b-instruct-q4_K_M

Force-rebuild the vector index after editing the PDF::

    python run.py --rebuild
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python run.py` from the project root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingest import DEFAULT_CHROMA_DIR  # noqa: E402
from src.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from src.rag import DEFAULT_LLM_MODEL, DEFAULT_TOP_K, RAGConfig  # noqa: E402
from src.transcribe import TranscriptionConfig  # noqa: E402
from src.utils import project_root  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Voice-powered investor support assistant (Sunrise AMC)."
    )
    parser.add_argument(
        "--audio", type=Path,
        default=root / "input" / "investor_sample.mp3",
        help="Path to the investor audio (default: input/investor_sample.mp3).",
    )
    parser.add_argument(
        "--pdf", type=Path,
        default=root / "input" / "SunriseAMC_FAQ.pdf",
        help="Path to the FAQ PDF (default: input/SunriseAMC_FAQ.pdf).",
    )
    parser.add_argument(
        "--whisper-model", default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Faster-Whisper model size. Default: base.",
    )
    parser.add_argument(
        "--llm", default=DEFAULT_LLM_MODEL,
        help="Ollama model tag. Default: %(default)s.",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help="How many FAQ chunks to retrieve (default: %(default)s).",
    )
    parser.add_argument(
        "--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR,
        help="Where Chroma persists the index.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "output",
        help="Where to write answer.json and transcript.json.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Wipe and rebuild the vector index before running.",
    )
    args = parser.parse_args(argv)

    cfg = PipelineConfig(
        audio_path=args.audio,
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        chroma_dir=args.chroma_dir,
        transcription=TranscriptionConfig(model_size=args.whisper_model),
        rag=RAGConfig(
            llm_model=args.llm,
            top_k=args.top_k,
            persist_dir=args.chroma_dir,
        ),
        rebuild_index=args.rebuild,
    )

    result = run_pipeline(cfg)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

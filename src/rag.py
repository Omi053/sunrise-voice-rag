import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, MutableMapping
from chromadb.api.models.Collection import Collection

try:
    import ollama
except ImportError as exc:
    raise RuntimeError("ollama not installed. Run: pip install ollama") from exc

from .ingest import CHROMA_DIR, get_or_create_collection
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .utils import Timer, get_logger

logger = get_logger(__name__)

DEFAULT_LLM_MODEL = "llama3.2:3b"
DEFAULT_TOP_K = 3
DEFAULT_OOS_THRESHOLD = 0.6

OUT_OF_SCOPE_REPLY = (
    "I'm sorry - I can only answer questions covered by the Sunrise AMC "
    "investor FAQ. Please contact support@sunriseamc.in for further help."
)


@dataclass
class RetrievedChunk:
    question_number: int
    question: str
    document: str
    distance: float

    @property
    def label(self) -> str:
        return f"FAQ Q{self.question_number}"


@dataclass
class RAGAnswer:
    query: str
    answer: str
    sources: list[str] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    out_of_scope: bool = False
    llm_model: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "out_of_scope": self.out_of_scope,
            "llm_model": self.llm_model,
            "error": self.error,
            "retrieved": [
                {
                    "question_number": r.question_number,
                    "question": r.question,
                    "distance": round(r.distance, 4),
                }
                for r in self.retrieved
            ],
        }


@dataclass
class RAGConfig:
    llm_model: str = DEFAULT_LLM_MODEL
    top_k: int = DEFAULT_TOP_K
    oos_threshold: float = DEFAULT_OOS_THRESHOLD
    persist_dir: Path = CHROMA_DIR
    temperature: float = 0.1

def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.label}] {c.document}" for c in chunks)

def retrieve(query: str, collection: Collection, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    
    if not query.strip():
        return []

    res = collection.query(query_texts=[query], n_results=top_k)
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    return [
        RetrievedChunk(
            question_number=int(meta["question_number"]),
            question=str(meta.get("question", "")),
            document=doc,
            distance=float(dist),
        )
        for doc, meta, dist in zip(docs, metas, dists)
    ]
    
def _call_ollama(system_prompt: str, user_prompt: str, model: str, temperature: float) -> str:

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": temperature},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Ollama call failed ({type(exc).__name__}): {exc}. "
            "Is `ollama serve` running and the model pulled?"
        ) from exc

    return response["message"]["content"].strip()

def answer_query(
    query: str,
    cfg: RAGConfig | None = None,
    collection: Collection | None = None,
    bench: MutableMapping[str, float] | None = None,
) -> RAGAnswer:
    
    cfg = cfg or RAGConfig()
    query = query.strip()

    if not query:
        return RAGAnswer(query="", answer="No query provided.", error="empty_query", llm_model=cfg.llm_model)

    collection = collection or get_or_create_collection(cfg.persist_dir)

    with Timer("retrieve", bench):
        retrieved = retrieve(query, collection, cfg.top_k)

    if not retrieved:
        return RAGAnswer(query=query, answer=OUT_OF_SCOPE_REPLY, out_of_scope=True, llm_model=cfg.llm_model)

    if retrieved[0].distance > cfg.oos_threshold:
        logger.info("OOS: top-1 distance %.3f > %.3f", retrieved[0].distance, cfg.oos_threshold)
        return RAGAnswer(query=query, answer=OUT_OF_SCOPE_REPLY, retrieved=retrieved, out_of_scope=True, llm_model=cfg.llm_model)

    user_prompt = USER_PROMPT_TEMPLATE.format(context=_format_context(retrieved), query=query)

    try:
        with Timer("generate", bench):
            answer_text = _call_ollama(SYSTEM_PROMPT, user_prompt, cfg.llm_model, cfg.temperature)
    except RuntimeError as exc:
        logger.error("LLM call failed: %s", exc)
        return RAGAnswer(
            query=query,
            answer="The local LLM is unavailable. Please ensure Ollama is running and try again.",
            retrieved=retrieved,
            error=str(exc),
            llm_model=cfg.llm_model,
        )

    return RAGAnswer(
        query=query,
        answer=answer_text,
        sources=[f"Q{c.question_number}" for c in retrieved],
        retrieved=retrieved,
        llm_model=cfg.llm_model,
    )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    bench: dict[str, float] = {}
    
    result = answer_query(args.query, RAGConfig(llm_model=args.model, top_k=args.top_k), bench=bench)
    out = result.to_dict()
    out["latency_sec"] = bench
    
    print(json.dumps(out, indent=2, ensure_ascii=False))
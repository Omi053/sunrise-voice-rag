import os
from pathlib import Path
import pytest

from src.ingest import build_index, parse_faq
from src.rag import RAGConfig, answer_query, retrieve

PDF_PATH = Path(__file__).resolve().parents[1] / "input" / "SunriseAMC_FAQ.pdf"
CHROMA_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma"

RETRIEVAL_CASES: list[tuple[str, int]] = [
    ("What documents do I need for KYC?", 1),
    ("How long does KYC take?", 2),
    ("Can my child invest in mutual funds?", 3),
    ("What is the minimum SIP amount?", 4),
    ("How do I pause my SIP for a couple of months?", 5),
    ("What happens if my SIP payment fails?", 6),
    ("Will I have to pay tax on my equity mutual fund gains?", 9),
    ("Is TDS deducted when I redeem my mutual funds?", 10),
]

@pytest.fixture(scope="session")
def collection():
    assert PDF_PATH.exists(), f"FAQ PDF missing at {PDF_PATH}"
    return build_index(PDF_PATH, persist_dir=CHROMA_DIR)

def test_parse_faq_finds_all_questions():
    chunks = parse_faq(PDF_PATH)
    qnums = sorted(c.question_number for c in chunks)
    assert len(chunks) >= 8
    assert len(qnums) == len(set(qnums))
    assert qnums[0] == 1

@pytest.mark.parametrize("query, expected_q", RETRIEVAL_CASES)
def test_retrieval_top1(collection, query: str, expected_q: int):
    chunks = retrieve(query, collection, top_k=3)
    assert chunks
    assert chunks[0].question_number == expected_q, (
        f"Top-1 was Q{chunks[0].question_number} for {query!r}, expected Q{expected_q}. "
        f"Top-3: {[(c.question_number, round(c.distance, 3)) for c in chunks]}"
    )

def test_oos_query(collection):
    result = answer_query(
        "What is the weather in Mumbai today?",
        cfg=RAGConfig(persist_dir=CHROMA_DIR, oos_threshold=0.6),
        collection=collection,
    )
    assert result.out_of_scope

def test_empty_query(collection):
    result = answer_query("   ", cfg=RAGConfig(persist_dir=CHROMA_DIR), collection=collection)
    assert result.error == "empty_query"

@pytest.mark.skipif(os.environ.get("RUN_LLM_TESTS") != "1", reason="Set RUN_LLM_TESTS=1")
def test_e2e_ollama(collection):
    
    result = answer_query(
        "How can I pause my SIP for a couple of months?",
        cfg=RAGConfig(persist_dir=CHROMA_DIR),
        collection=collection,
    )
    
    assert not result.out_of_scope
    assert result.error is None
    assert "Q5" in result.answer or "Q5" in result.sources
    
    for chunk in result.retrieved:
        assert chunk.document not in result.answer
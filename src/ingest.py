import argparse
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

from .utils import Timer, ensure_dir, get_logger

COLLECTION_NAME = "sunrise_faq"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = Path("data/chroma")

logger = get_logger(__name__)
QA_PATTERN = re.compile(r"Q(\d+)\.\s*(.+?)\n(.*?)(?=\nQ\d+\.|\Z)", re.DOTALL)

@dataclass(frozen=True)
class FAQChunk:
    question_number: int
    question: str
    answer: str

    @property
    def chroma_id(self):
        return f"Q{self.question_number}"

    @property
    def document(self):
        return f"Q{self.question_number}. {self.question}\nAnswer: {self.answer}"

    @property
    def metadata(self):
        return {"question_number": self.question_number, "question": self.question}


def parse_faq(pdf_path: Path) -> list[FAQChunk]:
    reader = PdfReader(str(pdf_path))
    raw = "\n".join((page.extract_text() or "") for page in reader.pages)

    chunks = []
    for match in QA_PATTERN.finditer(raw):
        question = re.sub(r"\s+", " ", match.group(2)).strip()
        answer = re.sub(r"\s+", " ", match.group(3)).strip()
        if answer:
            chunks.append(FAQChunk(int(match.group(1)), question, answer))

    if not chunks:
        raise ValueError("No Q&A pairs found in PDF.")

    return sorted(chunks, key=lambda c: c.question_number)

def build_index(
    pdf_path: Path,
    persist_dir: Path = CHROMA_DIR,
    rebuild: bool = False,
    bench: dict | None = None,
) -> chromadb.Collection:

    if bench is None:
        bench = {}

    if rebuild and persist_dir.exists():
        shutil.rmtree(persist_dir)

    ensure_dir(persist_dir)
    chunks = parse_faq(pdf_path)
    
    logger.info(f"Found {len(chunks)} from the provided pdf.")
    # logger.info(chunks[0])

    client = chromadb.PersistentClient(path=str(persist_dir))
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
        configuration={"hnsw": {"space": "cosine"}},
    )

    if collection.count() == len(chunks) and not rebuild:
        logger.info("Index already up to date (%d chunks). Skipping.", collection.count())
        bench.setdefault("ingest", 0.0)
        return collection

    with Timer("ingest", bench):
        collection.upsert(
            ids=[c.chroma_id for c in chunks],
            documents=[c.document for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    logger.info("Indexed %d chunks in %.2fs.", len(chunks), bench["ingest"])
    return collection

def get_or_create_collection(persist_dir: Path = CHROMA_DIR):

    client = chromadb.PersistentClient(path=str(persist_dir))
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
        configuration={"hnsw": {"space": "cosine"}}
    )
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--persist-dir", type=Path, default=CHROMA_DIR)
    args = parser.parse_args()

    bench: dict[str, float] = {}
    coll = build_index(args.pdf, args.persist_dir, args.rebuild, bench=bench)
    print(f"Collection '{coll.name}' has {coll.count()} chunks. Ingest: {bench.get('ingest', 0):.2f}s.")
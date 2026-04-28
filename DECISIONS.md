# DECISIONS.md

Engineering rationale for the Sunrise voice-RAG prototype. The goal of
this document is to make every non-trivial choice **defensible**: what we
picked, what we considered, what we sacrificed, and what we'd change if
this were running in production at Sunrise's actual scale (~thousands of
queries / month across KYC, SIP, redemptions, and taxation).

---

## 1 · Model selection

### 1.1 Speech-to-text: Faster-Whisper `base` (int8, CPU)

| Option            | Params | RAM (int8) | Rough WER (en) | Latency / 12s clip (M2 CPU) |
|-------------------|--------|------------|-----------------|------------------------------|
| `tiny`            |  39M   | ~75 MB     | 9–11%           | ~0.8 s                       |
| **`base`**        |  74M   | ~140 MB    | 6–8%            | **~2.1 s**                   |
| `small`           | 244M   | ~480 MB    | 4–5%            | ~5.5 s                       |
| `medium`          | 769M   | ~1.5 GB    | 3–4%            | ~14 s                        |

**Choice:** `base` with `compute_type="int8"`.

**Why:** investor queries are short, single-speaker English with mild
domain vocabulary. `base` lands well below the 10% WER threshold where
downstream RAG starts to silently miss. `small` halves WER but ~3× the
latency on CPU; not worth it for a laptop demo. `int8` quantization on
the CTranslate2 backend cuts RAM ~4× with negligible WER change — the
combination that *actually* runs on a developer laptop.

**VAD on by default.** `vad_filter=True` removes leading/trailing silence,
which prevents a well-known Whisper failure mode where the model
confidently hallucinates "thank you." or "you" on quiet audio.

### 1.2 Embedding model: `sentence-transformers/all-MiniLM-L6-v2`

| Option              | Dim | Size  | MTEB retrieval avg | CPU query latency |
|---------------------|-----|-------|---------------------|-------------------|
| **MiniLM-L6-v2**    | 384 | ~80 MB| 0.41                | <1 ms             |
| `bge-small-en-v1.5` | 384 | ~130 MB| 0.51               | ~2 ms             |
| `bge-base-en-v1.5`  | 768 | ~440 MB| 0.53               | ~6 ms             |

**Choice:** MiniLM-L6-v2.

**Why:** the corpus has **10 chunks**. Even a mediocre embedder hits
ceiling accuracy because the inter-topic distance (KYC vs SIP vs
taxation) dominates intra-topic noise. We verified empirically — MiniLM
yields top-1 = correct on every test query in `tests/test_eval.py`.
Spending 5× more RAM on `bge-base` would change zero outcomes here. If
the FAQ grew to ~10k items we'd reconsider.

### 1.3 LLM: Llama 3.2 3B (Q4_K_M via Ollama)

| Option                      | Params | RAM (Q4) | Quality on grounded QA | Tok/s (M2 CPU) |
|-----------------------------|--------|----------|--------------------------|------------------|
| `phi3:mini`                 | 3.8B   | ~2.4 GB  | competitive but more refusals | ~22         |
| **`llama3.2:3b`**           | 3B     | ~2.0 GB  | very good for short, grounded answers | ~28 |
| `mistral:7b-instruct-q4_K_M`| 7B     | ~4.5 GB  | marginally better wording | ~12          |
| `llama3.1:8b`               | 8B     | ~5.0 GB  | best, but heavy           | ~10          |

**Choice:** `llama3.2:3b` (Q4_K_M).

**Why:** the task is *extractive synthesis* over retrieved context, not
open-ended reasoning. 3B-class instruct models follow grounding
instructions reliably and produce concise paraphrases. The 7B/8B options
roughly *double* perceived latency for marginal answer-quality gains —
a poor trade-off when the user is waiting on a voice response. We also
need the pipeline to fit alongside Whisper on an 8GB laptop without
swapping.

**Fallback documented:** if a candidate's hardware can't run Ollama,
the assignment permits Groq's free tier with `llama-3.1-8b-instant` /
`mixtral-8x7b`. Drop-in via a single `ollama.chat → groq.chat` swap.
We did NOT need this for our run.

### 1.4 Retrieval `k` and OOS threshold

* **k = 3.** With only 10 chunks, k=1 is too brittle (a single bad
  retrieval kills the answer); k=5 starts to admit unrelated topics into
  the LLM context. k=3 reliably contains the correct chunk plus 1–2
  loosely related ones, which the prompt explicitly tells the model to
  ignore if not on point.
* **OOS distance > 0.6** (cosine). MiniLM's on-topic distances cluster at
  0.25–0.45 on this corpus; clearly off-topic queries land in 0.7–0.9.
  Choosing 0.6 rejects the noise band without losing valid paraphrases.
  This is configurable via `RAGConfig.oos_threshold`.

---

## 2 · Chunking strategy: one chunk per Q&A pair

The FAQ is authored as numbered, self-contained Q&A pairs:
`Q1.`, `Q2.`, ... Each answer is a complete thought; topics are mutually
exclusive (KYC ≠ SIP ≠ taxation).

| Strategy                 | Pros                              | Cons                                  |
|--------------------------|-----------------------------------|---------------------------------------|
| 500-char window, 50 overlap | Simple, generic                | Splits answers; mashes topics; kills citation; **assignment red flag** |
| Section-level (4 chunks) | Big contextual payloads           | Coarse — retrieval brings in irrelevant Q&As; longer LLM prompts |
| **One chunk per Q&A (10 chunks)** | Aligns to authored boundaries; precise citation; small prompts | Implementation needs a regex |
| Recursive token splitter | Adaptive to document structure    | Overkill for a 2-page, regex-friendly FAQ |

**Choice:** Q&A-pair chunking via `re.compile(r"Q(\d+)\.\s*(.+?)\n(.*?)(?=\nQ\d+\.|\Z)", DOTALL)`.

**Critical structural benefit:** the question number is stored as
ChromaDB metadata. The prompt template then injects it *as a label*
(`[FAQ Q5] …`) and the system prompt instructs the LLM to cite using
those labels verbatim. **Citation is a first-class metadata operation,
not regex post-processing of the model output.**

**Tradeoff accepted:** if the FAQ grows in size or starts containing
multi-paragraph answers with sub-topics, single-Q-per-chunk could become
too coarse. Mitigation noted in §4 below.

---

## 3 · Tradeoffs we made for simplicity / speed

| Sacrifice                                  | Why we accepted it                                              |
|--------------------------------------------|------------------------------------------------------------------|
| No reranker (e.g. cross-encoder)           | Useless on 10 chunks; pure latency cost                          |
| No streaming output                        | Adds protocol complexity; total latency is already < 8s          |
| English only                               | Whisper auto-detects language but the prompt and FAQ are English; multilingual would need LLM tuning |
| No chat-history / multi-turn               | Out of scope — assignment is single voice query → answer         |
| No HTTP server / Docker                    | Single-command CLI is what the rubric asks for                   |
| Synchronous Ollama calls                   | One concurrent user; async would just be ceremony                |
| `ffmpeg` system dependency                 | Standard for any Whisper-based stack; documented in README       |
| Local-only; no auth, no PII redaction       | Demo-grade; production needs both (see §4)                       |
| No reranker / answer-grounding verifier    | We rely on system-prompt discipline + temperature 0.1            |
| Eval is 8 hand-curated cases                | Sufficient to surface obvious regressions; not a benchmark suite |

---

## 4 · Production readiness: what would NOT scale

This section names specific failure modes if Sunrise dropped this code
into a production support channel. Each item is something we *would*
fix before launch.

### 4.1 Vector store

* `chromadb.PersistentClient` writes to local SQLite + parquet. **Breaks**
  with concurrent writers (locking) and at ~1M chunks (full-table scans).
* **Migration path:** Qdrant or Weaviate (HTTP, server-mode). The
  abstraction is already small enough — `ingest.py` knows about a
  `Collection` with `upsert` and `query`. ~80 LOC swap.

### 4.2 ASR

* Whisper-base struggles on heavy regional accents and mobile audio
  (codec artefacts). Production should:
  * Auto-route to `large-v3` on a T4/A10G when avg_confidence < 0.7.
  * Detect language explicitly; current `auto` defaults to English on
    short clips containing Hindi/Marathi proper nouns.
  * Add a **denoiser** (e.g. RNNoise) before Whisper for call-centre audio.
  * Add a **diarisation** stage if the channel ever carries 2+ speakers.

### 4.3 RAG quality / hallucination

* Single system prompt + temperature=0.1 gets us most of the way, but
  it's not a guarantee. Production needs:
  * **Answer-grounding verifier**: a second small-model pass that checks
    every claim in the answer is entailed by the retrieved chunks
    (NLI-style, e.g. `bart-large-mnli`). Reject and regenerate on fail.
  * **Citation enforcement check**: parse the `[FAQ Qn]` labels out of
    the answer and assert ≥ 1 valid match. We do this softly today.
  * **Refusal preference**: harden the prompt with few-shot OOS
    examples; current single-paragraph instruction is enough for an
    eval, not for adversarial users.

### 4.4 LLM serving

* Ollama on a laptop is single-tenant, blocking, no batching. Real
  deployments need vLLM or TGI for tensor-parallel batching, plus an
  autoscaler. Latency budget at 100 RPS would dominate.
* Cold start: Ollama loads model from disk on first request, ~3-6s.
  Production should keep the model warm and add a /healthz that
  pre-loads it.

### 4.5 Observability

* We log per-stage timings to stdout. Production needs:
  * OpenTelemetry traces with stage spans.
  * Per-query records: query, retrieved Q#s, distance scores, LLM
    response, latency, OOS flag, low-confidence flag. Goes to a
    column-store like ClickHouse.
  * **Online eval**: sample 1% of live traffic into a human-review
    queue with the retrieved chunks shown — measures grounding drift
    when the FAQ changes.

### 4.6 Security & compliance

* No auth on the CLI; voice samples may contain PII (PAN, Aadhaar
  numbers, account IDs).
* Production must:
  * Encrypt audio at rest and in transit; auto-redact PII from logs
    via regex pre-processors before ASR output is logged.
  * Apply SEBI-mandated retention rules to transcripts.
  * Add explicit consent banner before recording.

### 4.7 Cost & throughput envelope

| Metric                | Today (laptop) | Naïve scale to 10k queries / day |
|-----------------------|----------------|------------------------------------|
| ASR cost per query    | 0 (local)      | 0 (local on CPU box) → ~$0.0003 GPU-h  |
| LLM tok/s             | 28             | needs ≥1 vLLM replica on T4        |
| End-to-end p95        | ~7 s           | target 3 s with `whisper-small` GPU + vLLM |

### 4.8 What's *missing* from this prototype that I'd add Day 1 in production

1. Streaming ASR (`partial_results=True` UX) so the investor sees text
   the moment they finish speaking.
2. Continuous eval set checked into the repo and run on every PR.
3. A “fallback to human” escalation path when `out_of_scope=True` or
   `low_confidence` — a clean handoff into the existing support queue.
4. Versioned PDF — the rubric pins to v2.1; production needs the
   ingest job to record the FAQ version in chunk metadata so we can
   reproduce historical answers.
5. Cache embeddings per PDF SHA256 so re-ingestion is free.

---

## 5 · One-line summary

> *Q&A-level chunking + structural citation + a small but well-grounded
> local LLM is the right shape for a 10-question FAQ. Everything we
> didn't build was deliberately out of scope, and the section above
> lists exactly what production would need to add and where it would
> first break.*

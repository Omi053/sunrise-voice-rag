Decisions
Pipeline Overview\
The pipeline has 3 divisions - Ingestion, Transcription, and RAG.

Ingestion & Chunking Strategy\
Ingestion stores given data into a vector database, grouping similar vectors using ChromaDB. The core challenge here is how to chunk the PDF.
Brute force chunking was ruled out (chunks won't be effective enough for the assistant)).\
Q&A-based chunking was chosen instead. One chunk per Q/A pair (10 chunks for 10 questions) is ideal because it gives precise citation and smaller prompts. All other strategies would either break citation or retrieve irrelevant Q&A pairs, leading to longer, noisier LLM prompts.
\
A regex pattern is used to split the FAQ : simple and effective. The pattern detects where a question starts and ends, with the answer following directly below.\

Tradeoffs to be aware of:

If the FAQ grows significantly or contains multi-paragraph answers, single Q-chunks will become too coarse
We assume the FAQ and investor audio are both in English - otherwise, LLM tuning would be needed\


Vector Database
ChromaDB : simple, locally persisted, used for POC purposes. Chroma injects metadata as a label alongside each chunk.
\
Embedding Model
all-MiniLM-L6-v2
Chosen because it is efficient, fast, and balanced in performance. It maps sentences and paragraphs into dense vectors, making it ideal for fast semantic search and sentence similarity. Works well in a resource-constrained environment.

Model Selection : Transcription & LLM
Transcription: Faster Whisper (base)
All investor queries are expected to be short, general questions, so the base model is sufficient. BVase lands well, and since latency is a concern, a smaller model is preferred.
LLM: LLaMA 3.2:3b via Ollama

LLaMA is well-suited for short, grounded answers where the response needs to be extracted from a fixed source (the PDF), rather than open-ended reasoning
Mistral and LLaMA 3.1 are heavier (> 3 GB), which makes them less practical here
The assignment permits use of Groq with Ollama, which cannot be run in this setup


Production Additions
If this were to be extended for production, two additions stand out:

Streaming ASR : the investor can see the transcript the moment they finish speaking (partial results)
Automatic handoff : if a query is out of support scope or confidence is low, it routes to a support queue automatically


One-Line Summary
Q&A-level chunking + structural citation. A local LLM is the right fit for a 10-question FAQ. The sections above cover what can be added and where it can break.

Evaluation Design
The tests are split into two parts so failures are easier to trace.
Part 1- Does retrieval work?
We build the Chroma index once and then throw 8 different queries at it. Each query asks the same thing as a FAQ question but in different words — the way an investor would actually phrase it. The test passes if the right FAQ chunk comes back as the top result. This is the most important thing to get right in a RAG system.
Part 2- Do the failure cases behave correctly?
Two quick checks, no LLM involved:

Ask something completely off-topic: the system should reject it
Send an empty input: it should return the right error

There's also an end-to-end test that actually calls Ollama, but it only runs when you explicitly set RUN_LLM_TESTS=1, so the default test suite stays fast. That test checks that the answer references Q5 and that the model didn't just paste the FAQ chunk back word-for-word.
Why keep them separate? If retrieval fails, the issue is in chunking or embeddings. If retrieval works but the answer is bad, the issue is in the prompt or the model. Combining both into one test just makes it harder to figure out what broke.

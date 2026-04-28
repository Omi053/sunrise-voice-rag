Pipeline has 3 divisions- Ingestion, Transcription and Rag
Ingestion- Storing given data into a vector, which groups similar vectors- done using Chroma DB. 
We need to chunk that.


Brute Force Chunking- Chunks won't be effective enough for the assistant

Q and A-based sample set- Can store Q&A separately. For example- 10 Chunks for 10 questions
10 Chunks (one chunk per Q/a) is ideal as it has precise citation, smaller prompts, although its implementation needs a regex.
All other strategies would either kill citation or would bring irrelevant Q&a when retrieved + Long LLM Prompts

Regex used. 
WHY? Simple effective. Pattern deducted- Q start? end. 
Answer below THAT QUESTION

vector database: chroma db (simple, locally persisted, for poc purposes)
Chroma injects as a label. 

The tradeoff is that if the FAQ increases in size or contains multi-paragraph answers, these single Q- chunks will become too coarse

We also assume FAQS AND investor audio is English; otherwise, LLM TUNING WOULD BE NEEDED

embedding model: all-MiniLM-L6-v2 
Why?
The model is efficient, fast, and balanced in performance. 
Can map sentences/ paragraphs into dense vectors.
Ideal for fast semantic searches and sentence similarity. 
Really effective in a resource-constrained environment


A faster whisper base is used as all investor queries are short, general questions. BVase lands well, and since we have a latency issue, small is not optimal
A 
llama is very good for short, grounded answers. Mistral and Illama 3.1 are heavier in size (> 3 GB)
Also, ILLAMA 3.2 is ideal for situations when we have to extract the answers from someplace else (a PDF in this case). The question does not have open-ended reasoning

Assignment permits the use of GROQ with OLLAMA, which cannot be run. 


/What additional steps can be added for production- 
1) The investor can see the text the moment they finish speaking (partial results)- streaming ASR
2) if out of support or low confidence becomes true, automatic handoff to support queues


One-line summaries- 
Q&A Level chunking + structural citation- Local LLm is right for a 10-question FAQ
The section above states what can be added and where it can break.


Eval HAS two distinct layers-
Retrieval layer (offline, no LLM needed)
Builds the Chroma index once via the session fixture, then runs 8 parametrized queries against it. Each query is a natural language paraphrase of a FAQ question - deliberately worded differently from the source text - and asserts that the correct Q number comes back as top-1. This tests whether the embedding model (all-MiniLM-L6-v2) can semantically match investor intent to the right FAQ chunk, which is the core of the RAG system.

Behaviour layer
Two tests check the failure branches without touching the LLM: OOS gate rejects an off-topic query, and empty input returns the right error code. The test_e2e_ollama test only runs when RUN_LLM_TESTS=1 is set, keeping the default suite fast and Ollama-free. It checks that the answer cites Q5 and that the LLM didn't just copy the FAQ chunk verbatim (the red flag the rubric calls out explicitly).

The key design decision is to test retrieval and generation separately. If retrieval is wrong, you know the problem is in chunking or embeddings. If retrieval is right but the answer is bad, the problem is in the prompt or the model. Mixing them into one test makes failures harder to diagnose.

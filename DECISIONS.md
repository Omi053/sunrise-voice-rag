Pipeline has 3 divisions- Ingestion, Transcription and Rag
Ingestion- Storing given data into a vector, which groups similar vectors- done using Chroma DB. 
We need to chunk that.


Brute Force Chunking- Chunks won't be effective enough for the assistant

Q and A-based sample set- Can store Q&A separately. For example- 10 Chunks for 10 questions
Regex used. 
WHY? Simple effective. Pattern deducted- Q start ? end. 
Answer below THAT QUESTION

vector database: chroma db (simple, locally persisted, for poc purposes)
embedding model: all-MiniLM-L6-v2 
Why?
The model is efficient, fast, and balanced in performance. 
Can map sentences/ paragraphs into dense vectors.
Ideal for fast semantic searches and sentence similarity. 
Really effective in a resource-constrained environment


Faster whisper base is used as all investor queries are short general questions. BVase lands well and since we have a latency issues, samall is not optimal

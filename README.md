
🌅 Sunrise Project

Voice-powered investor Q&A — transcribes audio, retrieves answers from a PDF knowledge base, and returns structured JSON summaries via a local RAG pipeline.
Repository Layout-

Taking in the inputs- 
1) investor_sample.mp3
2) SunriseAMC_FAQ.pdf

Data, Output 
Source- Transcibe, Ingest , Rag, Pipeline , Utils
Transcribe- Faster Whisper
Ingest- PDF Broken down for Q&A chunks- ChromaDB
Rag- Retrieve+ Ollama


Installed FFMEG

Python 3.9
Setups done- 
cd sunrise-voice-rag

ls input/
# → investor_sample.mp3   SunriseAMC_FAQ.pdf

python -m venv .venv
source .venv/bin/activate       

pip install -r requirements.txt

ollama serve &                   
ollama pull llama3.2:3b 

oNCE MAIN CODE is ran-0 
command builds vector index- transcribes from voice to write output transcript
Sends transcript to RAG engine to write output (JSON)
Prints a summary (includes latency)

Failure Cases
A few things to watch out for:

Missing or silent audio file — pipeline will exit early
Query is out of scope — Ollama will say so rather than hallucinate
Ollama isn't running — you'll get a connection error, just run ollama serve
FAQ PDF in an unexpected format — ingest will skip bad sections and warn you

Notes
See Decisions.md for model choices and tradeoffs. The ChromaDB index is saved locally so subsequent runs are faster — delete chroma_store/ if you want to rebuild it from scratch.

Also see- Decisions.d for model selection, strategy, tradeoff. scalling issues


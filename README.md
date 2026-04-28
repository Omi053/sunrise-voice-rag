## 🌅 Sunrise Project

Voice-powered investor Q&A : transcribes audio, retrieves answers from a PDF knowledge base, and returns structured JSON summaries via a local RAG pipeline.

Codebase structure :

Inputs in `input` folder :

1. investor_sample.mp3
2. SunriseAMC_FAQ.pdf

Other folders :

Data, Output
Source - Transcibe, Ingest, Rag, Pipeline, Utils
Transcribe - Faster Whisper
Ingest - PDF Broken down for Q&A chunks - ChromaDB
Rag - Retrieve + ollama based answer generation

### Prequesities

Installed FFMEG
Python 3.9
ollama

### Setup

After cloning the repo please follow the instructions below :

```
cd sunrise-voice-rag
ls input/
 : investor_sample.mp3   SunriseAMC_FAQ.pdf
```

Create python environment either using venv or conda :

```
python -m venv .venv
source .venv/bin/activate
```

Install the requirements :

```
pip install -r requirements.txt
```

Make sure ollama is installed such as in ubuntu using :

```
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
ollama serve
ollama pull llama3.2:3b
```

The main entry point is `run.py` just use :

```
python run.py
```

This command builds vector index, transcribes from voice to write output transcript. Also, sends transcript to RAG engine to write output (JSON) and finally prints a summary (includes latency).

### Failure Cases

A few things to watch out for while running this code :

- Missing or silent audio file : pipeline will exit early
- Query is out of scope : `Ollama` will flag this as outu of scope, rather than hallucinate
- Ollama isn't running : ollama needs to be setup otherwise we'll get a connection error
- FAQ PDF in an unexpected format, ingest will skip bad sections and warn you

### Notes

See Decisions.md for model choices, tradeoffs and evaluation strategy. The ChromaDB index is saved locally so subsequent runs are faster : delete data/chroma if you want to rebuild it from scratch.

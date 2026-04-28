# Sunriseproject

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
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install -r requirements.txt

ollama serve &                    # runs the local model server
ollama pull llama3.2:3b 

oNCE MAIN CODE is ran-0 
command builds vector index- transcribes from voice to write output transcript
Sends transcript to RAG engine to write output (JSON)
Prints a summary (includes latency)

Failure cases would be-
File is missing, empty audio, File is pure silent, Whispering, query is out of scope
Ollama may not run, FAQ is in unexpected format


Also see- Decisions.d for model selection, strategy, tradeoff. scalling issues

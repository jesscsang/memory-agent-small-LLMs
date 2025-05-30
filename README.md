# Simplified Implementation of Agentic RAG-based Memory System

The self implemented agentic RAG-based memory system runs locally using any LLM hosted by Ollama.

## Quickstart

Following quick start to install the dependencies required for both the self implemented agentic RAG-based memory system and the jupyter notebook files.

### Setup Ollama

1. Download Ollama from https://ollama.com/download
2. Download the `llama3.2:3b-instruct-fp16` LLM and the `nomic-embed-text` embedding model

```bash
ollama pull llama3.2:3b-instruct-fp16
ollama pull nomic-embed-text
```

## Setup `.env` file

1. Create a `.env` file

```bash
cp .env.example .env
```

2. Create an account on LangSmith from https://www.langchain.com/langsmith.
3. Obtain a LangSmith API key then setup the key in the `.env` file.

```bash
LANGSMITH_API_KEY="lsv2_pt_XXX"
```

### Setup Python

1. Download `Python 3.11` from the official distribution or create virtual environment with `Python 3.11` using conda package manager (anaconda distribution)
   - **Official Distribution**: https://www.python.org/downloads/
   - **Anaconda Distribution**: https://www.anaconda.com/download/
2. (Recommended) Create a virtual environment

```bash
# Built-in venv maanger
python -m venv myenv
myenv\Scripts\activate

# Conda package manager
conda create -n myenv python=3.11
conda activate myenv
```

3. Install dependencies

```bash
pip install -r requirements.txt
pip install -U "langgraph-cli[inmem]"
```

4. Launch Langgraph server

```bash
langgraph dev
```

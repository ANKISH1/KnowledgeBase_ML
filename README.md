# KnowledgeBase — Chat With Your Documents

A personal AI-powered knowledge base that lets you upload any PDF or TXT file and ask questions about it in plain English. No more manually searching through documents.

## What It Does

- Upload any `.pdf` or `.txt` file
- Ask questions in natural language
- Get accurate answers from your own documents
- Remembers conversation history for follow-up questions
- Refuses to answer from outside your documents

## How It Works

```
Your document → chunked into pieces → stored in vector database
User asks question → finds relevant chunks → LLM answers using those chunks
```

This is called RAG (Retrieval Augmented Generation).

## Tech Stack

- **Python** — core language
- **Groq API** — LLM (LLaMA 3.3)
- **ChromaDB** — vector database for storing document embeddings
- **pypdf** — PDF text extraction
- **python-dotenv** — environment variable management

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/ANKISH1/KnowledgeBase_ML.git
cd KnowledgeBase_ML
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com)

## Usage

### Step 1 — Add documents to your knowledge base

```bash
python ingest.py
```

Enter the path to your PDF or TXT file when prompted.

### Step 2 — Chat with your documents

```bash
python chat.py
```

Ask anything about your uploaded documents. Type `quit` to exit.

## Example

```
Your question: What does the document say about sorting algorithms?
AI: Based on your notes, merge sort uses a divide and conquer approach...

Your question: What about its time complexity?
AI: Merge sort has a time complexity of O(n log n) in all cases...
```

## Project Structure

```
KnowledgeBase/
├── setup.py          # Chroma and Groq client initialization
├── ingest.py         # Read, chunk, and store documents
├── chat.py           # RAG-powered chat interface
├── requirements.txt  # Dependencies
└── .env              # API keys (not committed)
```

## Features

- Persistent storage — data survives between sessions
- Conversation memory — follow-up questions work naturally
- Strict context — only answers from your documents
- Supports PDF and TXT files
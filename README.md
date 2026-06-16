# AI RAG Service

A local Retrieval-Augmented Generation service for answering questions over PDF documents using:

* FastAPI
* Qdrant
* PyMuPDF
* Sentence Transformers
* BGE-M3 embeddings
* Ollama
* Local LLMs such as Qwen3

## Prerequisites

Install the following software:

* Python 3.12
* Docker Desktop or Rancher Desktop
* Ollama
* Git

Recommended Python version:

```bash
python --version
# Python 3.12.x
```

Do not use Python 3.14 for now, because some AI/ML dependencies may not support it reliably yet.

## Required Ollama Model

Pull the default LLM:

```bash
ollama pull qwen3:14b
```

Optional models:

```bash
ollama pull gemma3:12b-it-qat
ollama pull deepseek-r1:14b
```

Make sure Ollama is running before starting the service.

## Project Structure

```text
ai-rag-service/
  app/
    main.py
    config.py
    book_scanner.py
    pdf_loader.py
    chunking.py
    embeddings.py
    vector_store.py
    llm.py
    rag.py
  docker-compose.yml
  requirements.txt
```

## Setup

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### CUDA PyTorch for NVIDIA GPUs

For Windows with an NVIDIA GPU, install the CUDA-enabled PyTorch build:

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify PyTorch can see CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Embedding device and batch size can be configured with environment variables:

```powershell
$env:EMBEDDING_DEVICE="auto"
$env:EMBEDDING_DEVICE="cuda"
$env:EMBEDDING_BATCH_SIZE="64"
```

`EMBEDDING_DEVICE=auto` uses CUDA when PyTorch reports CUDA is available; otherwise it falls back to CPU.

## Start Qdrant

Start the vector database:

```bash
docker compose up -d
```

Check whether Qdrant is reachable:

```bash
curl http://localhost:6333
```

If Docker fails on Windows, make sure Docker Desktop or Rancher Desktop is running.

## Book Repository

PDF files are stored outside the project. By default the service reads from:

```text
C:\Users\alpha\OneDrive\Dokumente\Books
```

Expected structure:

```text
Books/
  default/
    de/
      <topic-subfolders>/
        *.pdf
    en/
      <topic-subfolders>/
        *.pdf
    fr/
      <topic-subfolders>/
        *.pdf

  sensitive/
    de/
      <topic-subfolders>/
        *.pdf
    en/
      <topic-subfolders>/
        *.pdf
    fr/
      <topic-subfolders>/
        *.pdf
```

The service recursively scans all PDF files below the supported repository/language folders.
Files outside `default` or `sensitive`, or outside `de`, `en`, or `fr`, are ignored.

You can override the location with the `BOOKS_ROOT` environment variable.

Windows PowerShell:

```powershell
$env:BOOKS_ROOT="D:\Books"
```

Linux / macOS:

```bash
export BOOKS_ROOT="/mnt/books"
```

## Start the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "qdrant": "ok",
  "collection": "books",
  "points_count": 12345,
  "embedding_model": "BAAI/bge-m3",
  "embedding_device_config": "auto",
  "embedding_device_selected": "cuda",
  "cuda_available": true,
  "cuda_version": "12.8",
  "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
  "default_llm": "qwen3:14b"
}
```

## Debugging Ingestion

Use these commands when `/ingest/all` appears slow or Qdrant still shows `points_count = 0`:

```powershell
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/debug/embedding
curl.exe http://localhost:8000/debug/config
curl.exe http://localhost:8000/debug/routes
curl.exe http://localhost:8000/books/hashes
curl.exe -X POST "http://localhost:8000/ingest/all?dry_run=true&limit=3"
curl.exe -X POST "http://localhost:8000/ingest/all?limit=1&sort_by=size&max_mb=20"
curl.exe -X POST "http://localhost:8000/ingest/test-one?repository=default&language=de"
curl.exe http://localhost:8000/books/no-text
```

Useful ingestion query parameters:

```text
limit=1
sort_by=path|size
max_mb=20
repository=default
language=de
dry_run=true
force_reindex=true
```

`force_reindex=true` disables the complete-document hash skip. It does not delete old chunks for changed files; it only writes the current document chunks.

List books and SHA-256 hashes:

```powershell
curl.exe http://localhost:8000/books/hashes
curl.exe "http://localhost:8000/books/hashes?repository=default&language=de&sort_by=path"
```

Response shape:

```json
{
  "status": "ok",
  "books_root": "C:\\Users\\alpha\\OneDrive\\Dokumente\\Books",
  "selected": 1,
  "books": [
    {
      "file": "default/de/philosophy/example.pdf",
      "repository": "default",
      "language": "de",
      "document_hash": "...",
      "topic_path": "philosophy",
      "file_name": "example.pdf"
    }
  ]
}
```

List PDFs that produced no extractable text during ingestion:

```powershell
curl.exe http://localhost:8000/books/no-text
curl.exe "http://localhost:8000/books/no-text?repository=default&language=de"
```

Remove a book from Qdrant by relative path or document hash:

```powershell
curl.exe -X POST http://localhost:8000/deindex `
  -H "Content-Type: application/json" `
  -d "{\"relative_path\":\"default/de/philosophy/example.pdf\"}"
```

```powershell
curl.exe -X POST http://localhost:8000/deindex `
  -H "Content-Type: application/json" `
  -d "{\"document_hash\":\"...\"}"
```

Embedding diagnostics:

```bash
curl http://localhost:8000/debug/embedding
```

Response shape:

```json
{
  "torch_version": "2.7.0+cu128",
  "cuda_available": true,
  "torch_cuda_version": "12.8",
  "device_count": 1,
  "devices": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 5070 Ti"
    }
  ],
  "selected_device": "cuda"
}
```

## Scan PDFs

Scan the external repository without indexing:

```bash
curl http://localhost:8000/books/scan
```

Response shape:

```json
{
  "status": "ok",
  "books_root": "C:\\Users\\alpha\\OneDrive\\Dokumente\\Books",
  "count": 1,
  "books": [
    {
      "absolute_path": "C:\\Users\\alpha\\OneDrive\\Dokumente\\Books\\default\\de\\philosophy\\example.pdf",
      "relative_path": "default/de/philosophy/example.pdf",
      "repository": "default",
      "language": "de",
      "topic_path": "philosophy",
      "file_name": "example.pdf",
      "sha256": "..."
    }
  ]
}
```

## Ingest a PDF

Single-file ingestion is still available. The file must live under `BOOKS_ROOT` in the expected repository/language layout.

Example using Windows PowerShell:

```powershell
curl -X POST http://localhost:8000/ingest `
  -H "Content-Type: application/json" `
  -d "{\"file_name\":\"de/philosophy/example.pdf\",\"repository\":\"default\"}"
```

Example using Bash:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_name":"de/philosophy/example.pdf","repository":"default"}'
```

## Ingest All New PDFs

Batch ingestion scans the full `BOOKS_ROOT`, indexes new PDFs, and skips files whose SHA-256 hash already exists in Qdrant.

```bash
curl -X POST http://localhost:8000/ingest/all
```

Response shape:

```json
{
  "status": "ok",
  "books_root": "C:\\Users\\alpha\\OneDrive\\Dokumente\\Books",
  "indexed": 3,
  "skipped": 12,
  "no_text": 0,
  "failed": 1,
  "details": [
    {
      "file": "default/de/philosophy/example.pdf",
      "repository": "default",
      "language": "de",
      "document_hash": "...",
      "status": "indexed",
      "chunks": 123
    }
  ]
}
```

## Ask a Question

Backward-compatible request using one repository:

Windows PowerShell:

```powershell
curl -X POST http://localhost:8000/chat `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"Was sagt der Autor über Moral?\",\"repository\":\"default\",\"model\":\"qwen3:14b\"}"
```

Bash:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Was sagt der Autor über Moral?","repository":"default","model":"qwen3:14b"}'
```

Multi-repository and language-filtered request:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Was sagt der Autor über Moral?","repositories":["default"],"languages":["de","en"],"model":"qwen3:14b"}'
```

Research mode can search both repositories:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Compare the views on ethics.","repositories":["default","sensitive"],"model":"qwen3:14b"}'
```

## Current Features

* PDF text extraction
* Text chunking
* BGE-M3 embeddings
* Qdrant vector storage
* Semantic search
* Ollama-based LLM response generation
* Source-based answers with retrieved text passages
* External book repository scanning
* SHA-256 document identity and duplicate skipping
* Repository and language retrieval filters

## Planned Features

* Role-based repository access
* Spring Boot / Vaadin frontend integration
* Model selection
* Chat history
* Better source citation rendering
* Reranking with `bge-reranker-v2-m3`
* Admin endpoint for reindexing documents

## Notes

The current version is an MVP. It is intended to prove the core RAG workflow:

```text
PDF → Chunks → Embeddings → Qdrant → Retrieval → LLM Answer
```

Security, authentication, repository permissions and public deployment should be added before exposing the service outside the local network.

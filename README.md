# AnyRAG – Retrieval-Augmented Generation with Local LLMs

**A document-grounded question-answering system integrating semantic search, vector databases, neural reranking, and locally hosted Large Language Models.**

## Overview

AnyRAG is a Retrieval-Augmented Generation (RAG) pipeline designed to extract, process, index, and retrieve information from PDF documents to generate contextually grounded answers using a locally hosted Large Language Model.

The system combines document preprocessing, sentence-aware text chunking, dense vector embeddings, FAISS similarity search, and cross-encoder reranking to retrieve relevant information from large documents.

Retrieved context is passed to a locally executed Qwen 2.5 3B Instruct model, which generates answers with source-level PDF page references.

The implementation is designed to run in Google Colab with GPU acceleration, demonstrating an end-to-end approach to building document intelligence systems without relying on external LLM inference APIs.

## Key Features

### 1. PDF Document Processing
- Extract text from PDF documents using PyMuPDF.
- Process documents page by page.
- Remove unnecessary headers, footers, and control characters.
- Detect and skip table-of-contents pages, index pages, and low-quality content.
- Preserve document metadata and original PDF page numbers.

### 2. Intelligent Text Chunking
- Sentence-aware text segmentation.
- Configurable chunk size and overlap.
- Word-boundary splitting for oversized sentences.
- Preserve document identifiers and page-level metadata.
- Generate uniquely identifiable text chunks.

### 3. Semantic Embedding Generation
- Generate dense vector representations using Sentence Transformers.
- Utilize `all-MiniLM-L6-v2` for semantic embeddings.
- Normalize embedding vectors.
- Batch-based embedding generation for efficient processing.

### 4. Vector Database and Similarity Search
- FAISS-based vector indexing.
- Implement cosine similarity search through normalized embeddings and inner-product indexing.
- Retrieve the top 20 candidate chunks.
- Cache vector indexes and metadata locally.
- Reuse previously generated indexes to reduce redundant processing.

### 5. Neural Reranking
- Integrate `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Evaluate query-document relevance.
- Reorder retrieved candidates based on reranking scores.
- Select the top 5 relevant context chunks for answer generation.

### 6. Local LLM Integration
- Integrate Qwen 2.5 3B Instruct.
- Load the model locally using Hugging Face Transformers.
- Use FP16 precision for GPU inference.
- Execute generation through PyTorch.
- Avoid external LLM inference APIs.

### 7. Context-Grounded Answer Generation
- Construct structured prompts using retrieved document excerpts.
- Generate answers exclusively from retrieved context.
- Include source references and original PDF page numbers.
- Provide an insufficient-context fallback.
- Reduce unsupported or hallucinated responses through explicit prompt constraints.

### 8. Efficient Processing
- Persistent document and vector index storage.
- Cached embedding and FAISS index reuse.
- Configurable chunking and retrieval parameters.
- GPU-accelerated model inference.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| PDF Processing | PyMuPDF |
| Text Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Similarity Search | Cosine Similarity |
| Reranking | Cross-Encoder |
| Reranking Model | ms-marco-MiniLM-L-6-v2 |
| Large Language Model | Qwen 2.5 3B Instruct |
| Deep Learning | PyTorch |
| Model Framework | Hugging Face Transformers |
| Model Precision | FP16 |
| Development Environment | Google Colab |
| GPU Acceleration | CUDA |

## System Architecture

```text
              PDF DOCUMENT
                    |
                    v
           DOCUMENT EXTRACTION
                PyMuPDF
                    |
                    v
            TEXT PREPROCESSING
                    |
                    v
           SENTENCE-AWARE CHUNKING
                    |
                    v
          METADATA PRESERVATION
                    |
                    v
          EMBEDDING GENERATION
          all-MiniLM-L6-v2
                    |
                    v
              FAISS INDEX
                    |
                    |
USER QUERY ---------+
    |
    v
QUERY EMBEDDING
    |
    v
DENSE RETRIEVAL (TOP 20)
    |
    v
CROSS-ENCODER RERANKING
    |
    v
TOP 5 RELEVANT CHUNKS
    |
    v
CONTEXT CONSTRUCTION
    |
    v
LOCAL LLM INFERENCE
    Qwen 2.5 3B
    |
    v
GROUNDED ANSWER GENERATION
    |
    v
ANSWER WITH PDF PAGE REFERENCES
```

## Retrieval and Generation Workflow

The pipeline follows a two-stage retrieval architecture combined with local language model inference.

**Stage 1: Document Ingestion**

1. Upload a PDF document.
2. Extract text from individual pages.
3. Clean extracted text and filter irrelevant pages.
4. Divide document content into sentence-aware chunks.
5. Preserve source document and PDF page metadata.

**Stage 2: Indexing**

1. Generate embeddings for each text chunk.
2. Normalize embedding vectors.
3. Create a FAISS `IndexFlatIP` index.
4. Store vector indexes and chunk metadata.
5. Reuse cached indexes when available.

**Stage 3: Retrieval**

1. Accept a natural language query.
2. Generate its semantic embedding.
3. Retrieve the top 20 candidate chunks using FAISS.
4. Rerank candidates using a cross-encoder.
5. Select the top 5 relevant chunks.

**Stage 4: Answer Generation**

1. Construct a structured prompt containing retrieved excerpts.
2. Include source identifiers and PDF page numbers.
3. Pass the context to the local Qwen model.
4. Generate an answer constrained to the retrieved content.
5. Return the answer alongside source references.

## Model Configuration

| Component | Model / Configuration |
|---|---|
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Reranking Model | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Generation Model | Qwen/Qwen2.5-3B-Instruct |
| Chunk Size | 1,000 characters |
| Chunk Overlap | 150 characters |
| Minimum Page Length | 100 characters |
| Initial Retrieval | Top 20 |
| Final Context | Top 5 |
| LLM Precision | FP16 |

## Installation and Execution

### Prerequisites

- Google Colab account.
- GPU-enabled runtime (NVIDIA T4 recommended).
- PDF document for processing.
- Internet connection for initial dependency and model downloads.

### Setup

Clone or download the repository and upload the notebook to Google Colab.

Install the required dependencies:

```bash
pip install pymupdf sentence-transformers faiss-cpu accelerate
```

### Execution Steps

1. Open `AnyRAG.ipynb` in Google Colab.
2. Select Runtime → Change runtime type.
3. Enable GPU acceleration.
4. Execute the notebook cells sequentially.
5. Upload the PDF document when prompted.
6. Allow the document extraction and preprocessing stages to complete.
7. Generate embeddings and construct the FAISS index.
8. Load the cross-encoder reranker.
9. Initialize the local Qwen 2.5 3B model.
10. Submit natural language questions and review the generated answers with source references.

## Project Structure

```text
AnyRAG/
│
├── AnyRAG.ipynb
│
└── doctor_rag/
    └── data/
        ├── processed/
        │   ├── cleaned_pages.json
        │   └── chunks.json
        │
        └── vectorstore/
            ├── anatomy.index
            └── chunks_metadata.json
```

*Note: The notebook uses `/content/doctor_rag` as its default working directory. The example structure reflects its generated runtime artifacts.*

## Example Queries

The notebook demonstrates the following questions using an Anatomy & Physiology document:

**Question 1:**
How does insulin help regulate blood glucose levels?

**Question 2:**
What is the difference between the pituitary's anterior and posterior lobes?

The system retrieves relevant document excerpts and generates answers with corresponding PDF page references.

## Future Enhancements

- Support for multiple PDF documents and document collections.
- Hybrid retrieval combining dense and sparse search.
- Advanced metadata-based filtering.
- Query expansion and contextual retrieval.
- Conversation memory for multi-turn question answering.
- Streamlit or FastAPI-based interactive interface.
- Document-level access control.
- Retrieval evaluation using Recall@K, MRR, and NDCG.
- Support for additional local and quantized LLMs.
- Improved citation validation and answer faithfulness evaluation.

## Project Objective

AnyRAG explores the implementation of an end-to-end Retrieval-Augmented Generation architecture with an emphasis on semantic retrieval, contextual relevance, and locally executed language models.

The project demonstrates how document intelligence systems can combine vector search, neural reranking, and LLM-based generation to produce grounded responses from large unstructured documents while maintaining traceability to their original sources.

## Author

**Venkata Raja B Y**

AI/ML Developer | Generative AI | Machine Learning | RAG Systems

GitHub: https://github.com/venkat2k25

---


# 🏢 SC-Literature Intelligence
### An LLM-Driven RAG System for Evidence-Based Supply Chain AI Research Synthesis

> **Setio Basuki, Amelia Khoidir, Daffa Nugraha**
> Universitas Muhammadiyah Malang, Malang, Indonesia
> `{setio_basuki, khoidir, daffa}@umm.ac.id`

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12.4-blue?logo=python" />
  <img src="https://img.shields.io/badge/Framework-Flask-black?logo=flask" />
  <img src="https://img.shields.io/badge/LLM-Llama--3.1--8b-purple?logo=meta" />
  <img src="https://img.shields.io/badge/VectorDB-ChromaDB-orange" />
  <img src="https://img.shields.io/badge/Evaluation-RAGAS-green" />
  <img src="https://img.shields.io/badge/Deploy-Docker-blue?logo=docker" />
</p>

---

## 📌 Overview

**SC-Literature Intelligence** is a Retrieval-Augmented Generation (RAG) system designed to synthesize and analyze scientific literature on **Artificial Intelligence applications in the Supply Chain** domain — particularly supply chain risk management (SCRM) and resilience.

### Problem Statement

The rapid growth of AI-related supply chain publications creates a critical challenge: thousands of scientific abstracts are published every year across hundreds of journals and conferences, making it practically impossible to manually synthesize knowledge, trace research trends, identify gaps, or compare methodologies at scale.

Conventional keyword-based search returns document lists but cannot reason across documents. Static topic modeling produces uninterpretable clusters. Manual literature review is time-consuming, non-scalable, and prone to subjective bias. LLMs used directly without retrieval produce hallucinations — fabricating information not grounded in actual sources.

### Solution

This system integrates **retrieval-based document grounding** with **LLM generative synthesis** via the RAG paradigm. Researchers can ask natural-language research questions and receive structured, evidence-based, citation-backed answers grounded strictly in 18,235 scientific papers.

### Four Core Synthesis Tasks

| Focus | Description | Example Query |
|---|---|---|
| 📈 **Trend Analysis** | Trace AI method evolution in supply chain over time | *"How has deep learning influenced supply chain resilience models?"* |
| 🔍 **Gap Detection** | Identify underexplored areas and missing research | *"Which AI techniques are rarely applied to SCRM problems?"* |
| ⚖️ **Comparative Synthesis** | Compare models, frameworks, and objectives across studies | *"How do ML and DL approaches differ in SCRM applications?"* |
| 📋 **Evidence-based QA** | Extract specific metrics, empirical findings, and statistics | *"What accuracy levels are reported for AI in disruption prediction?"* |

> Every response includes inline citations `[Author, Year]` and a Source Verification section, ensuring full scientific traceability and zero hallucination beyond retrieved context.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                     User Query  (Web Interface)                    │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
              ┌────────────────▼───────────────┐
              │        Embedding Model          │
              │   LaBSE  (768-dim)  /           │
              │   BGE-M3 (1024-dim)             │
              └────────────────┬───────────────┘
                               │  Query Vector
              ┌────────────────▼───────────────┐
              │  ChromaDB  (4 Vector Databases) │
              │  bge_m3/db_32  bge_m3/db_64    │
              │  labse/db_32   labse/db_64      │
              │  Cosine Similarity Search        │
              └────────────────┬───────────────┘
                               │  Top-K Chunks + Metadata
              ┌────────────────▼───────────────┐
              │    Prompt Engineering            │
              │    Context Injection             │
              │    (Structured System Prompt)    │
              └────────────────┬───────────────┘
                               │
              ┌────────────────▼───────────────┐
              │   Llama-3.1-8b-instant          │
              │   (via Groq API)                │
              └────────────────┬───────────────┘
                               │
              ┌────────────────▼───────────────┐
              │   Structured HTML Response      │
              │   Direct Summary                │
              │   Critical Analysis + Citations │
              │   Source Verification           │
              │   + Chart Visualization         │
              └────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Website-Supply-chain/
│
├── __pycache__/
│   └── rag_core.cpython-312.pyc
│
├── Chunking/
│   ├── Code/
│   │   └── chunking.ipynb              # Notebook: text chunking logic (32 & 64 tokens)
│   └── Dataset/
│       ├── corpus_chunks_32.json.json  # Chunked corpus — chunk size 32
│       ├── corpus_chunks_64.json.json  # Chunked corpus — chunk size 64
│       └── corpus_metadata.json.json   # Paper metadata (title, authors, year, journal)
│
├── Cleaning data/
│   └── Code/
│       ├── data_cleaning.ipynb         # Notebook v1: data cleaning pipeline
│       └── Data_Preprocessing.ipynb    # Notebook v2: improved cleaning pipeline
│
├── Dataset/
│   ├── Data/
│   │   ├── scopus_data_2015_2025.xlsx  # Raw Scopus export
│   │   └── supply_chain_scopus.csv     # CSV version of raw data
│   └── visualization/
│       └── Distribusi dataset.png      # Chart: paper distribution by year
│
├── embedding/
│   └── chroma_db/                      # ChromaDB vector databases (not tracked in Git)
│       ├── bge_m3/
│       │   ├── db_32/                  # BGE-M3 + chunk size 32
│       │   │   ├── e8443586-8cfb-4cee-8112-36aff5923bb7/
│       │   │   └── chroma.sqlite3
│       │   └── db_64/                  # BGE-M3 + chunk size 64
│       │       ├── 5f2c9fce-ef9f-489e-90f6-f06f231c37a9/
│       │       ├── 7de80f10-cf79-4b42-ab79-7f8f58d0a51a/
│       │       ├── f8bdae20-dd42-4bfb-adda-756acaaeb381/
│       │       └── chroma.sqlite3
│       └── labse/
│           ├── db_32/                  # LaBSE + chunk size 32
│           │   ├── <uuid-hash>/
│           │   └── chroma.sqlite3
│           └── db_64/                  # LaBSE + chunk size 64
│               ├── fdb24250-ae2b-4676-826b-5a3521ee89db/
│               └── chroma.sqlite3
│
├── Result/                             # Evaluation results output
│
├── static/
│   ├── js/
│   │   └── main.js                     # Frontend JS (AJAX, Chart.js rendering)
│   └── temp_images/                    # Temporary chart image storage
│
├── templates/
│   └── index.html                      # Main Jinja2 HTML template
│
├── .env
├── .gitignore
├── app.py                              # Flask web server — main entry point
├── Dockerfile                          # Docker container definition
├── dumb.js                             # JavaScript utility / testing script
├── dumb.py                             # Python utility / testing script
├── Embedding Access.txt                # Download link for pre-built vector databases
├── embedding.zip                       # Compressed vector database archive
├── Procfile                            # Heroku/Railway deployment config
├── rag_core.py                         # Core RAG pipeline (embed → retrieve → generate)
├── README.md
├── requirements.txt                    # Python dependencies
└── token.pickle
```

---

## 📊 Dataset

### Data Acquisition (Scopus)

Data was collected from **Scopus** using a structured Boolean query combining supply chain terminology with AI-related keywords:

```
TITLE-ABS-KEY(
  "supply chain" AND (
    "artificial intelligence" OR "machine learning" OR "deep learning" OR
    "predictive analytics" OR "big data" OR "automation" OR
    "smart warehouse" OR "digital twin" OR "AI"
  )
)
```

| Attribute | Value |
|---|---|
| **Total papers** | 18,235 |
| **Publication period** | 2015 – 2025 |
| **Journal papers** | 4,394 |
| **Conference papers** | 1,892 |
| **Fields captured** | Title, Authors, Year, Journal/Conference, Scopus Link, Abstract |

### Raw Data Schema

| Column | Type | Description |
|---|---|---|
| `Authors` | string | Author names, semicolon-separated |
| `Title` | string | Full paper title |
| `Year` | integer | Publication year (2015–2025) |
| `Journal` | string | Journal or conference venue |
| `Link` | string | Direct Scopus URL |
| `Abstract` | string | Full abstract text |

---

## 🗄️ Embedding & Vector Databases

### Pre-built Vector Databases

> ✅ **The four ChromaDB vector databases have already been built and are available for download.**
> Access the download link in [`Embedding Access.txt`](Embedding%20Access.txt), or use the included `embedding.zip` archive.
> Extract the contents into `embedding/chroma_db/` before running the app — **no need to rebuild from scratch**.

### Single Notebook — Build from Scratch (Optional)

If you prefer to rebuild the databases yourself, all four ChromaDB instances can be generated from the chunking notebook:

```
Chunking/Code/chunking.ipynb
```

This notebook iterates over both embedding models and both chunk sizes, building each database sequentially:

| Database Path | Embedding Model | Chunk Size | Vector Dimensions |
|---|---|---|---|
| `embedding/chroma_db/bge_m3/db_32/` | BGE-M3 | 32 tokens | 1024 |
| `embedding/chroma_db/bge_m3/db_64/` | BGE-M3 | 64 tokens | 1024 |
| `embedding/chroma_db/labse/db_32/` | LaBSE | 32 tokens | 768 |
| `embedding/chroma_db/labse/db_64/` | LaBSE | 64 tokens | 768 |

Each ChromaDB instance stores:
- **Document:** chunk text
- **Embedding:** dense vector (768 or 1024 dimensions)
- **Metadata:** `title`, `authors`, `year`, `journal`, `chunk_id`, `paper_id`

> ⚠️ The `embedding/chroma_db/` directory is **excluded from Git** (`.gitignore`) due to large file sizes.

### Embedding Models

| Model | Full Name | Dimensions | Strengths |
|---|---|---|---|
| **BGE-M3** | BAAI General Embedding M3 | 1024 | Stronger multilingual semantics, consistently higher RAGAS scores |
| **LaBSE** | Language-agnostic BERT Sentence Embedding | 768 | Multilingual dense retrieval, faster inference |

Both models support multilingual retrieval — important since the corpus contains papers in **English and Indonesian**.

---

## 💻 Code Overview

### `app.py` — Flask Web Application

Main web server. Responsibilities:
- Serve the frontend via `templates/index.html`
- Accept POST requests with: user query, embedding model choice, chunk size, Top-K value
- Select the correct ChromaDB instance (`bge_m3/db_32`, `bge_m3/db_64`, `labse/db_32`, or `labse/db_64`)
- Call `rag_core.py` and return the structured HTML response with chart data

### `rag_core.py` — Core RAG Pipeline

Central logic of the system:

```
query_text
    │
    ├─► embed_query(model: LaBSE | BGE-M3)
    │       → query_vector (768 or 1024 dims)
    │
    ├─► chromadb_client.query(
    │       collection = selected_db,
    │       query_embeddings = [query_vector],
    │       n_results = top_k               # 3 or 5
    │   )   → top_k_chunks + metadata
    │
    ├─► build_prompt(
    │       system_prompt = SYSTEM_PROMPT,
    │       context = top_k_chunks,
    │       user_query = query_text
    │   )   → final_messages
    │
    └─► groq_client.chat.completions.create(
            model = "llama-3.1-8b-instant",
            messages = final_messages
        )   → structured_html_response
```

### `static/js/` — Frontend Logic

- Sends AJAX POST requests to Flask backend
- Dynamically renders **Chart.js** visualizations (bar, line, pie) based on query type
- Handles citation highlighting, loading states, and error display

---

## 🤖 LLM Generation Model

| Model | Provider | Context Window | Throughput |
|---|---|---|---|
| **Llama-3.1-8b-instant** | Groq API | 128K tokens | ~750 tokens/sec |

Llama-3.1-8b-instant is chosen for its high inference speed on Groq's LPU hardware, enabling real-time interactive responses over large retrieved contexts.

---

## ⚙️ Setup & Installation

### Prerequisites

| Requirement | Version |
|---|---|
| **Python** | 3.12.4 |
| **Groq API Key** | Free at [console.groq.com](https://console.groq.com) |
| **Jupyter Notebook** | For running the embedding pipeline (optional) |

### 1. Clone the Repository

```bash
git clone https://github.com/Daffanugraha/Website-Supply-chain.git
cd Website-Supply-chain
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Prepare Vector Databases

**Option A — Use pre-built databases (recommended):**

Download the pre-built ChromaDB files from the link in `Embedding Access.txt`, or extract `embedding.zip` directly, then place into:

```
embedding/chroma_db/
├── bge_m3/db_32/
├── bge_m3/db_64/
├── labse/db_32/
└── labse/db_64/
```

**Option B — Rebuild from scratch:**

```bash
jupyter notebook Chunking/Code/chunking.ipynb
```

### 5. Launch the Application

```bash
python app.py
```

Open your browser at **`http://localhost:5000`**

---

## 🐳 Docker

```bash
# Build image
docker build -t sc-literature-intelligence .

# Run container
docker run -p 5000:5000 \
  -e GROQ_API_KEY=your_key_here \
  sc-literature-intelligence
```

---

## 🚀 Deployment

Deploy to **Heroku / Railway / Render** using the included `Procfile`:

```
web: python app.py
```

Set `GROQ_API_KEY` in your platform's environment variables panel. Note that the `chroma_db/` directory needs to be either committed (with Git LFS) or rebuilt via the embedding notebook as part of a pre-deploy step.

---

## 🧪 How to Use the System

### Step 1 — Open the Web Interface

Go to `http://localhost:5000` (or your deployed URL).

### Step 2 — Write a Research Query

The system accepts both **English and Indonesian** queries and responds in the same language.

**Example queries by category:**

| Category | Example Query |
|---|---|
| Trend Analysis | *"How has the application of AI in supply chain risk management evolved over time?"* |
| Trend Analysis | *"What major paradigm shifts occurred after COVID-19 in AI-based supply chain research?"* |
| Gap Detection | *"What aspects of supply chain risk are underexplored in current AI research?"* |
| Gap Detection | *"Are human-in-the-loop approaches missing in AI-based resilience studies?"* |
| Comparative Synthesis | *"How do machine learning and deep learning approaches differ in SCRM applications?"* |
| Comparative Synthesis | *"Compare AI approaches for short-term versus long-term supply chain resilience."* |
| Evidence-based QA | *"What accuracy levels are reported for AI models in supply chain disruption prediction?"* |
| Evidence-based QA | *"What datasets are commonly used in AI-based SCRM studies?"* |

### Step 3 — Configure RAG Parameters

| Parameter | Options | Recommended | Effect |
|---|---|---|---|
| **Embedding Model** | LaBSE, BGE-M3 | **BGE-M3** | BGE-M3 achieves higher RAGAS scores across all categories |
| **Chunk Size** | 32, 64 tokens | **64 tokens** | Larger chunks provide richer context for synthesis queries |
| **Top-K** | 3, 5 | **5** | More retrieved chunks improve recall, especially for trend queries |

### Step 4 — Interpret the Response

Every response follows a fixed three-part HTML structure:

| Section | Content |
|---|---|
| **① Direct Summary** | Concise 1–3 sentence answer with bolded key concepts |
| **② Detailed Synthesis** | Bullet-point critical analysis with numbered inline citations `[1]`, `[2]`, ... |
| **③ Source Verification** | Footer listing all papers (Author, Year) grounding the analysis |
| **④ Chart** *(if applicable)* | Auto-generated Chart.js visualization based on query type |

---

## 🎯 System Prompt

The following rules are injected at the start of every LLM call, governing all response generation:

| Rule | Description |
|---|---|
| **Language Protocol** | Output language strictly follows the query language. English query → 100% English. Indonesian query → 100% Indonesian. Technical terms (e.g. "RAG", "Bullwhip Effect") remain in English regardless. |
| **No Markdown Headers** | Do NOT use `###`, `##`, or `**Title**` formatting. Output must be raw HTML only. |
| **No Duplicate Titles** | Do NOT repeat section titles. Use only the designated HTML class tags. |
| **RAW HTML Only** | Response must start directly with `<p>` and use valid HTML tags throughout. |
| **No Outside Knowledge** | Answer strictly from retrieved context. If not found, output a red-text "not available" message. |
| **Citation Rule** | Every factual claim in the Detailed Synthesis section must include an inline citation `[i]`. |
| **Neutral Tone** | Maintain objective, academic tone throughout. |
| **No Hallucinations** | If an answer cannot be found in context, state it honestly rather than fabricating content. |



---

## 📈 Evaluation — RAGAS Results

The system was evaluated using **RAGAS** (Retrieval-Augmented Generation Assessment) — an automatic pipeline evaluation framework that does not require human-annotated ground truth.

### Metrics

| Metric | What It Measures | Range |
|---|---|---|
| **Faithfulness** | Are all factual claims in the answer supported by retrieved context? | 0 – 1 |
| **Answer Relevance** | Does the answer semantically match the user's query intent? | 0 – 1 |
| **Context Precision** | What proportion of retrieved chunks are actually relevant? | 0 – 1 |
| **Context Recall** | How complete is the retrieved context coverage? | 0 – 1 |

Threshold for acceptable performance per metric: **≥ 0.5**

### Test Query Set (80 queries)

| Category | Count |
|---|---|
| Trend Analysis | 20 |
| Gap Detection | 20 |
| Comparative Synthesis | 20 |
| Evidence-based QA | 20 |
| **Total** | **80** |

---

### Result 1 — Embedding Model Comparison (All 80 queries, averaged)

| Model | Chunk | Top-K | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|---|---|
| LaBSE | 32 | 3 | 0.619 | 0.662 | 0.570 | 0.638 |
| LaBSE | 32 | 5 | 0.612 | 0.723 | 0.594 | 0.732 |
| LaBSE | 64 | 3 | 0.601 | 0.618 | 0.575 | 0.691 |
| LaBSE | 64 | 5 | 0.620 | 0.653 | 0.612 | 0.723 |
| BGE-M3 | 32 | 3 | 0.703 | 0.824 | 0.801 | 0.674 |
| BGE-M3 | 32 | 5 | 0.762 | 0.856 | 0.800 | 0.702 |
| BGE-M3 | 64 | 3 | 0.687 | 0.814 | 0.829 | 0.764 |
| **BGE-M3** | **64** | **5** | **0.722** | **0.845** | **0.856** | **0.813** |

> ✅ **Best configuration: BGE-M3 · Chunk 64 · Top-K 5**

---

### Result 2 — BGE-M3 vs LaBSE per Query Category

| Model | Category | Chunk | Top-K | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|---|---|---|
| LaBSE | Trend Analysis | 32 | 3 | 0.612 | 0.732 | 0.595 | 0.704 |
| LaBSE | Gap Detection | 32 | 3 | 0.661 | 0.753 | 0.649 | 0.689 |
| LaBSE | Comparative Synthesis | 32 | 3 | 0.608 | 0.674 | 0.575 | 0.494 |
| LaBSE | Evidence-based QA | 32 | 3 | 0.596 | 0.489 | 0.462 | 0.667 |
| LaBSE | Trend Analysis | 64 | 5 | 0.609 | 0.783 | 0.650 | 0.787 |
| LaBSE | Gap Detection | 64 | 5 | 0.586 | 0.641 | 0.722 | 0.778 |
| LaBSE | Comparative Synthesis | 64 | 5 | 0.602 | 0.691 | 0.632 | 0.560 |
| LaBSE | Evidence-based QA | 64 | 5 | 0.682 | 0.499 | 0.442 | 0.767 |
| BGE-M3 | Trend Analysis | 32 | 3 | 0.636 | 0.911 | 0.895 | 0.753 |
| BGE-M3 | Gap Detection | 32 | 3 | 0.762 | 0.737 | 0.833 | 0.729 |
| BGE-M3 | Comparative Synthesis | 32 | 3 | 0.694 | 0.865 | 0.787 | 0.521 |
| BGE-M3 | Evidence-based QA | 32 | 3 | 0.721 | 0.781 | 0.687 | 0.693 |
| **BGE-M3** | **Trend Analysis** | **64** | **5** | 0.650 | 0.870 | 0.858 | **0.940** |
| **BGE-M3** | **Gap Detection** | **64** | **5** | **0.805** | 0.821 | **0.914** | 0.826 |
| **BGE-M3** | **Comparative Synthesis** | **64** | **5** | 0.674 | **0.888** | 0.904 | 0.788 |
| **BGE-M3** | **Evidence-based QA** | **64** | **5** | 0.758 | 0.793 | 0.746 | 0.697 |

### Key Findings

- **BGE-M3 consistently outperforms LaBSE** on all four RAGAS metrics across every configuration
- **Chunk 64 + Top-K 5** is the optimal setup — richer context chunks enable better synthesis
- **Gap Detection** shows the highest and most stable performance across both models
- **Comparative Synthesis** is the most challenging category — Context Recall remains lowest across all configs due to the need for cross-document reasoning
- **Trend Analysis** achieves the highest Context Recall (0.940) with BGE-M3 + Chunk 64 + Top-K 5
- Increasing Top-K from 3 → 5 reliably improves Context Recall for Trend Analysis and Gap Detection

---

## 🐞 Failure Analysis

### Error Classification Rules

| # | Error Type | Stage | Trigger Condition |
|---|---|---|---|
| 1 | **Retrieval Failure** | Retrieval | Context Recall < 0.5 **AND** Context Precision < 0.5 |
| 2 | **Context Noise** | Retrieval | Context Recall ≥ 0.5 **AND** Context Precision < 0.5 |
| 3 | **Hallucination** | Generation | Faithfulness < 0.5 |
| 4 | **Semantic Drift** | Generation | Faithfulness ≥ 0.5 **AND** Answer Relevance < 0.5 |
| 5 | **Incomplete Answer** | Generation | Faithfulness ≥ 0.5 **AND** Answer Relevance ≥ 0.5 (partial coverage) |
| — | **Correct Answer** | — | All metrics ≥ 0.5 |

### Error Frequency Results

> Evaluated across **all 8 configurations** (2 embedding models × 2 chunk sizes × 2 Top-K values) over **80 test queries** per configuration = **640 total evaluated cases**.

| Error Type | Stage | Cases |
|---|---|---|
| Correct Answer | — | 320 |
| Hallucination | Generation | 141 |
| Incomplete Answer | Generation | 96 |
| Semantic Drift | Generation | 46 |
| Retrieval Failure | Retrieval | 28 |
| Context Noise | Retrieval | 9 |
| **Total** | | **640** |

---

### Error Breakdown by Query Category

#### Correct Answer (320 cases)

| Query Category | Cases |
|---|---|
| Trend Analysis | 100 |
| Comparative Synthesis | 75 |
| Gap Detection | 74 |
| Evidence-based QA | 71 |
| **Total** | **320** |

#### Hallucination (141 cases)

| Query Category | Cases |
|---|---|
| Comparative Synthesis | 42 |
| Trend Analysis | 35 |
| Evidence-based QA | 33 |
| Gap Detection | 31 |
| **Total** | **141** |

#### Incomplete Answer (96 cases)

| Query Category | Cases |
|---|---|
| Gap Detection | 35 |
| Comparative Synthesis | 25 |
| Evidence-based QA | 24 |
| Trend Analysis | 12 |
| **Total** | **96** |

#### Semantic Drift (46 cases)

| Query Category | Cases |
|---|---|
| Evidence-based QA | 23 |
| Gap Detection | 12 |
| Trend Analysis | 7 |
| Comparative Synthesis | 4 |
| **Total** | **46** |

#### Retrieval Failure (28 cases)

| Query Category | Cases |
|---|---|
| Comparative Synthesis | 13 |
| Gap Detection | 5 |
| Evidence-based QA | 5 |
| Trend Analysis | 5 |
| **Total** | **28** |

#### Context Noise (9 cases)

| Query Category | Cases |
|---|---|
| Evidence-based QA | 4 |
| Gap Detection | 3 |
| Comparative Synthesis | 1 |
| Trend Analysis | 1 |
| **Total** | **9** |



## 📦 Dependencies

```
flask
groq
chromadb
sentence-transformers        # LaBSE
FlagEmbedding                # BGE-M3
ragas
langchain
langchain-community
pandas
numpy
openpyxl
python-dotenv
gunicorn
jupyter
ipykernel
```

> **Python version: 3.12.4**

---

## 📚 Related Resources

| Resource | Link |
|---|---|
| RAGAS Framework | [github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas) |
| BGE-M3 Model | [huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| LaBSE Model | [huggingface.co/sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE) |
| Llama 3.1 on Groq | [console.groq.com](https://console.groq.com) |
| ChromaDB Docs | [docs.trychroma.com](https://docs.trychroma.com) |
| Scopus Database | [scopus.com](https://www.scopus.com) |

---

## 📄 Citation

```bibtex
@article{basuki2026sc,
  title       = {SC-Literature Intelligence: An LLM-Driven RAG System for
                 Evidence-Based Supply Chain AI Research Synthesis},
  author      = {Basuki, Setio and Khoidir, Amelia and Nugraha, Daffa},
  year        = {2026},
  institution = {Universitas Muhammadiyah Malang},
  address     = {Malang, Indonesia}
}
```

---

## 👥 Contributors

| Name | Role | Contact |
|---|---|---|
| **Setio Basuki** | Research Lead | setio_basuki@umm.ac.id |
| **Amelia Khoidir** | Research Member | khoidir@umm.ac.id |
| **Daffa Nugraha** | Developer & Research Member | daffa@umm.ac.id |

---

## 📄 License

This project is developed for academic research purposes at **Universitas Muhammadiyah Malang**. Please contact the authors for licensing inquiries.

---

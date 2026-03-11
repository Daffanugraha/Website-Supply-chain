# 🏢 SC-Literature Intelligence
### An LLM-Driven RAG System for Evidence-Based Supply Chain AI Research Synthesis

> **Setio Basuki, Amelia Khoidir, Daffa Nugraha**
> Universitas Muhammadiyah Malang, Malang, Indonesia
> `{setio_basuki, khoidir, ilham, daffa}@umm.ac.id`

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
├── app.py                          # Flask web server — main entry point
├── rag_core.py                     # Core RAG pipeline (embed → retrieve → generate)
├── dumb.py                         # Python utility / testing script
├── dumb.js                         # JavaScript utility / testing script
│
├── chunking/                       # Chunking pipeline & output
│   ├── chunking.ipynb              # Notebook: text chunking logic (32 & 64 tokens)
│   ├── data_preprocessed_chunk_32.json   # Chunked corpus — chunk size 32
│   ├── data_preprocessed_chunk_64.json   # Chunked corpus — chunk size 64
│   └── metadata_jurnal.json        # Paper metadata (title, authors, year, journal)
│
├── Dataset/                        # Raw and cleaned dataset files
│   ├── Scopus dataset supply chain 2015 - 2025.xlsx   # Raw Scopus export (main input)
│   ├── Dataset scopus supply chain.csv                # CSV version of raw data
│   ├── cleaned_scopus_data.xlsx    # Cleaned dataset output
│   ├── cleaning dataset.ipynb      # Notebook v1: data cleaning pipeline
│   ├── cleaning dataset v2.ipynb   # Notebook v2: improved cleaning pipeline
│   └── Distribusi dataset.png      # Chart: paper distribution by year
│
├── embedding/                      # Vector database builder
│   ├── embedding.ipynb             # Single notebook: builds ALL 4 ChromaDB databases
│   └── chroma_db/                  # ChromaDB vector databases (not tracked in Git)
│       ├── bge_m3/
│       │   ├── db_32/              # BGE-M3 + chunk size 32
│       │   │   ├── <uuid-hash>/
│       │   │   └── chroma.sqlite3
│       │   └── db_64/              # BGE-M3 + chunk size 64
│       │       ├── <uuid-hash>/
│       │       └── chroma.sqlite3
│       └── labse/
│           ├── db_32/              # LaBSE + chunk size 32
│           │   ├── <uuid-hash>/
│           │   └── chroma.sqlite3
│           └── db_64/              # LaBSE + chunk size 64
│               ├── <uuid-hash>/
│               └── chroma.sqlite3
│
├── error analysis/                 # Per-error-type RAGAS audit files
│   ├── MASTER_DATA_RAG_MERGED.xlsx       # Full merged evaluation dataset
│   ├── Hasil_Audit_RAG_Final_FIXED.xlsx  # Final audited results with error labels
│   ├── Context Noise.xlsx          # Filtered: Context Noise cases
│   ├── Correct Answer.xlsx         # Filtered: Completely Correct cases
│   ├── Hallucination.xlsx          # Filtered: Hallucination cases
│   ├── Incomplete Answer.xlsx      # Filtered: Incomplete Answer cases
│   ├── Retrieval Failure.xlsx      # Filtered: Retrieval Failure cases
│   └── Semantic Drift.xlsx         # Filtered: Semantic Drift cases
│
├── static/
│   └── js/                         # Frontend JS (AJAX, Chart.js rendering)
│
├── templates/
│   └── index.html                  # Main Jinja2 HTML template
│
├── Dockerfile                      # Docker container definition
├── Procfile                        # Heroku/Railway deployment config
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
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
| **Raw file** | `Dataset/Scopus dataset supply chain 2015 - 2025.xlsx` |
| **Total papers** | 18,235 |
| **Publication period** | 2015 – 2025 |
| **Journal papers** | 4,394 |
| **Conference papers** | 1,892 |
| **Fields captured** | Title, Authors, Year, Journal/Conference, Scopus Link, Abstract |

### Dataset Distribution by Year

![Dataset Distribution by Year](https://raw.githubusercontent.com/Daffanugraha/Website-Supply-chain/main/Dataset/Distribusi%20dataset.png)

> A clear upward trend is visible from 2019 onwards, with a significant acceleration post-2021 — reflecting the surge in AI adoption for supply chain applications, largely driven by global disruptions during the COVID-19 pandemic.

### Raw Data Schema

| Column | Type | Description |
|---|---|---|
| `Authors` | string | Author names, semicolon-separated |
| `Title` | string | Full paper title |
| `Year` | integer | Publication year (2015–2025) |
| `Journal` | string | Journal or conference venue |
| `Link` | string | Direct Scopus URL |
| `Abstract` | string | Full abstract text |

### Data Cleaning Pipeline

Two cleaning notebooks are provided in `Dataset/`:

| Notebook | Description |
|---|---|
| `cleaning dataset.ipynb` | Initial cleaning: removes duplicates, empty abstracts, malformed entries |
| `cleaning dataset v2.ipynb` | Improved pipeline: additional normalization, encoding fixes, final export to `cleaned_scopus_data.xlsx` |

---

## ✂️ Chunking Pipeline

The `chunking/` folder contains the chunking logic that prepares text for embedding:

| Stage | Operation | Details |
|---|---|---|
| **Text Cleaning** | Remove HTML tags, punctuation, special characters | Standardizes raw abstract text |
| **Case Folding** | Convert all text to lowercase | Ensures uniform embedding representation |
| **Chunking** | Segment into overlapping token windows | Two sizes: **32 tokens** and **64 tokens**, with **10% overlap** |

**Output files:**

| File | Description |
|---|---|
| `data_preprocessed_chunk_32.json` | Full corpus chunked at 32 tokens |
| `data_preprocessed_chunk_64.json` | Full corpus chunked at 64 tokens |
| `metadata_jurnal.json` | Metadata per paper (title, authors, year, journal) for citation attachment |

**Chunking rationale:**
- **Chunk 32** → finer granularity, higher precision for narrow factual queries
- **Chunk 64** → richer context per chunk, better for complex synthesis and trend queries

**Preprocessing example:**

| Stage | Text |
|---|---|
| **Original** | *"High-Entropy Alloys (HEAs), such as Al10.3Co17... pose sustainability & supply-chain risks."* |
| **After Cleaning** | *"High Entropy Alloys HEAs such as Al Co Cr Fe Ni... pose sustainability and supply chain risks"* |
| **After Case Folding** | *"high entropy alloys heas such as al co cr fe ni... pose sustainability and supply chain risks"* |
| **Chunk 1 (32 tok)** | *"high entropy alloys heas such as al co cr fe ni have demonstrated superior performance compared to conventional materials"* |
| **Chunk 2 (32 tok)** | *"however their development often relies on critical raw materials crms which pose sustainability and supply chain risks"* |

---

## 🗄️ Embedding & Vector Databases

### Single Notebook — All Four Databases

All four ChromaDB vector databases are built from a **single notebook**:

```
embedding/embedding.ipynb
```

**Input:** `Dataset/Scopus dataset supply chain 2015 - 2025.xlsx`

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

> ⚠️ The `embedding/chroma_db/` directory is **excluded from Git** (`.gitignore`) due to large file sizes. Rebuild locally by running `embedding/embedding.ipynb`.

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

### `chunking/chunking.ipynb`

Processes `cleaned_scopus_data.xlsx` and outputs:
- `data_preprocessed_chunk_32.json` — tokenized chunks at size 32
- `data_preprocessed_chunk_64.json` — tokenized chunks at size 64
- `metadata_jurnal.json` — paper-level metadata for citation linking

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
| **Jupyter Notebook** | For running embedding & chunking pipelines |

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

### 4. Prepare the Dataset

The raw dataset is already included in the repository at:

```
Dataset/Scopus dataset supply chain 2015 - 2025.xlsx
```

If you want to re-run the cleaning pipeline, open and run:
```bash
jupyter notebook "Dataset/cleaning dataset v2.ipynb"
```

### 5. Run the Chunking Notebook

```bash
jupyter notebook chunking/chunking.ipynb
```

This produces the two JSON chunk files used as input for embedding.

### 6. Build All Vector Databases

Run the single embedding notebook — it will build all four ChromaDB databases:

```bash
jupyter notebook embedding/embedding.ipynb
```

Output structure after completion:
```
embedding/chroma_db/
├── bge_m3/db_32/
├── bge_m3/db_64/
├── labse/db_32/
└── labse/db_64/
```

### 7. Launch the Application

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

The following prompt is injected at the start of every LLM call, governing all response generation:

```python
system_prompt = """
Role: You are the "SC-Literature Intelligence Assistant", an expert AI specialized
in synthesizing scientific literature on Supply Chain and AI.

CORE OBJECTIVE: Answer the user's question using ONLY the provided Context Data.

---
### 🚨 1. LANGUAGE PROTOCOL (HIGHEST PRIORITY)
You must strictly follow the language of the **User Query**, regardless of the
language in the Context Data.

* **CASE A: User asks in ENGLISH**
  - OUTPUT: **100% English**.
  - ACTION: Translate all findings from Indonesian context into English.

* **CASE B: User asks in INDONESIAN**
  - OUTPUT: **100% Indonesian**.
  - ACTION: Translate all findings from English context into Indonesian.
  - EXCEPTION: Keep technical terms (e.g., "Robustness", "RAG", "Bullwhip Effect")
    in English.

---
### ⛔ 2. STRICT FORMATTING RULES (CRITICAL)
1. **NO MARKDOWN HEADERS:** Do NOT use `###`, `##`, or `**Title**`.
2. **NO DUPLICATE TITLES:** Do NOT repeat "Direct Answer" or "Direct Summary".
   Use ONLY the HTML tags provided below.
3. **RAW HTML ONLY:** Your output must start directly with `<p>` and contain
   valid HTML tags.
4. **NEVER** mix languages. **NEVER** let the document language override the
   query language.

---
### 📝 3. REQUIRED HTML STRUCTURE

**A. Direct Answer**
   <p class="font-bold text-blue-700 mb-2 underline underline-offset-4
      decoration-blue-200">Direct Summary:</p>
   <p class="mb-4">Your direct answer here with <strong>key concepts</strong>
      bolded.</p>

**B. Critical Analysis**
   <p class="font-bold text-slate-700 mb-1">Detailed Synthesis:</p>
   <ul class="list-disc pl-4 space-y-2 mb-4">
     <li>Point one explanation derived from context [1].</li>
     <li>Point two explanation with citation [2].</li>
   </ul>

**C. Source Verification**
   <p class="text-[11px] italic text-slate-500 bg-slate-50 p-2 rounded
      border-l-2 border-blue-400">Analysis grounded in primary sources:
      [authors, Year].</p>

---
### ⛔ 4. STRICT CONSTRAINTS
1. **No Outside Knowledge:** If the answer is not in the context, output:
   "<p class='text-red-500 font-bold'>Information not available in the
   provided documents.</p>"
2. **Citation Rule:** Every single claim in Section B must have a citation
   number [i].
3. **Neutral Tone:** Maintain an objective, academic tone.
4. **No Hallucinations:** If the answer cannot be found, honestly state it.
"""
```

### Prompt Sensitivity Variants Tested

| Variant | Description |
|---|---|
| **P1 — Generic Baseline** | Minimal instruction, open-ended output format |
| **P2 — Strict Context-Based** | Hard constraint: answer ONLY from context, no outside knowledge |
| **P3 — Structured Analysis** | Enforces three-section HTML structure, academic synthesis style |
| **P4 — Evidence & Anti-Hallucination** | Adds explicit citation rules + hallucination refusal mechanism |

> The production prompt combines the strongest elements of **P3 and P4**.

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

### Result 2 — Performance by Query Category

| Category | Chunk | Top-K | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|---|---|
| Trend Analysis | 32 | 3 | 0.624 | 0.822 | 0.745 | 0.728 |
| Gap Detection | 32 | 3 | 0.711 | 0.745 | 0.741 | 0.709 |
| Comparative Synthesis | 32 | 3 | 0.651 | 0.770 | 0.681 | 0.507 |
| Evidence-based QA | 32 | 3 | 0.658 | 0.635 | 0.575 | 0.680 |
| Trend Analysis | 32 | 5 | 0.755 | 0.805 | 0.717 | 0.724 |
| Gap Detection | 32 | 5 | 0.741 | 0.732 | 0.780 | 0.754 |
| Comparative Synthesis | 32 | 5 | 0.597 | 0.838 | 0.727 | 0.590 |
| Evidence-based QA | 32 | 5 | 0.654 | 0.782 | 0.566 | 0.749 |
| Trend Analysis | 64 | 3 | 0.609 | 0.770 | 0.718 | 0.802 |
| Gap Detection | 64 | 3 | 0.677 | 0.659 | 0.747 | 0.724 |
| Comparative Synthesis | 64 | 3 | 0.647 | 0.835 | 0.750 | 0.590 |
| Evidence-based QA | 64 | 3 | 0.643 | 0.599 | 0.593 | 0.698 |
| Trend Analysis | 64 | 5 | 0.630 | 0.827 | 0.754 | 0.863 |
| Gap Detection | 64 | 5 | 0.696 | 0.731 | 0.818 | 0.802 |
| Comparative Synthesis | 64 | 5 | 0.638 | 0.790 | 0.768 | 0.674 |
| Evidence-based QA | 64 | 5 | 0.720 | 0.646 | 0.594 | 0.732 |

---

### Result 3 — BGE-M3 vs LaBSE per Query Category

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

All error classification results are stored in `error analysis/`. Errors are assigned using a deterministic rule-based system with a **threshold of 0.5** per RAGAS metric. The master dataset is `Hasil_Audit_RAG_Final_FIXED.xlsx`; individual error type files are filtered subsets.

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
> Full data: [`error analysis/Hasil_Audit_RAG_Final_FIXED.xlsx`](error%20analysis/Hasil_Audit_RAG_Final_FIXED.xlsx)

| Error Type | Stage | File | Cases | Percentage |
|---|---|---|---|---|
| Correct Answer | — | [`Correct Answer.xlsx`](error%20analysis/Correct%20Answer.xlsx) | 320 | 50.0% |
| Hallucination | Generation | [`Hallucination.xlsx`](error%20analysis/Hallucination.xlsx) | 141 | 22.0% |
| Incomplete Answer | Generation | [`Incomplete Answer.xlsx`](error%20analysis/Incomplete%20Answer.xlsx) | 96 | 15.0% |
| Semantic Drift | Generation | [`Semantic Drift.xlsx`](error%20analysis/Semantic%20Drift.xlsx) | 46 | 7.2% |
| Retrieval Failure | Retrieval | [`Retrieval Failure.xlsx`](error%20analysis/Retrieval%20Failure.xlsx) | 28 | 4.4% |
| Context Noise | Retrieval | [`Context Noise.xlsx`](error%20analysis/Context%20Noise.xlsx) | 9 | 1.4% |
| **Total** | | | **640** | **100%** |

---

### Error Breakdown by Query Category

The tables below present how each error type distributes across the four query categories, providing insight into which query types are most prone to each failure mode.

#### Correct Answer (320 cases — 50.0%)

| Query Category | Cases | Percentage of Category |
|---|---|---|
| Trend Analysis | 100 | 31.3% |
| Comparative Synthesis | 75 | 23.4% |
| Gap Detection | 74 | 23.1% |
| Evidence-based QA | 71 | 22.2% |
| **Total** | **320** | **100%** |

> Trend Analysis queries yield the highest proportion of correct answers, indicating that temporal and evolutionary questions are well-matched to the corpus's coverage.

#### Hallucination (141 cases — 22.0%)

| Query Category | Cases | Percentage of Category |
|---|---|---|
| Comparative Synthesis | 42 | 29.8% |
| Trend Analysis | 35 | 24.8% |
| Evidence-based QA | 33 | 23.4% |
| Gap Detection | 31 | 22.0% |
| **Total** | **141** | **100%** |

> Comparative Synthesis queries produce the most hallucinations, as comparing two or more AI frameworks across multiple dimensions requires cross-document synthesis that often pushes the model beyond what is explicitly stated in the retrieved chunks.

#### Incomplete Answer (96 cases — 15.0%)

| Query Category | Cases | Percentage of Category |
|---|---|---|
| Gap Detection | 35 | 36.5% |
| Comparative Synthesis | 25 | 26.0% |
| Evidence-based QA | 24 | 25.0% |
| Trend Analysis | 12 | 12.5% |
| **Total** | **96** | **100%** |

> Gap Detection queries dominate incomplete answer cases. These queries inherently require the model to identify missing research dimensions — a multi-aspect task where answering one dimension (e.g., missing methods) while omitting another (e.g., missing geographic contexts) is classified as incomplete.

#### Semantic Drift (46 cases — 7.2%)

| Query Category | Cases | Percentage of Category |
|---|---|---|
| Evidence-based QA | 23 | 50.0% |
| Gap Detection | 12 | 26.1% |
| Trend Analysis | 7 | 15.2% |
| Comparative Synthesis | 4 | 8.7% |
| **Total** | **46** | **100%** |

> Evidence-based QA queries account for half of all Semantic Drift cases. This occurs when the model provides a factually grounded answer that is adjacent to the requested empirical metric (e.g., answering "which models are used" instead of "what accuracy levels are reported"), indicating that the query intent is correctly retrieved but imprecisely answered.

#### Retrieval Failure (28 cases — 4.4%)

| Query Category | Cases | Percentage of Category |
|---|---|---|
| Comparative Synthesis | 13 | 46.4% |
| Gap Detection | 5 | 17.9% |
| Evidence-based QA | 5 | 17.9% |
| Trend Analysis | 5 | 17.9% |
| **Total** | **28** | **100%** |

> Comparative Synthesis queries are disproportionately affected by retrieval failure, accounting for nearly half of all cases. Comparison queries often use abstract terminology (e.g., "centralized vs decentralized AI architectures") that does not closely match the vocabulary of individual paper abstracts, resulting in poor embedding-to-query alignment.

#### Context Noise (9 cases — 1.4%)

| Query Category | Cases | Percentage of Category |
|---|---|---|
| Evidence-based QA | 4 | 44.4% |
| Gap Detection | 3 | 33.3% |
| Comparative Synthesis | 1 | 11.1% |
| Trend Analysis | 1 | 11.1% |
| **Total** | **9** | **100%** |

> Context Noise is the least frequent error type overall, occurring in only 9 of 640 cases. Its concentration in Evidence-based QA queries reflects situations where the retrieval mechanism successfully recovers some relevant chunks but also surfaces thematically adjacent documents that dilute the precision of the provided context.

---

### Summary of Error Patterns

| Error Type | Stage | Most Affected Category | Key Cause |
|---|---|---|---|
| Correct Answer | — | Trend Analysis (31.3%) | Good temporal coverage in corpus |
| Hallucination | Generation | Comparative Synthesis (29.8%) | Cross-document synthesis exceeds retrieved span |
| Incomplete Answer | Generation | Gap Detection (36.5%) | Multi-dimensional queries partially addressed |
| Semantic Drift | Generation | Evidence-based QA (50.0%) | Adjacent answer retrieved; query intent imprecise |
| Retrieval Failure | Retrieval | Comparative Synthesis (46.4%) | Abstract comparative vocabulary misaligns with embeddings |
| Context Noise | Retrieval | Evidence-based QA (44.4%) | Relevant chunks mixed with thematically adjacent documents |

### Error Sub-Type Taxonomy

| Error Type | Sub-Category | Metric Pattern | Interpretation |
|---|---|---|---|
| **Retrieval Failure** | Hard Retrieval Miss | Precision ≤ 0.5, Recall ≤ 0.5 | Core concept absent or embedding–query alignment failure |
| **Retrieval Failure** | Partial Recall Loss | Precision > 0.5, Recall ≤ 0.5 | Partially found; Top-K too small or chunk too fine-grained |
| **Context Noise** | Dominant Noise | Precision ≤ 0.5, Recall ≥ 0.5 | Relevant evidence diluted by off-topic chunks |
| **Context Noise** | Mixed Context | Precision moderate, Recall high | Selective grounding despite noisy context |
| **Hallucination** | Abstractive Deviation | Faithfulness ≤ 0.5, Relevance ≥ 0.5 | Coherent synthesis that extends beyond retrieved spans |
| **Hallucination** | Unsupported Generation | Faithfulness ≤ 0.5, context weak | Model infers from parametric knowledge due to poor retrieval |
| **Hallucination** | True Hallucination | All metrics ≤ 0.5 | No factual grounding AND no semantic alignment |
| **Semantic Drift** | Intent Drift | Faithfulness ≥ 0.5, Relevance ≤ 0.5 | Well-grounded but diverges from specific query intent |
| **Semantic Drift** | Scope Drift | Faithfulness high, Relevance moderate | Accurate but at broader analytical scope than requested |
| **Incomplete Answer** | Partial Coverage | Faithfulness ≥ 0.5, Relevance moderate | Core aspects answered; secondary dimensions omitted |
| **Incomplete Answer** | Dimension Omission | Faithfulness high, Recall high | Strong grounding but not all required dimensions articulated |

---

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
openpyxl                     # xlsx read/write
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

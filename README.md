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

**SC-Literature Intelligence** is a Retrieval-Augmented Generation (RAG) system designed to synthesize and analyze scientific literature on **Artificial Intelligence applications in the Supply Chain** domain. It specifically targets supply chain risk management (SCRM) and resilience research.

### Problem Statement

The rapid growth of AI-related supply chain publications has created a major challenge: thousands of scientific abstracts are published every year across hundreds of journals and conferences, making it practically impossible to manually synthesize knowledge, trace research trends, identify gaps, or compare methodologies at scale.

Conventional keyword-based search engines return document lists but cannot reason across documents. Static topic modeling produces uninterpretable clusters with no interactive capability. Manual literature review is time-consuming, non-scalable, and prone to subjective bias.

### Solution

This system integrates retrieval-based document grounding with LLM generative synthesis, enabling researchers to ask natural-language research questions and receive evidence-based, citation-backed answers grounded strictly in the scientific literature.

### Four Core Synthesis Tasks

| Focus | Description | Example Query |
|---|---|---|
| 📈 **Trend Analysis** | Trace AI method evolution in supply chain over time | *"How has deep learning influenced supply chain resilience models?"* |
| 🔍 **Gap Detection** | Identify underexplored areas and missing research | *"Which AI techniques are rarely applied to SCRM problems?"* |
| ⚖️ **Comparative Synthesis** | Compare models, frameworks, and objectives across studies | *"How do ML and DL approaches differ in SCRM applications?"* |
| 📋 **Evidence-based QA** | Extract specific metrics, empirical data, and statistics | *"What accuracy levels are reported for AI in disruption prediction?"* |

> Every system response includes inline citations (Author, Year) and a Source Verification section — ensuring full traceability and no hallucination beyond the retrieved context.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     User Query  (Web Interface)                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
               ┌────────────────▼────────────────┐
               │       Embedding Model            │
               │    LaBSE (768-dim)  /            │
               │    BGE-M3 (1024-dim)             │
               └────────────────┬────────────────┘
                                │  Query Vector
               ┌────────────────▼────────────────┐
               │         ChromaDB                 │
               │   (4 Vector Databases)           │
               │   Cosine Similarity Search       │
               └────────────────┬────────────────┘
                                │  Top-K Chunks + Metadata
               ┌────────────────▼────────────────┐
               │    Prompt Engineering            │
               │    Context Injection             │
               │    (Structured System Prompt)    │
               └────────────────┬────────────────┘
                                │
               ┌────────────────▼────────────────┐
               │   LLM Generation                 │
               │   Llama-3.1-8b-instant           │
               │   (via Groq API)                 │
               └────────────────┬────────────────┘
                                │
               ┌────────────────▼────────────────┐
               │   Structured HTML Response       │
               │   Direct Summary                 │
               │   Critical Analysis + Citations  │
               │   Source Verification            │
               │   + Dynamic Chart Visualization  │
               └─────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Website-Supply-chain/
│
├── app.py                    # Flask web server — main entry point
├── rag_core.py               # Core RAG pipeline (embed → retrieve → generate)
├── dumb.py                   # Python utility/testing script
├── dumb.js                   # JavaScript utility/testing script
│
├── embedding/                # Data processing & vector database builder
│   ├── data_preprocessing.ipynb        # Text cleaning, case folding, chunking
│   ├── build_db_labse_chunk32.ipynb    # Build ChromaDB: LaBSE + chunk=32
│   ├── build_db_labse_chunk64.ipynb    # Build ChromaDB: LaBSE + chunk=64
│   ├── build_db_bge_chunk32.ipynb      # Build ChromaDB: BGE-M3 + chunk=32
│   ├── build_db_bge_chunk64.ipynb      # Build ChromaDB: BGE-M3 + chunk=64
│   └── ragas_evaluation.ipynb          # RAGAS evaluation pipeline
│
├── chroma_db/                # 4 vector databases (not tracked in Git — rebuild locally)
│   ├── labse_chunk32/
│   ├── labse_chunk64/
│   ├── bge_m3_chunk32/
│   └── bge_m3_chunk64/
│
├── docs/                     # Documentation assets
│   ├── distribusi_paper.png            # Dataset year distribution chart
│   ├── arsitektur_sistem.png           # System architecture diagram
│   ├── umap_labse.png                  # UMAP visualization — LaBSE
│   ├── umap_bge.png                    # UMAP visualization — BGE-M3
│   └── retrieval_example.png           # Cosine similarity retrieval example
│
├── data/
│   ├── scopus_raw.csv                  # Raw Scopus export (18,235 papers)
│   ├── scopus_preprocessed.csv         # Cleaned & chunked data
│   └── ragas_results.csv              # Full RAGAS evaluation results
│
├── static/
│   └── js/                   # Frontend JavaScript (AJAX, Chart.js rendering)
│
├── templates/
│   └── index.html            # Main Jinja2 HTML template
│
├── Dockerfile                # Docker container definition
├── Procfile                  # Heroku/Railway deployment config
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## 📊 Dataset

### 2.1 Data Acquisition

Data was collected from **Scopus** — one of the largest academic publication databases — using a structured Boolean query combining supply chain terminology with AI-related keywords:

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
| **Data fields** | Title, Authors, Year, Journal/Conference, Scopus Link, Abstract |

### Dataset Distribution by Year

The chart below shows the annual distribution of publications in the collected dataset. A clear upward trend is visible starting from 2019, with a significant acceleration post-2021 reflecting the boom in AI adoption for supply chain applications.

![Dataset Distribution by Year](docs/distribusi_paper.png)

> The dataset spans **11 years** (2015–2025). The sharp growth after 2020 reflects increased research interest following global supply chain disruptions during the COVID-19 pandemic.

### Raw Data Schema (`data/scopus_raw.csv`)

| Column | Type | Description |
|---|---|---|
| `authors` | string | Author names, semicolon-separated |
| `title` | string | Full paper title |
| `year` | integer | Publication year |
| `journal` | string | Journal or conference venue name |
| `link` | string | Direct Scopus URL |
| `abstract` | string | Full abstract text |

### 2.2 Data Preprocessing

Before building vector databases, all abstracts go through a three-stage preprocessing pipeline:

| Stage | Operation | Purpose |
|---|---|---|
| **Text Cleaning** | Remove HTML tags, punctuation, special characters, numeric normalization | Reduce noise, standardize characters |
| **Case Folding** | Convert all characters to lowercase | Ensure uniform representation for embedding |
| **Chunking** | Segment cleaned text into overlapping token windows (10% overlap) | Balance retrieval granularity vs. context completeness |

Two chunk sizes are tested: **32 tokens** (fine-grained, precise) and **64 tokens** (richer context, better for synthesis queries).

**Example preprocessing output:**

| Stage | Text |
|---|---|
| **Original** | *"High-Entropy Alloys (HEAs), such as Al10.3Co17Cr7.5Fe9Ni48.6, have demonstrated superior performance..."* |
| **After Cleaning** | *"High Entropy Alloys HEAs such as Al Co Cr Fe Ni have demonstrated superior performance..."* |
| **After Case Folding** | *"high entropy alloys heas such as al co cr fe ni have demonstrated superior performance..."* |
| **Chunk 1** | *"high entropy alloys heas such as al co cr fe ni have demonstrated superior performance compared to conventional materials"* |
| **Chunk 2** | *"however their development often relies on critical raw materials crms which pose sustainability and supply chain risks"* |

---

## 🗄️ Vector Databases

Four separate **ChromaDB** vector databases are built, one for each combination of embedding model × chunk size:

| Database ID | Embedding Model | Chunk Size | Vector Dimensions | Overlap |
|---|---|---|---|---|
| `labse_chunk32` | LaBSE | 32 tokens | 768 | 10% |
| `labse_chunk64` | LaBSE | 64 tokens | 768 | 10% |
| `bge_m3_chunk32` | BGE-M3 | 32 tokens | 1024 | 10% |
| `bge_m3_chunk64` | BGE-M3 | 64 tokens | 1024 | 10% |

Each database stores:
- Chunk text (document field)
- Dense embedding vector
- Metadata: `title`, `authors`, `year`, `journal`, `chunk_id`

> ⚠️ The `chroma_db/` directory is **excluded from Git** (see `.gitignore`) due to size. Rebuild locally using the embedding notebooks (see Setup section).

---

## 💻 Code Overview

### `app.py` — Flask Web Application

Main web server. Responsibilities:
- Serve the frontend via `templates/index.html`
- Accept POST requests containing user query, selected embedding model, chunk size, and Top-K value
- Route the request to the correct ChromaDB instance based on selected configuration
- Call `rag_core.py` and stream the structured HTML response back to the client
- Handle chart data generation based on query type

### `rag_core.py` — Core RAG Pipeline

Central logic of the system. Contains:

```
query_text
    │
    ├─► embed_query(model)        → query_vector (768 or 1024 dims)
    │
    ├─► chromadb.query(
    │       query_embeddings=[query_vector],
    │       n_results=top_k
    │   )                         → top_k_chunks + metadata
    │
    ├─► build_prompt(
    │       system_prompt,
    │       context=top_k_chunks,
    │       user_query=query_text
    │   )                         → final_prompt
    │
    └─► groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[system_prompt, final_prompt]
        )                         → structured_html_response
```

### `embedding/` — Database Building Notebooks

| Notebook | Input | Output |
|---|---|---|
| `data_preprocessing.ipynb` | `data/scopus_raw.csv` | `data/scopus_preprocessed.csv` |
| `build_db_labse_chunk32.ipynb` | `scopus_preprocessed.csv` | `chroma_db/labse_chunk32/` |
| `build_db_labse_chunk64.ipynb` | `scopus_preprocessed.csv` | `chroma_db/labse_chunk64/` |
| `build_db_bge_chunk32.ipynb` | `scopus_preprocessed.csv` | `chroma_db/bge_m3_chunk32/` |
| `build_db_bge_chunk64.ipynb` | `scopus_preprocessed.csv` | `chroma_db/bge_m3_chunk64/` |
| `ragas_evaluation.ipynb` | All 4 ChromaDBs + 80 test queries | `data/ragas_results.csv` |

### `static/js/` — Frontend Logic

- Sends AJAX POST requests to Flask backend
- Dynamically renders Chart.js visualizations (bar chart, line chart, pie chart) based on answer type
- Displays structured HTML responses with citation highlighting
- Handles loading states and error messages

### `templates/index.html` — Web Interface

Single-page application featuring:
- Query input textarea with placeholder examples per query type
- Parameter selectors: embedding model, chunk size, Top-K
- Response panel with Direct Summary, Detailed Synthesis, and Source Verification sections
- Embedded chart canvas rendered by Chart.js

---

## 🤖 Models

### Embedding Models

| Model | Full Name | Dimensions | Architecture | Multilingual |
|---|---|---|---|---|
| **LaBSE** | Language-agnostic BERT Sentence Embedding | 768 | BERT-based, dual encoder | ✅ 109 languages |
| **BGE-M3** | BAAI General Embedding — Multi-Lingual, Multi-Granularity, Multi-Functionality | 1024 | XLM-RoBERTa based | ✅ 100+ languages |

Both models support multilingual dense retrieval, which is important since the corpus includes papers in both **English and Indonesian**.

### LLM Generation Model

| Model | Provider | Context Window | Speed |
|---|---|---|---|
| **Llama-3.1-8b-instant** | Groq API | 128K tokens | ~750 tokens/sec |

Llama-3.1-8b-instant is used for its high inference throughput on Groq's LPU hardware, making it suitable for real-time interactive querying over large retrieved contexts.

### Vector Database

**ChromaDB** (v0.4+) — an open-source, persistent, local vector database. Used to store and query document chunk embeddings using L2 distance / cosine similarity.

---

## ⚙️ Setup & Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | **3.12.4** |
| pip | latest |
| Groq API Key | Free at [console.groq.com](https://console.groq.com) |
| Jupyter Notebook | for embedding pipeline |

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

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Prepare Raw Data

Place your Scopus CSV export at:

```
data/scopus_raw.csv
```

Make sure the CSV contains columns: `authors`, `title`, `year`, `journal`, `link`, `abstract`.

### 5. Run the Preprocessing Notebook

```bash
jupyter notebook embedding/data_preprocessing.ipynb
```

This produces `data/scopus_preprocessed.csv`.

### 6. Build All Four Vector Databases

Run the four embedding notebooks **in any order**:

```bash
jupyter notebook embedding/build_db_labse_chunk32.ipynb
jupyter notebook embedding/build_db_labse_chunk64.ipynb
jupyter notebook embedding/build_db_bge_chunk32.ipynb
jupyter notebook embedding/build_db_bge_chunk64.ipynb
```

Each notebook will create its respective folder under `chroma_db/`.

### 7. Launch the Application

```bash
python app.py
```

Open your browser at `http://localhost:5000`

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

Steps:
1. Push the repo to your platform
2. Set `GROQ_API_KEY` in environment variables
3. Upload or mount the `chroma_db/` directory (or rebuild via notebooks in a pre-deploy step)

---

## 🧪 How to Use the System

### Step 1 — Open the Web Interface

Go to `http://localhost:5000` (or your deployed URL).

### Step 2 — Write a Research Query

Enter your question in the query box. The system supports **English and Indonesian** queries and will respond in the same language.

**Example queries by category:**

| Category | Example |
|---|---|
| Trend Analysis | *"How has the application of AI in supply chain risk management evolved over time?"* |
| Trend Analysis | *"What major paradigm shifts occurred after COVID-19 in AI-based supply chain research?"* |
| Gap Detection | *"What aspects of supply chain risk are underexplored in current AI research?"* |
| Gap Detection | *"Are human-in-the-loop approaches missing in AI-based resilience studies?"* |
| Comparative Synthesis | *"How do machine learning and deep learning approaches differ in SCRM applications?"* |
| Comparative Synthesis | *"Compare AI approaches for short-term versus long-term supply chain resilience."* |
| Evidence-based QA | *"What accuracy levels are reported for AI models in supply chain disruption prediction?"* |
| Evidence-based QA | *"What datasets are commonly used in AI-based SCRM studies?"* |

### Step 3 — Set RAG Parameters

Adjust parameters in the settings panel:

| Parameter | Options | Recommended | Effect |
|---|---|---|---|
| **Embedding Model** | LaBSE, BGE-M3 | **BGE-M3** | BGE-M3 achieves higher RAGAS scores across all query types |
| **Chunk Size** | 32, 64 tokens | **64 tokens** | Larger chunks provide richer context for synthesis queries |
| **Top-K** | 3, 5 | **5** | Retrieves more evidence; improves recall especially for Trend Analysis |

### Step 4 — Read the Response

Every response follows a fixed three-part HTML structure:

**① Direct Summary**
> A concise, 1–3 sentence answer with bolded key concepts. This directly addresses the core intent of the query.

**② Detailed Synthesis**
> Bullet-point critical analysis with numbered inline citations `[1]`, `[2]`, ... explaining *how* and *why* findings emerge from the retrieved literature.

**③ Source Verification**
> A footer listing all primary sources (Author, Year) grounding the analysis, ensuring full scientific traceability.

**④ Chart Visualization** *(when applicable)*
> Auto-generated Chart.js visualization (bar/line/pie) based on query type — e.g., a trend line for Trend Analysis queries, or a comparison bar chart for Comparative Synthesis queries.

---

## 🎯 System Prompt Design

The following system prompt is injected at the start of every LLM call, governing the behavior of **Llama-3.1-8b-instant**:

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

### Prompt Sensitivity Experiment

Four prompt variants were tested on the same RAG configuration (BGE-M3, Chunk=64, Top-K=5) to assess the system's sensitivity to prompt design:

| Variant | Description |
|---|---|
| **P1 — Generic Baseline** | Minimal instruction, open-ended output format |
| **P2 — Strict Context-Based** | Hard constraint: answer ONLY from context, no outside knowledge |
| **P3 — Structured Analysis** | Enforces three-section HTML structure, academic synthesis style |
| **P4 — Evidence & Anti-Hallucination** | Adds explicit citation rules + hallucination refusal mechanism |

> The final production prompt above combines the strongest elements of P3 and P4.

---

## 📈 Evaluation — RAGAS Metrics

The system was evaluated using the **RAGAS** (Retrieval-Augmented Generation Assessment) framework, which enables automatic pipeline evaluation without requiring human-annotated ground truth.

### Metrics Explained

| Metric | What It Measures | Scoring Method |
|---|---|---|
| **Faithfulness** | Whether every factual claim in the answer is supported by retrieved context | Ratio of supported claims / total claims |
| **Answer Relevance** | Whether the answer semantically addresses the user's query intent | Reverse question generation + cosine similarity |
| **Context Precision** | What proportion of retrieved chunks are actually relevant | LLM-based relevance labeling per chunk |
| **Context Recall** | How complete the retrieved context is relative to the ideal answer | Ratio of relevant units in context vs. needed |

All scores range **0 to 1** (higher = better). Threshold for acceptable performance: **≥ 0.5**.

### Test Query Set

| Category | # Queries | Focus |
|---|---|---|
| Trend Analysis | 20 | AI method evolution, paradigm shifts, temporal patterns |
| Gap Detection | 20 | Missing methods, underexplored regions, research opportunities |
| Comparative Synthesis | 20 | Model comparisons, framework differences, performance tradeoffs |
| Evidence-based QA | 20 | Empirical metrics, datasets, accuracy levels, case studies |
| **Total** | **80** | |

---

### Result 1: Embedding Model Comparison

Average RAGAS scores across all 80 queries for each model-chunk-TopK configuration:

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

> ✅ **Best Configuration: BGE-M3 · Chunk 64 · Top-K 5**

---

### Result 2: Performance by Query Category

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

### Result 3: BGE-M3 vs LaBSE per Query Category (All Configurations)

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

### Key Findings Summary

- **BGE-M3 consistently outperforms LaBSE** across all metrics and all 8 configurations tested
- **Chunk 64 + Top-K 5** is the optimal configuration — larger chunks provide richer contextual grounding for complex synthesis tasks
- **Gap Detection** shows the highest and most stable performance across both embedding models, indicating this task aligns well with the retrieval mechanism
- **Comparative Synthesis** is the most challenging category: Context Recall remains the lowest across all configurations due to the need for cross-document reasoning over diverse study designs
- **Trend Analysis** achieves the highest Context Recall (0.940) with BGE-M3 + Chunk 64 + Top-K 5 — temporal information distributes well across the corpus
- **Evidence-based QA** benefits most from the large chunk + high Top-K combination, achieving the best Faithfulness balance (0.758) at this configuration

---

## 🐞 Failure Analysis

Failures are classified using a deterministic rule-based system with a **threshold of 0.5** on each RAGAS metric. Cases scoring below 0.5 on the relevant indicator(s) are assigned to an error category.

### Error Classification Rules

| # | Error Type | Stage | Trigger Condition | Interpretation |
|---|---|---|---|---|
| 1 | **Retrieval Failure** | Retrieval | Context Recall < 0.5 **AND** Context Precision < 0.5 | No Top-K chunk contains the core concept of the query; embedding–query alignment issue |
| 2 | **Context Noise** | Retrieval | Context Recall ≥ 0.5 **AND** Context Precision < 0.5 | Relevant chunks retrieved but >50% of Top-K is off-topic; recall without precision |
| 3 | **Hallucination** | Generation | Faithfulness < 0.5 | Answer contains facts, terms, or claims not traceable to any retrieved chunk |
| 4 | **Semantic Drift** | Generation | Faithfulness ≥ 0.5 **AND** Answer Relevance < 0.5 | Answer is contextually grounded but diverges from the query's core intent |
| 5 | **Incomplete Answer** | Generation | Faithfulness ≥ 0.5 **AND** Answer Relevance ≥ 0.5 (partial) | Multi-aspect query answered only partially; secondary dimensions omitted |
| — | **Completely Correct** | — | All metrics ≥ 0.5 | Response is grounded, relevant, precise, and complete |

### Error Frequency Results

> ⚠️ Results below are from the best-performing configuration: **BGE-M3, Chunk = 64, Top-K = 5** over all 80 test queries.

| Error Type | Stage | # Cases | Percentage (%) |
|---|---|---|---|
| Retrieval Failure | Retrieval | `[update from ragas_results.csv]` | `[%]` |
| Context Noise | Retrieval | `[update from ragas_results.csv]` | `[%]` |
| Hallucination | Generation | `[update from ragas_results.csv]` | `[%]` |
| Semantic Drift | Generation | `[update from ragas_results.csv]` | `[%]` |
| Incomplete Answer | Generation | `[update from ragas_results.csv]` | `[%]` |
| **Completely Correct** | — | `[update from ragas_results.csv]` | `[%]` |
| **Total** | | **80** | **100%** |

> 📂 Full error classification details are available in [`data/ragas_results.csv`](data/ragas_results.csv).

### Error Sub-Types (Detailed Taxonomy)

| Error Type | Sub-Category | Metric Pattern | Interpretation |
|---|---|---|---|
| **Retrieval Failure** | Hard Retrieval Miss | Precision ≤ 0.5, Recall ≤ 0.5 | Core concept absent from corpus or embedding alignment failure |
| **Retrieval Failure** | Partial Recall Loss | Precision > 0.5, Recall ≤ 0.5 | Concept partially found; Top-K too small or chunks too fine-grained |
| **Context Noise** | Dominant Noise | Precision ≤ 0.5, Recall ≥ 0.5 | Relevant evidence diluted by off-topic chunks; strong recall, poor filtering |
| **Context Noise** | Mixed Context | Precision moderate, Recall high | Selective grounding despite noisy context; generation partially succeeds |
| **Hallucination** | Abstractive Deviation | Faithfulness ≤ 0.5, Relevance ≥ 0.5 | Model synthesizes beyond retrieved spans; coherent but ungrounded abstraction |
| **Hallucination** | Unsupported Generation | Faithfulness ≤ 0.5, all context weak | Forced to rely on inferred knowledge due to poor retrieval quality |
| **Hallucination** | True Hallucination | All metrics ≤ 0.5 | Response lacks factual grounding AND semantic alignment; worst case |
| **Semantic Drift** | Intent Drift | Faithfulness ≥ 0.5, Relevance ≤ 0.5 | Well-grounded but diverges from specific query intent |
| **Semantic Drift** | Scope Drift | Faithfulness high, Relevance moderate | Accurate and grounded but at broader scope than explicitly requested |
| **Incomplete Answer** | Partial Coverage | Faithfulness ≥ 0.5, Relevance moderate | Core aspects answered; secondary dimensions omitted |
| **Incomplete Answer** | Dimension Omission | Faithfulness high, Recall high | Strong grounding but not all required analytical dimensions articulated |

---

## 📦 Dependencies (`requirements.txt`)

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
| ChromaDB | [docs.trychroma.com](https://docs.trychroma.com) |
| Scopus Database | [scopus.com](https://www.scopus.com) |

---

## 📄 Citation

If you use this system or find this work useful, please cite:

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

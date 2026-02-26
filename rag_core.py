import os
import chromadb
import numpy as np
import re
import ast
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# --- KONFIGURASI ---
PERSIST_DIRECTORY = "embedding/embedding/chroma_db/bge_m3/db_64"
COLLECTION_NAME = "jurnal_supply_chain"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
LLM_MODEL_NAME = "llama-3.1-8b-instant" 
TOP_K = 3
SIMILARITY_THRESHOLD = 0.30

# --- SYSTEM PROMPT (MERGED: Logic + HTML UI) ---
SYSTEM_INSTRUCTION = """
Role: You are the "SC-Literature Intelligence Assistant", an expert AI specialized in synthesizing scientific literature on Supply Chain and AI.

CORE OBJECTIVE: Answer the user's question using ONLY the provided Context Data.

---
### 🚨 1. LANGUAGE PROTOCOL (HIGHEST PRIORITY)
You must strictly follow the language of the **User Query**, regardless of the language in the Context Data.

* **CASE A: User asks in ENGLISH**
  - OUTPUT: **100% English**.
  - ACTION: Translate all findings from Indonesian context into English.

* **CASE B: User asks in INDONESIAN**
  - OUTPUT: **100% Indonesian**.
  - ACTION: Translate all findings from English context into Indonesian.
  - EXCEPTION: Keep technical terms (e.g., "Robustness", "RAG", "Bullwhip Effect") in English.

  
---
### ⛔ 2. STRICT FORMATTING RULES (CRITICAL)
1. **NO MARKDOWN HEADERS:** Do NOT use `###`, `##`, or `**Title**`.
2. **NO DUPLICATE TITLES:** Do NOT repeat "Direct Answer" or "Direct Summary". Use ONLY the HTML tags provided below.
3. **RAW HTML ONLY:** Your output must start directly with `<p>` and contain valid HTML tags.
4. **NEVER** mix languages. **NEVER** let the document language override the query language.


---
### 📝 3. REQUIRED HTML STRUCTURE
You must strictly follow this template:

**A. Direct Answer**
   - Provide a direct, concise answer (Max 3 sentences).
   - HTML STRUCTURE:
     <p class="font-bold text-blue-700 mb-2 underline underline-offset-4 decoration-blue-200">Direct Summary:</p>
     <p class="mb-4">Your direct answer here with <strong>key concepts</strong> bolded.</p>

**B. Critical Analysis**
   - Provide detailed synthesis using bullet points. Explain *how* and *why*.
   - HTML STRUCTURE:
     <p class="font-bold text-slate-700 mb-1">Detailed Synthesis:</p>
     <ul class="list-disc pl-4 space-y-2 mb-4">
       <li>Point one explanation derived from context [1].</li>
       <li>Point two explanation with citation [2].</li>
     </ul>

**C. Source Verification**
   - List primary authors used in the analysis.
   - HTML STRUCTURE:
     <p class="text-[11px] italic text-slate-500 bg-slate-50 p-2 rounded border-l-2 border-blue-400">Analysis grounded in primary sources: [authors, Year].</p>

---
### ⛔ 3. STRICT CONSTRAINTS
1. **No Outside Knowledge:** If the answer is not in the context, output: "<p class='text-red-500 font-bold'>Information not available in the provided documents.</p>"
2. **Citation Rule:** Every single claim in Section B must have a citation number `[i]`.
3. **Neutral Tone:** Maintain an objective, academic tone.
4. **No Hallucinations:** If the answer cannot be found, honestly state it.
"""


class RAGSystem:
    def __init__(self):
        print("[INIT] Loading Models & Connecting DB...")
        
        # Cek API Key
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY tidak ditemukan di file .env")

        # Load Resources
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
        self.chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
        self.groq_client = Groq(api_key=self.api_key)
        
        print("[INIT] System Ready.")

    def clean_html_tags(self, text):
        if not text: return ""
        return re.sub(r'\s+', ' ', re.sub(re.compile('<.*?>'), '', str(text))).strip()

    def format_metadata(self, meta):
        if not meta: return "Unknown", "n.d.", "Untitled"
        penulis = meta.get('penulis', meta.get('authors', 'Unknown authors'))
        
        # Handle format list string
        if isinstance(penulis, str) and penulis.startswith('['):
            try: penulis = ast.literal_eval(penulis)
            except: pass
        if isinstance(penulis, list): penulis = ", ".join(map(str, penulis))
        
        return str(penulis), str(meta.get('tahun', meta.get('year', 'n.d.'))), str(meta.get('judul', meta.get('title', 'Untitled')))

    def process_query(self, query):
        # 1. Retrieval
        print(f"[RAG] Processing query: {query}")
        q_vec = self.embed_model.encode(query, convert_to_tensor=False).tolist()
        results = self.collection.query(query_embeddings=[q_vec], n_results=TOP_K)

        context_str = ""
        sources_data = []

        # Parse hasil retrieval
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                dist = results['distances'][0][i]
                score = 1 - (dist / 2) 

                if score >= SIMILARITY_THRESHOLD:
                    raw_text = results['documents'][0][i]
                    meta = results['metadatas'][0][i]
                    
                    clean_text = self.clean_html_tags(raw_text)
                    authors, year, title = self.format_metadata(meta)
                    
                    # Format String yang Kaya Metadata
                    formatted_chunk = f"Title: {title}, Authors: {authors}, Year: {year}. Content: {clean_text}"

                    # Konteks untuk LLM (Prompt)
                    context_str += f"Doc [{i+1}] {formatted_chunk}\n\n"
                    
                    # Data source untuk UI
                    sources_data.append({
                        "id": i+1,
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "abstract": clean_text[:200] + "...",
                        "snippet": clean_text[:300] + "...",
                        "score": score
                    })

        # Jika tidak ada dokumen relevan
        if not context_str:
            return {
                "answer": "<p class='text-amber-600 font-bold'>No relevant academic documents found matching your query criteria (Similarity below threshold).</p>",
                "sources": []
            }

        # 2. Generation (LLM)
        # Hitung jumlah dokumen yang benar-benar di-retrieve
        num_docs = len(sources_data)
        
        full_prompt = (
            f"CONTEXT DATA:\n{context_str}\n\n"
            f"USER QUERY: {query}\n\n"
            f"CRITICAL INSTRUCTIONS:\n"
            f"1. Answer strictly in the same language as the USER QUERY (English query = English answer, Indonesian query = Indonesian answer)\n"
            f"2. Follow the HTML structure template exactly\n"
            f"3. In Detailed Synthesis: Provide EXACTLY {num_docs} bullet points (one for each document provided)\n"
            f"4. Each bullet point must synthesize findings from the corresponding document [1], [2], [3], etc.\n"
            f"5. Keep each bullet point concise (2-3 sentences) with proper citation"
        )
        
        try:
            completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": full_prompt}
                ],
                model=LLM_MODEL_NAME,
                temperature=0.0  # Strict factual
            )
            ans = completion.choices[0].message.content
        except Exception as e:
            ans = f"<p class='text-red-500'>LLM Generation Error: {str(e)}</p>"

        return {
            "answer": ans,
            "sources": sources_data
        }
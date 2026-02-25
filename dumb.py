import os
import chromadb
import numpy as np
import re
import ast
import datetime
import matplotlib
matplotlib.use('Agg') # Backend non-GUI agar server tidak crash
import matplotlib.pyplot as plt
import umap
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
# --- TAMBAHAN IMPORT UNTUK RAGAS ---
from datasets import Dataset 
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    ContextPrecision, 
    ContextRecall, 
    Faithfulness, 
    AnswerRelevancy
)
# Load environment variables
load_dotenv()
import math
from ragas.llms import LangchainLLMWrapper
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import warnings
# Supress warnings
warnings.filterwarnings('ignore')
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import gspread

# --- KONFIGURASI ---
# GANTI PATH INI SESUAI LOKASI CHROMA DB ANDA
PERSIST_DIRECTORY = r"D:\Penelitian supply chain\embedding\chroma_db\bge_m3\db_64" 
COLLECTION_NAME = "jurnal_supply_chain"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
LLM_MODEL_NAME = "llama-3.1-8b-instant" 
TOP_K = 3
SIMILARITY_THRESHOLD = 0.30
# --- KONFIGURASI YANG DIUPDATE ---
FOLDER_NAME_FORMAT = f"{EMBEDDING_MODEL_NAME.split('/')[-1]}_chunk64_top{TOP_K}"
# Note: Karena saya tidak tahu variabel CHUNK_SIZE di kode Anda, ganti 'ChunkUnknown' dengan variable chunk size Anda.

# ID Folder Induk (Dari link yang Anda kasih: 1PqJVOU_Vl8MFT-0DhMPiutW_U71_OwoV)
PARENT_DRIVE_FOLDER_ID = "1PqJVOU_Vl8MFT-0DhMPiutW_U71_OwoV"
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
        self.init_gsheet()
        
        print("[INIT] System Ready.")
    
    def init_gsheet(self):
        try:
            print("[INIT] Connecting to Google Services (User Mode)...")
            
            # SCOPES BARU (Yang menyebabkan error change scope jika token lama masih ada)
            SCOPES = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
                "https://www.googleapis.com/auth/drive.file"
            ]
            
            creds = None
            token_path = 'token.pickle'
            
            # 1. Coba Load Token Lama
            if os.path.exists(token_path):
                try:
                    with open(token_path, 'rb') as token:
                        creds = pickle.load(token)
                        
                    # Cek apakah scopes di token cocok dengan yang kita minta?
                    # Jika tidak cocok, google biasanya throw error nanti.
                    # Kita paksa validasi di sini:
                    if creds and creds.valid and not set(SCOPES).issubset(set(creds.scopes)):
                        raise Exception("Scope Mismatch") # Pemicu untuk login ulang
                        
                except Exception as e:
                    print(f"[INFO] Token lama tidak cocok atau rusak ({e}). Menghapus token...")
                    creds = None # Reset creds
            
            # 2. Jika Token Tidak Valid / Kosong / Mismatch -> LOGIN ULANG
            if not creds or not creds.valid:
                # Hapus file lama biar bersih
                if os.path.exists(token_path):
                    os.remove(token_path)
                    
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                    except:
                        creds = None # Refresh gagal, paksa login baru
                
                if not creds:
                    if not os.path.exists('client_secret.json'):
                        print("[ERROR] File 'client_secret.json' tidak ditemukan!")
                        return

                    # Buka Browser
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'client_secret.json', SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Simpan Token Baru
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)
                print("[INFO] Token baru berhasil dibuat.")

            # 3. Koneksi Layanan
            self.gs_client = gspread.authorize(creds)
            sheet_id = "1tri0q0D4ppRiHz0kIymhTjrOy4vIR4oWC8Mwg5-txKQ"
            self.sheet = self.gs_client.open_by_key(sheet_id).worksheet("BGE-M3 Top K = 3 , Chunk 64")
            
            self.drive_service = build('drive', 'v3', credentials=creds)
            
            print(f"[INIT] Connected as User. Ready to Upload.")

        except Exception as e:
            print(f"[ERROR] Connection Failed: {e}")
            # Hapus pickle jika error fatal
            if os.path.exists('token.pickle'):
                os.remove('token.pickle')
            self.sheet = None

    def get_or_create_folder(self, folder_name, parent_id):
        """Cari folder, kalau ga ada buat baru"""
        try:
            query = f"name = '{folder_name}' and '{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            results = self.drive_service.files().list(q=query, fields="files(id, name)").execute()
            files = results.get('files', [])

            if files:
                print(f"[DRIVE] Folder '{folder_name}' ditemukan.")
                return files[0]['id']
            else:
                print(f"[DRIVE] Membuat folder baru: '{folder_name}'...")
                file_metadata = {
                    'name': folder_name,
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [parent_id]
                }
                folder = self.drive_service.files().create(body=file_metadata, fields='id').execute()
                return folder.get('id')
        except Exception as e:
            print(f"[DRIVE ERROR] Gagal handle folder: {e}")
            return parent_id # Fallback ke folder induk

    def find_subfolder(self, folder_name, parent_id):
        """Mencari apakah subfolder sudah ada (Tanpa membuat baru)"""
        try:
            query = f"name = '{folder_name}' and '{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            results = self.drive_service.files().list(q=query, fields="files(id, name)").execute()
            files = results.get('files', [])
            if files:
                return files[0]['id'] # Kembalikan ID jika ketemu
            return None # Kembalikan None jika tidak ada
        except:
            return None

    def upload_image_to_drive(self, local_path, query_text):
        """
        Upload Cerdas: Cek subfolder dulu. Jika ada masuk situ, jika tidak masuk induk.
        """
        if not self.sheet: return ""
        
        try:
            # 1. Perbaikan Path
            safe_path = local_path.lstrip('/')
            if not os.path.exists(safe_path) and os.path.exists(f"static/{safe_path}"):
                safe_path = f"static/{safe_path}"

            if not os.path.exists(safe_path):
                print(f"[UPLOAD ERROR] File tidak ditemukan: {safe_path}")
                return ""

            # 2. LOGIKA CERDAS: Tentukan Tujuan Upload
            # Ganti FOLDER_NAME_FORMAT sesuai variabel config Anda
            # Contoh nama folder: "BAAI_bge-m3_Chunk512_Top5"
            target_folder_name = FOLDER_NAME_FORMAT 
            
            # Cari apakah folder spesifik itu ada?
            found_folder_id = self.find_subfolder(target_folder_name, PARENT_DRIVE_FOLDER_ID)
            
            if found_folder_id:
                # KASUS A: Folder ditemukan! Upload ke dalam subfolder biar rapi.
                print(f"[DRIVE] Folder '{target_folder_name}' ditemukan. Upload ke sana.")
                final_folder_id = found_folder_id
            else:
                # KASUS B: Folder belum dibuat user. Upload ke Induk saja biar gak Error.
                print(f"[DRIVE] Folder '{target_folder_name}' tidak ditemukan. Upload ke Induk.")
                final_folder_id = PARENT_DRIVE_FOLDER_ID

            # 3. Proses Upload
            safe_filename = "".join([c if c.isalnum() else "_" for c in query_text[:50]]) + ".png"
            
            file_metadata = {
                'name': safe_filename, 
                'parents': [final_folder_id] 
            }
            media = MediaFileUpload(safe_path, mimetype='image/png')
            
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            file_id = file.get('id')
            
            # 4. Set Permission Public
            self.drive_service.permissions().create(
                fileId=file_id,
                body={'type': 'anyone', 'role': 'reader'}
            ).execute()

            return f"https://drive.google.com/uc?export=view&id={file_id}"

        except Exception as e:
            print(f"[UPLOAD ERROR] Detail: {e}")
            return ""

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

 # --- METHOD RAGAS (UPDATED: FIX ERROR 400 GROQ) ---
 # --- METHOD RAGAS (FIX: MONKEY PATCHING UNTUK GROQ) ---
    def calculate_ragas_metrics(self, query, contexts, answer):
        print("[RAGAS] Calculating metrics sequentially...")
        
        try:
            # 1. Setup Resource
            groq_chat = ChatGroq(model=LLM_MODEL_NAME, temperature=0, api_key=self.api_key, max_tokens=4096)
            
            # Buat instance normal
            ragas_llm = LangchainLLMWrapper(groq_chat)
            
            # --- MONKEY PATCHING (SOLUSI PENGGANTI CLASS WRAPPER) ---
            # Kita 'bajak' metode generate milik instance ini agar selalu n=1
            original_generate = ragas_llm.generate

            def forceful_generate(prompts, n=1, **kwargs):
                # Paksa n=1 agar Groq tidak error 400, abaikan permintaan n dari Ragas
                return original_generate(prompts, n=1, **kwargs)
            
            # Tempelkan fungsi baru ke instance
            ragas_llm.generate = forceful_generate
            # --------------------------------------------------------

            ragas_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

            # 2. Siapkan Dataset
            data = {
                'question': [query],
                'answer': [answer],
                'contexts': [contexts], 
                'ground_truth': [answer]
            }
            dataset = Dataset.from_dict(data)

            # 3. Daftar Metrik
            metrics_map = {
                "context_precision": ContextPrecision(llm=ragas_llm),
                "context_recall": ContextRecall(llm=ragas_llm),
                "faithfulness": Faithfulness(llm=ragas_llm),
                "answer_relevancy": AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
            }

            results_accumulator = {}

            # 4. LOOPING: Jalankan SATU PER SATU
            for name, metric_obj in metrics_map.items():
                try:
                    print(f"   ... calculating {name} ...")
                    res = evaluate(
                        dataset=dataset,
                        metrics=[metric_obj],
                        raise_exceptions=False
                    )
                    
                    # Ambil nilai dengan aman
                    try:
                        val = res[name] 
                    except KeyError:
                        val = list(res.values())[0] if len(res) > 0 else 0.0

                    if isinstance(val, (list, np.ndarray)):
                        val = val[0] if len(val) > 0 else 0.0
                    
                    score_float = float(val)
                    results_accumulator[name] = score_float if not math.isnan(score_float) else 0.0
                    
                except Exception as e_metric:
                    print(f"   [SKIP] Error calculating {name}: {e_metric}")
                    results_accumulator[name] = 0.0

            return results_accumulator

        except Exception as e:
            print(f"[RAGAS FATAL ERROR] {e}")
            return None
    

    def generate_umap(self, query, query_vec, results, filename):
        """
        Membuat plot UMAP dengan gaya Heatmap & Anotasi Isi Teks
        (REVISI: Limit dihapus agar memetakan seluruh database)
        """
        try:
            # --- REVISI: Hapus limit=700 agar mengambil SEMUA data ---
            # Tanpa limit, chroma akan mengambil seluruh dokumen dalam collection
            print("[UMAP] Fetching ALL data points from DB (No Limit)...")
            db_data = self.collection.get(include=['embeddings', 'documents']) 
            
            doc_embeddings = np.array(db_data.get('embeddings'))
            doc_texts = db_data.get('documents') 
            
            # Cek jika data terlalu sedikit untuk di-plot
            if len(doc_embeddings) < 5: 
                print("[UMAP SKIP] Data terlalu sedikit (<5).")
                return None
            
            # 1. Hitung titik tengah (centroid) dari seluruh data
            centroid = np.mean(doc_embeddings, axis=0)
            
            # 2. Hitung jarak setiap dokumen ke titik tengah
            distances = np.linalg.norm(doc_embeddings - centroid, axis=1)
            
            # 3. Tentukan batas ambang (Threshold). 
            # Kita akan membuang 2% data terjauh (98th percentile).
            # Jika masih ada titik nyasar, turunkan jadi 95.
            threshold = np.percentile(distances, 98) 
            
            # 4. Buat "Masker" untuk memilih data yang BAGUS saja (jarak < threshold)
            mask = distances < threshold
            
            # 5. Terapkan filter
            doc_embeddings = doc_embeddings[mask]
            
            # Hati-hati: List Python tidak support boolean masking langsung, jadi pakai list comprehension
            doc_texts = [text for i, text in enumerate(doc_texts) if mask[i]]

            # 2. HITUNG SIMILARITY (Tetap sama)
            query_vec_np = np.array(query_vec).reshape(1, -1)
            similarities = cosine_similarity(query_vec_np, doc_embeddings)[0]
            
            # Cari dokumen dengan skor tertinggi di populasi INI
            top_idx = np.argmax(similarities)
            top_score = similarities[top_idx]
            
            full_text = doc_texts[top_idx]
            display_text = (full_text[:60] + '...') if len(full_text) > 60 else full_text
            display_text = " ".join(display_text.split())

            # 3. Proses UMAP (Tetap sama)
            combined = np.vstack([doc_embeddings, query_vec])
            
            # Adaptasi n_neighbors jika data masih sedikit meski sudah all
            n_neighbors = min(15, len(combined) - 2)
            
            reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=0.1, metric='cosine', random_state=42)
            embedding_2d = reducer.fit_transform(combined)

            doc_coords = embedding_2d[:-1]
            query_coord = embedding_2d[-1]
            top_match_coord = doc_coords[top_idx]

            # 4. PLOTTING (Tetap sama)
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sc = ax.scatter(doc_coords[:, 0], doc_coords[:, 1], 
                            c=similarities, cmap='turbo', 
                            s=60, alpha=0.9, edgecolors='none')
            
            ax.scatter(query_coord[0], query_coord[1], 
                       c='white', s=400, marker='*', 
                       edgecolors='red', linewidth=2, zorder=10, label='User Query')
                       

            # Anotasi Query
            display_query = query[:25] + "..." if len(query) > 25 else query
            ax.annotate(f"❓ ANDA: \"{display_query}\"", 
                        xy=(query_coord[0], query_coord[1]),
                        xytext=(20, 20), textcoords='offset points',
                        color='white', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.4", fc="#b91c1c", ec="white", alpha=0.9),
                        arrowprops=dict(arrowstyle="-", color="white", alpha=0.5))

            # Anotasi Dokumen Paling Mirip
            ax.annotate(f"🥇 RANK #1 ({top_score:.2f})\n\"{display_text}\"", 
                        xy=(top_match_coord[0], top_match_coord[1]),
                        xytext=(-20, -40), textcoords='offset points',
                        color='cyan', fontsize=8, fontweight='bold', family='monospace',
                        bbox=dict(boxstyle="round,pad=0.4", fc="black", ec="cyan", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", color="cyan", connectionstyle="arc3,rad=0.2"))

            cbar = plt.colorbar(sc)
            cbar.set_label('Tingkat Relevansi (Merah = Sangat Cocok)', rotation=270, labelpad=20, color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
# 1. Buat versi pendek dari query agar judul tidak kepanjangan
            short_query = query[:50] + "..." if len(query) > 50 else query

            # 2. Set judul menggunakan query tersebut
            ax.set_title(f"UMAP Projection for: \"{short_query}\"", color='white', fontsize=12, fontweight='bold', pad=15)
            ax.axis('off')
            
            filepath = os.path.join('static', 'temp_images', filename)
            plt.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='black')
            plt.close()
            
            return f"/static/temp_images/{filename}"

        except Exception as e:
            print(f"UMAP Error: {e}")
            return None
        
    def log_to_sheet(self, query, answer, sources, metrics, image_path):
        if not self.sheet:
            print("[SKIP] Tidak bisa simpan ke Sheet karena koneksi gagal.")
            return

        try:
            # 1. Hitung Nomor Urut
            all_values = self.sheet.get_all_values()
            current_row_count = len(all_values)
            
            # Jika sheet kosong banget (belum ada header), anggap row 1
            if current_row_count == 0: 
                next_no = 1
            else:
                # Jika sudah ada header, nomor urut adalah baris saat ini (karena row 1 itu header)
                # Contoh: Header (row 1). Next data masuk row 2. ID nya harus 1.
                # Jadi ID = current_row_count. 
                # Kalau Header + Data1 (row 2). Next data masuk row 3. ID harus 2.
                next_no = current_row_count
            
            # 2. Bersihkan Format Answer
            clean_answer = self.clean_html_tags(answer)

            # 3. Format Top K Sources (Sesuai Request: Rank, Chunk, Title, Score)
            formatted_sources = []
            for idx, s in enumerate(sources):
                # Buat blok text untuk setiap source
                # Menggunakan s['snippet'] dan s['score'] yang dikirim dari process_query
                entry = (
                    f"Rank #{idx + 1}\n"
                    f"Potongan Chunk: {s.get('snippet', '-')}\n"
                    f"Title: {s['title']}\n"
                    f"Authors: {s['authors']}\n"
                    f"Score: {s.get('score', 0):.4f}" 
                )
                formatted_sources.append(entry)
            
            # Gabungkan dengan pemisah double enter agar rapi di dalam sel (Wrap Text)
            source_str = "\n\n-----------------\n\n".join(formatted_sources)

            # 4. Handle Image
            # Catatan: Karena ini file lokal, Google Sheet tidak bisa merender gambar dari "D:/" atau "localhost".
            # Gambar hanya akan muncul jika URL-nya publik (http...).
            # Namun sesuai request "jangan file path", kita masukkan sebagai Formula IMAGE.
            # Kita ganti backslash windows (\) dengan slash (/)
            print("[DRIVE] Sedang mengupload gambar UMAP...")
            drive_image_url = self.upload_image_to_drive(image_path, query)
            
            if drive_image_url:
                image_formula = f'=IMAGE("{drive_image_url}")'
            else:
                image_formula = "Upload Failed"

            # 5. Susun Data Row
            row_data = [
                next_no,                            # No
                query,                              # Question
                clean_answer,                       # Answer
                source_str,                         # Top_K_Sources (Format Baru)
                metrics.get('faithfulness', 0),     # Faithfulness
                metrics.get('answer_relevancy', 0), # Answer_Relevancy
                metrics.get('context_precision', 0),# Context_Precision
                metrics.get('context_recall', 0),   # Context_Recall
                image_formula                       # Image (Formula)
            ]

            # 6. Append ke Sheet dengan Opsi USER_ENTERED agar formula terbaca
            self.sheet.append_row(row_data, value_input_option='USER_ENTERED')
            print(f"[GSHEET] Sukses menyimpan log ke baris {next_no}")

        except Exception as e:
            print(f"[GSHEET ERROR] Gagal menyimpan data: {e}")
        
    def process_query(self, query):
        # 1. Retrieval
        print(f"[RAG] Processing query: {query}")
        q_vec = self.embed_model.encode(query, convert_to_tensor=False).tolist()
        results = self.collection.query(query_embeddings=[q_vec], n_results=TOP_K)

  # ... (bagian query chroma db sebelumnya) ...

        context_str = ""
        sources_data = []
        list_contexts_for_ragas = [] # List baru khusus buat Ragas

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
                    
                    # 1. Format String yang Kaya Metadata (Ini kuncinya!)
                    # Kita masukkan info penulis ke dalam teks yang akan dibaca Ragas
                    formatted_chunk = f"Title: {title}, Authors: {authors}, Year: {year}. Content: {clean_text}"
                    
                    # Simpan ke list untuk Ragas
                    list_contexts_for_ragas.append(formatted_chunk)

                    # Konteks untuk LLM (Prompt)
                    context_str += f"Doc [{i+1}] {formatted_chunk}\n\n"
                    
                    # Data source untuk UI
                    sources_data.append({
                        "id": i+1,
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "abstract": clean_text[:200] + "...",
                        "snippet": clean_text[:300] + "...", # Potongan chunk lebih panjang utk Excel
                        "score": score
                    })


        # Jika tidak ada dokumen relevan
        if not context_str:
            return {
                "answer": "<p class='text-amber-600 font-bold'>No relevant academic documents found matching your query criteria (Similarity below threshold).</p>",
                "sources": [],
                "image": None
            }

        # 2. Generation (LLM)
# --- UPDATE: Tambahkan instruksi penegas di akhir prompt ---
        full_prompt = (
            f"CONTEXT DATA:\n{context_str}\n\n"
            f"USER QUERY: {query}\n\n"
            f"INSTRUCTION: Please answer the query strictly in the same language as the USER QUERY above. "
            f"If the query is English, answer in English. If Indonesian, answer in Indonesian."
        )
        
        try:
            completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": full_prompt}
                ],
                model=LLM_MODEL_NAME,
                temperature=0.0 # Strict factual
            )
            ans = completion.choices[0].message.content
        except Exception as e:
            ans = f"<p class='text-red-500'>LLM Generation Error: {str(e)}</p>"

        # 3. Visualisasi UMAP
        img_filename = f"umap_{datetime.datetime.now().strftime('%H%M%S')}.png"
        img_url = self.generate_umap(query, [q_vec], results, img_filename)

        # 4. RAGAS EVALUATION
        ragas_scores = {
            "context_precision": 0.0, "context_recall": 0.0, 
            "faithfulness": 0.0, "answer_relevancy": 0.0
        }

        if list_contexts_for_ragas:
            try:
                # Panggil fungsi sequential yang baru
                # Fungsi ini sekarang me-return dict { 'faithfulness': 0.8, ... } yang sudah bersih
                computed_scores = self.calculate_ragas_metrics(query, list_contexts_for_ragas, self.clean_html_tags(ans))
                
                if computed_scores:
                    # Update skor default dengan hasil perhitungan
                    ragas_scores.update(computed_scores)
                    
                    # Rounding agar rapi
                    for k, v in ragas_scores.items():
                        ragas_scores[k] = round(v, 3)
                        
                    print(f"[RAGAS] Final Scores: {ragas_scores}")

            except Exception as e:
                print(f"[RAGAS FAIL] Skipped: {e}")
        
        self.log_to_sheet(
            query=query,
            answer=ans,
            sources=sources_data,
            metrics=ragas_scores,
            image_path=img_url
        )
        
        return {
            "answer": ans,
            "sources": sources_data,
            "image": img_url,
            "metrics": ragas_scores
        }
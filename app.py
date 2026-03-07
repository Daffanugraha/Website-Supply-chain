from flask import Flask, render_template, request, jsonify
from rag_core import RAGSystem
import os
import zipfile
import gdown
import glob
import shutil

# ==========================================
# FUNGSI PENJEMPUT DATABASE DARI GOOGLE DRIVE
# ==========================================
def siapkan_database():
    folder_db = 'embedding/chroma_db/bge_m3/db_64'
    file_zip = 'embedding.zip'
    file_id = '1lb7iOt_z56NXAJ54xgy_PG_VHD2zNgGF'
    url = f'https://drive.google.com/uc?id={file_id}'

    if os.path.exists(folder_db) and os.path.exists(os.path.join(folder_db, 'chroma.sqlite3')):
        print("✅ Database ChromaDB sudah tersedia dan siap.")
        return

    print("⏳ Database ChromaDB belum ditemukan. Mengunduh dari Google Drive...")
    temp_dir = 'embedding/temp_extract'

    try:
        os.makedirs('embedding', exist_ok=True)
        gdown.download(url, file_zip, quiet=False)

        print("📦 Mengekstrak database...")
        with zipfile.ZipFile(file_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Debug: tampilkan semua file hasil ekstrak
        print("🔍 Isi hasil ekstrak:")
        for root, dirs, files in os.walk(temp_dir):
            for f in files:
                print(f"   {os.path.join(root, f)}")

        # Cari semua chroma.sqlite3, prioritaskan yang ada di folder 'db_64'
        db_files = glob.glob(f'{temp_dir}/**/chroma.sqlite3', recursive=True)
        print(f"🔍 Ditemukan chroma.sqlite3 di: {db_files}")

        actual_db_dir = None
        for db_file in db_files:
            parent = os.path.dirname(db_file)
            if os.path.basename(parent) == 'db_64':
                actual_db_dir = parent
                break

        # Fallback: ambil yang pertama jika tidak ada yang namanya db_64
        if actual_db_dir is None and db_files:
            actual_db_dir = os.path.dirname(db_files[0])

        if actual_db_dir is None:
            print("❌ ERROR: File chroma.sqlite3 tidak ditemukan di dalam ZIP!")
            return

        print(f"✅ Menggunakan folder sumber: {actual_db_dir}")
        print(f"   Isinya: {os.listdir(actual_db_dir)}")

        # Hapus target lama jika ada, lalu salin SELURUH folder (termasuk uuid subfolder)
        if os.path.exists(folder_db):
            shutil.rmtree(folder_db)

        os.makedirs(os.path.dirname(folder_db), exist_ok=True)
        shutil.copytree(actual_db_dir, folder_db)

        print(f"✅ Database berhasil diposisikan!")
        print(f"   Isi {folder_db}: {os.listdir(folder_db)}")

    except Exception as e:
        print(f"❌ Gagal mengunduh/mengekstrak database: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if os.path.exists(file_zip):
            os.remove(file_zip)


# Jalankan setup database sebelum Flask start
siapkan_database()

# ==========================================
# APLIKASI FLASK
# ==========================================
app = Flask(__name__)

print("⏳ Initializing RAG Engine. Please wait...")
try:
    rag_engine = RAGSystem()
    print("✅ RAG Engine initialized successfully.")
except Exception as e:
    rag_engine = None
    print(f"❌ Failed to initialize RAG Engine: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global rag_engine
    if not rag_engine:
        return jsonify({"error": "System initialization failed. Check server logs."}), 503

    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        result = rag_engine.process_query(query)
        return jsonify(result)
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_corpus_stats():
    try:
        results = rag_engine.collection.get(include=["metadatas"])
        unique_papers = set()
        if results and 'metadatas' in results and results['metadatas']:
            for meta in results['metadatas']:
                if meta:
                    title = meta.get('judul', meta.get('title', ''))
                    if title and title.strip().lower() != 'untitled':
                        unique_papers.add(title.strip().lower())
        return jsonify({"total_papers": len(unique_papers)})
    except Exception as e:
        print(f"Error fetching stats: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # HF Spaces kadang memberikan port lewat environment variable, default ke 7860
    port = int(os.environ.get("PORT", 7860)) 
    app.run(host='0.0.0.0', port=port, debug=False)
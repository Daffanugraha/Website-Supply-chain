from flask import Flask, render_template, request, jsonify
from rag_core import RAGSystem
import os
import zipfile
import gdown

# ==========================================
# FUNGSI PENJEMPUT DATABASE DARI GOOGLE DRIVE
# ==========================================
def siapkan_database():
    folder_db = 'embedding/chroma_db'
    file_zip = 'chroma_db.zip'
    
    # ⚠️ GANTI ID INI DENGAN ID FILE .ZIP DI GOOGLE DRIVE KAMU
    # Menggunakan ID dari link Google Drive kamu
    file_id = '1lb7iOt_z56NXAJ54xgy_PG_VHD2zNgGF'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # Mengecek apakah database sudah ada (agar tidak download berulang kali)
    if not os.path.exists(folder_db):
        print("⏳ Database ChromaDB belum ditemukan. Mengunduh dari Google Drive...")
        
        # Buat folder embedding jika belum ada
        os.makedirs('embedding', exist_ok=True)
        
        try:
            # Download file zip dari Drive
            gdown.download(url, file_zip, quiet=False)
            
            print("📦 Mengekstrak database...")
            with zipfile.ZipFile(file_zip, 'r') as zip_ref:
                # Ekstrak ke dalam folder embedding
                zip_ref.extractall('embedding/')
                
            # Hapus file zip setelah diekstrak agar server tidak penuh
            os.remove(file_zip)
            print("✅ Database berhasil disiapkan!")
        except Exception as e:
            print(f"❌ Gagal mengunduh/mengekstrak database: {e}")
    else:
        print("✅ Database ChromaDB sudah tersedia.")

# Panggil fungsi ini tepat saat aplikasi pertama kali dijalankan (sebelum Flask jalan)
siapkan_database()

# ==========================================
# APLIKASI FLASK
# ==========================================
app = Flask(__name__)

# Variabel global untuk instance RAG
rag_engine = None

@app.before_request
def initialize_system():
    """Inisialisasi sistem RAG hanya sekali saat request pertama masuk"""
    global rag_engine
    if rag_engine is None:
        # Buat folder gambar jika belum ada
        if not os.path.exists('static/temp_images'):
            os.makedirs('static/temp_images')
        
        try:
            rag_engine = RAGSystem()
            print("✅ RAG Engine initialized successfully.")
        except Exception as e:
            print(f"❌ Failed to initialize RAG Engine: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global rag_engine
    if not rag_engine:
        return jsonify({"error": "System initializing, please try again in a moment."}), 503

    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # Panggil otak RAG
        result = rag_engine.process_query(query)
        return jsonify(result)
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Debug=False sangat disarankan untuk produksi/deployment
    app.run(debug=False, port=5000)
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
    folder_db = 'embedding/chroma_db'
    file_zip = 'chroma_db.zip'
    
    # ID dari link Google Drive kamu
    file_id = '1lb7iOt_z56NXAJ54xgy_PG_VHD2zNgGF'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # Mengecek apakah database sudah benar-benar ada dan lengkap
    if not os.path.exists(folder_db) or not os.path.exists(os.path.join(folder_db, 'chroma.sqlite3')):
        print("⏳ Database ChromaDB belum ditemukan. Mengunduh dari Google Drive...")
        
        os.makedirs('embedding', exist_ok=True)
        temp_dir = 'embedding/temp_extract'
        
        try:
            # Download file zip
            gdown.download(url, file_zip, quiet=False)
            
            print("📦 Mengekstrak database...")
            with zipfile.ZipFile(file_zip, 'r') as zip_ref:
                # Ekstrak ke folder sementara (agar tidak berantakan)
                zip_ref.extractall(temp_dir)
                
            print("🔍 Mencari lokasi file chroma.sqlite3...")
            # Radar otomatis pencari file database
            db_files = glob.glob(f'{temp_dir}/**/chroma.sqlite3', recursive=True)
            
            if db_files:
                # Ambil folder tempat file sqlite3 itu berada
                actual_db_dir = os.path.dirname(db_files[0])
                
                # Buat folder target jika belum ada
                os.makedirs(folder_db, exist_ok=True)
                
                # Pindahkan semua isinya ke embedding/chroma_db
                for filename in os.listdir(actual_db_dir):
                    source = os.path.join(actual_db_dir, filename)
                    destination = os.path.join(folder_db, filename)
                    # Timpa jika ada file lama
                    if os.path.exists(destination):
                        if os.path.isdir(destination): shutil.rmtree(destination)
                        else: os.remove(destination)
                    shutil.move(source, destination)
                
                print("✅ Database berhasil diposisikan ke jalur yang benar!")
            else:
                print("❌ ERROR: File chroma.sqlite3 tidak ditemukan di dalam file ZIP!")

            # Bersihkan file sisa zip dan folder sementara
            shutil.rmtree(temp_dir, ignore_errors=True)
            if os.path.exists(file_zip):
                os.remove(file_zip)
                
        except Exception as e:
            print(f"❌ Gagal mengunduh/mengekstrak database: {e}")
    else:
        print("✅ Database ChromaDB sudah tersedia dan siap.")

# Jalankan fungsi
siapkan_database()

# ==========================================
# APLIKASI FLASK
# ==========================================
app = Flask(__name__)

# Inisialisasi RAG Engine saat aplikasi start (Global Scope)
print("⏳ Initializing RAG Engine. Please wait...")
try:
    rag_engine = RAGSystem()
    print("✅ RAG Engine initialized successfully.")
except Exception as e:
    rag_engine = None
    print(f"❌ Failed to initialize RAG Engine: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global rag_engine
    if not rag_engine:
        # Jika gagal saat startup, pesan ini akan muncul, 
        # dan kamu harus cek "Logs" di Hugging Face Spaces.
        return jsonify({"error": "System initialization failed. Check server logs."}), 503

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
    # Debug=False sangat disarankan untuk produksi/deployment di Hugging Face
    app.run(debug=False, port=5000)
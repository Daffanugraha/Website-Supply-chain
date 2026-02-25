from flask import Flask, render_template, request, jsonify
from rag_core import RAGSystem
import os

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
    # Debug=True agar auto-reload saat koding
    app.run(debug=False, port=5000)
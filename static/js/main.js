document.addEventListener('DOMContentLoaded', () => {
    // Inisialisasi Icon
    lucide.createIcons();

    const messagesList = document.getElementById('messages-list');
    const chatInput = document.getElementById('chat-input');
    const sourcePanel = document.getElementById('source-panel');
    const sourceContent = document.getElementById('source-content');
    const chatContainer = document.getElementById('chat-container');

    // Fungsi Render Pesan ke Chat Bubble
// Fungsi Render Pesan ke Chat Bubble
    // PERBAIKAN: Tambahkan parameter 'metrics = null' di akhir
    window.appendMessage = function(role, content, sources = [], imageUrl = null, metrics = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `flex ${role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`;
        
        // --- HTML UNTUK METRICS ---
        let metricsHtml = '';
        if (metrics && role === 'assistant') {
            // Helper untuk membulatkan angka jika valid, atau tampilkan 0.000
            const fmt = (val) => (typeof val === 'number' ? val.toFixed(3) : '0.000');

            metricsHtml = `
                <div class="mt-3 pt-3 border-t border-slate-200 grid grid-cols-2 gap-2">
                    <div class="bg-slate-50 p-2 rounded border border-slate-100">
                        <div class="text-[9px] uppercase text-slate-400 font-bold">Faithfulness</div>
                        <div class="text-xs font-mono font-bold ${metrics.faithfulness > 0.7 ? 'text-green-600' : 'text-amber-500'}">
                            ${fmt(metrics.faithfulness)}
                        </div>
                    </div>
                    <div class="bg-slate-50 p-2 rounded border border-slate-100">
                        <div class="text-[9px] uppercase text-slate-400 font-bold">Ans Relevancy</div>
                        <div class="text-xs font-mono font-bold ${metrics.answer_relevancy > 0.5 ? 'text-blue-600' : 'text-amber-500'}">
                            ${fmt(metrics.answer_relevancy)}
                        </div>
                    </div>
                    <div class="bg-slate-50 p-2 rounded border border-slate-100">
                        <div class="text-[9px] uppercase text-slate-400 font-bold">Ctx Precision</div>
                        <div class="text-xs font-mono font-bold text-slate-700">
                            ${fmt(metrics.context_precision)}
                        </div>
                    </div>
                    <div class="bg-slate-50 p-2 rounded border border-slate-100">
                        <div class="text-[9px] uppercase text-slate-400 font-bold">Ctx Recall*</div>
                        <div class="text-xs font-mono font-bold text-slate-700">
                            ${fmt(metrics.context_recall)}
                        </div>
                    </div>
                </div>
                <div class="text-[8px] text-slate-300 italic mt-1 text-center">*Recall using synthetic GT</div>
            `;
        }

        let imageHtml = '';
        if (imageUrl) {
            imageHtml = `
                <div class="mt-4 border border-slate-200 rounded-xl overflow-hidden shadow-sm">
                    <div class="bg-slate-100 px-3 py-1 text-[10px] font-bold text-slate-500 uppercase border-b border-slate-200 flex justify-between items-center">
                        <span>Semantic Projection (UMAP)</span>
                        <a href="${imageUrl}" target="_blank" class="text-blue-500 hover:text-blue-700"><i data-lucide="maximize-2" class="w-3 h-3"></i></a>
                    </div>
                    <img src="${imageUrl}" class="w-full h-auto object-cover opacity-95 hover:opacity-100 transition-opacity bg-black">
                </div>
            `;
        }

        // Tampilan Role User vs AI
        const avatar = role === 'assistant' 
            ? `<div class="flex items-center gap-2 mb-2 ml-1"><div class="w-6 h-6 bg-blue-600 rounded flex items-center justify-center text-white text-[10px] font-bold">RAG</div><span class="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Verified Response</span></div>`
            : '';

        const bubbleStyle = role === 'user' 
            ? 'bg-blue-600 text-white rounded-tr-none' 
            : 'bg-white border border-slate-200 text-slate-800 rounded-tl-none';

        // HTML untuk sumber (Sources)
        let sourcesHtml = '';
        if (sources && sources.length > 0) {
            sourcesHtml = `
                <div class="mt-4 flex flex-wrap gap-2 pt-3 border-t border-slate-100">
                    <div class="text-[9px] font-bold text-slate-400 uppercase self-center bg-slate-100 px-2 py-1 rounded">Sources:</div>
                    ${sources.map((s) => `
                        <button onclick='openSourcePanel(${JSON.stringify(s).replace(/'/g, "&apos;")})' class="flex items-center gap-1 px-3 py-1.5 bg-white border border-slate-200 rounded-lg text-[10px] font-bold text-blue-700 hover:bg-blue-50 transition-all">
                            <span class="bg-blue-100 w-4 h-4 rounded flex items-center justify-center mr-1">[${s.id}]</span>
                            ${s.authors ? s.authors.substring(0, 15) : 'Unknown'}...
                        </button>
                    `).join('')}
                </div>
            `;
        }

        messageDiv.innerHTML = `
            <div class="max-w-[85%] flex flex-col ${role === 'user' ? 'items-end' : 'items-start'}">
                ${avatar}
                <div class="rounded-2xl px-5 py-4 text-sm leading-relaxed shadow-sm ${bubbleStyle}">
                    ${content}
                    ${imageHtml}
                    ${metricsHtml}
                </div>
                ${sourcesHtml}
            </div>
        `;
        
        messagesList.appendChild(messageDiv);
        lucide.createIcons(); // Refresh icons
        chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
    }
    
    // Animasi Loading
    window.showTyping = function() {
        const id = 'typing-indicator';
        const typingDiv = document.createElement('div');
        typingDiv.id = id;
        typingDiv.className = 'flex justify-start items-center gap-3 mb-4';
        typingDiv.innerHTML = `
            <div class="w-6 h-6 bg-slate-200 rounded animate-pulse"></div>
            <div class="bg-white border border-slate-200 rounded-2xl rounded-tl-none px-4 py-3 flex gap-1.5 shadow-sm">
                <span class="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce"></span>
                <span class="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce" style="animation-delay: 0.2s"></span>
                <span class="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce" style="animation-delay: 0.4s"></span>
            </div>
        `;
        messagesList.appendChild(typingDiv);
        chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
    }

    window.removeTyping = function() {
        const el = document.getElementById('typing-indicator');
        if (el) el.remove();
    }

    // Fungsi Kirim ke Python Backend
    window.handleSend = async function() {
        const text = chatInput.value.trim();
        if (!text) return;
        
        appendMessage('user', text);
        chatInput.value = '';
        showTyping();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: text })
            });
            const data = await response.json();
            
            removeTyping();
            
            // Di dalam window.handleSend ...
            if (data.error) {
                appendMessage('assistant', `<span class="text-red-500 font-bold">Error: ${data.error}</span>`);
            } else {
                // TAMBAHKAN argumen ke-4 (data.metrics)
                appendMessage('assistant', data.answer, data.sources, data.image, data.metrics);
            }
            
        } catch (error) {
            removeTyping();
            appendMessage('assistant', `<span class="text-red-500">Connection Error: Ensure Flask server is running.</span>`);
            console.error(error);
        }
    }

    window.handleQuickAnalysis = function(query) {
        chatInput.value = query;
        handleSend();
    }

    // Panel Sumber/Sitasi
    window.openSourcePanel = function(source) {
        sourcePanel.classList.remove('hidden');
        sourceContent.innerHTML = `
            <div class="mb-6">
                <span class="inline-block px-2 py-0.5 bg-blue-50 text-blue-600 text-[10px] font-bold rounded mb-2 uppercase tracking-widest">Full Title</span>
                <h4 class="text-xl font-bold text-slate-900 leading-tight">${source.title}</h4>
            </div>
            <div class="grid grid-cols-2 gap-6 mb-8 border-y border-slate-100 py-6">
                <div><span class="text-[10px] font-bold uppercase text-slate-400 block mb-1">authors</span><p class="text-sm font-bold text-slate-800">${source.authors}</p></div>
                <div><span class="text-[10px] font-bold uppercase text-slate-400 block mb-1">Year</span><p class="text-sm font-bold text-slate-800">${source.year}</p></div>
            </div>
            <div class="bg-slate-50 p-6 rounded-2xl border border-slate-100">
                <span class="text-[10px] font-bold uppercase text-blue-500 block mb-3">Content Snippet</span>
                <p class="text-sm text-slate-700 leading-relaxed italic">"${source.abstract}"</p>
            </div>
        `;
    }

    window.closeSourcePanel = function() { 
        sourcePanel.classList.add('hidden'); 
    }

    // Listener Enter Key
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });
});
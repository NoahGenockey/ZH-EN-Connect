// LinguaBridge Local - Web Frontend JavaScript

// ========== Configuration ==========
const API_BASE_URL = 'http://localhost:8000';
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000;

// ========== State Management ==========
const state = {
    translationCount: 0,
    cacheHits: 0,
    apiOnline: false,
    isTranslating: false
};

// ========== DOM Elements ==========
const elements = {
    sourceText: document.getElementById('sourceText'),
    targetText: document.getElementById('targetText'),
    translateBtn: document.getElementById('translateBtn'),
    clearBtn: document.getElementById('clearBtn'),
    pasteBtn: document.getElementById('pasteBtn'),
    copyBtn: document.getElementById('copyBtn'),
    downloadBtn: document.getElementById('downloadBtn'),
    sourceCount: document.getElementById('sourceCount'),
    targetCount: document.getElementById('targetCount'),
    statusMessage: document.getElementById('statusMessage'),
    translationCount: document.getElementById('translationCount'),
    cacheHits: document.getElementById('cacheHits'),
    apiStatus: document.getElementById('apiStatus'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    toast: document.getElementById('toast')
};

// ========== API Functions ==========

/**
 * Check API health status
 */
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (response.ok) {
            const data = await response.json();
            state.apiOnline = data.model_loaded;
            updateAPIStatus(true, data);
            return true;
        }
    } catch (error) {
        console.error('API health check failed:', error);
        updateAPIStatus(false);
        return false;
    }
}

/**
 * Translate text via API
 */
async function translateText(text, retries = 0) {
    try {
        const response = await fetch(`${API_BASE_URL}/translate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                use_cache: true
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Translation failed');
        }
        
        const data = await response.json();
        
        // Update statistics
        state.translationCount++;
        if (data.cached) {
            state.cacheHits++;
        }
        updateStatistics();
        
        return data;
    } catch (error) {
        // Retry logic
        if (retries < MAX_RETRIES) {
            console.log(`Retrying... (${retries + 1}/${MAX_RETRIES})`);
            await sleep(RETRY_DELAY);
            return translateText(text, retries + 1);
        }
        throw error;
    }
}

/**
 * Get cache statistics
 */
async function getCacheStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/cache/stats`);
        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.error('Failed to get cache stats:', error);
    }
    return null;
}

// ========== UI Functions ==========

/**
 * Update API status indicator
 */
function updateAPIStatus(online, data = null) {
    const statusDot = elements.apiStatus.querySelector('.status-dot');
    
    if (online) {
        statusDot.classList.add('online');
        elements.apiStatus.innerHTML = `
            <span class="status-dot online"></span> 
            ${data?.model_loaded ? 'Online' : 'Loading...'}
        `;
    } else {
        statusDot.classList.remove('online');
        elements.apiStatus.innerHTML = `
            <span class="status-dot"></span> Offline
        `;
        showToast('API server is offline. Please start the server.', 'error');
    }
}

/**
 * Update statistics display
 */
function updateStatistics() {
    elements.translationCount.textContent = state.translationCount;
    elements.cacheHits.textContent = state.cacheHits;
}

/**
 * Update character counts
 */
function updateCharCount(element, countElement) {
    const text = element.value.trim();
    const count = text.length;
    countElement.textContent = `${count} character${count !== 1 ? 's' : ''}`;
}

/**
 * Show loading overlay
 */
function showLoading() {
    elements.loadingOverlay.classList.add('active');
    elements.translateBtn.disabled = true;
    state.isTranslating = true;
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    elements.loadingOverlay.classList.remove('active');
    elements.translateBtn.disabled = false;
    state.isTranslating = false;
}

/**
 * Show toast notification
 */
function showToast(message, type = 'success') {
    elements.toast.textContent = message;
    elements.toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        elements.toast.classList.remove('show');
    }, 3000);
}

/**
 * Update status message
 */
function updateStatus(message, type = 'info') {
    elements.statusMessage.textContent = message;
    elements.statusMessage.style.color = 
        type === 'error' ? 'var(--error-color)' :
        type === 'success' ? 'var(--success-color)' :
        'var(--text-secondary)';
}

// ========== Event Handlers ==========

/**
 * Handle translation
 */
async function handleTranslate() {
    const text = elements.sourceText.value.trim();
    
    // Validation
    if (!text) {
        showToast('Please enter text to translate', 'warning');
        return;
    }
    
    if (!state.apiOnline) {
        showToast('API server is not available', 'error');
        return;
    }
    
    if (state.isTranslating) {
        return;
    }
    
    showLoading();
    updateStatus('Translating...');
    
    try {
        const startTime = Date.now();
        const result = await translateText(text);
        const elapsed = Date.now() - startTime;
        
        // Display translation
        elements.targetText.value = result.translation;
        updateCharCount(elements.targetText, elements.targetCount);
        
        // Update status
        const statusMsg = result.cached 
            ? `✨ Cached translation (${elapsed}ms)`
            : `✅ Translated in ${elapsed}ms`;
        updateStatus(statusMsg, 'success');
        
        if (!result.cached) {
            showToast('Translation complete!', 'success');
        }
    } catch (error) {
        console.error('Translation error:', error);
        updateStatus(`❌ Error: ${error.message}`, 'error');
        showToast(`Translation failed: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

/**
 * Handle clear
 */
function handleClear() {
    elements.sourceText.value = '';
    elements.targetText.value = '';
    updateCharCount(elements.sourceText, elements.sourceCount);
    updateCharCount(elements.targetText, elements.targetCount);
    updateStatus('');
    showToast('Cleared', 'success');
}

/**
 * Handle paste
 */
async function handlePaste() {
    try {
        const text = await navigator.clipboard.readText();
        elements.sourceText.value = text;
        updateCharCount(elements.sourceText, elements.sourceCount);
        showToast('Pasted from clipboard', 'success');
    } catch (error) {
        console.error('Paste failed:', error);
        showToast('Paste failed. Please use Ctrl+V', 'error');
    }
}

/**
 * Handle copy translation
 */
async function handleCopy() {
    const text = elements.targetText.value.trim();
    
    if (!text) {
        showToast('No translation to copy', 'warning');
        return;
    }
    
    try {
        await navigator.clipboard.writeText(text);
        showToast('Copied to clipboard!', 'success');
    } catch (error) {
        console.error('Copy failed:', error);
        
        // Fallback: select text
        elements.targetText.select();
        showToast('Please use Ctrl+C to copy', 'warning');
    }
}

/**
 * Handle download translation
 */
function handleDownload() {
    const source = elements.sourceText.value.trim();
    const translation = elements.targetText.value.trim();
    
    if (!translation) {
        showToast('No translation to download', 'warning');
        return;
    }
    
    const content = `English:\n${source}\n\nChinese:\n${translation}`;
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `translation_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showToast('Translation downloaded!', 'success');
}

// ========== Keyboard Shortcuts ==========

function handleKeyboardShortcuts(e) {
    // Ctrl+Enter: Translate
    if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        handleTranslate();
    }
    
    // Ctrl+K: Clear
    if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        handleClear();
    }
    
    // Ctrl+C: Copy (when target is focused)
    if (e.ctrlKey && e.key === 'c' && document.activeElement === elements.targetText) {
        handleCopy();
    }
}

// ========== Utility Functions ==========

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ========== Event Listeners ==========

// Button clicks
elements.translateBtn.addEventListener('click', handleTranslate);
elements.clearBtn.addEventListener('click', handleClear);
elements.pasteBtn.addEventListener('click', handlePaste);
elements.copyBtn.addEventListener('click', handleCopy);
elements.downloadBtn.addEventListener('click', handleDownload);

// Text input changes
elements.sourceText.addEventListener('input', () => {
    updateCharCount(elements.sourceText, elements.sourceCount);
});

elements.targetText.addEventListener('input', () => {
    updateCharCount(elements.targetText, elements.targetCount);
});

// Keyboard shortcuts
document.addEventListener('keydown', handleKeyboardShortcuts);

// Enter key in source text (translate)
elements.sourceText.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        handleTranslate();
    }
});

// ========== Initialization ==========

async function initialize() {
    console.log('🌉 LinguaBridge Local - Initializing...');
    
    // Check API health
    updateStatus('Connecting to API...');
    const isOnline = await checkAPIHealth();
    
    if (isOnline) {
        updateStatus('Ready to translate');
        console.log('✅ API is online and ready');
    } else {
        updateStatus('API offline - Please start the server', 'error');
        console.error('❌ API is offline');
    }
    
    // Periodic health check
    setInterval(checkAPIHealth, 10000); // Every 10 seconds
    
    // Initialize character counts
    updateCharCount(elements.sourceText, elements.sourceCount);
    updateCharCount(elements.targetText, elements.targetCount);
    
    console.log('✨ Initialization complete');
}

// Start the application
initialize();

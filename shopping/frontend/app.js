/**
 * Shopping AI Agent - Frontend Application
 * Gemini Flash를 활용한 프론트엔드 로직
 */

// Configuration
const CONFIG = {
    BACKEND_URL: 'http://localhost:8000',
    GOOGLE_API_KEY: '', // 여기에 Google API Key 입력 또는 .env에서 로드
};

// DOM Elements
const chatContainer = document.getElementById('chatContainer');
const productsGrid = document.getElementById('productsGrid');
const productCount = document.getElementById('productCount');
const chatForm = document.getElementById('chatForm');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const loadingOverlay = document.getElementById('loadingOverlay');

// State
let isLoading = false;
let conversationHistory = [];

/**
 * Initialize the application
 */
function init() {
    // Event listeners
    chatForm.addEventListener('submit', handleSubmit);
    userInput.addEventListener('keydown', handleKeydown);
    
    // Focus on input
    userInput.focus();
    
    // Show empty state
    showEmptyProducts();
    
    console.log('Shopping AI Agent initialized');
}

/**
 * Handle form submission
 */
async function handleSubmit(e) {
    e.preventDefault();
    
    const message = userInput.value.trim();
    if (!message || isLoading) return;
    
    // Clear input
    userInput.value = '';
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Start loading
    setLoading(true);
    
    try {
        // Call backend API
        const response = await searchProducts(message);
        
        if (response.success) {
            // Add AI response
            addMessage(response.message, 'assistant');
            
            // Display products
            displayProducts(response.products);
        } else {
            addMessage(response.message || '검색 중 오류가 발생했습니다.', 'assistant', true);
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('서비스 연결에 실패했습니다. 백엔드 서버가 실행 중인지 확인해주세요.', 'assistant', true);
    } finally {
        setLoading(false);
    }
}

/**
 * Handle keyboard shortcuts
 */
function handleKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
}

/**
 * Call backend API to search products
 */
async function searchProducts(query) {
    const response = await fetch(`${CONFIG.BACKEND_URL}/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: query }),
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
}

/**
 * Add a message to the chat
 */
function addMessage(content, type, isError = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const avatar = type === 'user' ? '👤' : '🤖';
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content${isError ? ' error' : ''}">
            <p>${escapeHtml(content)}</p>
        </div>
    `;
    
    chatContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    // Save to history
    conversationHistory.push({ role: type, content });
}

/**
 * Display products in the grid
 */
function displayProducts(products) {
    if (!products || products.length === 0) {
        showEmptyProducts();
        return;
    }
    
    productsGrid.innerHTML = '';
    productCount.textContent = `${products.length}개 제품`;
    
    products.forEach((product, index) => {
        const card = createProductCard(product, index + 1);
        productsGrid.appendChild(card);
    });
}

/**
 * Create a product card element
 */
function createProductCard(product, rank) {
    const card = document.createElement('a');
    card.className = 'product-card';
    card.href = product.link;
    card.target = '_blank';
    card.rel = 'noopener noreferrer';
    
    // Format price
    const price = formatPrice(product.lprice);
    
    // Get score display
    const scoreDisplay = product.total_score 
        ? `<span class="product-score">⭐ ${product.total_score.toFixed(1)}</span>`
        : '';
    
    card.innerHTML = `
        <img 
            class="product-image" 
            src="${product.image || 'data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22%3E%3Crect fill=%22%231a1a2e%22 width=%22100%22 height=%22100%22/%3E%3Ctext x=%2250%22 y=%2250%22 text-anchor=%22middle%22 dy=%22.3em%22 fill=%22%2364748b%22%3E🛒%3C/text%3E%3C/svg%3E'}" 
            alt="${escapeHtml(product.title)}"
            onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22%3E%3Crect fill=%22%231a1a2e%22 width=%22100%22 height=%22100%22/%3E%3Ctext x=%2250%22 y=%2250%22 text-anchor=%22middle%22 dy=%22.3em%22 fill=%22%2364748b%22%3E🛒%3C/text%3E%3C/svg%3E'"
        >
        <div class="product-info">
            <span class="product-rank">${rank}</span>
            <h3 class="product-title">${escapeHtml(product.title)}</h3>
            <p class="product-price">${price}</p>
            <div class="product-meta">
                ${product.mall_name ? `<span class="product-tag">${escapeHtml(product.mall_name)}</span>` : ''}
                ${product.brand ? `<span class="product-tag">${escapeHtml(product.brand)}</span>` : ''}
                ${scoreDisplay}
            </div>
        </div>
    `;
    
    return card;
}

/**
 * Show empty products state
 */
function showEmptyProducts() {
    productsGrid.innerHTML = `
        <div class="empty-state">
            <div class="empty-state-icon">🔍</div>
            <p>검색 결과가 여기에 표시됩니다</p>
            <p>원하시는 제품을 입력해보세요!</p>
        </div>
    `;
    productCount.textContent = '0개 제품';
}

/**
 * Set loading state
 */
function setLoading(loading) {
    isLoading = loading;
    
    if (loading) {
        loadingOverlay.classList.add('active');
        sendButton.disabled = true;
        userInput.disabled = true;
    } else {
        loadingOverlay.classList.remove('active');
        sendButton.disabled = false;
        userInput.disabled = false;
        userInput.focus();
    }
}

/**
 * Format price with comma separators
 */
function formatPrice(price) {
    if (!price || price === 0) return '가격 정보 없음';
    return price.toLocaleString('ko-KR') + '원';
}

/**
 * Escape HTML special characters
 */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Gemini Flash를 활용한 추가 기능 (향후 확장용)
 * 예: 이미지 분석, 음성 입력 등
 */
const GeminiHelper = {
    /**
     * 사용자 입력을 개선 (향후 Gemini Flash 연동)
     */
    async enhanceQuery(query) {
        // TODO: Gemini Flash API 연동
        // 사용자 입력을 더 구체적인 검색어로 변환
        return query;
    },
    
    /**
     * 제품 설명 생성 (향후 Gemini Flash 연동)
     */
    async generateDescription(product) {
        // TODO: Gemini Flash API 연동
        // 제품에 대한 AI 설명 생성
        return null;
    }
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', init);

// AI Chat UI - Modal interface for AI assistant
// Handles chat modal, messages, voice controls

class AIChatUI {
    constructor() {
        this.isOpen = false;
        this.messages = [];
        this.modal = null;
        this.currentQuestionContext = null;
    }

    // Open chat modal with question context
    open(questionData) {
        if (this.isOpen) return;

        this.currentQuestionContext = questionData;
        this.messages = [];

        // Set context in AI assistant
        aiAssistant.setContext(
            questionData.question,
            questionData.userAnswer,
            questionData.correctAnswer,
            questionData.explanation
        );

        // Create modal
        this.createModal();

        // Add greeting message
        this.addAIMessage(aiAssistant.getGreeting());

        this.isOpen = true;

        // Speak greeting if auto-play is on
        if (voiceOutput.getAutoPlay()) {
            voiceOutput.speak(aiAssistant.getGreeting());
        }
    }

    // Close chat modal
    close() {
        if (!this.isOpen) return;

        // Stop any ongoing speech
        voiceOutput.stop();
        voiceInput.stop();

        // Remove modal
        if (this.modal) {
            this.modal.remove();
            this.modal = null;
        }

        this.isOpen = false;
        this.messages = [];
        this.currentQuestionContext = null;
    }

    // Create modal HTML
    createModal() {
        const modal = document.createElement('div');
        modal.className = 'ai-chat-modal';
        modal.id = 'ai-chat-modal';

        const voiceSupport = checkVoiceSupport();
        const autoSpeakOn = voiceOutput.getAutoPlay();

        modal.innerHTML = `
            <div class="ai-chat-overlay" onclick="closeAIChat()"></div>
            <div class="ai-chat-container">
                <div class="ai-chat-header">
                    <div class="ai-chat-title">
                        <span class="ai-icon">🤖</span>
                        <span data-i18n="ai.title">Assistente de Estudo IA</span>
                    </div>
                    <div class="ai-chat-controls">
                        ${voiceSupport.speechSynthesis ? `
                            <button class="voice-toggle-btn ${autoSpeakOn ? 'active' : ''}" 
                                    onclick="toggleAutoSpeak()" 
                                    title="${autoSpeakOn ? 'Auto-speak: ON' : 'Auto-speak: OFF'}">
                                <span class="icon">🔊</span>
                                <span class="label">${autoSpeakOn ? 'ON' : 'OFF'}</span>
                            </button>
                        ` : ''}
                        <button class="ai-close-btn" onclick="closeAIChat()">✕</button>
                    </div>
                </div>
                
                <div class="ai-chat-context">
                    <div class="context-label">📝 <span data-i18n="ai.context">Contexto</span>:</div>
                    <div class="context-question">${this.currentQuestionContext.question}</div>
                </div>
                
                <div class="ai-chat-messages" id="ai-chat-messages">
                    <!-- Messages will be added here -->
                </div>
                
                <div class="ai-quick-suggestions" id="ai-quick-suggestions">
                    <!-- Suggestions will be added here -->
                </div>
                
                <div class="ai-chat-input-container">
                    ${voiceSupport.speechRecognition ? `
                        <button class="mic-btn" 
                                onmousedown="startVoiceInput(event)" 
                                onmouseup="stopVoiceInput(event)"
                                ontouchstart="startVoiceInput(event)"
                                ontouchend="stopVoiceInput(event)"
                                title="Hold to speak">
                            <span class="mic-icon">🎤</span>
                            <span class="recording-pulse"></span>
                        </button>
                    ` : ''}
                    <input type="text" 
                           class="ai-chat-input" 
                           id="ai-chat-input" 
                           placeholder="${currentLanguage === 'pt' ? 'Digite ou fale sua dúvida...' : 'Type or speak your question...'}"
                           onkeypress="handleChatInputKeypress(event)">
                    <button class="send-btn" onclick="sendChatMessage()">
                        <span>➤</span>
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
        this.modal = modal;

        // Add quick suggestions
        this.renderQuickSuggestions();

        // Focus input
        setTimeout(() => {
            document.getElementById('ai-chat-input')?.focus();
        }, 100);
    }

    // Render quick suggestions
    renderQuickSuggestions() {
        const container = document.getElementById('ai-quick-suggestions');
        if (!container) return;

        const suggestions = aiAssistant.getQuickSuggestions();

        container.innerHTML = suggestions.map(suggestion => `
            <button class="quick-suggestion-btn" onclick="selectQuickSuggestion('${suggestion.replace(/'/g, "\\'")}')">
                ${suggestion}
            </button>
        `).join('');
    }

    // Add user message
    addUserMessage(text) {
        this.messages.push({ role: 'user', content: text });
        this.renderMessage('user', text);
    }

    // Add AI message
    addAIMessage(text) {
        this.messages.push({ role: 'assistant', content: text });
        this.renderMessage('assistant', text);
    }

    // Render message in chat
    renderMessage(role, text) {
        const messagesContainer = document.getElementById('ai-chat-messages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `ai-message ${role}-message`;

        const voiceSupport = checkVoiceSupport();

        messageDiv.innerHTML = `
            <div class="message-content">${this.formatMessage(text)}</div>
            ${role === 'assistant' && voiceSupport.speechSynthesis ? `
                <button class="replay-btn" onclick="replayMessage('${text.replace(/'/g, "\\'")}')">
                    <span class="icon">🔊</span>
                </button>
            ` : ''}
        `;

        messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Format message text (support basic markdown)
    formatMessage(text) {
        // Simple markdown support
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }

    // Show typing indicator
    showTyping() {
        const messagesContainer = document.getElementById('ai-chat-messages');
        if (!messagesContainer) return;

        const typingDiv = document.createElement('div');
        typingDiv.className = 'ai-message assistant-message typing-indicator';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="message-content">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
            </div>
        `;

        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Hide typing indicator
    hideTyping() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    // Send message to AI
    async sendMessage(text) {
        if (!text.trim()) return;

        // Add user message
        this.addUserMessage(text);

        // Clear input
        const input = document.getElementById('ai-chat-input');
        if (input) input.value = '';

        // Show typing indicator
        this.showTyping();

        try {
            // Get AI response
            const response = await aiAssistant.sendMessage(text);

            // Hide typing indicator
            this.hideTyping();

            // Add AI response
            this.addAIMessage(response);

            // Speak response if auto-play is on
            if (voiceOutput.getAutoPlay()) {
                voiceOutput.speak(response);
            }

        } catch (error) {
            this.hideTyping();

            const errorMessage = currentLanguage === 'pt'
                ? `Desculpe, ocorreu um erro: ${error.message}`
                : `Sorry, an error occurred: ${error.message}`;

            this.addAIMessage(errorMessage);
        }
    }
}

// Initialize chat UI
const aiChatUI = new AIChatUI();

// Global functions for event handlers
function openAIChat(questionData) {
    aiChatUI.open(questionData);
}

function closeAIChat() {
    aiChatUI.close();
}

function sendChatMessage() {
    const input = document.getElementById('ai-chat-input');
    if (input && input.value.trim()) {
        aiChatUI.sendMessage(input.value.trim());
    }
}

function handleChatInputKeypress(event) {
    if (event.key === 'Enter') {
        sendChatMessage();
    }
}

function selectQuickSuggestion(suggestion) {
    aiChatUI.sendMessage(suggestion);
}

function toggleAutoSpeak() {
    const newState = voiceOutput.toggleAutoPlay();
    const btn = document.querySelector('.voice-toggle-btn');
    if (btn) {
        btn.classList.toggle('active', newState);
        btn.querySelector('.label').textContent = newState ? 'ON' : 'OFF';
        btn.title = newState ? 'Auto-speak: ON' : 'Auto-speak: OFF';
    }
}

function replayMessage(text) {
    voiceOutput.speak(text);
}

// Voice input handlers
let voiceInputActive = false;

function startVoiceInput(event) {
    event.preventDefault();
    if (voiceInputActive) return;

    voiceInputActive = true;
    const micBtn = event.currentTarget;
    micBtn.classList.add('recording');

    const input = document.getElementById('ai-chat-input');

    voiceInput.start(
        // On interim result
        (transcript) => {
            if (input) input.value = transcript;
        },
        // On final result
        (transcript) => {
            if (input) input.value = transcript;
            micBtn.classList.remove('recording');
            voiceInputActive = false;
        }
    ).catch(error => {
        console.error('Voice input error:', error);
        micBtn.classList.remove('recording');
        voiceInputActive = false;
    });
}

function stopVoiceInput(event) {
    event.preventDefault();
    if (!voiceInputActive) return;

    voiceInput.stop();
    const micBtn = event.currentTarget;
    micBtn.classList.remove('recording');
    voiceInputActive = false;
}

// Expose globally
window.aiChatUI = aiChatUI;
window.openAIChat = openAIChat;
window.closeAIChat = closeAIChat;

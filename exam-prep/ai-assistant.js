// AI Assistant - Gemini API Integration
// Handles chat functionality and LLM communication

// IMPORTANT: API Key is now loaded from config.js (local) or environment
let GEMINI_API_KEY = typeof CONFIG !== 'undefined' ? CONFIG.GEMINI_API_KEY : 'MISSING_API_KEY';

if (GEMINI_API_KEY === 'MISSING_API_KEY') {
    console.warn('Gemini API Key missing! Check config.js');
}

const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent';

class AIAssistant {
    constructor() {
        this.conversationHistory = [];
        this.currentContext = null;
        this.rateLimiter = {
            requests: [],
            maxPerMinute: 10
        };
    }

    // Set context for current question
    setContext(question, userAnswer, correctAnswer, explanation) {
        this.currentContext = {
            question,
            userAnswer,
            correctAnswer,
            explanation,
            language: currentLanguage
        };
        this.conversationHistory = []; // Reset conversation for new question
    }

    // Build system prompt with context
    buildSystemPrompt() {
        const lang = this.currentContext.language;

        if (lang === 'pt') {
            return `Você é um tutor de IA ajudando estudantes a se prepararem para o exame de certificação CAIP-210.

Contexto da questão:
Pergunta: ${this.currentContext.question}
Resposta do aluno: ${this.currentContext.userAnswer}
Resposta correta: ${this.currentContext.correctAnswer}
Explicação: ${this.currentContext.explanation}

Seu papel:
- Explicar conceitos de AI/ML de forma clara e concisa
- Ajudar o aluno a entender POR QUE errou
- Fornecer exemplos práticos
- Ser encorajador e solidário
- Manter respostas com menos de 150 palavras
- Usar analogias quando útil
- Sugerir tópicos relacionados para estudar

Seja direto, claro e educativo.`;
        } else {
            return `You are an AI tutor helping students prepare for the CAIP-210 certification exam.

Question context:
Question: ${this.currentContext.question}
Student's answer: ${this.currentContext.userAnswer}
Correct answer: ${this.currentContext.correctAnswer}
Explanation: ${this.currentContext.explanation}

Your role:
- Explain AI/ML concepts clearly and concisely
- Help the student understand WHY they got it wrong
- Provide practical examples
- Be encouraging and supportive
- Keep responses under 150 words
- Use analogies when helpful
- Suggest related topics to study

Be direct, clear, and educational.`;
        }
    }

    // Check rate limiting
    canMakeRequest() {
        const now = Date.now();
        this.rateLimiter.requests = this.rateLimiter.requests.filter(t => now - t < 60000);
        return this.rateLimiter.requests.length < this.rateLimiter.maxPerMinute;
    }

    recordRequest() {
        this.rateLimiter.requests.push(Date.now());
    }

    // Send message to Gemini API
    async sendMessage(userMessage) {
        // Check API key
        if (GEMINI_API_KEY === 'YOUR_GEMINI_API_KEY_HERE') {
            throw new Error('Please configure your Gemini API key in ai-assistant.js');
        }

        // Check rate limiting
        if (!this.canMakeRequest()) {
            throw new Error('Rate limit exceeded. Please wait a moment.');
        }

        // Build conversation with language-specific labels
        const systemPrompt = this.buildSystemPrompt();
        const lang = this.currentContext.language;

        // Add explicit language instruction
        const languageInstruction = lang === 'pt'
            ? '\n\nIMPORTANTE: Responda SEMPRE em Português (PT-BR).'
            : '\n\nIMPORTANT: Always respond in English.';

        const studentLabel = lang === 'pt' ? 'Aluno' : 'Student';
        const tutorLabel = lang === 'pt' ? 'Tutor' : 'Tutor';

        const fullPrompt = `${systemPrompt}${languageInstruction}\n\n${studentLabel}: ${userMessage}\n\n${tutorLabel}:`;

        try {
            this.recordRequest();

            const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    contents: [{
                        parts: [{
                            text: fullPrompt
                        }]
                    }],
                    generationConfig: {
                        temperature: 0.7,
                        maxOutputTokens: 500,
                        topP: 0.8,
                        topK: 40
                    },
                    safetySettings: [
                        {
                            category: 'HARM_CATEGORY_HARASSMENT',
                            threshold: 'BLOCK_MEDIUM_AND_ABOVE'
                        },
                        {
                            category: 'HARM_CATEGORY_HATE_SPEECH',
                            threshold: 'BLOCK_MEDIUM_AND_ABOVE'
                        }
                    ]
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error?.message || 'API request failed');
            }

            const data = await response.json();

            if (!data.candidates || data.candidates.length === 0) {
                throw new Error('No response from AI');
            }

            const aiResponse = data.candidates[0].content.parts[0].text;

            // Store in conversation history
            this.conversationHistory.push({
                role: 'user',
                content: userMessage
            });
            this.conversationHistory.push({
                role: 'assistant',
                content: aiResponse
            });

            return aiResponse;

        } catch (error) {
            console.error('AI Assistant error:', error);
            throw error;
        }
    }

    // Get greeting message
    getGreeting() {
        if (currentLanguage === 'pt') {
            return 'Olá! Vi que você errou essa questão. Como posso ajudar você a entender melhor?';
        } else {
            return 'Hi! I saw you got this question wrong. How can I help you understand it better?';
        }
    }

    // Get quick suggestions
    getQuickSuggestions() {
        if (currentLanguage === 'pt') {
            return [
                'Explique de forma mais simples',
                'Dê um exemplo prático',
                'Por que minha resposta está errada?',
                'Qual é o conceito principal?'
            ];
        } else {
            return [
                'Explain it more simply',
                'Give a practical example',
                'Why is my answer wrong?',
                'What is the main concept?'
            ];
        }
    }
}

// Initialize AI Assistant
const aiAssistant = new AIAssistant();

// Expose globally
window.aiAssistant = aiAssistant;

// CAIP-210 Internationalization (i18n) System
// Bilingual support: Portuguese (PT-BR) and English (EN)

// Current language - default to Portuguese or load from localStorage
let currentLanguage = localStorage.getItem('caip210_language') || 'pt';

// Translation dictionary
const TRANSLATIONS = {
    // Navigation
    'nav.home': { pt: 'Início', en: 'Home' },
    'nav.practice': { pt: 'Modo Prática', en: 'Practice Mode' },
    'nav.exam': { pt: 'Simulado', en: 'Exam Simulation' },
    'nav.study': { pt: 'Estudar', en: 'Study' },
    'nav.topics': { pt: 'Por Tópicos', en: 'By Topics' },
    'nav.stats': { pt: 'Estatísticas', en: 'Statistics' },
    'nav.progress': { pt: 'Progresso Geral', en: 'Overall Progress' },
    'nav.totalQuestions': { pt: 'Total de Questões', en: 'Total Questions' },

    // Home view
    'home.welcome': { pt: 'Bem-vindo ao', en: 'Welcome to' },
    'home.title': { pt: 'Preparação CAIP-210', en: 'CAIP-210 Preparation' },
    'home.subtitle': { pt: 'Sua jornada para a certificação CertNexus Certified Artificial Intelligence Practitioner', en: 'Your journey to CertNexus Certified Artificial Intelligence Practitioner certification' },
    'home.quickPractice': { pt: 'Prática Rápida', en: 'Quick Practice' },
    'home.quickPracticeDesc': { pt: '10 questões aleatórias', en: '10 random questions' },
    'home.startExam': { pt: 'Iniciar Simulado', en: 'Start Exam' },
    'home.startExamDesc': { pt: '60 questões • 90 min', en: '60 questions • 90 min' },
    'home.domains': { pt: 'Domínios do Exame', en: 'Exam Domains' },
    'home.questions': { pt: 'questões', en: 'questions' },

    // Quiz view
    'quiz.question': { pt: 'Questão', en: 'Question' },
    'quiz.of': { pt: 'de', en: 'of' },
    'quiz.confirm': { pt: 'Confirmar', en: 'Confirm' },
    'quiz.next': { pt: 'Próxima', en: 'Next' },
    'quiz.finish': { pt: 'Finalizar', en: 'Finish' },
    'quiz.correct': { pt: 'Correto!', en: 'Correct!' },
    'quiz.incorrect': { pt: 'Incorreto', en: 'Incorrect' },
    'quiz.correctAnswer': { pt: 'Resposta correta', en: 'Correct answer' },
    'quiz.explanation': { pt: 'Explicação', en: 'Explanation' },
    'quiz.quit': { pt: 'Sair', en: 'Quit' },
    'quiz.timeRemaining': { pt: 'Tempo restante', en: 'Time remaining' },

    // Results view
    'results.title': { pt: 'Resultado do Quiz', en: 'Quiz Results' },
    'results.score': { pt: 'Pontuação', en: 'Score' },
    'results.correct': { pt: 'Corretas', en: 'Correct' },
    'results.incorrect': { pt: 'Incorretas', en: 'Incorrect' },
    'results.time': { pt: 'Tempo', en: 'Time' },
    'results.passed': { pt: 'Aprovado!', en: 'Passed!' },
    'results.failed': { pt: 'Continue estudando', en: 'Keep studying' },
    'results.passThreshold': { pt: 'Nota de corte: 70%', en: 'Pass threshold: 70%' },
    'results.reviewAnswers': { pt: 'Revisar Respostas', en: 'Review Answers' },
    'results.backHome': { pt: 'Voltar ao Início', en: 'Back to Home' },
    'results.tryAgain': { pt: 'Tentar Novamente', en: 'Try Again' },

    // Study view
    'study.title': { pt: 'Material de Estudo', en: 'Study Material' },
    'study.subtitle': { pt: 'Escolha um domínio para estudar conceitos, exemplos e casos reais', en: 'Choose a domain to study concepts, examples, and real cases' },
    'study.backToDomains': { pt: '← Voltar aos Domínios', en: '← Back to Domains' },
    'study.concept': { pt: 'Conceito', en: 'Concept' },
    'study.keyPoints': { pt: 'Pontos-Chave', en: 'Key Points' },
    'study.practicalExample': { pt: 'Exemplo Prático', en: 'Practical Example' },
    'study.realCase': { pt: 'Caso Real', en: 'Real Case' },
    'study.impact': { pt: 'Impacto', en: 'Impact' },
    'study.topics': { pt: 'tópicos', en: 'topics' },

    // Topics view
    'topics.title': { pt: 'Estudo por Tópicos', en: 'Study by Topics' },
    'topics.selectDomain': { pt: 'Selecione um domínio para praticar questões específicas', en: 'Select a domain to practice specific questions' },
    'topics.startQuiz': { pt: 'Iniciar Quiz', en: 'Start Quiz' },

    // Stats view
    'stats.title': { pt: 'Suas Estatísticas', en: 'Your Statistics' },
    'stats.totalAnswered': { pt: 'Total Respondido', en: 'Total Answered' },
    'stats.accuracy': { pt: 'Taxa de Acerto', en: 'Accuracy Rate' },
    'stats.studyTime': { pt: 'Tempo de Estudo', en: 'Study Time' },
    'stats.domainPerformance': { pt: 'Desempenho por Domínio', en: 'Performance by Domain' },
    'stats.reset': { pt: '🗑️ Resetar Estatísticas', en: '🗑️ Reset Statistics' },
    'stats.resetConfirm': { pt: 'Tem certeza que deseja resetar todas as estatísticas?', en: 'Are you sure you want to reset all statistics?' },

    // General
    'general.loading': { pt: 'Carregando...', en: 'Loading...' },
    'general.error': { pt: 'Erro', en: 'Error' },
    'general.cancel': { pt: 'Cancelar', en: 'Cancel' },
    'general.save': { pt: 'Salvar', en: 'Save' },
    'general.close': { pt: 'Fechar', en: 'Close' },

    // AI Assistant
    'ai.button': { pt: '🤖 Perguntar à IA', en: '🤖 Ask AI' },
    'ai.title': { pt: 'Assistente de Estudo IA', en: 'AI Study Assistant' },
    'ai.context': { pt: 'Contexto', en: 'Context' },
    'ai.greeting': {
        pt: 'Olá! Vi que você errou essa questão. Como posso ajudar você a entender melhor?',
        en: 'Hi! I saw you got this question wrong. How can I help you understand it better?'
    },
    'ai.placeholder': {
        pt: 'Digite ou fale sua dúvida...',
        en: 'Type or speak your question...'
    },
    'ai.autoSpeak': { pt: 'Auto-falar', en: 'Auto-speak' },
    'ai.recording': { pt: 'Gravando...', en: 'Recording...' },
    'ai.thinking': { pt: 'Pensando...', en: 'Thinking...' },
    'ai.error': { pt: 'Desculpe, ocorreu um erro', en: 'Sorry, an error occurred' },
    'ai.apiKeyMissing': {
        pt: 'Configure sua chave Gemini API em ai-assistant.js',
        en: 'Configure your Gemini API key in ai-assistant.js'
    },
    'ai.rateLimit': {
        pt: 'Limite de requisições atingido. Aguarde um momento.',
        en: 'Rate limit exceeded. Please wait a moment.'
    },

    // Language toggle
    'lang.toggle': { pt: 'EN', en: 'PT' },
    'lang.current': { pt: '🇧🇷 Português', en: '🇺🇸 English' },
    'lang.switchTo': { pt: 'Switch to English', en: 'Mudar para Português' }
};

// Get translation for a key
function t(key) {
    const translation = TRANSLATIONS[key];
    if (!translation) {
        console.warn(`Missing translation for key: ${key}`);
        return key;
    }
    return translation[currentLanguage] || translation['en'] || key;
}

// Toggle language between PT and EN
function toggleLanguage() {
    currentLanguage = currentLanguage === 'pt' ? 'en' : 'pt';
    localStorage.setItem('caip210_language', currentLanguage);
    updateAllTranslations();
    updateLanguageToggleButton();
}

// Update the language toggle button
function updateLanguageToggleButton() {
    const toggleBtn = document.getElementById('language-toggle');
    if (toggleBtn) {
        toggleBtn.innerHTML = `
            <span class="lang-text">${currentLanguage === 'pt' ? 'PT' : 'EN'}</span>
        `;
        toggleBtn.title = t('lang.switchTo');
    }
}

// Update all translatable elements on the page
function updateAllTranslations() {
    // Update elements with data-i18n attribute
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.dataset.i18n;
        el.textContent = t(key);
    });

    // Update elements with data-i18n-placeholder attribute
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
        const key = el.dataset.i18nPlaceholder;
        el.placeholder = t(key);
    });

    // Update elements with data-i18n-title attribute
    document.querySelectorAll('[data-i18n-title]').forEach(el => {
        const key = el.dataset.i18nTitle;
        el.title = t(key);
    });

    // Re-render current view to update dynamic content
    if (typeof state !== 'undefined' && state.currentView) {
        // Update questions language
        if (typeof updateQuestionsLanguage === 'function') {
            updateQuestionsLanguage();
        }

        if (state.currentView === 'study') {
            renderStudyView();
        } else if (state.currentView === 'topics') {
            renderTopicsView();
        } else if (state.currentView === 'stats') {
            renderStatsView();
        } else if (state.currentView === 'quiz' && state.questions && state.questions.length > 0) {
            // Reload questions for current quiz with new language
            // Only reload if quiz hasn't been answered yet
            if (state.answers && state.answers.every(a => a === null)) {
                if (state.quizMode === 'exam') {
                    state.questions = getExamQuestions();
                } else if (state.quizMode === 'practice') {
                    state.questions = getRandomQuestions(10);
                } else if (state.quizMode === 'topic' && state.currentDomain) {
                    state.questions = getQuestionsByDomain(state.currentDomain);
                }
                state.answers = new Array(state.questions.length).fill(null);
                document.getElementById('total-questions').textContent = state.questions.length;
            }
            renderQuestion();
        }
    }

    // Update document title based on language
    document.title = currentLanguage === 'pt'
        ? 'CAIP-210 Exam Prep | Quiz de Preparação'
        : 'CAIP-210 Exam Prep | Preparation Quiz';
}

// Initialize language system on page load
function initLanguageSystem() {
    updateLanguageToggleButton();
    updateAllTranslations();
}

// Expose functions globally
window.t = t;
window.toggleLanguage = toggleLanguage;
window.currentLanguage = currentLanguage;
window.initLanguageSystem = initLanguageSystem;

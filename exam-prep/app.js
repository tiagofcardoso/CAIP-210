// CAIP-210 Exam Prep Application Logic

// ===== State Management =====
let state = {
    currentView: 'home',
    quizMode: null, // 'practice', 'exam', 'topic'
    currentDomain: null,
    questions: [],
    currentQuestionIndex: 0,
    answers: [],
    score: 0,
    startTime: null,
    timerInterval: null,
    timeRemaining: 0,
    isPracticeMode: true
};

// Load stats from localStorage
let stats = JSON.parse(localStorage.getItem('caip210_stats')) || {
    totalAnswered: 0,
    totalCorrect: 0,
    studyTime: 0,
    domainStats: {
        1: { answered: 0, correct: 0 },
        2: { answered: 0, correct: 0 },
        3: { answered: 0, correct: 0 },
        4: { answered: 0, correct: 0 }
    },
    quizHistory: []
};

// ===== Navigation =====

// ===== Navigation =====
function switchView(viewId) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById(viewId + '-view').classList.add('active');

    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.querySelector(`[data-view="${viewId}"]`)?.classList.add('active');

    state.currentView = viewId;

    if (viewId === 'topics') {
        renderTopicsView();
    } else if (viewId === 'stats') {
        renderStatsView();
    } else if (viewId === 'study') {
        renderStudyView();
    }

    updateProgress();

    // Close mobile menu if open
    document.getElementById('nav-menu').classList.remove('active');
    document.getElementById('mobile-menu-btn').classList.remove('active');
    document.body.style.overflow = '';
}

function toggleMobileMenu() {
    const menu = document.getElementById('nav-menu');
    const btn = document.getElementById('mobile-menu-btn');

    menu.classList.toggle('active');
    btn.classList.toggle('active');

    // Prevent scrolling when menu is open
    if (menu.classList.contains('active')) {
        document.body.style.overflow = 'hidden';
    } else {
        document.body.style.overflow = '';
    }
}

function goHome() {
    switchView('home');
}


// Initialize navigation
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        const view = item.dataset.view;
        if (view === 'practice') {
            startQuickPractice();
        } else if (view === 'exam') {
            startExam();
        } else {
            switchView(view);
        }
    });
});

// ===== Quiz Functions =====
function startQuickPractice() {
    state.quizMode = 'practice';
    state.isPracticeMode = true;
    state.questions = getRandomQuestions(10);
    state.currentDomain = null;

    initQuiz('Modo Prática', '10 questões aleatórias');
}

function startExam() {
    state.quizMode = 'exam';
    state.isPracticeMode = false;
    state.questions = getExamQuestions();
    state.currentDomain = null;
    state.timeRemaining = 90 * 60; // 90 minutes in seconds

    initQuiz('Simulado Completo', '60 questões • 90 min');
    startTimer();
}

function startTopicQuiz(domainId) {
    state.quizMode = 'topic';
    state.isPracticeMode = true;
    state.currentDomain = domainId;
    state.questions = getQuestionsByDomain(domainId);

    const domain = DOMAINS[domainId];
    initQuiz(`Domínio ${domainId}`, domain.name);
}

function initQuiz(typeText, domainText) {
    state.currentQuestionIndex = 0;
    state.answers = new Array(state.questions.length).fill(null);
    state.score = 0;
    state.startTime = Date.now();

    document.getElementById('quiz-type').textContent = typeText;
    document.getElementById('quiz-domain').textContent = domainText;
    document.getElementById('total-questions').textContent = state.questions.length;
    document.getElementById('score-display').textContent = 'Acertos: 0';

    // Show/hide timer based on mode
    const timer = document.getElementById('timer');
    if (state.quizMode === 'exam') {
        timer.classList.remove('hidden');
    } else {
        timer.classList.add('hidden');
    }

    switchView('quiz');
    renderQuestion();
}

function renderQuestion() {
    const question = state.questions[state.currentQuestionIndex];
    const index = state.currentQuestionIndex;

    // Update progress
    document.getElementById('current-question').textContent = index + 1;
    document.getElementById('quiz-progress').style.width =
        ((index + 1) / state.questions.length * 100) + '%';
    document.getElementById('question-number').textContent = `Q${index + 1}`;
    document.getElementById('question-text').textContent = question.question;

    // Render options
    const container = document.getElementById('options-container');
    container.innerHTML = '';

    // Check if already answered (defined before loop so it can be used after)
    const previousAnswer = state.answers[index];

    const letters = ['A', 'B', 'C', 'D'];
    question.options.forEach((opt, i) => {
        const btn = document.createElement('button');
        btn.className = 'option-btn';
        btn.dataset.index = i;

        // Apply styles if already answered
        if (previousAnswer !== null) {
            btn.classList.add('disabled');
            if (i === question.correct) {
                btn.classList.add('correct');
            } else if (i === previousAnswer && previousAnswer !== question.correct) {
                btn.classList.add('incorrect');
            }
        }

        btn.innerHTML = `
            <span class="option-letter">${letters[i]}</span>
            <span class="option-text">${opt}</span>
        `;

        btn.addEventListener('click', () => selectOption(i));
        container.appendChild(btn);
    });

    // Show feedback if already answered
    const feedbackContainer = document.getElementById('feedback-container');
    if (previousAnswer !== null) {
        showFeedback(previousAnswer === question.correct);
    } else {
        feedbackContainer.classList.add('hidden');
    }

    // Update navigation
    updateNavigation();
}

function selectOption(optionIndex) {
    const question = state.questions[state.currentQuestionIndex];

    // Check if already answered
    if (state.answers[state.currentQuestionIndex] !== null) return;

    // Record answer
    state.answers[state.currentQuestionIndex] = optionIndex;
    const isCorrect = optionIndex === question.correct;

    if (isCorrect) {
        state.score++;
        document.getElementById('score-display').textContent = `Acertos: ${state.score}`;
    }

    // Update stats
    stats.totalAnswered++;
    if (isCorrect) stats.totalCorrect++;
    stats.domainStats[question.domain].answered++;
    if (isCorrect) stats.domainStats[question.domain].correct++;
    saveStats();

    // Update UI
    const buttons = document.querySelectorAll('.option-btn');
    buttons.forEach((btn, i) => {
        btn.classList.add('disabled');
        if (i === question.correct) {
            btn.classList.add('correct');
        } else if (i === optionIndex && !isCorrect) {
            btn.classList.add('incorrect');
        }
    });

    showFeedback(isCorrect);
    updateNavigation();
}

function showFeedback(isCorrect) {
    const question = state.questions[state.currentQuestionIndex];
    const container = document.getElementById('feedback-container');
    const card = document.getElementById('feedback-card');

    container.classList.remove('hidden');
    card.className = 'feedback-card ' + (isCorrect ? 'correct' : 'incorrect');

    document.getElementById('feedback-icon').textContent = isCorrect ? '✓' : '✗';
    document.getElementById('feedback-text').textContent = isCorrect ? 'Correto!' : 'Incorreto';
    document.getElementById('explanation').textContent = question.explanation;

    // Add AI Assistant button if answer is incorrect
    const explanationEl = document.getElementById('explanation');
    const existingAIBtn = document.getElementById('ai-assistant-btn');

    if (!isCorrect && !existingAIBtn) {
        const aiButton = document.createElement('button');
        aiButton.id = 'ai-assistant-btn';
        aiButton.className = 'ai-assistant-btn';
        aiButton.innerHTML = `
            <span class="ai-icon">🤖</span>
            <span data-i18n="ai.button">Perguntar à IA</span>
            <span class="ai-badge">BETA</span>
        `;
        aiButton.onclick = function () {
            openAIChat({
                question: question.question,
                userAnswer: question.options[state.selectedOption],
                correctAnswer: question.options[question.correct],
                explanation: question.explanation
            });
        };
        explanationEl.parentNode.appendChild(aiButton);
    } else if (isCorrect && existingAIBtn) {
        // Remove AI button if answer is correct
        existingAIBtn.remove();
    }
}

function nextQuestion() {
    if (state.currentQuestionIndex < state.questions.length - 1) {
        state.currentQuestionIndex++;
        renderQuestion();
    } else if (!state.isPracticeMode || state.answers.every(a => a !== null)) {
        // All questions answered or end of exam
        showFinishButton();
    }
}

function prevQuestion() {
    if (state.currentQuestionIndex > 0) {
        state.currentQuestionIndex--;
        renderQuestion();
    }
}

function updateNavigation() {
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const finishBtn = document.getElementById('finish-btn');

    prevBtn.disabled = state.currentQuestionIndex === 0;

    const isLastQuestion = state.currentQuestionIndex === state.questions.length - 1;
    const allAnswered = state.answers.every(a => a !== null);

    if (isLastQuestion && allAnswered) {
        nextBtn.classList.add('hidden');
        finishBtn.classList.remove('hidden');
    } else {
        nextBtn.classList.remove('hidden');
        finishBtn.classList.add('hidden');
    }
}

function showFinishButton() {
    document.getElementById('next-btn').classList.add('hidden');
    document.getElementById('finish-btn').classList.remove('hidden');
}

function finishQuiz() {
    if (state.timerInterval) {
        clearInterval(state.timerInterval);
        state.timerInterval = null;
    }

    const totalTime = Math.floor((Date.now() - state.startTime) / 1000);
    const answered = state.answers.filter(a => a !== null).length;
    const scorePercent = Math.round((state.score / answered) * 100) || 0;

    // Update study time
    stats.studyTime += totalTime;

    // Add to history
    stats.quizHistory.push({
        date: new Date().toISOString(),
        mode: state.quizMode,
        domain: state.currentDomain,
        score: scorePercent,
        correct: state.score,
        total: answered,
        time: totalTime
    });
    saveStats();

    // Render results
    renderResults(scorePercent, answered, totalTime);
    switchView('results');
}

function renderResults(scorePercent, answered, totalTime) {
    const scoreCircle = document.getElementById('score-circle');
    const pass = scorePercent >= 70;

    scoreCircle.className = 'score-circle ' + (pass ? 'pass' : 'fail');
    document.getElementById('final-score').textContent = scorePercent + '%';
    document.getElementById('correct-count').textContent = state.score;
    document.getElementById('incorrect-count').textContent = answered - state.score;

    const minutes = Math.floor(totalTime / 60);
    const seconds = totalTime % 60;
    document.getElementById('time-spent').textContent =
        `${minutes}:${seconds.toString().padStart(2, '0')}`;

    // Message
    let message = '';
    if (scorePercent >= 90) {
        message = '🎉 Excelente! Você está muito bem preparado para o exame!';
    } else if (scorePercent >= 70) {
        message = '✅ Bom trabalho! Você atingiu a pontuação de aprovação.';
    } else if (scorePercent >= 50) {
        message = '📚 Continue estudando! Você está no caminho certo.';
    } else {
        message = '💪 Não desista! Revise o material e tente novamente.';
    }
    document.getElementById('results-message').textContent = message;

    // Domain performance
    renderDomainStats();
}

function renderDomainStats() {
    const container = document.getElementById('domain-stats');
    container.innerHTML = '';

    // Calculate performance for this quiz by domain
    const domainPerf = {};
    state.questions.forEach((q, i) => {
        if (!domainPerf[q.domain]) {
            domainPerf[q.domain] = { correct: 0, total: 0 };
        }
        if (state.answers[i] !== null) {
            domainPerf[q.domain].total++;
            if (state.answers[i] === q.correct) {
                domainPerf[q.domain].correct++;
            }
        }
    });

    Object.keys(domainPerf).forEach(domainId => {
        const domain = DOMAINS[domainId];
        const perf = domainPerf[domainId];
        const percent = perf.total > 0 ? Math.round((perf.correct / perf.total) * 100) : 0;
        const color = percent >= 70 ? 'var(--success)' : percent >= 50 ? 'var(--warning)' : 'var(--error)';

        const div = document.createElement('div');
        div.className = 'domain-stat';
        div.innerHTML = `
            <span class="domain-stat-name">${domain.icon} ${domain.name}</span>
            <div class="domain-stat-bar">
                <div class="domain-stat-fill" style="width: ${percent}%; background: ${color}"></div>
            </div>
            <span class="domain-stat-percent">${percent}%</span>
        `;
        container.appendChild(div);
    });
}

function reviewAnswers() {
    state.currentQuestionIndex = 0;
    switchView('quiz');
    renderQuestion();
}

function quitQuiz() {
    if (confirm('Tem certeza que deseja sair? Seu progresso será perdido.')) {
        if (state.timerInterval) {
            clearInterval(state.timerInterval);
            state.timerInterval = null;
        }
        switchView('home');
    }
}

// ===== Timer =====
function startTimer() {
    const timerDisplay = document.getElementById('timer-display');
    const timer = document.getElementById('timer');

    // Clear any existing timer interval
    if (state.timerInterval) {
        clearInterval(state.timerInterval);
        state.timerInterval = null;
    }

    // Initialize timer display immediately
    const initMinutes = Math.floor(state.timeRemaining / 60);
    const initSeconds = state.timeRemaining % 60;
    timerDisplay.textContent = `${initMinutes}:${initSeconds.toString().padStart(2, '0')}`;
    timer.classList.remove('warning');

    console.log('Timer started with', state.timeRemaining, 'seconds');

    state.timerInterval = setInterval(() => {
        state.timeRemaining--;

        const minutes = Math.floor(state.timeRemaining / 60);
        const seconds = state.timeRemaining % 60;
        timerDisplay.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;

        // Warning when less than 10 minutes
        if (state.timeRemaining <= 600) {
            timer.classList.add('warning');
        }

        // Time's up
        if (state.timeRemaining <= 0) {
            clearInterval(state.timerInterval);
            finishQuiz();
        }
    }, 1000);
}

// ===== Topics View =====
function renderTopicsView() {
    const container = document.getElementById('topics-list');
    container.innerHTML = '';

    Object.keys(DOMAINS).forEach(domainId => {
        const domain = DOMAINS[domainId];
        const questions = getQuestionsByDomain(parseInt(domainId));
        const stats = getLocalStats().domainStats[domainId];
        const progress = stats.answered > 0 ? Math.round((stats.correct / stats.answered) * 100) : 0;

        const div = document.createElement('div');
        div.className = 'topic-item';
        div.onclick = () => startTopicQuiz(parseInt(domainId));
        div.innerHTML = `
            <span class="topic-icon">${domain.icon}</span>
            <div class="topic-info">
                <h4>${domain.name}</h4>
                <p>${domain.description}</p>
            </div>
            <span class="topic-count">${questions.length} questões • ${progress}% acertos</span>
        `;
        container.appendChild(div);
    });
}

// ===== Stats View =====
function renderStatsView() {
    const s = getLocalStats();

    document.getElementById('total-answered').textContent = s.totalAnswered;

    const avgScore = s.totalAnswered > 0
        ? Math.round((s.totalCorrect / s.totalAnswered) * 100)
        : 0;
    document.getElementById('avg-score').textContent = avgScore + '%';

    const hours = Math.floor(s.studyTime / 3600);
    const minutes = Math.floor((s.studyTime % 3600) / 60);
    document.getElementById('study-time').textContent =
        hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;

    // Domain breakdown
    const container = document.getElementById('stats-domains');
    container.innerHTML = '';

    Object.keys(DOMAINS).forEach(domainId => {
        const domain = DOMAINS[domainId];
        const ds = s.domainStats[domainId];
        const percent = ds.answered > 0 ? Math.round((ds.correct / ds.answered) * 100) : 0;
        const color = percent >= 70 ? 'var(--success)' : percent >= 50 ? 'var(--warning)' : 'var(--error)';

        const div = document.createElement('div');
        div.className = 'domain-stat';
        div.innerHTML = `
            <span class="domain-stat-name">${domain.icon} ${domain.name}</span>
            <div class="domain-stat-bar">
                <div class="domain-stat-fill" style="width: ${percent}%; background: ${color}"></div>
            </div>
            <span class="domain-stat-percent">${ds.correct}/${ds.answered}</span>
        `;
        container.appendChild(div);
    });
}

function resetStats() {
    if (confirm('Tem certeza que deseja resetar todas as estatísticas?')) {
        stats = {
            totalAnswered: 0,
            totalCorrect: 0,
            studyTime: 0,
            domainStats: {
                1: { answered: 0, correct: 0 },
                2: { answered: 0, correct: 0 },
                3: { answered: 0, correct: 0 },
                4: { answered: 0, correct: 0 }
            },
            quizHistory: []
        };
        saveStats();
        renderStatsView();
        updateProgress();
    }
}

// ===== Study View =====
let currentStudyDomain = null;
let currentStudyTopic = 0;

function renderStudyView() {
    showStudyDomains();
}

function showStudyDomains() {
    currentStudyDomain = null;
    currentStudyTopic = 0;

    document.getElementById('study-back-btn').style.display = 'none';
    document.getElementById('study-title').textContent = t('study.title');
    document.getElementById('study-subtitle').textContent = t('study.subtitle');

    document.getElementById('study-domains').classList.remove('hidden');
    document.getElementById('study-content').classList.add('hidden');

    const container = document.getElementById('study-domains');
    container.innerHTML = '';

    const topicsLabel = t('study.topics');

    Object.keys(STUDY_CONTENT).forEach(domainId => {
        const domain = STUDY_CONTENT[domainId];
        const topicsCount = domain.topics.length;

        const card = document.createElement('div');
        card.className = 'study-domain-card';
        card.onclick = () => showDomainTopics(parseInt(domainId));
        card.innerHTML = `
            <div class="study-domain-header">
                <span class="study-domain-icon">${domain.icon}</span>
                <div class="study-domain-title">
                    <h3>${domain.name}</h3>
                    <span>${domain.weight}</span>
                </div>
            </div>
            <p>${topicsCount} ${topicsLabel}</p>
            <div class="study-domain-topics">
                ${domain.topics.slice(0, 3).map(t => `• ${t.title}`).join('<br>')}
                ${topicsCount > 3 ? '<br>• ...' : ''}
            </div>
        `;
        container.appendChild(card);
    });
}

function showDomainTopics(domainId) {
    currentStudyDomain = domainId;
    currentStudyTopic = 0;

    const domain = STUDY_CONTENT[domainId];

    document.getElementById('study-back-btn').style.display = 'block';
    document.getElementById('study-title').textContent = `${domain.icon} ${domain.name}`;
    document.getElementById('study-subtitle').textContent = `${domain.weight} do exame • ${domain.topics.length} tópicos`;

    document.getElementById('study-domains').classList.add('hidden');
    document.getElementById('study-content').classList.remove('hidden');

    renderStudyContent();
}

function renderStudyContent() {
    const domain = STUDY_CONTENT[currentStudyDomain];
    const container = document.getElementById('study-content');
    container.innerHTML = '';

    // Topic navigation tabs
    const nav = document.createElement('div');
    nav.className = 'study-topic-nav';
    domain.topics.forEach((topic, index) => {
        const btn = document.createElement('button');
        btn.className = `topic-nav-btn ${index === currentStudyTopic ? 'active' : ''}`;
        btn.textContent = topic.title;
        btn.onclick = () => {
            currentStudyTopic = index;
            renderStudyContent();
        };
        nav.appendChild(btn);
    });
    container.appendChild(nav);

    // Topic content
    const topic = domain.topics[currentStudyTopic];
    const content = document.createElement('div');
    content.className = 'study-topic-content';

    // Apply glossary tooltips to text content
    const conceptWithTooltips = applyGlossaryTooltips(topic.concept);
    const keyPointsWithTooltips = topic.keyPoints.map(point => applyGlossaryTooltips(point));
    const descriptionWithTooltips = applyGlossaryTooltips(topic.realCase.description);

    content.innerHTML = `
        <!-- Concept Section -->
        <div class="concept-section">
            <h3>📘 Conceito</h3>
            <div class="concept-text">${conceptWithTooltips}</div>
        </div>
        
        <!-- Key Points -->
        <div class="key-points">
            <h4>🎯 Pontos-Chave</h4>
            <ul>
                ${keyPointsWithTooltips.map(point => `<li>${point}</li>`).join('')}
            </ul>
        </div>
        
        <!-- Code Example -->
        <div class="code-example">
            <div class="code-example-header">
                <h4>💻 Exemplo Prático</h4>
            </div>
            <pre><code>${escapeHtml(topic.example)}</code></pre>
        </div>
        
        <!-- Real Case Study -->
        <div class="real-case">
            <div class="real-case-header">
                <span>🏢</span>
                <h4>Caso Real: ${topic.realCase.title}</h4>
            </div>
            <p>${descriptionWithTooltips}</p>
            <div class="real-case-impact">📊 Impacto: ${topic.realCase.impact}</div>
        </div>
        
        <!-- Navigation -->
        <div class="topic-navigation">
            <button class="topic-nav-arrow" onclick="prevTopic()" ${currentStudyTopic === 0 ? 'disabled' : ''}>
                ← Anterior
            </button>
            <button class="topic-nav-arrow" onclick="nextTopic()" ${currentStudyTopic === domain.topics.length - 1 ? 'disabled' : ''}>
                Próximo →
            </button>
        </div>
    `;

    container.appendChild(content);
}

function prevTopic() {
    if (currentStudyTopic > 0) {
        currentStudyTopic--;
        renderStudyContent();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

function nextTopic() {
    const domain = STUDY_CONTENT[currentStudyDomain];
    if (currentStudyTopic < domain.topics.length - 1) {
        currentStudyTopic++;
        renderStudyContent();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===== Utility Functions =====
function saveStats() {
    localStorage.setItem('caip210_stats', JSON.stringify(stats));
}

function getLocalStats() {
    return JSON.parse(localStorage.getItem('caip210_stats')) || stats;
}

function updateProgress() {
    const s = getLocalStats();
    const totalQuestions = QUESTIONS.length;
    const uniqueAnswered = Math.min(s.totalAnswered, totalQuestions);
    const progress = Math.round((uniqueAnswered / totalQuestions) * 100);

    document.getElementById('overall-progress').style.width = progress + '%';
    document.getElementById('progress-text').textContent = progress + '%';

    // Update domain cards on home
    Object.keys(DOMAINS).forEach(domainId => {
        const ds = s.domainStats[domainId];
        const domainQuestions = getQuestionsByDomain(parseInt(domainId)).length;
        const domainProgress = ds.answered > 0
            ? Math.round((ds.correct / ds.answered) * 100)
            : 0;

        const fill = document.querySelector(`.mini-fill[data-domain="${domainId}"]`);
        if (fill) fill.style.width = domainProgress + '%';
    });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    updateProgress();
});

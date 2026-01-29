// CAIP-210 Advanced Questions - Extra Set 3
// Based on grok-CAIP-210.txt advanced topics
// Manually curated for quality and completeness

const questionsExtra3 = [
    // Domain 1: AI & ML Fundamentals
    {
        id: 129,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Uma empresa global de logística enfrenta flutuações imprevisíveis na demanda devido a eventos geopolíticos e mudanças climáticas. Qual abordagem integrada seria mais apropriada para maximizar valor de negócio enquanto mitiga riscos éticos?",
        options: [
            "Formular como regressão simples, ignorando ética para focar em precisão",
            "Usar Design of Experiments (DOE) para variáveis controláveis, integrando análise de viés e transparência como métricas de performance",
            "Aplicar clustering não supervisionado sem formulação explícita",
            "Priorizar reinforcement learning sem considerar stakeholders externos"
        ],
        correct: 1,
        explanation: "A formulação de problemas ML deve integrar DOE para variáveis independentes/dependentes, enquanto ética (viés, transparência) deve ser considerada desde o início. Isso preserva integridade e alinha com stakeholders."
    },
    {
        id: 130,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Qual é o impacto mais crítico da aleatoriedade e incerteza em um modelo ML estocástico ao resolver problemas como previsão de crises de saúde pública?",
        options: [
            "Garante previsões perfeitas para eventos individuais",
            "Permite capturar padrões gerais, mas exige mitigação via ensemble methods para reduzir variância",
            "Elimina a necessidade de experimentação DOE",
            "Torna todos os modelos determinísticos com dados suficientes"
        ],
        correct: 1,
        explanation: "Modelos estocásticos capturam padrões gerais, mas aleatoriedade causa variância. Ensemble methods (como Random Forest, Gradient Boosting) mitigam essa variância, especialmente em cenários incertos."
    },

    // Domain 2: Data Preparation
    {
        id: 131,
        domain: 2,
        domainName: "Data Preparation",
        question: "Uma empresa de saúde coleta dados de wearables com 30% de missing values e outliers de sensores defeituosos. Qual estratégia integrada minimiza viés preservando integridade estatística?",
        options: [
            "Excluir todos dados com missing values para simplicidade",
            "Usar imputação mediana para outliers, Box-Cox para normalização, e embedding para dados não estruturados, com verificação ética para PII",
            "Aplicar deduplication sem transformação",
            "Ignorar dados não estruturados, focando apenas em numéricos"
        ],
        correct: 1,
        explanation: "Preparação robusta inclui: imputação mediana (robusta a outliers), Box-Cox para distribuições skewed, embedding para dados não estruturados, e verificação ética para proteger informações pessoais identificáveis (PII)."
    },
    {
        id: 132,
        domain: 2,
        domainName: "Data Preparation",
        question: "Qual é o desafio mais crítico ao trabalhar com dados de áudio não estruturados em um sistema de detecção de fraude vocal?",
        options: [
            "Baixa amplitude ignora sampling rate",
            "Necessidade de Fourier transformation e MFCCs para feature extraction, lidando com noise e periodicity",
            "Conversão direta para texto sem preprocessing",
            "Ignorar spectrograms para velocidade"
        ],
        correct: 1,
        explanation: "Dados de áudio requerem Fourier transformation para análise de frequência e MFCCs (Mel-Frequency Cepstral Coefficients) para extração de features. Noise e periodicity afetam significativamente a qualidade."
    },
    {
        id: 133,
        domain: 2,
        domainName: "Data Preparation",
        question: "Em um dataset com 50% de texto não estruturado de reviews, qual sequência de preprocessing é mais eficaz para sentiment analysis?",
        options: [
            "Tokenization → Stemming → Bag of words → Stop words removal",
            "Stop words removal → Tokenization → Lemmatization → Embedding",
            "Embedding direto sem tokenization",
            "Deduplication → Normalization → Binning"
        ],
        correct: 1,
        explanation: "A sequência correta para text preprocessing é: remover stop words, tokenizar, aplicar lemmatization (melhor que stemming para preservar significado), e então criar embeddings para representação vetorial densa."
    },
    {
        id: 134,
        domain: 2,
        domainName: "Data Preparation",
        question: "Qual é o impacto da curse of dimensionality em large datasets?",
        options: [
            "Aumenta performance automaticamente",
            "Reduz capacidade de aprender padrões, exigindo dimensionality reduction",
            "Elimina necessidade de feature selection",
            "Torna todas features relevantes"
        ],
        correct: 1,
        explanation: "A maldição da dimensionalidade ocorre quando muitas features em relação às amostras reduzem a capacidade do modelo de aprender padrões úteis. Técnicas como PCA, feature selection, ou regularização são necessárias."
    },

    // Domain 3: Training & Tuning
    {
        id: 135,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Um modelo de detecção de câncer tem 98% accuracy no treino mas cai para 70% em validação devido a imbalance e noise. Qual estratégia resolve overfitting e melhora generalização?",
        options: [
            "Aumentar epochs sem cross-validation",
            "Usar k-fold stratified cross-validation, learning curves para bias-variance, e regularization",
            "Ignorar métricas como F1, focando apenas em accuracy",
            "Reduzir dados para simplicidade"
        ],
        correct: 1,
        explanation: "Stratified k-fold mantém proporção de classes, learning curves diagnosticam bias-variance tradeoff, e regularização (L1/L2) previne overfitting. Accuracy sozinha é inadequada para dados imbalanced."
    },
    {
        id: 136,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Qual é o papel das learning curves em diagnosticar irreducible error?",
        options: [
            "Mostram quando mais dados não reduzem error",
            "Ignoram bias",
            "Sempre indicam underfitting",
            "Substituem cross-validation"
        ],
        correct: 0,
        explanation: "Learning curves mostram como erro de treino e validação evoluem com mais dados. Quando ambas as curvas convergem para um patamar, indica irreducible error (erro inerente ao problema)."
    },
    {
        id: 137,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Em um modelo black box, como avaliar performance além de accuracy?",
        options: [
            "Apenas com AUC",
            "Usar Precision-Recall Curve para imbalance, F1 para tradeoff, e explainability tools",
            "Ignorar variance",
            "Focar apenas em training time"
        ],
        correct: 1,
        explanation: "Para avaliação completa: PRC para dados imbalanced, F1 score para balance precision-recall, e ferramentas de explainability (SHAP, LIME) para entender decisões do modelo."
    },
    {
        id: 138,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Cenário: Treino iterativo com overfitting. Como otimizar hyperparameters eficientemente?",
        options: [
            "Grid search exaustivo",
            "Bayesian optimization para efficiency em large spaces",
            "Randomized search sem distribuição",
            "Manual tuning"
        ],
        correct: 1,
        explanation: "Bayesian optimization usa probabilidades para explorar o espaço de hiperparâmetros de forma inteligente, sendo muito mais eficiente que grid search em espaços grandes."
    },
    {
        id: 139,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Qual métrica é melhor para imbalanced classification?",
        options: [
            "Accuracy",
            "F1 score",
            "MSE",
            "R²"
        ],
        correct: 1,
        explanation: "F1 score é a média harmônica de precision e recall, sendo ideal para dados imbalanced. Accuracy pode ser enganosa (ex: 99% accuracy prevendo sempre a classe majoritária)."
    },

    // Domain 3: Advanced Algorithms
    {
        id: 140,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Modelo de precificação de imóveis com alta multicolinearidade e outliers extremos. Qual abordagem é mais robusta para obter coeficientes interpretáveis?",
        options: [
            "Regressão linear simples sem regularização",
            "Ridge regression (L2) combinada com análise VIF e remoção seletiva de variáveis colineares",
            "Lasso regression (L1) sem análise prévia",
            "Batch Gradient Descent sem regularização"
        ],
        correct: 1,
        explanation: "Ridge (L2) reduz impacto da multicolinearidade sem zerar coeficientes (mantém interpretabilidade). VIF (Variance Inflation Factor) ajuda a identificar e tratar colinearidade explicitamente."
    },
    {
        id: 141,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Qual afirmação é FALSA sobre regularização em regressão linear?",
        options: [
            "Ridge penaliza a soma dos quadrados dos coeficientes",
            "Lasso pode zerar coeficientes e realizar seleção automática de variáveis",
            "Elastic Net nunca zera coeficientes quando λ é pequeno",
            "Ridge mantém todos os coeficientes diferentes de zero (exceto em casos extremos)"
        ],
        correct: 2,
        explanation: "Elastic Net PODE zerar coeficientes (herda propriedade do Lasso), mas de forma mais controlada que Lasso puro. Combina L1 e L2 para balancear feature selection e estabilidade."
    },
    {
        id: 142,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Previsão de vendas com sazonalidade semanal, feriados e promoções. Qual pipeline de modelagem é mais apropriado?",
        options: [
            "ARIMA simples sem diferenciação",
            "SARIMA com componentes sazonais + regressores exógenos (SARIMAX)",
            "VAR sem verificação de estacionariedade",
            "ARIMA com ordem (0,0,0)"
        ],
        correct: 1,
        explanation: "SARIMAX lida com sazonalidade (componente S), tendência (via diferenciação I), e variáveis exógenas (X) como promoções e feriados — ideal para forecasting multivariado com sazonalidade."
    },
    {
        id: 143,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Qual teste estatístico verifica estacionariedade em séries temporais antes de aplicar ARIMA?",
        options: [
            "Teste de Dickey-Fuller aumentado (ADF)",
            "Teste de Shapiro-Wilk",
            "Teste de Levene",
            "Teste qui-quadrado"
        ],
        correct: 0,
        explanation: "ADF (Augmented Dickey-Fuller) testa presença de raiz unitária, indicando não-estacionariedade. É o teste padrão antes de aplicar ARIMA em séries temporais."
    },
    {
        id: 144,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Classificação de risco de crédito com 3% de inadimplentes e 200 features. Qual combinação oferece melhor trade-off entre interpretabilidade e performance?",
        options: [
            "Logistic Regression + SMOTE + L1 regularization",
            "k-NN com k=1 + sem balanceamento",
            "Logistic Regression sem regularização + undersampling aleatório",
            "k-NN com k=50 + sem feature selection"
        ],
        correct: 0,
        explanation: "Logistic Regression oferece interpretabilidade via coeficientes; L1 (Lasso) faz feature selection automática; SMOTE (Synthetic Minority Over-sampling) lida com imbalance preservando informação."
    },
    {
        id: 145,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Em classificação binária, mover o threshold de 0.5 para 0.7 geralmente causa:",
        options: [
            "Aumento de recall e diminuição de precision",
            "Aumento de precision e diminuição de recall",
            "Aumento simultâneo de ambos",
            "Diminuição simultânea de ambos"
        ],
        correct: 1,
        explanation: "Threshold mais alto (0.7) torna o modelo mais conservador → classifica positivo apenas com alta confiança → menos falsos positivos → maior precision, mas menor recall (perde verdadeiros positivos)."
    },
    {
        id: 146,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Qual linkage é mais robusto a outliers em hierarchical clustering?",
        options: [
            "Single linkage",
            "Complete linkage",
            "Average linkage",
            "Ward linkage"
        ],
        correct: 1,
        explanation: "Complete linkage considera a distância máxima entre pontos de clusters diferentes, sendo menos sensível a outliers que single linkage (que usa distância mínima e pode criar cadeias)."
    },
    {
        id: 147,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Modelo de churn com 1.2M linhas e 300 features com alta variância. Qual ensemble oferece melhor equilíbrio entre performance e tempo de treinamento?",
        options: [
            "Uma única árvore CART profunda",
            "Random Forest com 500 árvores + max_features = sqrt(n)",
            "Gradient Boosting com learning rate 0.01 e 2000 estimadores",
            "Bagging com árvores rasas sem feature subsampling"
        ],
        correct: 1,
        explanation: "Random Forest reduz variância via bagging + random feature selection, é escalável para grandes datasets, e fornece feature importance. Gradient Boosting seria muito lento com 2000 estimadores."
    },
    {
        id: 148,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Em SVM para regressão (SVR), o parâmetro ε (epsilon) controla:",
        options: [
            "A largura da margem de tolerância",
            "A penalidade por violações da margem",
            "O grau do kernel polinomial",
            "O raio do kernel RBF"
        ],
        correct: 0,
        explanation: "Em SVR, ε define a faixa de erro aceitável sem penalidade (tubo epsilon). Predições dentro de ±ε do valor real não são penalizadas, permitindo controle sobre a sensibilidade do modelo."
    },
    {
        id: 149,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Detecção de defeitos em imagens industriais de alta resolução. Qual arquitetura é mais indicada para capturar texturas finas com eficiência computacional?",
        options: [
            "MLP com 5 camadas densas",
            "CNN com várias camadas convolucionais + batch normalization + global average pooling",
            "RNN com LSTM para pixels sequenciais",
            "Transformer sem convolução"
        ],
        correct: 1,
        explanation: "CNN é padrão ouro para visão computacional: camadas convolucionais capturam padrões locais e hierárquicos, batch normalization estabiliza treinamento, e global average pooling reduz parâmetros mantendo informação espacial."
    },

    // Domain 4: MLOps & Production
    {
        id: 150,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Qual é a principal vantagem de usar Docker + Kubernetes para deployment de modelos ML comparado a servidores bare-metal?",
        options: [
            "Menor latência em predição",
            "Portabilidade, escalabilidade horizontal e isolamento de dependências",
            "Menor consumo de memória",
            "Maior interpretabilidade do modelo"
        ],
        correct: 1,
        explanation: "Docker fornece containerização (isolamento de dependências e portabilidade), enquanto Kubernetes oferece orquestração (escalabilidade horizontal automática, self-healing, load balancing) — essenciais para MLOps moderno."
    }
];

// Merge with main questions array
questions.push(...questionsExtra3);


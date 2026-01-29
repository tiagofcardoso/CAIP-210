// Additional CAIP-210 Questions - Set 3
// Based on CertNexus Certified AI Practitioner official course material

const questionsExtra2 = [
    // More Domain 1 questions
    {
        id: 101,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "O que é Computer Vision e quais técnicas inclui?",
        options: [
            "Técnica de visualização de dados",
            "Tipo de monitor de computador",
            "Área de AI que processa dados visuais (imagens/vídeos), incluindo object recognition, motion detection e image generation",
            "Técnica de debugging visual"
        ],
        correct: 2,
        explanation: "Computer Vision é um termo guarda-chuva para técnicas que processam dados visuais: object recognition, object classification, image generation, motion detection, trajectory estimation, video tracking e navigation."
    },
    {
        id: 102,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "O que é NLP (Natural Language Processing)?",
        options: [
            "Um tipo de banco de dados",
            "Área de AI onde computadores trabalham com linguagens humanas usando técnicas de ML",
            "Uma linguagem de programação",
            "Uma ferramenta de visualização"
        ],
        correct: 1,
        explanation: "NLP é o termo geral para tarefas onde computadores trabalham com linguagens humanas usando ML. Inclui: speech recognition, text analysis, natural language understanding/generation/translation."
    },
    {
        id: 103,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Quando NÃO usar AI/ML para resolver um problema?",
        options: [
            "Quando há muitos dados disponíveis",
            "Quando o problema pode ser resolvido com lógica tradicional simples ou falta dados suficientes",
            "Quando há budget disponível",
            "Quando o problema é complexo"
        ],
        correct: 1,
        explanation: "AI/ML pode ser caro, demorado e arriscado. Se o problema pode ser resolvido com programação tradicional simples, ou se não há dados suficientes, AI/ML pode não ser justificável."
    },
    {
        id: 104,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "O que representam 'labels' em supervised learning?",
        options: [
            "Os nomes das colunas",
            "As respostas corretas conhecidas (ground truth) usadas para treinar o modelo",
            "Os tipos de dados",
            "Os nomes das features"
        ],
        correct: 1,
        explanation: "Labels são as respostas corretas conhecidas (ground truth) em um dataset de supervised learning. O modelo aprende comparando suas predições com esses labels durante o treino."
    },

    // More Domain 2 questions
    {
        id: 105,
        domain: 2,
        domainName: "Data Preparation",
        question: "O que é 'Cold-deck Imputation' para missing values?",
        options: [
            "Remover valores ausentes",
            "Deixar os valores ausentes como estão",
            "Usar valores de um dataset externo similar para preencher valores ausentes",
            "Usar a média do próprio dataset"
        ],
        correct: 2,
        explanation: "Cold-deck imputation usa valores de um dataset externo similar para preencher valores ausentes, diferente de hot-deck que usa valores do mesmo dataset."
    },
    {
        id: 106,
        domain: 2,
        domainName: "Data Preparation",
        question: "O que é 'Regression Imputation' para missing values?",
        options: [
            "Usar a mediana",
            "Substituir por zero",
            "Remover linhas com valores ausentes",
            "Usar um modelo de regressão treinado em outras features para prever o valor ausente"
        ],
        correct: 3,
        explanation: "Regression imputation treina um modelo de regressão usando outras features para prever o valor ausente. É mais sofisticado que mean/mode imputation mas pode introduzir viés."
    },
    {
        id: 107,
        domain: 2,
        domainName: "Data Preparation",
        question: "Qual transformação é usada para dados com skewness negativa?",
        options: [
            "Exponential ou square transformation",
            "Standardization",
            "One-hot encoding",
            "Log transformation"
        ],
        correct: 0,
        explanation: "Para skewness negativa (cauda à esquerda), transformações exponenciais ou quadráticas ajudam. Log transformation é usada para skewness positiva."
    },
    {
        id: 108,
        domain: 2,
        domainName: "Data Preparation",
        question: "O que é 'Box-Cox Transformation'?",
        options: [
            "Técnica de encoding",
            "Uma técnica de visualização",
            "Método de feature selection",
            "Transformação paramétrica que encontra automaticamente a melhor power transformation para normalizar dados"
        ],
        correct: 3,
        explanation: "Box-Cox é uma transformação paramétrica que encontra automaticamente o melhor parâmetro lambda para transformar dados skewed em distribuição mais normal."
    },
    {
        id: 109,
        domain: 2,
        domainName: "Data Preparation",
        question: "O que é 't-SNE' usado para?",
        options: [
            "Treinamento de modelos",
            "Imputação de missing values",
            "Redução de dimensionalidade para visualização, preservando estrutura local dos dados",
            "Encoding de categorias"
        ],
        correct: 2,
        explanation: "t-SNE (t-Distributed Stochastic Neighbor Embedding) é uma técnica de redução de dimensionalidade usada principalmente para visualização de dados de alta dimensionalidade em 2D ou 3D."
    },
    {
        id: 110,
        domain: 2,
        domainName: "Data Preparation",
        question: "O que é 'Target Encoding' para variáveis categóricas?",
        options: [
            "Remover categorias raras",
            "Criar colunas dummy",
            "Ordenar categorias alfabeticamente",
            "Substituir categorias pela média do target para aquela categoria"
        ],
        correct: 3,
        explanation: "Target encoding substitui cada categoria pela média do target variable para aquela categoria. Útil para high cardinality, mas pode causar data leakage se não feito corretamente."
    },

    // More Domain 3 questions
    {
        id: 111,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é 'Stratified K-Fold Cross Validation'?",
        options: [
            "Cross-validation simples",
            "Cross-validation que mantém a mesma proporção de classes em cada fold",
            "Validação apenas no test set",
            "Validação sem folds"
        ],
        correct: 1,
        explanation: "Stratified K-Fold garante que cada fold mantenha aproximadamente a mesma proporção de classes que o dataset original. Importante para datasets desbalanceados."
    },
    {
        id: 112,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é 'Grid Search' para tuning de hiperparâmetros?",
        options: [
            "Busca manual",
            "Busca exaustiva que testa todas as combinações possíveis de hiperparâmetros especificados",
            "Otimização automática",
            "Busca aleatória de parâmetros"
        ],
        correct: 1,
        explanation: "Grid Search testa exaustivamente todas as combinações de hiperparâmetros em uma grade especificada. É completo mas pode ser computacionalmente caro."
    },
    {
        id: 113,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é 'Random Search' para tuning de hiperparâmetros?",
        options: [
            "Amostra aleatória de combinações de hiperparâmetros, geralmente mais eficiente que grid search",
            "Busca exaustiva",
            "Não faz busca, usa defaults",
            "Busca manual"
        ],
        correct: 0,
        explanation: "Random Search amostra aleatoriamente combinações de hiperparâmetros. Estudos mostram que é frequentemente mais eficiente que grid search para encontrar bons hiperparâmetros."
    },
    {
        id: 114,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é a métrica 'AUC-ROC'?",
        options: [
            "Accuracy Under Curve",
            "Average User Count",
            "Automated Utility Check",
            "Área sob a curva ROC, mede capacidade do modelo de distinguir entre classes"
        ],
        correct: 3,
        explanation: "AUC-ROC (Area Under the ROC Curve) mede a capacidade do classificador de distinguir entre classes. Valor de 0.5 = random, 1.0 = perfeito. Útil para dados desbalanceados."
    },
    {
        id: 115,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é 'Early Stopping' no treinamento de redes neurais?",
        options: [
            "Parar o treino quando a performance no validation set para de melhorar, prevenindo overfitting",
            "Parar quando acabar os dados",
            "Nunca parar o treinamento",
            "Parar após um tempo fixo"
        ],
        correct: 0,
        explanation: "Early stopping monitora a performance no validation set e para o treino quando ela para de melhorar (ou começa a piorar), prevenindo overfitting."
    },
    {
        id: 116,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é 'Learning Rate' em gradient descent?",
        options: [
            "O tamanho do passo dado na direção do gradiente a cada iteração",
            "A velocidade do computador",
            "O número de epochs",
            "O tamanho do batch"
        ],
        correct: 0,
        explanation: "Learning rate determina o tamanho do passo na direção oposta ao gradiente. Muito alto pode não convergir, muito baixo pode ser lento e ficar preso em mínimos locais."
    },
    {
        id: 117,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é 'Ensemble Learning'?",
        options: [
            "Usar múltiplas GPUs",
            "Treinar em múltiplos datasets",
            "Combinar múltiplos modelos para obter melhor performance que qualquer modelo individual",
            "Usar um único modelo forte"
        ],
        correct: 2,
        explanation: "Ensemble learning combina previsões de múltiplos modelos (bagging, boosting, stacking) para obter melhor generalização e performance que modelos individuais."
    },
    {
        id: 118,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é 'Bagging' (Bootstrap Aggregating)?",
        options: [
            "Treinar múltiplos modelos em amostras bootstrap dos dados e agregar suas previsões",
            "Aumentar dados artificialmente",
            "Reduzir features",
            "Treinar modelos sequencialmente"
        ],
        correct: 0,
        explanation: "Bagging treina múltiplos modelos independentemente em diferentes amostras bootstrap (com reposição) dos dados, e agrega suas previsões (voto/média). Random Forest usa bagging."
    },
    {
        id: 119,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é 'Boosting'?",
        options: [
            "Treinar modelos em paralelo",
            "Treinar modelos sequencialmente, cada um focando nos erros do anterior",
            "Comprimir modelos",
            "Aumentar a velocidade de treino"
        ],
        correct: 1,
        explanation: "Boosting treina modelos sequencialmente, onde cada modelo tenta corrigir os erros do anterior. Exemplos: AdaBoost, Gradient Boosting, XGBoost."
    },
    {
        id: 120,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é 'LSTM' em redes neurais?",
        options: [
            "Long Short-Term Memory - tipo de RNN que resolve o problema de vanishing gradient para sequências longas",
            "Um tipo de rede convolucional",
            "Um otimizador",
            "Uma técnica de regularização"
        ],
        correct: 0,
        explanation: "LSTM (Long Short-Term Memory) é um tipo de RNN com células de memória e gates que permitem aprender dependências de longo prazo, resolvendo o problema de vanishing gradient."
    },

    // More Domain 4 questions
    {
        id: 121,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'Infrastructure as Code (IaC)' em MLOps?",
        options: [
            "Scripts de teste",
            "Gerenciar e provisionar infraestrutura através de código versionado (Terraform, CloudFormation)",
            "Código que roda em servidores",
            "Código de machine learning"
        ],
        correct: 1,
        explanation: "IaC permite gerenciar e provisionar toda a infraestrutura (servidores, redes, storage) através de arquivos de configuração versionados, garantindo reprodutibilidade e automação."
    },
    {
        id: 122,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'model throughput'?",
        options: [
            "A precisão do modelo",
            "O tamanho do modelo",
            "O tempo de treinamento",
            "O número de requisições/previsões que o modelo pode processar por unidade de tempo"
        ],
        correct: 3,
        explanation: "Throughput mede quantas requisições o modelo pode processar por segundo/minuto. Importante para sistemas de alto volume onde muitas previsões são necessárias simultaneamente."
    },
    {
        id: 123,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'model observability'?",
        options: [
            "Monitorar apenas erros",
            "Ver as previsões do modelo",
            "Capacidade de entender o estado interno e comportamento do modelo em produção através de métricas e logs",
            "Visualizar o código do modelo"
        ],
        correct: 2,
        explanation: "Observability é a capacidade de entender o estado interno do sistema através de outputs externos. Inclui métricas (latência, throughput), logs e traces para debugging."
    },
    {
        id: 124,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'model quantization'?",
        options: [
            "Aumentar a complexidade do modelo",
            "Treinar com mais dados",
            "Dividir o modelo em partes",
            "Reduzir a precisão numérica dos pesos (ex: float32 para int8) para diminuir tamanho e aumentar velocidade"
        ],
        correct: 3,
        explanation: "Quantization reduz a precisão numérica dos pesos e ativações (ex: de 32-bit float para 8-bit int), diminuindo tamanho do modelo e acelerando inferência, com perda mínima de accuracy."
    },
    {
        id: 125,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'model pruning'?",
        options: [
            "Adicionar mais camadas",
            "Treinar por mais tempo",
            "Remover conexões/neurônios com pesos próximos de zero para reduzir tamanho do modelo",
            "Aumentar learning rate"
        ],
        correct: 2,
        explanation: "Pruning remove pesos, neurônios ou camadas que contribuem pouco para a previsão (geralmente com valores próximos de zero), reduzindo o tamanho do modelo sem perder muita accuracy."
    },
    {
        id: 126,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'knowledge distillation' em model compression?",
        options: [
            "Treinar um modelo menor (student) para imitar um modelo maior (teacher)",
            "Extrair features do modelo",
            "Treinar um modelo do zero",
            "Documentar conhecimento sobre o modelo"
        ],
        correct: 0,
        explanation: "Knowledge distillation treina um modelo menor (student) para reproduzir o comportamento de um modelo maior e mais complexo (teacher), transferindo 'conhecimento' para um modelo mais eficiente."
    },
    {
        id: 127,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que são 'SLAs' (Service Level Agreements) para modelos em produção?",
        options: [
            "Logs de sistema",
            "Algoritmos de ML",
            "Acordos formais sobre níveis de serviço esperados (latência, uptime, accuracy)",
            "Scripts de automação"
        ],
        correct: 2,
        explanation: "SLAs são acordos formais que definem níveis de serviço esperados: tempo de resposta máximo, disponibilidade mínima, taxa de erro aceitável, etc. Crucial para aplicações de produção."
    },
    {
        id: 128,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'edge deployment' para modelos de ML?",
        options: [
            "Deploy em datacenters",
            "Deploy em ambiente de teste",
            "Deploy gradual",
            "Deploy de modelos em dispositivos locais (smartphones, IoT) em vez de servidores na nuvem"
        ],
        correct: 3,
        explanation: "Edge deployment executa o modelo diretamente em dispositivos locais (edge devices), reduzindo latência, custos de rede e permitindo funcionamento offline. Requer modelos otimizados."
    }
];

// Merge with main questions array
questions.push(...questionsExtra2);

// Additional CAIP-210 Questions - Set 2
// Based on CertNexus Certified AI Practitioner official course material

const questionsExtra = [
    // ===== DOMAIN 1: AI & ML FUNDAMENTALS - Additional Questions =====
    {
        id: 61,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Qual é a diferença entre collaborative filtering e content filtering em sistemas de recomendação?",
        options: [
            "Collaborative filtering é mais lento; content filtering é mais rápido",
            "Não há diferença, são a mesma técnica",
            "Collaborative filtering usa similaridade entre usuários; content filtering analisa atributos dos itens",
            "Collaborative filtering requer mais dados que content filtering"
        ],
        correct: 2,
        explanation: "Collaborative filtering encontra usuários similares que 'gostam' das mesmas coisas e recomenda baseado nisso. Content filtering analisa os atributos do item (gênero, autor, etc.) e compara com o perfil do usuário."
    },
    {
        id: 62,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Em Reinforcement Learning, o que é um 'reward'?",
        options: [
            "O modelo final treinado",
            "Uma função que indica ao agente quais ações são desejáveis e devem ser repetidas",
            "O dataset usado para treinamento",
            "O ambiente onde o agente opera"
        ],
        correct: 1,
        explanation: "O reward é uma função que indica ao agente como ele deve agir - quais ações são desejáveis e valem a pena repetir no futuro. O objetivo do agente é maximizar os rewards acumulados."
    },
    {
        id: 63,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "O que é 'forecasting' em machine learning?",
        options: [
            "Estimar a magnitude de um número baseado em mudanças ao longo do tempo",
            "Agrupar dados similares",
            "Classificar dados em categorias",
            "Detectar anomalias nos dados"
        ],
        correct: 0,
        explanation: "Forecasting é estimar a magnitude de algum número baseado na mudança de algum evento ao longo do tempo. Por exemplo, prever a variação do preço de uma ação no dia seguinte."
    },
    {
        id: 64,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "O que é um sistema de 'diagnosis' em AI/ML?",
        options: [
            "Um sistema que traduz linguagens",
            "Um sistema que recomenda produtos",
            "Um sistema que determina a causa e natureza de comportamentos anômalos em um ambiente",
            "Um sistema que prevê valores futuros"
        ],
        correct: 2,
        explanation: "Um sistema de diagnóstico determina a causa e natureza de comportamentos, atividades ou condições anômalas em um ambiente (corpo humano, dispositivo físico, software, etc.)."
    },
    {
        id: 65,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "O que é 'self-training' em semi-supervised learning?",
        options: [
            "Criar um baseline model com dados rotulados e usá-lo para gerar pseudo-labels nos dados não rotulados",
            "Deixar o modelo treinar indefinidamente",
            "Treinar o modelo sem nenhum dado",
            "Treinar múltiplos modelos em paralelo"
        ],
        correct: 0,
        explanation: "Self-training cria um modelo baseline com dados rotulados, usa esse modelo para estimar labels (pseudo-labels) nos dados não rotulados com alta confiança, e então retreina o modelo com o dataset expandido."
    },
    {
        id: 66,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "O que é 'co-training' em semi-supervised learning?",
        options: [
            "Criar dois modelos separados em duas porções diferentes dos dados (views), onde pseudo-labels de um treinam o outro",
            "Treinar dois usuários juntos",
            "Combinar dois datasets diferentes",
            "Treinar um modelo duas vezes"
        ],
        correct: 0,
        explanation: "Co-training cria dois modelos separados de duas 'views' do dataset (mesmas observações, features diferentes). Os pseudo-labels gerados por um modelo são usados para treinar o outro, e vice-versa."
    },
    {
        id: 67,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Qual técnica de ML é mais apropriada para sistemas autônomos como robôs?",
        options: [
            "Semi-supervised learning",
            "Reinforcement learning",
            "Unsupervised learning apenas",
            "Supervised learning apenas"
        ],
        correct: 1,
        explanation: "Reinforcement learning é comumente usado para construir o aspecto autônomo de robôs. O agente aprende continuamente do ambiente enquanto ele muda de estado, maximizando rewards."
    },
    {
        id: 68,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "O que significa criar um 'proof of concept (POC)' em projetos de ML?",
        options: [
            "O deploy do modelo em produção",
            "Uma solução preliminar em pequena escala para avaliar a viabilidade de diferentes algoritmos",
            "A documentação completa do projeto",
            "O modelo final pronto para produção"
        ],
        correct: 1,
        explanation: "Um POC é uma solução preliminar em pequena quantidade de dados (reais ou artificiais) para avaliar a performance de diferentes algoritmos e determinar a probabilidade de sucesso."
    },

    // ===== DOMAIN 2: DATA PREPARATION - Additional Questions =====
    {
        id: 69,
        domain: 2,
        domainName: "Data Preparation",
        question: "Qual a diferença entre dados estruturados e não estruturados?",
        options: [
            "Dados estruturados são sempre numéricos",
            "Dados estruturados são maiores que não estruturados",
            "Dados estruturados facilitam busca e filtragem (como planilhas); não estruturados não (como imagens, vídeos)",
            "Não há diferença prática entre eles"
        ],
        correct: 2,
        explanation: "Dados estruturados estão em formato que facilita busca, filtragem e extração (como planilhas ou databases). Dados não estruturados (imagens, vídeos, texto livre) não estão em containers predefinidos."
    },
    {
        id: 70,
        domain: 2,
        domainName: "Data Preparation",
        question: "O que é 'ETL' no contexto de preparação de dados?",
        options: [
            "Extract, Transform, Load - processo de combinar dados de múltiplas fontes e prepará-los",
            "Encoding, Tokenization, Lemmatization",
            "Evaluate, Train, Learn",
            "Error Testing and Logging"
        ],
        correct: 0,
        explanation: "ETL (Extract, Transform, Load) é o processo de extrair dados de várias fontes, transformá-los em formato adequado, e carregá-los no destino final para análise e modelagem."
    },
    {
        id: 71,
        domain: 2,
        domainName: "Data Preparation",
        question: "O que são 'outliers' em um dataset?",
        options: [
            "Valores no formato incorreto",
            "Valores fora da distribuição principal, desviando significativamente do resto",
            "Valores ausentes",
            "Valores duplicados"
        ],
        correct: 1,
        explanation: "Outliers são valores fora da distribuição principal, desviando significativamente do resto dos valores no dataset. Podem ser causados por erros de medição ou execução."
    },
    {
        id: 72,
        domain: 2,
        domainName: "Data Preparation",
        question: "O que é 'noise' no contexto de machine learning?",
        options: [
            "Interferência de hardware",
            "Dados ausentes",
            "Erros de programação",
            "Dados que não contribuem para fazer boas estimativas e podem atrapalhar o aprendizado"
        ],
        correct: 3,
        explanation: "Noise são dados (valores, features ou exemplos) que não são necessários para fazer boas estimativas. Podem dificultar que o algoritmo 'ouça' os padrões relevantes nos dados."
    },
    {
        id: 73,
        domain: 2,
        domainName: "Data Preparation",
        question: "Qual a diferença entre dados quantitativos e qualitativos?",
        options: [
            "Não há diferença, são sinônimos",
            "Quantitativos expressam magnitude numérica; qualitativos representam categorias sem ranking",
            "Quantitativos são sempre inteiros; qualitativos são decimais",
            "Quantitativos são mais importantes que qualitativos"
        ],
        correct: 1,
        explanation: "Dados quantitativos (numéricos) expressam magnitude (ex: milhas percorridas). Dados qualitativos (categóricos) representam categorias limitadas sem ranking inerente (ex: tipo de veículo)."
    },
    {
        id: 74,
        domain: 2,
        domainName: "Data Preparation",
        question: "O que são dados 'ordinais'?",
        options: [
            "Dados que podem ser ordenados/rankeados, mas não medem magnitude",
            "Dados numéricos contínuos",
            "Dados categóricos sem ordem",
            "Dados binários (0 ou 1)"
        ],
        correct: 0,
        explanation: "Dados ordinais podem ser ordenados, mas não medem magnitude. Ex: 'Excelente', 'Bom', 'Regular', 'Ruim' - há uma ordem, mas não uma quantidade mensurável entre eles."
    },
    {
        id: 75,
        domain: 2,
        domainName: "Data Preparation",
        question: "O que é 'data wrangling' ou 'data munging'?",
        options: [
            "O processo manual e tedioso de preparação e limpeza de dados",
            "Análise estatística de dados",
            "Treinamento de modelos",
            "Visualização de dados"
        ],
        correct: 0,
        explanation: "Data wrangling (ou data munging) é o processo de preparação de dados, particularmente quando feito manualmente ou fora de processos formais. Reflete o desafio e trabalho envolvido na tarefa."
    },
    {
        id: 76,
        domain: 2,
        domainName: "Data Preparation",
        question: "O que é um 'Data Lake'?",
        options: [
            "Uma técnica de feature engineering",
            "Um tipo de banco de dados relacional",
            "Um tipo de modelo de ML",
            "Repositório para ML e big data analytics que armazena dados estruturados e não estruturados em seu formato original"
        ],
        correct: 3,
        explanation: "Data Lake é usado para ML, big data analytics e data discovery. Armazena dados estruturados e não estruturados de várias fontes em seus formatos originais para possíveis usos futuros."
    },
    {
        id: 77,
        domain: 2,
        domainName: "Data Preparation",
        question: "Qual cálculo estatístico é válido para dados categóricos (qualitativos)?",
        options: [
            "Moda (mode)",
            "Média (mean)",
            "Mediana (median)",
            "Desvio padrão"
        ],
        correct: 0,
        explanation: "Para dados categóricos, a maioria das medidas de centro e dispersão são inapropriadas. A moda (valor mais frequente) é a única medida válida para dados puramente categóricos."
    },
    {
        id: 78,
        domain: 2,
        domainName: "Data Preparation",
        question: "O que é PII (Personally Identifiable Information) e por que é importante em ML?",
        options: [
            "Parâmetros de inicialização do modelo",
            "Informação sobre infraestrutura",
            "Informação sobre performance do modelo",
            "Dados sensíveis que identificam pessoas e devem ser protegidos por privacidade"
        ],
        correct: 3,
        explanation: "PII são dados que podem identificar, contatar ou localizar um indivíduo (nome, email, endereço, CPF). Devem ser protegidos para garantir privacidade e compliance com regulações como GDPR."
    },
    {
        id: 79,
        domain: 2,
        domainName: "Data Preparation",
        question: "O que significa 'data bias' na coleta de dados?",
        options: [
            "Dados no formato errado",
            "Dados muito grandes para processar",
            "Dados que exibem tendências ou preconceitos que podem não representar verdadeiramente a população",
            "Dados duplicados"
        ],
        correct: 2,
        explanation: "Data bias ocorre quando os dados exibem potencial para vieses sociais - podem ter sido registrados por pessoas com preconceitos ou o próprio processo de coleta pode ser tendencioso."
    },
    {
        id: 80,
        domain: 2,
        domainName: "Data Preparation",
        question: "O que é 'imbalanced data' e qual problema pode causar?",
        options: [
            "Dados sem valores numéricos",
            "Dados com muitas colunas",
            "Dados sem labels",
            "Dados com frequência desproporcional de certos valores, podendo causar super/subestimação de fatores"
        ],
        correct: 3,
        explanation: "Imbalanced data tem frequência desproporcional de certos valores, especialmente em variáveis categóricas alvo. O modelo pode superestimar ou subestimar a importância de certos fatores."
    },

    // ===== DOMAIN 3: TRAINING & TUNING - Additional Questions =====
    {
        id: 81,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é 'bias' no contexto de machine learning models?",
        options: [
            "Erro de hardware",
            "Erro sistemático do modelo que resulta em underfitting",
            "Preconceito nos dados",
            "Variação nos resultados"
        ],
        correct: 1,
        explanation: "Bias é o erro sistemático que um modelo comete. Alto bias resulta em underfitting - o modelo é muito simples e não captura os padrões nos dados."
    },
    {
        id: 82,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é 'variance' no contexto de machine learning?",
        options: [
            "O tamanho do dataset",
            "A velocidade de treinamento",
            "A média dos erros",
            "A sensibilidade do modelo a flutuações nos dados de treino, podendo causar overfitting"
        ],
        correct: 3,
        explanation: "Variance mede a sensibilidade do modelo a flutuações nos dados de treino. Alta variance resulta em overfitting - o modelo memoriza o ruído dos dados de treino."
    },
    {
        id: 83,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é o 'bias-variance tradeoff'?",
        options: [
            "Escolher entre velocidade e precisão",
            "A troca entre treino e teste",
            "O equilíbrio entre underfitting (alto bias) e overfitting (alta variance)",
            "O custo computacional vs qualidade"
        ],
        correct: 2,
        explanation: "O bias-variance tradeoff é o equilíbrio entre ter um modelo muito simples (alto bias, underfitting) e muito complexo (alta variance, overfitting). O objetivo é minimizar ambos."
    },
    {
        id: 84,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é Naive Bayes classifier?",
        options: [
            "Um classificador probabilístico baseado no teorema de Bayes, assumindo independência entre features",
            "Um algoritmo de clustering",
            "Um algoritmo de regressão",
            "Uma rede neural simples"
        ],
        correct: 0,
        explanation: "Naive Bayes é um classificador probabilístico baseado no teorema de Bayes. É 'naive' porque assume independência entre as features, o que simplifica os cálculos."
    },
    {
        id: 85,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é 'Silhouette Score' usado para avaliar?",
        options: [
            "Erro de regressão",
            "Performance de classificação",
            "Tempo de treinamento",
            "Qualidade dos clusters em algoritmos de clustering"
        ],
        correct: 3,
        explanation: "O Silhouette Score avalia a qualidade dos clusters, medindo quão similar um objeto é ao seu próprio cluster comparado a outros clusters. Varia de -1 a 1, onde maior é melhor."
    },
    {
        id: 86,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é 'Inertia' em k-Means clustering?",
        options: [
            "O número de clusters",
            "A variância total dos dados",
            "A soma das distâncias quadradas de cada ponto ao seu centroide",
            "A velocidade do algoritmo"
        ],
        correct: 2,
        explanation: "Inertia é a soma das distâncias quadradas de cada ponto ao centroide do seu cluster. Menor inertia indica clusters mais compactos. É usada no Elbow Method para escolher K."
    },
    {
        id: 87,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é DBSCAN clustering?",
        options: [
            "Um algoritmo de classificação supervisionada",
            "Um tipo de rede neural",
            "Um algoritmo baseado em densidade que encontra clusters de forma arbitrária e identifica outliers",
            "Um algoritmo que requer o número de clusters como entrada"
        ],
        correct: 2,
        explanation: "DBSCAN (Density-Based Spatial Clustering of Applications with Noise) encontra clusters baseado em densidade, pode descobrir clusters de formas arbitrárias e identifica outliers automaticamente."
    },
    {
        id: 88,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é 'Hierarchical Clustering'?",
        options: [
            "Clustering que requer K pré-definido",
            "Clustering que cria uma hierarquia de clusters, representada por um dendrograma",
            "Classificação em múltiplos níveis",
            "Clustering para dados hierárquicos apenas"
        ],
        correct: 1,
        explanation: "Hierarchical Clustering cria uma hierarquia de clusters, podendo ser aglomerativo (bottom-up) ou divisivo (top-down). O resultado é visualizado em um dendrograma."
    },
    {
        id: 89,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é a função 'softmax' em redes neurais?",
        options: [
            "Um otimizador",
            "Uma função de ativação que converte outputs em probabilidades que somam 1",
            "Uma função de perda",
            "Uma técnica de regularização"
        ],
        correct: 1,
        explanation: "Softmax é usada na camada de saída para classificação multi-classe. Converte os outputs em probabilidades que somam 1, permitindo interpretar como a probabilidade de cada classe."
    },
    {
        id: 90,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é 'dropout' em redes neurais?",
        options: [
            "Reduzir a taxa de aprendizado",
            "Remover camadas inteiras da rede",
            "Desativar aleatoriamente neurônios durante o treino para prevenir overfitting",
            "Parar o treinamento cedo"
        ],
        correct: 2,
        explanation: "Dropout é uma técnica de regularização que desativa aleatoriamente uma fração dos neurônios durante o treinamento, forçando a rede a não depender demais de nenhum neurônio específico."
    },
    {
        id: 91,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é 'batch size' no treinamento de redes neurais?",
        options: [
            "O tamanho da saída",
            "O número de exemplos processados antes de atualizar os pesos",
            "O tamanho total do dataset",
            "O número de camadas da rede"
        ],
        correct: 1,
        explanation: "Batch size é o número de exemplos de treino usados em uma iteração antes de atualizar os pesos do modelo. Afeta velocidade de treino, uso de memória e convergência."
    },
    {
        id: 92,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é um 'epoch' no treinamento de modelos?",
        options: [
            "O tempo de treinamento",
            "Um único forward pass",
            "O número de iterações",
            "Uma passagem completa por todo o dataset de treinamento"
        ],
        correct: 3,
        explanation: "Um epoch é uma passagem completa por todo o dataset de treinamento. Múltiplos epochs são normalmente necessários para o modelo convergir para bons pesos."
    },

    // ===== DOMAIN 4: MLOps & PRODUCTION - Additional Questions =====
    {
        id: 93,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'Blue-Green Deployment'?",
        options: [
            "Deploy em dois datacenters diferentes",
            "Deploy durante o dia e a noite",
            "Manter dois ambientes idênticos, alternando tráfego entre eles para deploys zero-downtime",
            "Deploy em ambientes de teste"
        ],
        correct: 2,
        explanation: "Blue-Green deployment mantém dois ambientes de produção idênticos. Um recebe tráfego enquanto o outro é atualizado. Após validação, o tráfego é redirecionado, permitindo rollback instantâneo."
    },
    {
        id: 94,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'data drift' vs 'concept drift'?",
        options: [
            "Concept drift é mais comum que data drift",
            "São a mesma coisa",
            "Data drift é mais grave que concept drift",
            "Data drift: distribuição dos inputs muda. Concept drift: relação entre inputs e outputs muda"
        ],
        correct: 3,
        explanation: "Data drift ocorre quando a distribuição das features de entrada muda. Concept drift ocorre quando a relação entre inputs e outputs muda. Ambos degradam performance do modelo."
    },
    {
        id: 95,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é um 'Model Registry'?",
        options: [
            "Registro de erros do modelo",
            "Repositório centralizado para gerenciar versões de modelos, metadados e lifecycle",
            "Documentação do modelo",
            "Registro de usuários do modelo"
        ],
        correct: 1,
        explanation: "Model Registry é um repositório central para armazenar e gerenciar modelos de ML, incluindo versões, metadados, métricas e status do lifecycle (staging, production, archived)."
    },
    {
        id: 96,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'model serving'?",
        options: [
            "Treinar o modelo",
            "Salvar o modelo em disco",
            "Avaliar o modelo",
            "Disponibilizar o modelo para fazer inferências/previsões em tempo real ou batch"
        ],
        correct: 3,
        explanation: "Model serving é o processo de disponibilizar o modelo treinado para que ele possa receber requisições e retornar previsões, seja em tempo real (API) ou em batch."
    },
    {
        id: 97,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'containerization' no contexto de MLOps?",
        options: [
            "Empacotar modelo com todas suas dependências em containers (como Docker) para deploy consistente",
            "Comprimir modelos",
            "Armazenar dados em containers",
            "Dividir dados em partes menores"
        ],
        correct: 0,
        explanation: "Containerization empacota o modelo junto com todas suas dependências, bibliotecas e configurações em um container (Docker). Isso garante que o modelo rode de forma consistente em qualquer ambiente."
    },
    {
        id: 98,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'model latency' e por que é importante?",
        options: [
            "O tamanho do modelo em disco",
            "A idade do modelo",
            "O tempo que o modelo leva para retornar uma previsão após receber uma requisição",
            "O tempo de treinamento do modelo"
        ],
        correct: 2,
        explanation: "Model latency é o tempo entre receber uma requisição e retornar a previsão. É crítico para aplicações real-time onde usuários esperam respostas rápidas (< segundos)."
    },
    {
        id: 99,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'model compression'?",
        options: [
            "Backups do modelo",
            "Combinar múltiplos modelos",
            "Criptografar o modelo",
            "Técnicas para reduzir o tamanho do modelo mantendo performance (quantization, pruning)"
        ],
        correct: 3,
        explanation: "Model compression são técnicas para reduzir o tamanho do modelo (quantization, pruning, distillation) mantendo performance aceitável. Importante para deploy em dispositivos com recursos limitados."
    },
    {
        id: 100,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'adversarial attack' em modelos de ML?",
        options: [
            "Inputs maliciosamente modificados para enganar o modelo e produzir outputs incorretos",
            "Ataque de força bruta",
            "Ataque à infraestrutura de rede",
            "Roubo de código"
        ],
        correct: 0,
        explanation: "Adversarial attacks são inputs especialmente criados para enganar o modelo. Pequenas perturbações imperceptíveis a humanos podem fazer o modelo classificar erroneamente."
    }
];

// Merge with main questions array
questions.push(...questionsExtra);

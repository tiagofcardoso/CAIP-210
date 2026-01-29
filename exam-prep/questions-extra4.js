// CAIP-210 Questions - Extra Set 4
// Domain 1 (AI & ML Fundamentals) and Domain 4 (MLOps & Production)
// Created to balance question distribution

const questionsExtra4 = [
    // ========================================
    // DOMAIN 1: AI & ML FUNDAMENTALS (10 questions)
    // ========================================

    {
        id: 151,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Uma empresa de varejo quer prever a demanda de produtos para otimizar estoque. Qual tipo de problema de ML é mais apropriado e por quê?",
        options: [
            "Classificação, porque precisa categorizar produtos em alta ou baixa demanda",
            "Regressão, porque a demanda é um valor numérico contínuo que precisa ser previsto",
            "Clustering, porque precisa agrupar produtos similares",
            "Reinforcement learning, porque precisa aprender com feedback do cliente"
        ],
        correct: 1,
        explanation: "Previsão de demanda é um problema de regressão porque o objetivo é prever um valor numérico contínuo (quantidade de produtos). Classificação seria usada se o objetivo fosse apenas categorizar como 'alta' ou 'baixa' demanda."
    },
    {
        id: 152,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Qual é a principal diferença entre aprendizado supervisionado e não supervisionado?",
        options: [
            "Supervisionado usa mais dados que não supervisionado",
            "Supervisionado requer dados rotulados (labels) enquanto não supervisionado descobre padrões sem labels",
            "Não supervisionado é sempre mais preciso",
            "Supervisionado só funciona com dados numéricos"
        ],
        correct: 1,
        explanation: "A diferença fundamental é que aprendizado supervisionado requer dados rotulados (exemplos com respostas conhecidas) para treinar, enquanto aprendizado não supervisionado descobre padrões e estruturas nos dados sem labels pré-definidos."
    },
    {
        id: 153,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Uma fintech quer detectar transações fraudulentas em tempo real. Qual consideração ética é MAIS crítica neste cenário?",
        options: [
            "Maximizar lucro da empresa acima de tudo",
            "Garantir que o modelo não discrimine grupos específicos de clientes e seja transparente nas decisões",
            "Usar o máximo de dados pessoais possível para melhor precisão",
            "Manter o modelo completamente secreto para segurança"
        ],
        correct: 1,
        explanation: "Em aplicações financeiras, é crítico evitar viés discriminatório (fairness) e manter transparência nas decisões que afetam clientes. Isso inclui não discriminar por raça, gênero, localização, etc., e permitir que decisões sejam explicáveis e contestáveis."
    },
    {
        id: 154,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Qual métrica de negócio seria mais apropriada para avaliar um sistema de recomendação de produtos em e-commerce?",
        options: [
            "Apenas accuracy do modelo",
            "Taxa de conversão (conversion rate) e aumento de receita por usuário",
            "Apenas tempo de treinamento do modelo",
            "Número de features utilizadas"
        ],
        correct: 1,
        explanation: "Métricas de negócio como taxa de conversão e receita por usuário são mais relevantes que métricas técnicas isoladas. Um modelo com alta accuracy técnica pode não gerar valor de negócio se não aumentar vendas ou engajamento."
    },
    {
        id: 155,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Em um projeto de ML para diagnóstico médico, qual é o maior risco ético a ser mitigado?",
        options: [
            "Custo computacional alto",
            "Viés nos dados de treinamento que pode levar a diagnósticos incorretos para grupos sub-representados",
            "Tempo de inferência lento",
            "Complexidade do código"
        ],
        correct: 1,
        explanation: "Em saúde, viés nos dados pode resultar em diagnósticos incorretos ou tratamentos inadequados para grupos sub-representados (ex: minorias étnicas, mulheres). Isso pode ter consequências graves para a saúde dos pacientes e perpetuar desigualdades."
    },
    {
        id: 156,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Qual tipo de aprendizado de máquina é mais adequado para treinar um agente que aprende a jogar xadrez através de tentativa e erro?",
        options: [
            "Supervised learning com dataset de partidas históricas",
            "Unsupervised learning para descobrir padrões",
            "Reinforcement learning onde o agente recebe recompensas por boas jogadas",
            "Semi-supervised learning"
        ],
        correct: 2,
        explanation: "Reinforcement learning é ideal para cenários onde um agente aprende através de interação com um ambiente, recebendo recompensas (vitórias) ou penalidades (derrotas). O agente aprende a política ótima através de tentativa e erro."
    },
    {
        id: 157,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Uma startup quer usar ML para automatizar triagem de currículos. Qual é o principal risco de viés a considerar?",
        options: [
            "O modelo pode ser lento demais",
            "O modelo pode perpetuar vieses históricos de contratação, discriminando candidatos por gênero, raça ou idade",
            "O modelo pode usar muita memória",
            "O modelo pode precisar de muitos dados"
        ],
        correct: 1,
        explanation: "Se os dados históricos de contratação contêm vieses (ex: preferência por determinado gênero ou raça), o modelo aprenderá e perpetuará esses vieses. É essencial auditar os dados e o modelo para garantir fairness e compliance com leis anti-discriminação."
    },
    {
        id: 158,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Qual é a diferença entre um problema de classificação binária e classificação multiclasse?",
        options: [
            "Binária usa apenas 2 features, multiclasse usa mais",
            "Binária prevê entre 2 classes (ex: spam/não-spam), multiclasse prevê entre 3+ classes (ex: tipo de flor)",
            "Multiclasse é sempre mais precisa",
            "Binária só funciona com dados numéricos"
        ],
        correct: 1,
        explanation: "Classificação binária prevê entre duas classes possíveis (sim/não, verdadeiro/falso), enquanto classificação multiclasse prevê entre três ou mais classes (ex: classificar tipo de animal: gato, cachorro, pássaro, etc.)."
    },
    {
        id: 159,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Ao formular um problema de ML, qual é o primeiro passo mais importante?",
        options: [
            "Escolher o algoritmo mais complexo disponível",
            "Definir claramente o objetivo de negócio e como o sucesso será medido",
            "Coletar o máximo de dados possível",
            "Implementar o modelo imediatamente"
        ],
        correct: 1,
        explanation: "A formulação correta do problema começa com definir claramente o objetivo de negócio e as métricas de sucesso. Isso guia todas as decisões subsequentes sobre dados, features, algoritmos e avaliação."
    },
    {
        id: 160,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Qual cenário é um exemplo clássico de aprendizado não supervisionado?",
        options: [
            "Prever preço de casas baseado em características",
            "Classificar emails como spam ou não-spam",
            "Segmentar clientes em grupos com comportamentos similares sem categorias pré-definidas",
            "Prever se um paciente tem ou não uma doença"
        ],
        correct: 2,
        explanation: "Segmentação de clientes (clustering) é aprendizado não supervisionado porque não há categorias pré-definidas. O algoritmo descobre grupos naturais nos dados baseado em similaridades. Os outros exemplos são supervisionados (regressão ou classificação)."
    },

    // ========================================
    // DOMAIN 4: MLOps & PRODUCTION (15 questions)
    // ========================================

    {
        id: 161,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Qual estratégia de deployment permite testar um novo modelo com uma pequena porcentagem de usuários antes do rollout completo?",
        options: [
            "Big bang deployment",
            "Canary deployment, onde o novo modelo é gradualmente exposto a uma porcentagem crescente de tráfego",
            "Rollback deployment",
            "Batch deployment"
        ],
        correct: 1,
        explanation: "Canary deployment expõe o novo modelo a uma pequena porcentagem de tráfego inicialmente (ex: 5%), monitora performance, e gradualmente aumenta se tudo estiver funcionando bem. Isso minimiza risco de falhas em produção."
    },
    {
        id: 162,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é model drift e por que é importante monitorá-lo?",
        options: [
            "Erro de sintaxe no código do modelo",
            "Degradação da performance do modelo ao longo do tempo devido a mudanças nos dados de produção",
            "Aumento do tempo de inferência",
            "Problema de memória no servidor"
        ],
        correct: 1,
        explanation: "Model drift ocorre quando a distribuição dos dados de produção muda em relação aos dados de treinamento, causando degradação de performance. Monitorar drift é essencial para saber quando retreinar o modelo."
    },
    {
        id: 163,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Em um pipeline de CI/CD para ML, qual etapa é ESSENCIAL antes de fazer deploy em produção?",
        options: [
            "Apenas verificar se o código compila",
            "Executar testes automatizados incluindo validação de performance do modelo em dados de teste",
            "Fazer deploy direto sem testes",
            "Apenas verificar se o modelo treina sem erros"
        ],
        correct: 1,
        explanation: "CI/CD para ML deve incluir testes automatizados que validam não apenas o código, mas também a performance do modelo (accuracy, latência, etc.) em dados de teste antes de permitir deploy em produção."
    },
    {
        id: 164,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Qual é a principal vantagem de usar um model registry em MLOps?",
        options: [
            "Apenas armazenar modelos",
            "Versionamento centralizado de modelos, metadados, e rastreabilidade de quais modelos estão em produção",
            "Treinar modelos mais rápido",
            "Reduzir custo de infraestrutura"
        ],
        correct: 1,
        explanation: "Model registry fornece versionamento centralizado, armazenamento de metadados (métricas, hiperparâmetros), rastreabilidade de lineage, e controle de quais versões estão em staging/produção. Essencial para governança e reprodutibilidade."
    },
    {
        id: 165,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Qual métrica é mais importante monitorar em produção para detectar data drift?",
        options: [
            "Apenas CPU usage",
            "Distribuição estatística das features de entrada comparada com dados de treinamento",
            "Apenas tempo de resposta",
            "Número de requests"
        ],
        correct: 1,
        explanation: "Data drift é detectado comparando a distribuição estatística das features de entrada em produção com a distribuição dos dados de treinamento. Testes como Kolmogorov-Smirnov ou chi-quadrado podem identificar mudanças significativas."
    },
    {
        id: 166,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Em uma estratégia de Blue-Green deployment, qual é o principal benefício?",
        options: [
            "Usar menos servidores",
            "Permitir rollback instantâneo para a versão anterior em caso de problemas",
            "Treinar modelos mais rápido",
            "Reduzir custo de armazenamento"
        ],
        correct: 1,
        explanation: "Blue-Green deployment mantém duas versões completas do ambiente (blue = atual, green = nova). O tráfego é redirecionado para green após validação. Se houver problemas, rollback para blue é instantâneo, minimizando downtime."
    },
    {
        id: 167,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Qual ferramenta é essencial para rastrear experimentos de ML (hiperparâmetros, métricas, artifacts)?",
        options: [
            "Apenas planilhas Excel",
            "Plataforma de experiment tracking como MLflow, Weights & Biases, ou Neptune",
            "Apenas logs de texto",
            "Não é necessário rastrear experimentos"
        ],
        correct: 1,
        explanation: "Ferramentas de experiment tracking como MLflow, W&B, ou Neptune permitem rastrear hiperparâmetros, métricas, código, dados, e artifacts de forma estruturada, facilitando comparação de experimentos e reprodutibilidade."
    },
    {
        id: 168,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Qual é o propósito de um feature store em arquitetura MLOps?",
        options: [
            "Apenas armazenar dados brutos",
            "Centralizar, versionar e servir features processadas para treinamento e inferência, garantindo consistência",
            "Apenas fazer backup de modelos",
            "Substituir bancos de dados tradicionais"
        ],
        correct: 1,
        explanation: "Feature store centraliza features engenheiradas, garante consistência entre treinamento e inferência (evitando training-serving skew), permite reuso de features, e fornece versionamento e lineage."
    },
    {
        id: 169,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Ao fazer deploy de um modelo em produção, qual consideração de segurança é CRÍTICA?",
        options: [
            "Apenas usar HTTPS",
            "Implementar autenticação, autorização, criptografia de dados, e proteção contra adversarial attacks",
            "Não é necessário se preocupar com segurança",
            "Apenas usar firewall"
        ],
        correct: 1,
        explanation: "Segurança em ML inclui: autenticação/autorização de APIs, criptografia de dados em trânsito e em repouso, proteção contra adversarial attacks (inputs maliciosos), e compliance com regulações (GDPR, LGPD)."
    },
    {
        id: 170,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Qual é a diferença entre batch inference e real-time inference?",
        options: [
            "Não há diferença",
            "Batch processa múltiplas predições de uma vez (ex: diariamente), real-time responde a requests individuais imediatamente",
            "Batch é sempre mais preciso",
            "Real-time usa menos recursos"
        ],
        correct: 1,
        explanation: "Batch inference processa grandes volumes de dados periodicamente (ex: previsões diárias de demanda). Real-time inference responde a requests individuais com baixa latência (ex: detecção de fraude em transações). Cada um tem trade-offs de latência, throughput e custo."
    },
    {
        id: 171,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Qual métrica de SLA (Service Level Agreement) é mais crítica para um modelo de ML em produção?",
        options: [
            "Apenas uptime do servidor",
            "Combinação de latência (tempo de resposta), throughput (requests/segundo), e accuracy do modelo",
            "Apenas custo de infraestrutura",
            "Apenas número de features"
        ],
        correct: 1,
        explanation: "SLA para ML deve incluir: latência (ex: p95 < 100ms), throughput (ex: 1000 req/s), uptime (ex: 99.9%), e performance do modelo (ex: accuracy > 90%). Todos são críticos para garantir qualidade do serviço."
    },
    {
        id: 172,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é A/B testing no contexto de deployment de modelos ML?",
        options: [
            "Testar dois algoritmos durante treinamento",
            "Expor diferentes versões do modelo a diferentes grupos de usuários e comparar performance de negócio",
            "Testar apenas em ambiente de desenvolvimento",
            "Não é aplicável a ML"
        ],
        correct: 1,
        explanation: "A/B testing expõe diferentes versões do modelo (A = controle, B = novo) a grupos aleatórios de usuários em produção, permitindo comparar métricas de negócio (conversão, receita) de forma estatisticamente rigorosa antes de decidir qual versão manter."
    },
    {
        id: 173,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Qual é o propósito de model monitoring contínuo em produção?",
        options: [
            "Apenas verificar se o servidor está online",
            "Detectar degradação de performance, drift, anomalias, e disparar retreinamento quando necessário",
            "Apenas salvar logs",
            "Não é necessário monitorar após deploy"
        ],
        correct: 1,
        explanation: "Monitoring contínuo detecta: model drift (mudança nos dados), concept drift (mudança na relação features-target), degradação de performance, anomalias, e dispara alertas ou retreinamento automático. Essencial para manter qualidade ao longo do tempo."
    },
    {
        id: 174,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Qual é a melhor prática para retreinamento de modelos em produção?",
        options: [
            "Nunca retreinar",
            "Retreinar periodicamente (schedule-based) ou quando drift/degradação é detectado (trigger-based)",
            "Retreinar apenas quando o modelo falha completamente",
            "Retreinar aleatoriamente"
        ],
        correct: 1,
        explanation: "Retreinamento pode ser: schedule-based (ex: mensal) para dados com sazonalidade, ou trigger-based (quando drift/degradação é detectado). A escolha depende da natureza dos dados e criticidade da aplicação."
    },
    {
        id: 175,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Qual é o papel de observability em sistemas de ML em produção?",
        options: [
            "Apenas coletar logs",
            "Fornecer visibilidade completa do sistema através de logs, métricas, traces, e dashboards para debugging e otimização",
            "Apenas monitorar CPU",
            "Não é necessário em ML"
        ],
        correct: 1,
        explanation: "Observability em ML inclui: logs estruturados, métricas de modelo e infraestrutura, distributed tracing, dashboards, e alertas. Permite entender o comportamento do sistema, debugar problemas, e otimizar performance."
    }
];

// Merge with main questions array
questions.push(...questionsExtra4);

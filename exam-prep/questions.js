// CAIP-210 Exam Questions Database
// Based on CertNexus Certified AI Practitioner official course material

const questions = [
    // ===== DOMAIN 1: AI & ML FUNDAMENTALS (26%) =====
    {
        id: 1,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Qual é a principal diferença entre machine learning e programação tradicional?",
        options: [
            "Machine learning requer mais hardware",
            "Machine learning faz previsões baseadas em dados sem instruções explícitas",
            "Machine learning só funciona com Big Data",
            "Programação tradicional é mais lenta"
        ],
        correct: 1,
        explanation: "Machine learning se diferencia da programação tradicional porque os computadores fazem previsões e decisões baseadas em conjuntos de dados, sem instruções explícitas fornecidas por humanos. Isso permite automatizar processos de tomada de decisão de forma mais rápida e eficiente."
    },
    {
        id: 2,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "O que é Deep Learning em relação ao Machine Learning?",
        options: [
            "Um subconjunto do ML que usa redes neurais artificiais complexas",
            "Uma forma de armazenamento em nuvem",
            "Um método de visualização de dados",
            "Uma técnica de otimização de banco de dados"
        ],
        correct: 0,
        explanation: "Deep Learning é um subconjunto do machine learning que envolve o uso de redes neurais artificiais complexas. Essas redes são ainda mais eficazes na resolução de problemas complexos."
    },
    {
        id: 3,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Na hierarquia DIK (Data, Information, Knowledge), o que transforma dados em informação?",
        options: [
            "Backup dos dados",
            "Armazenamento em banco de dados",
            "Compressão dos arquivos",
            "Agregação, organização e interpretação dos dados"
        ],
        correct: 3,
        explanation: "Dados brutos geralmente têm pouco contexto. Quando são agregados, organizados e interpretados, tornam-se informação útil para decisões de negócio."
    },
    {
        id: 4,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Qual é um exemplo de problema de REGRESSÃO em machine learning?",
        options: [
            "Agrupar clientes em segmentos de marketing",
            "Classificar emails como spam ou não spam",
            "Prever o preço de fechamento do índice Dow Jones",
            "Detectar anomalias em transações bancárias"
        ],
        correct: 2,
        explanation: "Regressão é usada para estimar valores numéricos. Prever preços de ações é um exemplo clássico de regressão, pois o resultado é um número contínuo, não uma categoria."
    },
    {
        id: 5,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Qual é um exemplo de problema de CLASSIFICAÇÃO em machine learning?",
        options: [
            "Prever o salário de um funcionário",
            "Calcular a temperatura média de amanhã",
            "Estimar o tempo de vida útil de uma máquina",
            "Classificar emails como spam (1) ou não spam (0)"
        ],
        correct: 3,
        explanation: "Classificação identifica a qual classe uma instância de dados pertence. Classificar emails como spam ou não spam é um exemplo de classificação binária."
    },
    {
        id: 6,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "O que é CLUSTERING em machine learning?",
        options: [
            "Prever valores numéricos futuros",
            "Transformar dados categóricos em numéricos",
            "Agrupar dados semelhantes sem conhecimento prévio das classes",
            "Classificar dados em categorias pré-definidas"
        ],
        correct: 2,
        explanation: "Clustering agrupa componentes que pertencem juntos, sem conhecimento prévio de uma variável alvo. É útil quando você não sabe quais grupos existem nos dados."
    },
    {
        id: 7,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Quais são os principais riscos éticos em AI/ML?",
        options: [
            "Apenas custos de infraestrutura",
            "Privacidade, accountability, transparência, fairness e segurança",
            "Apenas violações de privacidade",
            "Apenas problemas de performance"
        ],
        correct: 1,
        explanation: "Os principais riscos éticos em AI/ML incluem: Privacy (proteção de dados pessoais), Accountability (responsabilização por decisões), Transparency/Explainability (possibilidade de entender as decisões), Fairness (tratamento justo sem discriminação), e Safety/Security (minimização de danos)."
    },
    {
        id: 8,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "O que significa 'stochastic' no contexto de modelos de machine learning?",
        options: [
            "Amostras individuais são aleatórias, mas o conjunto segue um padrão geral",
            "Os modelos são determinísticos e sempre produzem o mesmo resultado",
            "Os modelos requerem supervisão humana constante",
            "Os modelos não podem aprender com dados"
        ],
        correct: 0,
        explanation: "Modelos estocásticos reconhecem que amostras individuais são inerentemente aleatórias, mas o conjunto de dados segue padrões gerais que permitem fazer estimativas úteis."
    },
    {
        id: 9,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Na formulação de problemas de ML, o que representa 'Task' no framework Task-Experience-Performance?",
        options: [
            "O hardware necessário para processamento",
            "A métrica de avaliação do modelo",
            "O dataset utilizado para treinamento",
            "O que a solução deve realizar (ex: prever o preço de uma casa)"
        ],
        correct: 3,
        explanation: "No framework TEP, Task define o que a solução deve realizar (ex: 'Prever o preço de venda de uma casa'), Experience define qual dataset será usado para aprendizado, e Performance define como avaliar o desempenho."
    },
    {
        id: 10,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Quando NÃO é recomendado usar AI/ML para resolver um problema?",
        options: [
            "Quando há padrões não óbvios nos dados",
            "Quando o problema pode ser resolvido com lógica de programação tradicional mais simples",
            "Quando há grandes volumes de dados disponíveis",
            "Quando é necessário tomar decisões complexas"
        ],
        correct: 1,
        explanation: "AI/ML pode ser caro, demorado e arriscado. Se o problema pode ser resolvido com programação tradicional mais simples (ex: roteamento de tickets baseado em regras), AI/ML pode não ser justificável."
    },
    {
        id: 11,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "O que são variáveis independentes (input/predictor variables) no Design of Experiments?",
        options: [
            "Variáveis que resultam de outros cálculos",
            "Variáveis que você pode alterar diretamente para ver seu impacto",
            "Variáveis que você não pode controlar",
            "Variáveis que são sempre constantes"
        ],
        correct: 1,
        explanation: "Variáveis independentes são aquelas que você pode alterar diretamente no experimento. Variáveis dependentes (output/response) são as que mudam indiretamente como resultado."
    },
    {
        id: 12,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Qual stakeholder é responsável por fornecer insight sobre ferramentas, tecnologias e recursos necessários para o projeto?",
        options: [
            "Team Members (practitioners)",
            "Governments",
            "Customers/End Users",
            "Sponsors/Champions"
        ],
        correct: 0,
        explanation: "Team Members são os praticantes que trabalham diretamente no desenvolvimento do projeto e podem fornecer insights sobre as ferramentas, tecnologias e recursos necessários para o sucesso."
    },

    // ===== DOMAIN 2: DATA PREPARATION (20%) =====
    {
        id: 13,
        domain: 2,
        domainName: "Data Preparation",
        question: "Qual método de imputação encontra registros similares no mesmo dataset para preencher valores ausentes?",
        options: [
            "Hot-deck Imputation",
            "Mean/Mode Imputation",
            "Cold-deck Imputation",
            "Regression Imputation"
        ],
        correct: 0,
        explanation: "Hot-deck imputation encontra registros no mesmo sample que têm valores similares em outras features, e copia o valor faltante de um desses registros similares."
    },
    {
        id: 14,
        domain: 2,
        domainName: "Data Preparation",
        question: "Qual é a fórmula correta para NORMALIZAÇÃO (min-max scaling)?",
        options: [
            "x' = log(x)",
            "x' = (x - min) / (max - min)",
            "x' = x / max",
            "x' = (x - μ) / σ"
        ],
        correct: 1,
        explanation: "Normalização transforma valores para o intervalo [0, 1] usando a fórmula: x' = (x - min) / (max - min), onde min e max são os valores mínimo e máximo da feature."
    },
    {
        id: 15,
        domain: 2,
        domainName: "Data Preparation",
        question: "Qual é a fórmula correta para PADRONIZAÇÃO (z-score)?",
        options: [
            "x' = x^(1/3)",
            "x' = log10(x)",
            "x' = (x - min) / (max - min)",
            "x' = (x - μ) / σ"
        ],
        correct: 3,
        explanation: "Padronização calcula o z-score: x' = (x - μ) / σ, onde μ é a média e σ é o desvio padrão. Isso centraliza os dados em 0 com desvio padrão 1."
    },
    {
        id: 16,
        domain: 2,
        domainName: "Data Preparation",
        question: "Quando o scaling de features é MENOS importante?",
        options: [
            "Ao usar Decision Trees e Random Forests",
            "Ao usar Support Vector Machines (SVM)",
            "Ao usar k-Nearest Neighbor (k-NN)",
            "Ao usar redes neurais"
        ],
        correct: 0,
        explanation: "Algoritmos baseados em árvores (Decision Trees, Random Forests) não requerem que features sejam escalonadas. Já algoritmos baseados em distância (k-NN, SVM) requerem scaling."
    },
    {
        id: 17,
        domain: 2,
        domainName: "Data Preparation",
        question: "Qual método de encoding é mais apropriado quando as categorias NÃO têm ordem ou ranking natural?",
        options: [
            "One-hot Encoding",
            "Target Encoding",
            "Hash Encoding",
            "Label Encoding (Ordinal Encoding)"
        ],
        correct: 0,
        explanation: "One-hot encoding cria colunas dummy para cada classe, atribuindo 1 ou 0. Isso evita que o algoritmo interprete uma ordem/ranking entre as categorias."
    },
    {
        id: 18,
        domain: 2,
        domainName: "Data Preparation",
        question: "O que é 'discretização' de uma variável contínua?",
        options: [
            "Converter uma variável de string para número",
            "Remover valores duplicados",
            "Calcular a média da variável",
            "Converter uma variável contínua em intervalos discretos (bins)"
        ],
        correct: 3,
        explanation: "Discretização (ou data binning) é o processo de converter uma variável contínua em intervalos discretos. Por exemplo, transformar idade exata em faixas etárias (18-24, 25-34, etc.)."
    },
    {
        id: 19,
        domain: 2,
        domainName: "Data Preparation",
        question: "O que é a 'maldição da dimensionalidade' (curse of dimensionality)?",
        options: [
            "A dificuldade de processar dados em tempo real",
            "A redução na capacidade do modelo de aprender quando há muitas features em relação às amostras",
            "Ter poucos dados para treinar um modelo",
            "O alto custo de armazenamento de dados"
        ],
        correct: 1,
        explanation: "A maldição da dimensionalidade ocorre quando adicionar mais features (sem aumentar as amostras) começa a reduzir a capacidade do modelo de aprender padrões úteis."
    },
    {
        id: 20,
        domain: 2,
        domainName: "Data Preparation",
        question: "Qual algoritmo de redução de dimensionalidade seleciona features que contribuem com a maior variância linear nos dados?",
        options: [
            "Random Forest",
            "t-SNE",
            "PCA (Principal Component Analysis)",
            "k-Means"
        ],
        correct: 2,
        explanation: "PCA projeta dados de alta dimensionalidade em um espaço de menor dimensionalidade, selecionando as features que contribuem com a maior variância linear."
    },
    {
        id: 21,
        domain: 2,
        domainName: "Data Preparation",
        question: "Qual transformação ajuda a reduzir skewness positiva em dados não normalmente distribuídos?",
        options: [
            "One-hot encoding",
            "Standardization",
            "Log transformation",
            "Target encoding"
        ],
        correct: 2,
        explanation: "A transformação logarítmica (log) ajuda a reduzir skewness positiva em datasets não normalmente distribuídos, aproximando-os de uma distribuição normal."
    },
    {
        id: 22,
        domain: 2,
        domainName: "Data Preparation",
        question: "Qual função do pandas é usada para identificar valores ausentes em um DataFrame?",
        options: [
            "df.dropna()",
            "df.isna() ou df.isnull()",
            "df.fillna()",
            "df.duplicated()"
        ],
        correct: 1,
        explanation: "pandas.DataFrame.isna() retorna um DataFrame de booleanos indicando quais valores estão formatados como tipo ausente (None, NaN)."
    },
    {
        id: 23,
        domain: 2,
        domainName: "Data Preparation",
        question: "Quando uma coluna tem mais de 70% de valores ausentes, qual é a abordagem recomendada?",
        options: [
            "Usar mean imputation para todos os valores",
            "Converter para categoria 'unknown'",
            "Duplicar valores de outras colunas",
            "Dropar (remover) a coluna inteira"
        ],
        correct: 3,
        explanation: "Quando uma coluna tem grande percentual de valores ausentes (como 70% ou mais), é recomendado remover (drop) a coluna inteira, pois a imputação pode introduzir muito ruído."
    },
    {
        id: 24,
        domain: 2,
        domainName: "Data Preparation",
        question: "O que é Feature Selection vs Feature Extraction na redução de dimensionalidade?",
        options: [
            "São a mesma coisa, apenas nomes diferentes",
            "Selection remove outliers; Extraction remove duplicatas",
            "Selection é manual; Extraction é automática",
            "Selection escolhe um subset das features originais; Extraction deriva novas features combinando as originais"
        ],
        correct: 3,
        explanation: "Feature Selection seleciona um subset das features originais (excluindo redundantes/irrelevantes). Feature Extraction deriva novas features combinando múltiplas features correlacionadas em uma."
    },

    // ===== DOMAIN 3: TRAINING & TUNING (24%) =====
    {
        id: 25,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é OVERFITTING em machine learning?",
        options: [
            "Quando o modelo é muito simples para capturar padrões nos dados",
            "Quando o modelo leva muito tempo para treinar",
            "Quando não há dados suficientes para treinamento",
            "Quando o modelo se ajusta demais aos dados de treinamento e performa mal em novos dados"
        ],
        correct: 3,
        explanation: "Overfitting ocorre quando o modelo aprende os dados de treinamento tão bem (incluindo ruído) que falha ao generalizar para dados novos não vistos."
    },
    {
        id: 26,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é UNDERFITTING em machine learning?",
        options: [
            "Quando o modelo é muito simples para capturar padrões subjacentes nos dados",
            "Quando o modelo é muito complexo",
            "Quando o modelo treina muito rápido",
            "Quando há dados demais para treinamento"
        ],
        correct: 0,
        explanation: "Underfitting ocorre quando o modelo é muito simples (alto bias) e não consegue capturar os padrões subjacentes nos dados, resultando em baixo desempenho tanto em treino quanto em teste."
    },
    {
        id: 27,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que representa a métrica R² (R-squared) em regressão?",
        options: [
            "O número de iterações necessárias para convergir",
            "O erro médio absoluto do modelo",
            "A taxa de aprendizado do modelo",
            "A proporção da variância na variável dependente que é explicada pelo modelo"
        ],
        correct: 3,
        explanation: "R² (coeficiente de determinação) mede a proporção da variância na variável dependente que é explicada pelas variáveis independentes do modelo. Valores mais próximos de 1 indicam melhor fit."
    },
    {
        id: 28,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Qual métrica é mais apropriada quando é crucial minimizar FALSOS NEGATIVOS?",
        options: [
            "Recall (Sensitivity)",
            "Specificity",
            "Precision",
            "Accuracy"
        ],
        correct: 0,
        explanation: "Recall (Sensitivity) mede a proporção de positivos reais corretamente identificados. É crucial quando falsos negativos são perigosos (ex: não detectar uma doença grave)."
    },
    {
        id: 29,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Qual métrica é mais apropriada quando é crucial minimizar FALSOS POSITIVOS?",
        options: [
            "F1-Score",
            "Recall",
            "Precision",
            "Sensitivity"
        ],
        correct: 2,
        explanation: "Precision mede a proporção de previsões positivas que estão corretas. É crucial quando falsos positivos são custosos (ex: classificar email legítimo como spam)."
    },
    {
        id: 30,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é Cross-Validation e para que serve?",
        options: [
            "Uma técnica para visualizar dados",
            "Uma técnica que divide os dados em múltiplos folds para avaliar o modelo de forma mais robusta",
            "Uma técnica para coletar mais dados",
            "Uma técnica para remover outliers"
        ],
        correct: 1,
        explanation: "Cross-validation divide os dados em múltiplos folds, usando cada fold como teste enquanto os demais são usados para treino. Isso fornece uma avaliação mais robusta do desempenho do modelo."
    },
    {
        id: 31,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é Regularização (L1/Lasso, L2/Ridge) em modelos de regressão?",
        options: [
            "Uma técnica para aumentar a complexidade do modelo",
            "Uma técnica para acelerar o treinamento",
            "Uma técnica que adiciona uma penalidade para prevenir overfitting",
            "Uma técnica para aumentar os dados de treino"
        ],
        correct: 2,
        explanation: "Regularização adiciona uma penalidade (termo de regularização) à função de custo para reduzir a complexidade do modelo e prevenir overfitting. L1 (Lasso) pode zerar coeficientes; L2 (Ridge) os reduz."
    },
    {
        id: 32,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Em k-Nearest Neighbors (k-NN), o que acontece quando K é muito pequeno?",
        options: [
            "O tempo de treinamento aumenta significativamente",
            "O modelo fica muito generalista (underfitting)",
            "O modelo fica muito sensível a ruído (overfitting)",
            "O modelo deixa de funcionar"
        ],
        correct: 2,
        explanation: "Com K pequeno, o modelo considera poucos vizinhos, tornando-o muito sensível a pontos individuais (incluindo ruído), resultando em overfitting."
    },
    {
        id: 33,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é uma Confusion Matrix?",
        options: [
            "Uma tabela de hiperparâmetros",
            "Uma tabela que mostra True Positives, False Positives, True Negatives e False Negatives",
            "Uma matriz de transformação de dados",
            "Uma matriz que mostra a correlação entre features"
        ],
        correct: 1,
        explanation: "A Confusion Matrix é uma tabela que resume o desempenho de um modelo de classificação, mostrando TP (verdadeiros positivos), FP (falsos positivos), TN (verdadeiros negativos) e FN (falsos negativos)."
    },
    {
        id: 34,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Qual algoritmo de classificação cria fronteiras de decisão que maximizam a margem entre classes?",
        options: [
            "Logistic Regression",
            "Naive Bayes",
            "Support Vector Machines (SVM)",
            "k-Nearest Neighbors"
        ],
        correct: 2,
        explanation: "SVMs encontram o hiperplano que maximiza a margem (distância) entre as classes, tornando-os robustos para classificação."
    },
    {
        id: 35,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que são 'support vectors' em um SVM?",
        options: [
            "Os pontos de dados mais próximos da fronteira de decisão que definem o hiperplano",
            "As features mais importantes",
            "Os centróides dos clusters",
            "Todos os pontos do dataset"
        ],
        correct: 0,
        explanation: "Support vectors são os pontos de dados mais próximos da fronteira de decisão (hyperplane). Eles são críticos porque definem a posição e orientação do hiperplano de separação."
    },
    {
        id: 36,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é o 'kernel trick' em SVMs?",
        options: [
            "Uma técnica para acelerar o treinamento",
            "Uma técnica de feature selection",
            "Uma técnica que permite encontrar fronteiras não-lineares mapeando dados para dimensões superiores",
            "Uma técnica de regularização"
        ],
        correct: 2,
        explanation: "O kernel trick permite que SVMs encontrem fronteiras de decisão não-lineares, mapeando implicitamente os dados para espaços de dimensionalidade superior onde podem ser linearmente separáveis."
    },
    {
        id: 37,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Qual é a principal vantagem de Random Forests sobre uma única Decision Tree?",
        options: [
            "Não requer dados numéricos",
            "É mais rápido para treinar",
            "Usa menos memória",
            "Reduz overfitting ao combinar múltiplas árvores (ensemble)"
        ],
        correct: 3,
        explanation: "Random Forests combinam múltiplas árvores de decisão (ensemble), cada uma treinada em subsets diferentes dos dados. Isso reduz a variância e o overfitting comparado a uma única árvore."
    },
    {
        id: 38,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é k-Means Clustering?",
        options: [
            "Um algoritmo de redução de dimensionalidade",
            "Um algoritmo de classificação supervisionada",
            "Um algoritmo de clustering que particiona dados em K clusters baseado em centroides",
            "Um algoritmo de regressão"
        ],
        correct: 2,
        explanation: "k-Means é um algoritmo de clustering não-supervisionado que particiona n observações em K clusters, onde cada observação pertence ao cluster com o centroide mais próximo."
    },
    {
        id: 39,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é o 'Elbow Method' usado para determinar?",
        options: [
            "A taxa de aprendizado ideal",
            "O número de epochs para treinamento",
            "O número ótimo de clusters (K) em k-Means",
            "O melhor algoritmo de ML para usar"
        ],
        correct: 2,
        explanation: "O Elbow Method plota a inércia (soma das distâncias ao centroides) vs número de clusters. O ponto onde a curva forma um 'cotovelo' indica o número ótimo de clusters."
    },
    {
        id: 40,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que são Multi-Layer Perceptrons (MLPs)?",
        options: [
            "Um método de feature engineering",
            "Redes neurais com múltiplas camadas conectadas (feedforward)",
            "Um tipo de regularização",
            "Um tipo de algoritmo de clustering"
        ],
        correct: 1,
        explanation: "MLPs são redes neurais artificiais com múltiplas camadas (input, hidden, output) conectadas de forma feedforward. São a base para deep learning."
    },
    {
        id: 41,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Qual tipo de rede neural é mais apropriada para processamento de IMAGENS?",
        options: [
            "Multi-Layer Perceptrons (MLP)",
            "Recurrent Neural Networks (RNN)",
            "Autoencoders",
            "Convolutional Neural Networks (CNN)"
        ],
        correct: 3,
        explanation: "CNNs são projetadas para processamento de dados em grid (como imagens). Usam camadas de convolução para detectar padrões locais e hierárquicos."
    },
    {
        id: 42,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Qual tipo de rede neural é mais apropriada para dados SEQUENCIAIS (como texto ou séries temporais)?",
        options: [
            "Convolutional Neural Networks (CNN)",
            "Autoencoders",
            "Generative Adversarial Networks (GAN)",
            "Recurrent Neural Networks (RNN)"
        ],
        correct: 3,
        explanation: "RNNs são projetadas para processar dados sequenciais, mantendo 'memória' de inputs anteriores. São ideais para NLP, tradução e séries temporais."
    },
    {
        id: 43,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é Gradient Descent?",
        options: [
            "Um tipo de regularização",
            "Uma técnica de feature selection",
            "Um método de validação cruzada",
            "Um algoritmo de otimização que ajusta parâmetros para minimizar a função de custo"
        ],
        correct: 3,
        explanation: "Gradient Descent é um algoritmo de otimização iterativo que ajusta os parâmetros do modelo na direção oposta ao gradiente da função de custo, buscando minimizá-la."
    },
    {
        id: 44,
        domain: 3,
        domainName: "Training & Tuning",
        question: "O que é a função de ativação em redes neurais?",
        options: [
            "Uma função que inicializa os pesos",
            "Uma função que determina o tamanho do batch",
            "Uma função que introduz não-linearidade, permitindo que a rede aprenda padrões complexos",
            "Uma função que calcula a perda do modelo"
        ],
        correct: 2,
        explanation: "Funções de ativação (como ReLU, Sigmoid, Tanh) introduzem não-linearidade na rede neural, permitindo que ela aprenda relações complexas além de transformações lineares."
    },

    // ===== DOMAIN 4: MLOps & PRODUCTION (30%) =====
    {
        id: 45,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é MLOps?",
        options: [
            "Uma técnica de feature engineering",
            "A prática de combinar Machine Learning com DevOps para automatizar o ciclo de vida de ML",
            "Um tipo de algoritmo de machine learning",
            "Um framework de deep learning"
        ],
        correct: 1,
        explanation: "MLOps é a prática de aplicar princípios de DevOps ao machine learning, automatizando o desenvolvimento, deployment, e manutenção de modelos em produção."
    },
    {
        id: 46,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'model drift' ou 'concept drift'?",
        options: [
            "Quando o modelo aprende muito rápido",
            "Quando o modelo é transferido para outro servidor",
            "Quando o código do modelo é modificado",
            "Quando os dados ou relações mudam ao longo do tempo, degradando o desempenho do modelo"
        ],
        correct: 3,
        explanation: "Model/concept drift ocorre quando os padrões nos dados mudam ao longo do tempo, fazendo com que um modelo treinado em dados antigos perca eficácia em dados novos."
    },
    {
        id: 47,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é um 'ML Pipeline'?",
        options: [
            "Uma sequência automatizada de etapas desde preparação de dados até deployment do modelo",
            "Um tipo de rede neural",
            "Um hardware específico para ML",
            "Uma ferramenta de visualização"
        ],
        correct: 0,
        explanation: "Um ML Pipeline é uma sequência automatizada e reprodutível de etapas que inclui coleta de dados, preprocessing, treinamento, avaliação e deployment do modelo."
    },
    {
        id: 48,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'model versioning' e por que é importante?",
        options: [
            "Dar nomes diferentes para modelos",
            "Rastrear diferentes versões de modelos, dados e código para reprodutibilidade",
            "Criar backups de modelos",
            "Atualizar a documentação do modelo"
        ],
        correct: 1,
        explanation: "Model versioning rastreia diferentes versões de modelos, datasets e código. É crucial para reprodutibilidade, rollback em caso de problemas, e auditoria."
    },
    {
        id: 49,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'A/B Testing' no contexto de deployment de modelos?",
        options: [
            "Testar o modelo em dois datasets diferentes",
            "Comparar dois modelos/versões servindo a diferentes grupos de usuários simultaneamente",
            "Dividir dados em treino e teste",
            "Testar accuracy e precision separadamente"
        ],
        correct: 1,
        explanation: "A/B Testing no contexto de ML significa servir duas versões diferentes de um modelo para diferentes grupos de usuários, comparando seu desempenho em produção."
    },
    {
        id: 50,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'Canary Deployment'?",
        options: [
            "Deploy em ambiente de desenvolvimento",
            "Deploy gradual do novo modelo para uma pequena porcentagem de usuários antes do rollout completo",
            "Deploy em múltiplos servidores simultaneamente",
            "Deploy automático sem testes"
        ],
        correct: 1,
        explanation: "Canary deployment libera o novo modelo para uma pequena porcentagem de tráfego primeiro, permitindo detectar problemas antes de impactar todos os usuários."
    },
    {
        id: 51,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Qual é a importância do monitoramento de modelos em produção?",
        options: [
            "Para detectar degradação de performance, drift, e garantir que o modelo continua atendendo aos requisitos",
            "Para medir tempo de treinamento",
            "Apenas para logging de erros",
            "Apenas para medir custos de infraestrutura"
        ],
        correct: 0,
        explanation: "Monitoramento contínuo é essencial para detectar degradação de performance, data/concept drift, anomalias, e garantir que o modelo continua atendendo aos requisitos de negócio."
    },
    {
        id: 52,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'model retraining' e quando deve ser feito?",
        options: [
            "Testar o modelo antes do deploy",
            "Documentar o modelo existente",
            "Atualizar periodicamente o modelo com dados novos para manter performance",
            "Treinar o modelo uma única vez"
        ],
        correct: 2,
        explanation: "Model retraining é o processo de atualizar o modelo com dados mais recentes. Deve ser feito periodicamente ou quando métricas de monitoramento indicarem degradação (drift)."
    },
    {
        id: 53,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que são 'Feature Stores' no contexto de MLOps?",
        options: [
            "Repositórios centralizados para armazenar, gerenciar e servir features para modelos de ML",
            "Databases tradicionais",
            "Lojas online de features",
            "Bibliotecas de algoritmos"
        ],
        correct: 0,
        explanation: "Feature Stores são repositórios centralizados que armazenam, gerenciam, e servem features computadas para treino e inferência de modelos, garantindo consistência."
    },
    {
        id: 54,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Qual é uma consideração importante de segurança para ML pipelines?",
        options: [
            "Proteger dados sensíveis, controlar acesso, e garantir integridade dos modelos",
            "Manter todos os modelos públicos",
            "Usar apenas dados públicos",
            "Não usar criptografia para velocidade"
        ],
        correct: 0,
        explanation: "Segurança em ML pipelines inclui: proteção de dados sensíveis, controle de acesso, criptografia, integridade dos modelos (prevenir adversarial attacks), e auditoria."
    },
    {
        id: 55,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'model explainability' e por que é importante em produção?",
        options: [
            "Documentar o código do modelo",
            "A velocidade de inferência do modelo",
            "A capacidade de explicar como e por que um modelo tomou uma decisão específica",
            "O tamanho do modelo em megabytes"
        ],
        correct: 2,
        explanation: "Model explainability é a capacidade de entender e explicar as decisões do modelo. É crucial para compliance regulatório, debugging, confiança dos usuários, e identificação de bias."
    },
    {
        id: 56,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'batch inference' vs 'real-time inference'?",
        options: [
            "Batch processa múltiplas previsões de uma vez; real-time processa previsões individuais imediatamente",
            "São a mesma coisa",
            "Batch é para treino; real-time é para teste",
            "Batch é mais preciso; real-time é menos preciso"
        ],
        correct: 0,
        explanation: "Batch inference processa grandes volumes de dados de uma vez (ex: overnight). Real-time inference processa previsões individuais imediatamente quando requisitadas (ex: recomendações)."
    },
    {
        id: 57,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é CI/CD no contexto de MLOps?",
        options: [
            "Continuous Intelligence / Continuous Data",
            "Customer Interface / Customer Development",
            "Code Inspection / Code Debugging",
            "Continuous Integration / Continuous Deployment - automação de build, teste e deploy"
        ],
        correct: 3,
        explanation: "CI/CD (Continuous Integration / Continuous Deployment) automatiza o processo de build, teste e deployment de código e modelos, garantindo entregas mais rápidas e confiáveis."
    },
    {
        id: 58,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'data lineage' e por que é importante?",
        options: [
            "O formato dos dados",
            "O tamanho dos dados",
            "O rastreamento da origem, transformações e movimentação dos dados através do pipeline",
            "A qualidade dos dados"
        ],
        correct: 2,
        explanation: "Data lineage rastreia de onde os dados vieram, como foram transformados, e para onde foram. É importante para debugging, auditoria, compliance, e reprodutibilidade."
    },
    {
        id: 59,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Qual é uma prática recomendada para rollback de modelos em produção?",
        options: [
            "Fazer rollback apenas manualmente",
            "Manter versões anteriores disponíveis e ter um processo automatizado de rollback",
            "Deletar versões antigas imediatamente",
            "Nunca fazer rollback, sempre ir para frente"
        ],
        correct: 1,
        explanation: "É importante manter versões anteriores do modelo e ter um processo automatizado de rollback para poder reverter rapidamente caso o novo modelo apresente problemas."
    },
    {
        id: 60,
        domain: 4,
        domainName: "MLOps & Production",
        question: "O que é 'shadow deployment' (shadow mode)?",
        options: [
            "Executar o novo modelo em paralelo com o atual, sem afetar usuários, para comparar resultados",
            "Deploy com features escondidas",
            "Deploy em servidores de backup",
            "Deploy do modelo apenas durante a noite"
        ],
        correct: 0,
        explanation: "Shadow deployment executa o novo modelo em paralelo com o modelo atual, processando as mesmas requisições, mas sem retornar seus resultados aos usuários. Permite comparar performance em produção real sem riscos."
    }
];

// Export questions for use in app.js
const QUESTIONS_PT = questions;
let QUESTIONS = questions; // Will be updated based on language

// Domain information - bilingual
const DOMAINS = {
    1: {
        name: "AI & ML Fundamentals",
        weight: "26%",
        icon: "🧠",
        description: {
            pt: "Conceitos de AI, machine learning, formulação de problemas e stakeholders",
            en: "AI concepts, machine learning, problem formulation and stakeholders"
        }
    },
    2: {
        name: "Data Preparation",
        weight: "20%",
        icon: "🔧",
        description: {
            pt: "Coleta, transformação, feature engineering e preprocessing",
            en: "Collection, transformation, feature engineering and preprocessing"
        }
    },
    3: {
        name: "Training & Tuning",
        weight: "24%",
        icon: "⚙️",
        description: {
            pt: "Treinamento, avaliação, algoritmos de ML e neural networks",
            en: "Training, evaluation, ML algorithms and neural networks"
        }
    },
    4: {
        name: "MLOps & Production",
        weight: "30%",
        icon: "🚀",
        description: {
            pt: "Deploy, automação, pipelines e manutenção de modelos",
            en: "Deploy, automation, pipelines and model maintenance"
        }
    }
};

// Get domain description based on current language
function getDomainDescription(domainId) {
    const domain = DOMAINS[domainId];
    if (typeof domain.description === 'object') {
        return domain.description[currentLanguage] || domain.description.en;
    }
    return domain.description;
}

// Update questions based on language
function updateQuestionsLanguage() {
    if (typeof currentLanguage !== 'undefined' && currentLanguage === 'en') {
        // Use English questions if available
        if (typeof questions_en !== 'undefined') {
            QUESTIONS = [...questions_en];
            // Add extra questions if available
            if (typeof questionsExtra_en !== 'undefined') {
                QUESTIONS.push(...questionsExtra_en);
            }
            if (typeof questionsExtra2_en !== 'undefined') {
                QUESTIONS.push(...questionsExtra2_en);
            }
            if (typeof questionsExtra3_en !== 'undefined') {
                QUESTIONS.push(...questionsExtra3_en);
            }
            if (typeof questionsExtra4_en !== 'undefined') {
                QUESTIONS.push(...questionsExtra4_en);
            }
        }
    } else {
        // Use Portuguese questions (already merged by questions-extra.js and questions-extra2.js push())
        QUESTIONS = [...questions];
    }
    console.log('Questions loaded:', QUESTIONS.length, 'Language:', currentLanguage);
}

// Get questions by domain
function getQuestionsByDomain(domainId) {
    updateQuestionsLanguage();
    return QUESTIONS.filter(q => q.domain === domainId);
}

// Get random questions
function getRandomQuestions(count, domainId = null) {
    updateQuestionsLanguage();
    let pool = domainId ? getQuestionsByDomain(domainId) : [...QUESTIONS];
    let shuffled = pool.sort(() => Math.random() - 0.5);
    return shuffled.slice(0, Math.min(count, shuffled.length));
}

// Get all questions for exam simulation
function getExamQuestions() {
    return getRandomQuestions(60);
}


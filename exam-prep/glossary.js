// CAIP-210 Glossário Completo - Termos com definições COMPLETAS do eBook oficial
// Baseado no Glossário CertNexus AIP-210 páginas 579-591
// TRADUZIDO PARA PORTUGUÊS BRASILEIRO

const GLOSSARY = {
    // A
    "AI": "Inteligência Artificial (IA) - A capacidade das máquinas de exibir inteligência semelhante à humana, bem como a disciplina científica relacionada a essa ideia. IA engloba aprendizado de máquina, deep learning, robótica, processamento de linguagem natural e visão computacional.",
    "algorithm": "Algoritmo - Um conjunto de regras que define como uma operação de resolução de problemas é executada. Em sua essência, um algoritmo é uma fórmula matemática ou grupo de fórmulas que recebe entrada e produz saída. Algoritmos suportam o aspecto de 'aprendizado' do ML ao atualizar crenças e suposições sobre os dados ao longo do tempo.",
    "ANN": "Rede Neural Artificial (RNA) - Uma aproximação computacional de redes neurais biológicas. Usada em deep learning. ANNs consistem em camadas de nós (neurônios) interconectados que processam informações usando funções de ativação.",
    "augmentation": "Aumento de Dados - O processo de criar múltiplas transformações de dados (ex: perturbar uma imagem de diferentes formas) para aumentar a quantidade de dados de entrada para um modelo. Técnicas comuns incluem rotação, espelhamento, recorte e ajustes de cor para imagens.",
    "authentication": "Autenticação - O ato de verificar a identidade de alguém ou algo. Em sistemas de ML, a autenticação garante que apenas usuários autorizados possam acessar modelos e dados.",
    "authorization": "Autorização - O ato de atribuir permissões e direitos pelos quais um usuário autenticado pode interagir com um sistema protegido.",
    "amplitude": "Amplitude - No processamento de áudio, a magnitude de um sinal sonoro. A amplitude representa a força ou volume de uma onda sonora.",

    // B
    "backpropagation": "Retropropagação - Um método de treinamento de rede neural que começa calculando o gradiente de erro dos neurônios na última camada oculta, depois a penúltima, e assim por diante, até atingir a camada de entrada. Os pesos das conexões entre neurônios são então atualizados com base nesses gradientes.",
    "bag of words": "Saco de Palavras - Uma abordagem para representar conteúdo textual como uma lista de palavras individuais, independente de outros componentes linguísticos como gramática e pontuação. Cada documento se torna um vetor de contagem de palavras.",
    "bagging": "Bootstrap Aggregating (Agregação por Bootstrap) - Uma técnica de ensemble learning para amostragem de dados com reposição. Múltiplos modelos são treinados em diferentes amostras bootstrap e suas previsões são agregadas (votadas ou médias).",
    "Bayesian optimization": "Otimização Bayesiana - Um método de otimização de hiperparâmetros que determina o próximo espaço ótimo de hiperparâmetros a ser amostrado usando resultados passados para influenciar onde a amostragem é conduzida nas iterações subsequentes. Mais eficiente que grid search para modelos custosos.",
    "BCSS": "Soma dos Quadrados Entre Clusters - Uma métrica de avaliação de clustering que mede a separação entre clusters. BCSS maior indica clusters mais bem separados.",
    "BGD": "Gradient Descent em Lote - Uma abordagem de gradient descent que usa todo o conjunto de treinamento para calcular os gradientes. Estável mas computacionalmente caro para grandes datasets.",
    "bias": "Viés - No aprendizado de máquina, um tipo de erro que ocorre quando as estimativas de um modelo são sistematicamente diferentes da verdade real. Alto viés leva ao underfitting. Também se refere ao termo de viés em redes neurais.",
    "black box": "Caixa Preta - A propriedade pela qual as decisões de um modelo de ML são difíceis de entender ou explicar. Redes neurais profundas são frequentemente consideradas modelos caixa preta, levando à demanda por IA Explicável (XAI).",
    "Box-Cox": "Transformação Box-Cox - Uma função de transformação que obtém uma distribuição normal dos dados usando transformações log e de potência. Útil para estabilizar variância e tornar dados mais normalmente distribuídos.",
    "BPTT": "Retropropagação Através do Tempo - Um método de treinamento de RNN onde a sequência temporal das camadas RNN é primeiro desenrolada, e então a retropropagação é executada.",
    "boosting": "Boosting - Uma técnica de ensemble learning onde modelos são treinados sequencialmente, com cada modelo tentando corrigir os erros dos modelos anteriores. Exemplos incluem AdaBoost, Gradient Boosting e XGBoost.",

    // C
    "C4.5": "C4.5 - Um algoritmo de árvore de decisão que usa ganho de informação relativo para divisão de dados para resolver problemas de classificação ou regressão. Melhoria sobre o algoritmo ID3.",
    "CACE": "Mudar Qualquer Coisa Muda Tudo - O princípio pelo qual qualquer mudança nos dados ou hiperparâmetros de um modelo pode alterar consideravelmente o modelo em si. Destaca a necessidade de controle de versão cuidadoso.",
    "CART": "Árvore de Classificação e Regressão - Um algoritmo de árvore de decisão que usa o índice Gini para divisão de dados. Um dos algoritmos de árvore mais amplamente usados.",
    "CD": "Entrega Contínua - O processo pelo qual componentes de software são automaticamente implementados em um ambiente em ciclos curtos e repetidos. Parte das práticas de MLOps.",
    "centroid": "Centróide - Em um modelo de clustering, a média de todos os pontos de dados que o cluster contém, através de todas as features. K-means atualiza centróides iterativamente.",
    "CI": "Integração Contínua - O processo pelo qual engenheiros de software escrevem, testam e mesclam código em um repositório centralizado frequentemente. Essencial para desenvolvimento de modelos ML.",
    "classification": "Classificação - Um tipo de tarefa de ML onde um exemplo de dados é colocado em uma ou mais categorias. Classificação binária tem duas classes; multi-classe tem três ou mais.",
    "clustering": "Agrupamento (Clustering) - Um tipo de tarefa de ML não supervisionado que coloca exemplos de dados em grupos (clusters) baseado em suas similaridades. Exemplos incluem k-means e clustering hierárquico.",
    "CNN": "Rede Neural Convolucional - Um tipo de rede neural artificial mais comumente usado para processar dados de pixels. Usa operações de convolução para detectar padrões como bordas, texturas e formas em imagens.",
    "co-training": "Co-treinamento - Uma abordagem de aprendizado semi-supervisionado onde dois modelos treinam em duas visões diferentes dos dados, então treinam um ao outro baseado em pseudo-rótulos gerados de dados não rotulados.",
    "coefficient of determination": "Coeficiente de Determinação (R²) - Uma medida estatística que indica quanta variância da variável dependente é explicável pelas variáveis independentes. Varia de 0 a 1, sendo 1 previsão perfeita.",
    "collaborative filtering": "Filtragem Colaborativa - Uma abordagem onde um sistema de recomendação faz recomendações baseadas em usuários com interesses similares. 'Usuários que gostaram de X também gostaram de Y.'",
    "collinearity": "Colinearidade - Quando duas ou mais features são altamente correlacionadas entre si. Veja multicolinearidade.",
    "computer vision": "Visão Computacional - Um conjunto de técnicas pelas quais computadores processam imagens e outros dados visuais. Inclui detecção de objetos, classificação de imagens, segmentação e reconhecimento facial.",
    "concept drift": "Deriva de Conceito - Quando os padrões subjacentes nos dados mudam ao longo do tempo. Veja deriva de modelo.",
    "confusion matrix": "Matriz de Confusão - Um método de visualizar os resultados de um problema de classificação. Mostra Verdadeiros Positivos, Verdadeiros Negativos, Falsos Positivos e Falsos Negativos em formato de matriz.",
    "content filtering": "Filtragem de Conteúdo - Uma abordagem onde um sistema de recomendação faz recomendações baseadas no perfil do usuário comparado ao conteúdo considerado para recomendação.",
    "continuous variable": "Variável Contínua - Uma variável quantitativa cujos valores são incontáveis e podem se estender infinitamente. Exemplos incluem preço, temperatura e tempo.",
    "convolutional layer": "Camada Convolucional - Um tipo de camada em uma CNN onde os neurônios escaneiam uma porção da imagem de entrada para dados dentro do filtro dos neurônios.",
    "cost function": "Função de Custo - Uma função que tenta quantificar o erro entre valores estimados e valores reais de treinamento. Também chamada função de perda ou função objetivo. Exemplos incluem MSE e cross-entropy.",
    "CPU": "Unidade Central de Processamento - O chip de computador que funciona como componente central em um computador de propósito geral. Adequado para processamento sequencial mas mais lento que GPUs para operações matriciais paralelas.",
    "cross-entropy": "Entropia Cruzada - Uma função de custo usada para avaliar o desempenho de uma função softmax penalizando scores de baixa probabilidade para uma classe particular. Função de perda comum para problemas de classificação.",
    "cross-validation": "Validação Cruzada - Um conjunto de métodos para particionar dados de modo que um modelo consiga generalizar para novos dados de teste. K-fold divide dados em K partes e treina K vezes.",
    "curse of dimensionality": "Maldição da Dimensionalidade - O fenômeno pelo qual adicionar mais dimensões (features) a um dataset reduz a capacidade de um modelo de aprender padrões dos dados. Mais features requerem exponencialmente mais dados.",

    // D
    "DAI": "Inteligência Artificial Distribuída - Uma abordagem avançada de implantação onde 'agentes' de aprendizado executam operações paralelas de localizações geograficamente dispersas.",
    "data binning": "Binning de Dados - O processo de discretizar uma variável contínua colocando seus valores em intervalos específicos (bins). Converte dados contínuos em categóricos.",
    "data cleaning": "Limpeza de Dados - O processo de localizar e tratar erros e inconsistências em dados. Inclui tratamento de valores ausentes, duplicatas e outliers.",
    "data drift": "Deriva de Dados - Quando as propriedades estatísticas dos dados de entrada mudam ao longo do tempo, potencialmente degradando o desempenho do modelo. Requer monitoramento e retreinamento.",
    "data encoding": "Codificação de Dados - O processo de converter dados de um certo tipo em um valor codificado de tipo diferente. Exemplos incluem one-hot encoding e label encoding.",
    "data munging": "Munging de Dados - Veja data wrangling. O processo de transformar dados brutos em formato usável.",
    "data preparation": "Preparação de Dados - O processo de alterar dados para que suportem mais efetivamente tarefas como análise de dados e modelagem. Engloba limpeza, transformação e engenharia de features.",
    "data preprocessing": "Pré-processamento de Dados - A tarefa de aplicar várias técnicas de transformação e codificação aos dados para que possam ser interpretados e analisados por um algoritmo de ML.",
    "data wrangling": "Wrangling de Dados - O processo de transformar dados em formato usável. Também chamado data munging. Inclui limpar, remodelar e combinar dados de múltiplas fontes.",
    "Davies-Bouldin index": "Índice Davies-Bouldin - Uma métrica de avaliação de clustering que calcula a razão média entre distância intra-cluster e distância entre clusters para cada cluster comparado ao cluster mais similar. Menor é melhor.",
    "decision boundary": "Fronteira de Decisão - A linha divisória (ou hiperplano em dimensões maiores) que separa classes negativas e positivas em um problema de classificação.",
    "decision tree": "Árvore de Decisão - Um arranjo de declarações condicionais e suas conclusões em uma estrutura de galhos e folhas. Fácil de interpretar e visualizar. Propensa a overfitting sem poda.",
    "deduplication": "Deduplicação - O processo de identificar e remover entradas duplicadas de um dataset.",
    "deep learning": "Aprendizado Profundo - Um tipo de aprendizado de máquina que usa redes neurais artificiais com múltiplas camadas ocultas para tomar decisões complexas. Excelente em reconhecimento de padrões em imagens, texto e áudio.",
    "dendrogram": "Dendrograma - Um diagrama que representa uma hierarquia em forma de árvore, comumente usado para visualizar tarefas de clustering hierárquico. Mostra como clusters se fundem em diferentes limiares de distância.",
    "dependent variable": "Variável Dependente - Em um experimento, a variável que está sendo estudada e é afetada por uma ou mais variáveis independentes. Também chamada variável alvo ou variável resposta em ML.",
    "deployment": "Implantação (Deploy) - O processo pelo qual um modelo de ML é colocado em ambiente de produção, onde pode receber entrada e produzir saída.",
    "diagnosis": "Diagnóstico - A tarefa de determinar a causa de um problema em algum ambiente. Uma aplicação comum de modelos de classificação ML.",
    "dimension": "Dimensão - No aprendizado de máquina, o número de features em um dataset ou modelo treinado nesse dataset.",
    "dimensionality reduction": "Redução de Dimensionalidade - Uma tarefa que minimiza elementos irrelevantes ou desnecessários de um dataset para melhorar o processo de ML. Exemplos incluem PCA e seleção de features.",
    "discrete variable": "Variável Discreta - Uma variável quantitativa cujos valores são contáveis e limitados, pois há uma lacuna definida entre cada valor em um intervalo. Exemplos incluem dados de contagem.",
    "discretization": "Discretização - O processo de converter uma variável contínua em variável discreta através de binning.",
    "discriminator": "Discriminador - Metade de uma GAN que estima se uma imagem é real ou falsa. Tenta distinguir imagens geradas das reais.",
    "Docker": "Docker - Uma plataforma open-source para construir e manter containers virtuais. Amplamente usado para empacotar modelos ML com suas dependências para deploy.",
    "DOE": "Design de Experimentos - Uma abordagem para identificar, analisar e controlar variáveis usadas em um experimento. Também referido como design experimental.",
    "Dunn index": "Índice de Dunn - Uma métrica de avaliação de clustering que calcula a razão entre a menor distância entre dois exemplos em clusters diferentes, e a maior distância entre exemplos no mesmo cluster. Maior é melhor.",

    // E
    "elastic net": "Elastic Net - Uma técnica de regularização que usa uma média ponderada de regressão ridge (L2) e regressão lasso (L1) ao treinar um modelo.",
    "elbow point": "Ponto do Cotovelo - Em clustering, o ponto em que a distância média entre cada exemplo e seu centróide associado não mais diminui significativamente. Usado para selecionar número ótimo de clusters.",
    "embedding": "Embedding (Incorporação) - O processo de condensar um vocabulário linguístico em vetores de dimensões relativamente pequenas. Word embeddings como Word2Vec capturam relações semânticas entre palavras.",
    "endogenous": "Endógeno - A propriedade pela qual uma variável é explicada por outras variáveis em um modelo. Em séries temporais, variáveis endógenas são influenciadas por outras variáveis no sistema.",
    "endpoint": "Endpoint - Um intermediário com o qual consumidores e sistemas interagem para enviar e receber entrada e saída através de uma rede. APIs REST expõem endpoints para servir modelos.",
    "ensemble learning": "Aprendizado de Ensemble - Uma aplicação de ML onde as estimativas de múltiplos modelos são consideradas em combinação. Tipicamente produz melhores resultados que modelos únicos.",
    "entropy": "Entropia - No contexto de árvores de decisão, uma medida de impureza. Um nó com apenas uma classe tem entropia 0; um nó com distribuição igual tem entropia máxima.",
    "epoch": "Época - Em uma rede neural, uma única passagem de treinamento (ida e volta) através de todo o conjunto de dados de entrada.",
    "error": "Erro - Valores incorretos ou ausentes em dados. Também se refere à diferença entre valores previstos e reais.",
    "ETL": "Extrair, Transformar e Carregar - O processo de combinar dados de múltiplas fontes, preparar os dados e carregar os dados resultantes em um formato destino.",
    "evaluation metric": "Métrica de Avaliação - Um método de avaliar a habilidade, desempenho e características de um modelo baseado em uma medição específica. Exemplos incluem acurácia, F1-score e RMSE.",
    "exogenous": "Exógeno - A propriedade pela qual uma variável não é explicada por outras variáveis em um modelo mas vem de fora do sistema.",

    // F
    "F1 score": "F1 Score - A média harmônica ponderada de precisão e recall. F1 = 2 × (Precisão × Recall) / (Precisão + Recall). Varia de 0 a 1.",
    "feature": "Feature (Característica) - No aprendizado de máquina, uma propriedade mensurável de um exemplo no conjunto de treinamento. Também chamada atributo, variável ou preditor.",
    "feature engineering": "Engenharia de Features - A técnica de gerar e extrair features de dados para melhorar a capacidade de um modelo de ML fazer estimativas. Frequentemente a etapa de maior impacto em pipelines de ML.",
    "feature extraction": "Extração de Features - Um tipo de redução de dimensionalidade onde novas features são derivadas das features originais. PCA é uma técnica comum de extração de features.",
    "feature map": "Mapa de Features - Uma representação de imagem que foca em qualquer feature que um filtro de convolução procura em uma CNN.",
    "feature scaling": "Escalonamento de Features - A tarefa de transformar valores de múltiplas features para que estejam em escala similar. Inclui normalização e padronização.",
    "feature selection": "Seleção de Features - Um tipo de redução de dimensionalidade onde um subconjunto das features originais é selecionado. Métodos incluem abordagens filter, wrapper e embedded.",
    "filter": "Filtro - Em CNNs, a porção do campo receptivo que um neurônio de camada convolucional usa para escanear a imagem em camadas anteriores. Filtros detectam padrões específicos.",
    "fitting": "Ajuste - Veja treinamento. O processo de aprender parâmetros do modelo a partir de dados.",
    "FNN": "Rede Neural Feedforward - Um tipo de RNA onde informação flui em uma única direção, da entrada para a saída.",
    "forecasting": "Previsão (Forecasting) - Uma tarefa que envolve fazer previsões sobre eventos futuros baseado na análise de eventos passados relevantes. Comum em análise de séries temporais.",
    "Fourier transformation": "Transformação de Fourier - O processo de decompor um sinal de áudio em suas frequências constituintes. Converte domínio do tempo para domínio da frequência.",
    "FPR": "Taxa de Falso Positivo - Uma medida de quão frequentemente o modelo classificou incorretamente instâncias negativas como positivas. FPR = FP / (FP + TN).",
    "frame rate": "Taxa de Quadros - O número de frames (imagens) exibidos por segundo em um vídeo. Taxas comuns são 24, 30 e 60 fps.",
    "frequency": "Frequência - No processamento de áudio, o número de ondas sonoras repetidas por segundo, medido em Hertz (Hz).",

    // G
    "GAN": "Rede Adversária Generativa - Uma arquitetura de rede neural que coloca duas redes diferentes uma contra a outra (gerador e discriminador), tipicamente para gerar imagens realistas.",
    "Gaussian RBF kernel": "Kernel RBF Gaussiano - Um método de kernel trick que projeta um novo espaço de features em dimensões maiores medindo a distância entre todos os exemplos e centros definidos.",
    "generalization": "Generalização - A capacidade de um modelo de se adaptar adequadamente a dados novos, nunca vistos. O objetivo do ML é boa generalização, não apenas memorizar dados de treino.",
    "generator": "Gerador - Metade de uma GAN que cria dados sintéticos (ex: imagens) e tenta enganar o discriminador fazendo-o acreditar que são reais.",
    "genetic algorithm": "Algoritmo Genético - Uma abordagem de otimização inspirada na teoria da seleção natural. Usa mutação, crossover e seleção para evoluir soluções.",
    "Gini index": "Índice Gini - Uma métrica de divisão de árvore de decisão que mede a 'pureza' dos nós de decisão calculando a probabilidade de classificação incorreta. Menor é melhor.",
    "Goodhart's law": "Lei de Goodhart - Um princípio que afirma: 'Quando uma medida se torna alvo, ela deixa de ser uma boa medida.' Lembrete para não depender muito de métricas únicas ao avaliar modelos ML.",
    "GPU": "Unidade de Processamento Gráfico - O chip de computador tipicamente usado para renderização gráfica, mas também excelente para operações matriciais paralelas requeridas no treinamento de deep learning.",
    "gradient boosting": "Gradient Boosting - Um método de ensemble learning iterativo que constrói múltiplas árvores de decisão em sucessão, onde cada árvore tenta reduzir os erros da árvore anterior. Exemplos incluem XGBoost e LightGBM.",
    "gradient descent": "Descida de Gradiente - Um método de minimizar uma função de custo onde os parâmetros do modelo são ajustados em várias iterações tomando 'passos' graduais descendo uma inclinação, em direção a um valor mínimo de erro.",
    "grid search": "Busca em Grade - Um método de otimização de hiperparâmetros que pega um conjunto (ou grade) de combinações de parâmetros, treina um modelo usando cada combinação, e retorna a que melhor otimiza uma métrica especificada.",
    "GRU cell": "Célula GRU - Gated Recurrent Unit - Uma versão simplificada da célula LSTM usada em RNNs. Mais rápida para treinar que LSTM.",

    // H
    "HAC": "Clustering Aglomerativo Hierárquico - Um tipo de algoritmo de clustering que inicializa cada exemplo em seu próprio cluster, então gradualmente mescla os exemplos e clusters mais próximos.",
    "hard-margin classification": "Classificação de Margem Rígida - Um tipo de classificação em SVMs onde todos os exemplos devem estar fora das margens. Só funciona para dados linearmente separáveis.",
    "hashing": "Hashing - Um processo ou função que transforma entrada em texto plano em uma saída de comprimento fixo indecifrável e garante que este processo não pode ser reversivelmente executado. Usado para privacidade e segurança.",
    "HDC": "Clustering Divisivo Hierárquico - Um tipo de algoritmo de clustering que inicializa todos os exemplos em um único cluster, então gradualmente divide os dados em mais clusters.",
    "hidden layer": "Camada Oculta - Uma camada de neurônios em uma rede neural que não está diretamente exposta à entrada ou saída. Camadas ocultas aprendem representações intermediárias dos dados.",
    "holdout": "Holdout - Um método de amostragem onde o dataset é dividido em dois: conjunto de treino e conjunto de teste. Simples mas pode ter alta variância.",
    "hyperparameter": "Hiperparâmetro - Um parâmetro externo ao modelo de ML (definido no algoritmo antes do treino, não aprendido dos dados). Exemplos incluem taxa de aprendizado e profundidade da árvore.",
    "hyperparameter optimization": "Otimização de Hiperparâmetros - O processo de alterar repetidamente os hiperparâmetros que um algoritmo usa para treinar um modelo a fim de determinar o conjunto que leva ao melhor desempenho.",
    "hyperplane": "Hiperplano - Em SVMs, uma fronteira de decisão que separa classes. Em espaço n-dimensional, um hiperplano é (n-1) dimensional.",

    // I
    "ID3": "ID3 - Um algoritmo de árvore de decisão que usa ganho de informação para divisão de dados para resolver problemas de classificação. Predecessor do C4.5.",
    "identity matrix": "Matriz Identidade - Uma matriz quadrada de zeros exceto pela diagonal principal, que consiste de todos 1s. Usada em várias operações de álgebra linear.",
    "image resolution": "Resolução de Imagem - O número total de pixels em uma imagem, geralmente expresso como largura × altura (ex: 1920×1080).",
    "imputation": "Imputação - O processo de preencher valores ausentes de dados usando cálculos estatísticos para determinar quais deveriam ser os valores ausentes. Métodos incluem média, mediana, moda e imputação KNN.",
    "independent variable": "Variável Independente - Em um experimento, uma variável que pode ter efeito na variável dependente. Também chamada feature, preditor ou variável de entrada em ML.",
    "information gain": "Ganho de Informação - Uma métrica de divisão de árvore de decisão que mede a redução na entropia após uma divisão. Maior ganho de informação indica melhores divisões.",
    "input layer": "Camada de Entrada - Uma camada de neurônios em uma rede neural que recebe os dados brutos de entrada e os passa para as camadas ocultas.",
    "irreducible error": "Erro Irredutível - Erros que não podem ser reduzidos ao ajustar um modelo de ML, devido ao ruído inerente nos dados ou fatores desconhecidos.",
    "iteration": "Iteração - Em uma rede neural, uma única passagem de treino através de apenas um lote do conjunto de dados.",

    // K
    "k-fold cross-validation": "Validação Cruzada K-Fold - Um método de validação cruzada onde o dataset é dividido em k grupos (folds). Cada grupo é usado uma vez como conjunto de teste enquanto os k-1 restantes formam o conjunto de treino.",
    "k-means clustering": "Clustering K-means - Um tipo de algoritmo de clustering não supervisionado que atualiza iterativamente centróides de clusters baseado no valor médio de cada exemplo no cluster do centróide.",
    "k-NN": "k-Vizinhos Mais Próximos - Um algoritmo que classifica exemplos de dados baseado em suas similaridades com outros exemplos no espaço de features. Usa métricas de distância como distância Euclidiana.",
    "kernel trick": "Truque do Kernel - Um grupo de métodos matemáticos para representar eficientemente dados não linearmente separáveis em espaço de maior dimensão sem computar explicitamente a transformação.",

    // L
    "label": "Rótulo - No aprendizado supervisionado, a variável verdade fundamental que um modelo deve estimar para novos dados. Também chamado alvo ou resposta.",
    "lasso regression": "Regressão Lasso (Regularização L1) - Uma técnica de regularização que usa norma L1 para reduzir features irrelevantes a exatamente 0, efetivamente realizando seleção de features.",
    "latent representation": "Representação Latente - A forma simplificada e reduzida de features dos dados modelada por uma rede neural, tipicamente nas camadas ocultas.",
    "learning curve": "Curva de Aprendizado - Um método de comparar visualmente a mudança no score ou erro de um modelo ao número de exemplos de dados usados ou iterações de treinamento.",
    "learning mode": "Modo de Aprendizado - A forma pela qual um modelo de ML aprende dos dados — supervisionado, não supervisionado, semi-supervisionado ou aprendizado por reforço.",
    "learning rate": "Taxa de Aprendizado - Na descida de gradiente, o tamanho de cada 'passo' descendo a inclinação. Muito alto causa oscilação; muito baixo causa convergência lenta.",
    "lemmatization": "Lematização - O processo de usar morfologia linguística para determinar a forma base de dicionário de uma palavra flexionada. 'Correndo' → 'correr'.",
    "linear kernel": "Kernel Linear - Um método de kernel trick simples e rápido que se aplica apenas a dados linearmente separáveis. Apenas computa o produto escalar dos vetores.",
    "linear regression": "Regressão Linear - Um tipo de análise de regressão onde há uma relação linear entre variáveis independentes e a variável dependente.",
    "logistic function": "Função Logística (Sigmoid) - A função que produz valores entre 0 e 1, tomando forma de S. Usada em regressão logística para estimativa de probabilidade.",
    "logistic regression": "Regressão Logística - Um tipo de algoritmo de classificação (apesar do nome) onde a saída é uma probabilidade entre 0 e 1, tipicamente usado para classificação binária.",
    "LOOCV": "Leave-One-Out Cross-Validation - Um método de validação cruzada onde cada ponto de dados individual é usado uma vez como conjunto de teste. Mais minucioso mas computacionalmente caro.",
    "LPOCV": "Leave-P-Out Cross-Validation - Um método de validação cruzada k-fold com n − p sendo o conjunto de treino e p sendo o conjunto de teste.",
    "LSTM": "Long Short-Term Memory (Memória de Longo e Curto Prazo) - Um tipo de célula RNN que usa portões para preservar informação importante através de longas sequências enquanto 'esquece' informação irrelevante. Resolve o problema do gradiente desvanecente.",

    // M  
    "machine learning": "Aprendizado de Máquina - Uma disciplina de IA onde uma máquina é capaz de gradualmente melhorar suas capacidades de estimativa sem receber instruções de programação explícitas. O sistema aprende dos dados.",
    "MAE": "Erro Absoluto Médio - Uma função de custo que calcula a diferença absoluta média entre valores estimados e reais. Menos sensível a outliers que MSE.",
    "MBGD": "Gradient Descent de Mini-Lote - Uma abordagem de descida de gradiente que seleciona um pequeno grupo de exemplos aleatoriamente do dataset, então usa para calcular os gradientes. Equilibra velocidade e estabilidade.",
    "memory cell": "Célula de Memória - Um componente de uma RNN que mantém informação de estado através de passos temporais.",
    "MLOps": "Operações de Machine Learning - A disciplina de aplicar práticas de DevOps para automatizar desenvolvimento, teste, deploy e manutenção de modelos de ML em produção.",
    "MLP": "Perceptron Multi-Camada - Uma arquitetura de rede neural com múltiplas camadas distintas de neurônios (entrada, ocultas, saída) conectadas de forma feedforward.",
    "model drift": "Deriva de Modelo - Um processo pelo qual os padrões inicialmente usados para treinar um modelo de ML mudam ao longo do tempo de modo que o modelo não mais performa bem com novos dados. Requer retreinamento.",
    "model parameter": "Parâmetro do Modelo - Um parâmetro interno a um modelo de ML (derivado do modelo durante o processo de treinamento). Exemplos incluem pesos e vieses.",
    "MSE": "Erro Quadrático Médio - Uma função de custo que eleva ao quadrado o erro entre valores estimados e reais, então calcula a média. Penaliza erros maiores mais pesadamente.",
    "multi-class classification": "Classificação Multi-classe - Um problema de classificação onde um exemplo pode ser colocado em uma de três ou mais classes (ex: classificar imagens como gato, cachorro ou pássaro).",
    "multi-label classification": "Classificação Multi-rótulo - Um problema de classificação onde um exemplo pode receber múltiplos rótulos simultaneamente (ex: um filme pode ser 'ação' e 'comédia').",
    "multicollinearity": "Multicolinearidade - A propriedade que descreve múltiplas variáveis exibindo uma relação linear entre si. Pode causar problemas em modelos de regressão.",
    "multinomial logistic regression": "Regressão Logística Multinomial - Um algoritmo comumente usado para resolver problemas de classificação multi-classe estendendo regressão logística para múltiplas classes.",
    "multivariate": "Multivariado - A propriedade de um dataset ter múltiplas variáveis sendo estudadas ou usadas como features.",
    "multivariate regression": "Regressão Multivariada - Um tipo de análise de regressão que envolve múltiplas variáveis independentes prevendo uma ou mais variáveis dependentes.",

    // N
    "NLP": "Processamento de Linguagem Natural - Um conjunto de técnicas pelas quais computadores trabalham com e analisam linguagens humanas. Inclui classificação de texto, análise de sentimento, tradução e resposta a perguntas.",
    "noise": "Ruído - Valores de dados irrelevantes ou irregulares que dificultam identificar padrões significativos nos dados. Variação aleatória que não representa o sinal verdadeiro.",
    "non-parametric": "Não-paramétrico - Uma descrição de um algoritmo de ML que pode gerar um número potencialmente infinito de parâmetros baseado nos dados. Exemplos incluem k-NN e árvores de decisão.",
    "normal equation": "Equação Normal - Uma solução de forma fechada para problemas de regressão linear que computa diretamente os pesos ótimos sem iteração. Funciona para datasets pequenos.",
    "normalization": "Normalização (Min-Max) - Uma técnica de escalonamento de features onde as features são escaladas para que o menor valor seja 0 e o maior seja 1.",
    "neural network": "Rede Neural - Um modelo computacional inspirado em neurônios biológicos. Consiste em camadas de nós conectados que aprendem a transformar entradas em saídas através de treinamento.",

    // O
    "offline model": "Modelo Offline - Um modelo de ML implantado de forma que treina com novos dados periodicamente (ex: diariamente ou semanalmente) ao invés de continuamente. Também chamado batch learning.",
    "one-hot encoding": "One-Hot Encoding - Uma técnica para converter variáveis categóricas em vetores binários, com uma coluna por categoria contendo 1 para presença e 0 para ausência.",
    "online model": "Modelo Online - Um modelo de ML que treina com novos dados continuamente conforme chegam, atualizando parâmetros incrementalmente. Também chamado online learning.",
    "ordinal data": "Dados Ordinais - Dados categóricos que podem ser colocados em uma ordem significativa (ex: 'baixo', 'médio', 'alto').",
    "outlier": "Outlier (Valor Atípico) - Um valor fora da distribuição principal, desviando significativamente do resto dos valores no dataset. Pode ser erro ou valor extremo válido.",
    "output layer": "Camada de Saída - Uma camada de neurônios em uma rede neural que formata e produz as previsões ou classificações finais.",
    "overfitting": "Overfitting (Sobreajuste) - Um problema em ML onde as estimativas de um modelo se ajustam bem aos dados de treino mas falham em generalizar para novos dados. Caracterizado por alta variância e baixo viés.",

    // P
    "padding": "Padding - Em CNNs, adicionar pixels (geralmente zeros) ao redor de uma imagem de entrada para preservar suas dimensões após operações de convolução.",
    "parallelization": "Paralelização - A técnica de dividir tarefas de processamento entre múltiplos processadores para aumentar o desempenho. GPUs permitem paralelização para deep learning.",
    "parametric": "Paramétrico - Uma descrição de um algoritmo de ML que gera um número fixo de parâmetros do modelo independente do tamanho do dataset. Exemplos incluem regressão linear.",
    "PCA": "Análise de Componentes Principais - Uma técnica de redução de dimensionalidade que transforma features em componentes principais que capturam a maior variância.",
    "penetration test": "Teste de Penetração - Um teste de segurança que usa ferramentas ativas para avaliar segurança simulando um ataque autorizado em um sistema. Importante para sistemas ML que lidam com dados sensíveis.",
    "perceptron": "Perceptron - A forma mais simples de rede neural, consistindo de um único neurônio que faz classificações binárias baseado em uma combinação linear de entradas.",
    "periodicity": "Periodicidade - Em um modelo de previsão sazonal, o número de observações que compreendem uma única estação (ex: 12 para dados mensais com sazonalidade anual).",
    "perturbation": "Perturbação - Métodos para modificar levemente dados (especialmente imagens) sem mudar significativamente seu significado. Usado em aumento de dados e testes adversários.",
    "PII": "Informação Pessoalmente Identificável - Dados que devem ser protegidos para garantir a privacidade de indivíduos, como nomes, endereços e números de documentos.",
    "pipeline": "Pipeline - Um conjunto sequencial de tarefas que automatizam o processo de ML alimentando a saída de uma tarefa na entrada da próxima tarefa.",
    "POC": "Prova de Conceito - Evidência que suporta a viabilidade de uma solução de ML antes do desenvolvimento completo.",
    "polynomial kernel": "Kernel Polinomial - Um método de kernel trick que usa combinações polinomiais de features para classificação não linear em SVMs.",
    "pooling layer": "Camada de Pooling - Uma camada em uma CNN que aplica uma função de agregação (max ou média) para reduzir as dimensões espaciais dos mapas de features.",
    "PRC": "Curva Precisão-Recall - Um método de visualizar o tradeoff entre precisão e recall em diferentes limiares de classificação.",
    "precision": "Precisão - Uma medida de quão frequentemente os positivos identificados pelo modelo são realmente positivos. Precisão = VP / (VP + FP).",
    "prediction": "Predição - A tarefa de ML de estimar o estado de algo (geralmente no futuro) baseado em dados passados ou atuais.",
    "principle of least privilege": "Princípio do Menor Privilégio - O princípio de segurança pelo qual um usuário ou sistema recebe apenas as permissões mínimas necessárias para executar sua função.",
    "privacy by design": "Privacidade por Design - Uma abordagem de desenvolvimento de software que incorpora considerações de privacidade em cada fase do desenvolvimento, não apenas no final.",
    "problem formulation": "Formulação do Problema - O processo de identificar uma questão que deve ser abordada e colocá-la em termos compreensíveis e acionáveis para aprendizado de máquina.",
    "pruning": "Poda - O processo de reduzir o tamanho de uma árvore de decisão eliminando nós, galhos e folhas que fornecem pouco valor, reduzindo overfitting.",
    "pseudo-label": "Pseudo-rótulo - No aprendizado semi-supervisionado, uma previsão feita por um modelo que é usada como rótulo para treinar outro modelo.",

    // Q
    "qualitative data": "Dados Qualitativos - Dados que contêm valores categóricos representando qualidades ou características ao invés de quantidades.",
    "quantitative data": "Dados Quantitativos - Dados que contêm valores numéricos que representam magnitude ou quantidade.",
    "quasi-identifier": "Quasi-identificador - Um valor de dados que não contém diretamente PII mas pode ser combinado com outros valores para identificar um indivíduo (ex: CEP + data de nascimento).",

    // R
    "R2": "R-quadrado (Coeficiente de Determinação) - Uma métrica indicando quanta variância da variável alvo é explicada pelo modelo. Varia de 0 a 1 para modelos razoáveis.",
    "random forest": "Random Forest (Floresta Aleatória) - Um método de ensemble learning que agrega múltiplos modelos de árvore de decisão, tipicamente usando bagging, e seleciona o classificador ou preditor ótimo através de votação ou média.",
    "randomized search": "Busca Aleatória - Um método de otimização de hiperparâmetros que amostra combinações de parâmetros aleatoriamente de distribuições, frequentemente mais eficiente que grid search para espaços grandes.",
    "recall": "Recall (Sensibilidade) - Uma medida da porcentagem de instâncias positivas que são corretamente identificadas pelo modelo. Recall = VP / (VP + FN).",
    "recommendation system": "Sistema de Recomendação - Um sistema que sugere itens, serviços ou conteúdo de interesse para usuários baseado em suas preferências, histórico ou similaridade com outros usuários.",
    "regression": "Regressão - Um tipo de tarefa de ML que mede a relação entre variáveis e produz uma estimativa para uma variável numérica (contínua).",
    "regularization": "Regularização - A técnica de simplificar um modelo de ML restringindo parâmetros do modelo, ajudando o modelo a evitar overfitting aos dados de treino. Exemplos incluem regularização L1 e L2.",
    "reinforcement learning": "Aprendizado por Reforço - Um tipo de ML onde um agente de software aprende tomando ações em um ambiente e recebendo recompensas ou penalidades baseado nos resultados.",
    "ReLU": "Unidade Linear Retificada - Uma função de ativação que produz a entrada diretamente se positiva, caso contrário produz 0. f(x) = max(0, x). Ativação mais popular para camadas ocultas.",
    "ridge regression": "Regressão Ridge (Regularização L2) - Uma técnica de regularização que usa norma L2 para restringir pesos de features, reduzindo overfitting sem eliminar features completamente.",
    "RMSE": "Raiz do Erro Quadrático Médio - A raiz quadrada do MSE, trazendo a métrica de erro de volta às mesmas unidades da variável alvo.",
    "RNN": "Rede Neural Recorrente - Um tipo de RNA onde informação pode fluir em loops, permitindo que a rede mantenha memória de entradas anteriores. Boa para dados sequenciais.",
    "robotics": "Robótica - A disciplina que envolve estudar, projetar e operar robôs. Frequentemente incorpora IA para percepção, tomada de decisão e controle.",
    "ROC curve": "Curva ROC (Receiver Operating Characteristic) - Um gráfico da taxa de verdadeiro positivo (recall) versus taxa de falso positivo em vários limiares de classificação.",

    // S
    "SAG": "Gradiente Médio Estocástico - Uma abordagem de otimização similar ao SGD mas mantém memória de computações de gradiente passadas para convergência mais rápida.",
    "seasonality": "Sazonalidade - A propriedade pela qual uma série temporal exibe padrões recorrentes em intervalos fixos (ex: vendas maiores todo dezembro).",
    "self-training": "Auto-treinamento - Uma abordagem de aprendizado semi-supervisionado onde um modelo treinado em dados rotulados gera pseudo-rótulos para dados não rotulados, que são então usados para treinamento adicional.",
    "semi-structured data": "Dados Semi-estruturados - Dados parcialmente organizados, tornando alguns elementos fáceis de buscar e extrair enquanto outros não são (ex: JSON, XML).",
    "semi-supervised learning": "Aprendizado Semi-supervisionado - Um tipo de ML que usa uma pequena quantidade de dados rotulados junto com uma grande quantidade de dados não rotulados para treinamento.",
    "sensitivity": "Sensibilidade - Veja recall. A taxa de verdadeiro positivo.",
    "SGD": "Descida de Gradiente Estocástica - Uma abordagem de descida de gradiente que seleciona aleatoriamente um exemplo por vez do dataset para calcular gradientes. Mais rápida mas mais ruidosa que GD em lote.",
    "sigmoid": "Sigmoid - Uma função de ativação que produz valores entre 0 e 1 em forma de curva S. σ(x) = 1 / (1 + e^(-x)). Usada para saídas de classificação binária.",
    "sigmoid kernel": "Kernel Sigmoid - Um método de kernel trick que usa função tangente hiperbólica para criar um equivalente de uma rede neural simples em SVMs.",
    "silhouette analysis": "Análise de Silhueta - Um método de calcular quão bem um exemplo particular se encaixa em seu cluster atribuído comparado a clusters vizinhos. Score varia de -1 a 1.",
    "skillful": "Habilidoso - Usado para descrever um modelo que é útil para sua tarefa pretendida, performando melhor que baselines ingênuos. Modelos têm graus de habilidade.",
    "SMOTE": "Técnica de Sobre-amostragem de Minoria Sintética - Um método para lidar com datasets desbalanceados criando exemplos sintéticos da classe minoritária.",
    "soft-margin classification": "Classificação de Margem Suave - Uma abordagem de classificação SVM que permite que alguns exemplos caiam dentro ou do lado errado da margem, permitindo classificação de dados não linearmente separáveis.",
    "softmax": "Softmax - Uma função de ativação que converte um vetor de números em uma distribuição de probabilidade onde todos os valores somam 1. Usada para saídas de classificação multi-classe.",
    "specificity": "Especificidade (Taxa de Verdadeiro Negativo) - Uma medida de quão frequentemente um modelo identifica corretamente instâncias negativas reais. Especificidade = VN / (VN + FP).",
    "spectrogram": "Espectrograma - Uma representação visual de áudio mostrando tempo no eixo x, frequência no eixo y, e amplitude como intensidade de cor.",
    "stakeholder": "Stakeholder (Parte Interessada) - Uma pessoa que tem interesse no resultado de um projeto ou está ativamente envolvida em seu trabalho. Inclui clientes, patrocinadores e membros da equipe.",
    "standardization": "Padronização (Z-score) - Uma técnica de escalonamento de features onde features são escaladas para ter média 0 e desvio padrão 1.",
    "stationarity": "Estacionariedade - A propriedade de uma série temporal onde atributos estatísticos (média, variância, covariância) são constantes ao longo do tempo ao invés de ter tendência.",
    "stemming": "Stemming - O processo de remover o afixo de uma palavra para recuperar a raiz da palavra. 'Correndo' → 'Corr'. Mais grosseiro que lematização.",
    "stochastic": "Estocástico - A propriedade de aleatoriedade em um processo. Métodos estocásticos usam amostragem aleatória, introduzindo variância mas potencialmente melhor exploração.",
    "stop word": "Stop Word (Palavra de Parada) - Uma palavra comum em texto (como 'o', 'é', 'em') que é tipicamente removida durante pré-processamento pois carrega pouco significado.",
    "stratified sampling": "Amostragem Estratificada - Uma técnica de amostragem que mantém a mesma proporção de classes em cada divisão como no dataset original. Importante para datasets desbalanceados.",
    "stride": "Stride (Passo) - Em CNNs, a distância (número de pixels) que o filtro move entre operações de convolução. Stride maior = mais downsampling.",
    "structured data": "Dados Estruturados - Dados organizados em um formato que facilita busca, filtragem e extração, como dados em bancos de dados relacionais ou arquivos CSV.",
    "supervised learning": "Aprendizado Supervisionado - Um tipo de ML onde os dados de treino incluem rótulos (respostas corretas), então o modelo aprende a prever esses rótulos para novos dados.",
    "SVM": "Máquina de Vetores de Suporte - Um algoritmo de aprendizado supervisionado que encontra o hiperplano ótimo para separar classes com máxima margem. Efetivo em espaços de alta dimensão.",

    // T
    "tanh": "Tangente Hiperbólica - Uma função de ativação que produz valores entre -1 e 1. Similar ao sigmoid mas centrada em zero.",
    "target function": "Função Alvo - O mapeamento ideal entre variáveis de entrada e variáveis de saída que um modelo de ML tenta aproximar.",
    "target variable": "Variável Alvo - Veja variável dependente. A variável que um modelo está tentando prever.",
    "TF-IDF": "Frequência de Termo-Frequência Inversa de Documento - Um método de vetorização de texto que pondera palavras por sua frequência em um documento relativa à sua frequência em todos os documentos.",
    "threshold": "Limiar - Um valor usado por um modelo de classificação para determinar pertencimento a classe. Previsões acima do limiar são classificadas como positivas.",
    "time series": "Série Temporal - Uma sequência de pontos de dados ordenados no tempo, onde a ordenação temporal é significativa para análise e previsão.",
    "TLU": "Unidade Lógica de Limiar - O bloco de construção básico de perceptrons que computa uma soma ponderada de entradas e aplica uma função degrau.",
    "TNR": "Taxa de Verdadeiro Negativo - Veja especificidade.",
    "tokenization": "Tokenização - O processo de dividir texto em unidades menores (tokens), tipicamente palavras ou subpalavras, para processamento de NLP.",
    "TPR": "Taxa de Verdadeiro Positivo - Veja recall.",
    "training": "Treinamento - O processo pelo qual um modelo de ML aprende padrões de dados de entrada ajustando seus parâmetros para minimizar erro.",
    "transfer learning": "Aprendizado por Transferência - Uma técnica onde um modelo treinado em uma tarefa é reutilizado como ponto de partida para um modelo em uma tarefa diferente mas relacionada. Comum em deep learning.",

    // U
    "underfitting": "Underfitting (Subajuste) - Um problema em ML onde um modelo não pode fazer estimativas efetivas porque é muito simples para capturar os padrões subjacentes. Caracterizado por alto viés e baixa variância.",
    "univariate": "Univariado - A propriedade de um dataset ou análise envolvendo apenas uma única variável.",
    "unstructured data": "Dados Não Estruturados - Dados que não estão organizados em um formato pré-definido, tornando difícil buscar e analisar diretamente. Exemplos incluem imagens, áudio e texto livre.",
    "unsupervised learning": "Aprendizado Não Supervisionado - Um tipo de ML onde dados de treino não incluem rótulos, então o modelo deve descobrir padrões e estrutura por conta própria.",

    // V
    "validation set": "Conjunto de Validação - Uma porção de dados separada do treino, usada para ajustar hiperparâmetros e seleção de modelo antes da avaliação final no conjunto de teste.",
    "VAR": "Vetor Autoregressivo - Um algoritmo comum para executar previsão de séries temporais multivariadas onde múltiplas variáveis influenciam umas às outras.",
    "variance": "Variância - Em estatística, uma medida de dispersão em um dataset. Em ML, variância se refere a quanto as previsões do modelo mudam com diferentes conjuntos de treino. Alta variância indica overfitting.",

    // W
    "waveform": "Forma de Onda - No processamento de áudio, uma representação visual da amplitude do sinal ao longo do tempo, mostrando a forma da onda sonora.",
    "WCSS": "Soma dos Quadrados Dentro do Cluster - Uma métrica de avaliação de clustering que mede a compacidade dos clusters. A soma das distâncias quadradas de cada ponto ao centróide de seu cluster.",

    // Z
    "z-score": "Z-score - O número de desvios padrão que um valor de amostra está acima ou abaixo da média. Usado para padronização e detecção de outliers."
};

// Função para aplicar tooltips ao conteúdo de texto
function applyGlossaryTooltips(text) {
    // Ordenar termos por tamanho (maior primeiro) para evitar substituições parciais
    const sortedTerms = Object.keys(GLOSSARY).sort((a, b) => b.length - a.length);

    // Usar placeholders para evitar substituições aninhadas
    const placeholders = [];
    let result = text;

    sortedTerms.forEach((term, index) => {
        // Criar padrão case-insensitive com limites de palavra
        const pattern = new RegExp(`\\b(${escapeRegExp(term)})\\b`, 'gi');

        // Substituir com placeholder primeiro (apenas primeira ocorrência para evitar poluição)
        let replaced = false;
        result = result.replace(pattern, (match) => {
            if (!replaced) {
                replaced = true;
                const placeholder = `__GLOSSARY_${index}_${placeholders.length}__`;
                const tooltip = GLOSSARY[term].replace(/"/g, '&quot;').replace(/'/g, '&#39;');
                placeholders.push({
                    placeholder,
                    replacement: `<span class="glossary-term" data-tooltip="${tooltip}">${match}</span>`
                });
                return placeholder;
            }
            return match; // Não substituir ocorrências subsequentes
        });
    });

    // Agora substituir todos os placeholders com HTML real
    placeholders.forEach(({ placeholder, replacement }) => {
        result = result.replace(placeholder, replacement);
    });

    return result;
}

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

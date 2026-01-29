// CAIP-210 Study Content - Lessons 7-9
// Clustering, Decision Trees, SVM

// Lesson 7: Building Clustering Models
STUDY_CONTENT[7] = {
    name: "Lesson 7: Building Clustering Models",
    icon: "🔵",
    weight: "Focus Areas: k-Means, Hierarchical Clustering, Evaluation",
    topics: [
        {
            title: "k-Means Clustering",
            concept: `K-Means agrupa dados em k clusters baseado em distância:

🔄 ALGORITMO:
1. Escolher k centróides iniciais (aleatório ou k-means++)
2. Atribuir cada ponto ao centróide mais próximo
3. Recalcular centróides como média dos pontos
4. Repetir 2-3 até convergência

📊 ESCOLHENDO k:

ELBOW METHOD:
• Plotar WCSS vs. k
• WCSS = Within-Cluster Sum of Squares
• Procurar "cotovelo" onde redução desacelera

SILHOUETTE SCORE:
• Mede quão similar ponto é ao seu cluster vs. outros
• Score: -1 a 1 (maior = melhor)
• Usar k com maior silhouette médio

⚙️ k-MEANS++:
• Inicialização mais inteligente
• Primeiro centróide aleatório
• Próximos proporcionais à distância
• Evita convergência ruim

⚠️ LIMITAÇÕES:
• Assume clusters esféricos
• Sensível a escala
• Sensível a outliers
• Número k deve ser especificado`,
            keyPoints: [
                "K-means minimiza distância intra-cluster",
                "Elbow method: procurar ponto de inflexão",
                "Silhouette score: qualidade dos clusters",
                "k-means++: inicialização melhor que aleatória",
                "ESCALONAR features antes de clustering"
            ],
            example: `from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Escalonar dados (OBRIGATÓRIO!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
wcss = []
silhouettes = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

# Plotar
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(K, wcss, 'bo-')
axes[0].set_xlabel('k')
axes[0].set_ylabel('WCSS')
axes[0].set_title('Elbow Method')

axes[1].plot(K, silhouettes, 'go-')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis')

# Melhor k (maior silhouette)
best_k = K[np.argmax(silhouettes)]
print(f"K ótimo por silhouette: {best_k}")

# Modelo final
kmeans = KMeans(n_clusters=best_k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Análise dos clusters
for i in range(best_k):
    print(f"Cluster {i}: {np.sum(clusters == i)} pontos")
    print(f"  Centróide: {scaler.inverse_transform([kmeans.cluster_centers_[i]])[0]}")`,
            realCase: {
                title: "Segmentação de Clientes RFM",
                description: "Varejistas usam k-means em features RFM (Recency, Frequency, Monetary) para segmentar clientes em grupos como 'Champions', 'Loyal', 'At Risk', 'Lost'. Cada segmento recebe marketing diferenciado.",
                impact: "Aumenta ROI de marketing ao personalizar mensagens por segmento"
            }
        },
        {
            title: "Hierarchical Clustering",
            concept: `Cria hierarquia de clusters sem especificar k:

📊 DOIS TIPOS:

AGLOMERATIVO (bottom-up):
1. Cada ponto é um cluster
2. Mesclar clusters mais próximos
3. Repetir até um cluster
4. Dendrograma registra hierarquia

DIVISIVO (top-down):
1. Todos pontos em um cluster
2. Dividir cluster menos coeso
3. Repetir até clusters individuais

🔗 LINKAGE (critério de proximidade):

SINGLE (nearest):
• Distância mínima entre pontos
• Pode criar clusters alongados

COMPLETE (farthest):
• Distância máxima entre pontos
• Clusters mais compactos

AVERAGE:
• Média das distâncias
• Equilíbrio

WARD:
• Minimiza variância intra-cluster
• Clusters esféricos, mais usado

📈 DENDROGRAMA:
• Visualiza hierarquia
• Cortar em altura h → k clusters
• Escolher h onde gap é grande`,
            keyPoints: [
                "Não precisa especificar k antecipadamente",
                "Dendrograma mostra toda hierarquia",
                "Ward linkage: clusters compactos e esféricos",
                "Cortar dendrograma na altura desejada",
                "Computacionalmente caro para grandes datasets"
            ],
            example: `from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# Calcular linkage matrix
Z = linkage(X_scaled, method='ward')

# Plotar dendrograma
plt.figure(figsize=(12, 6))
dendrogram(Z, 
           truncate_mode='level', 
           p=5,
           leaf_rotation=90)
plt.title('Dendrograma Hierárquico')
plt.xlabel('Samples')
plt.ylabel('Distância')
plt.axhline(y=15, color='r', linestyle='--', label='Corte k=3')
plt.legend()

# Cortar em altura específica
clusters_h = fcluster(Z, t=15, criterion='distance')

# Ou especificar número de clusters
agg = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)
clusters = agg.fit_predict(X_scaled)

# Comparar diferentes linkages
for linkage_method in ['single', 'complete', 'average', 'ward']:
    agg = AgglomerativeClustering(n_clusters=3, linkage=linkage_method)
    labels = agg.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"{linkage_method}: Silhouette = {score:.3f}")`,
            realCase: {
                title: "Filogenia em Biologia",
                description: "Biólogos usam clustering hierárquico para construir árvores filogenéticas mostrando relações evolutivas entre espécies. O dendrograma representa a história evolutiva.",
                impact: "Fundamental para biologia evolutiva e taxonomia"
            }
        },
        {
            title: "Clustering Evaluation Metrics",
            concept: `Métricas para avaliar qualidade de clusters:

📊 MÉTRICAS INTERNAS (sem labels):

SILHOUETTE SCORE:
• s = (b - a) / max(a, b)
• a = distância média intra-cluster
• b = distância média ao cluster mais próximo
• Range: -1 a 1 (maior = melhor)

DAVIES-BOULDIN INDEX:
• Razão dispersão intra / separação inter
• Menor = melhor (clusters compactos e separados)

CALINSKI-HARABASZ (Variance Ratio):
• BCSS / WCSS × (n - k) / (k - 1)
• Maior = melhor

📈 MÉTRICAS EXTERNAS (com labels):

ADJUSTED RAND INDEX (ARI):
• Similaridade com clustering "verdadeiro"
• Range: -1 a 1 (1 = perfeito)

NORMALIZED MUTUAL INFORMATION (NMI):
• Informação compartilhada vs. labels
• Range: 0 a 1

⚠️ CONSIDERAÇÕES:
• Métricas internas preferem clusters esféricos
• Externas requerem labels (raramente disponíveis)
• Combinar múltiplas métricas`,
            keyPoints: [
                "Silhouette: mais usado, interpretável [-1, 1]",
                "Davies-Bouldin: menor é melhor",
                "Calinski-Harabasz: maior é melhor",
                "Métricas externas precisam de labels",
                "Não confiar em uma métrica apenas"
            ],
            example: `from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                               calinski_harabasz_score,
                               adjusted_rand_score, 
                               normalized_mutual_info_score)

# Comparar diferentes k
results = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    results.append({
        'k': k,
        'silhouette': silhouette_score(X_scaled, labels),
        'davies_bouldin': davies_bouldin_score(X_scaled, labels),
        'calinski_harabasz': calinski_harabasz_score(X_scaled, labels)
    })

import pandas as pd
df_results = pd.DataFrame(results)
print(df_results)

# Se tiver labels verdadeiros (raro em clustering real)
if y_true is not None:
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    print(f"ARI: {ari:.3f}")
    print(f"NMI: {nmi:.3f}")

# Silhouette por sample (identificar outliers)
from sklearn.metrics import silhouette_samples
sample_silhouettes = silhouette_samples(X_scaled, labels)
low_silhouette = np.where(sample_silhouettes < 0)[0]
print(f"Pontos mal clusterizados: {len(low_silhouette)}")`,
            realCase: {
                title: "Validação de Segmentação de Mercado",
                description: "Empresas de pesquisa de mercado usam múltiplas métricas para validar segmentações. Além de métricas estatísticas, consideram interpretabilidade de negócio e acionabilidade dos segmentos.",
                impact: "Métricas técnicas + validação de negócio = segmentação útil"
            }
        }
    ]
};

// Lesson 8: Decision Trees and Random Forests
STUDY_CONTENT[8] = {
    name: "Lesson 8: Decision Trees & Random Forests",
    icon: "🌲",
    weight: "Focus Areas: Tree Algorithms, Ensemble Learning, Feature Importance",
    topics: [
        {
            title: "Decision Trees",
            concept: `Árvores de decisão dividem dados recursivamente:

📊 ESTRUTURA:
• Nó raiz: todo o dataset
• Nós internos: condições de divisão
• Folhas: previsões finais
• Ramos: resultados das condições

🔀 CRITÉRIOS DE DIVISÃO:

GINI IMPURITY:
• Gini = 1 - Σpᵢ²
• 0 = puro (uma classe), 0.5 = máxima impureza
• Usado por CART

ENTROPY / INFORMATION GAIN:
• Entropy = -Σpᵢ log₂(pᵢ)
• IG = Entropy(pai) - Σ(nⱼ/n)×Entropy(filhoⱼ)
• Usado por ID3, C4.5

⚙️ HIPERPARÂMETROS:
• max_depth: profundidade máxima
• min_samples_split: mínimo para dividir
• min_samples_leaf: mínimo nas folhas
• max_features: features consideradas

💡 VANTAGENS:
• Altamente interpretáveis
• Não requer escalonamento
• Captura não-linearidades
• Feature importance built-in

⚠️ DESVANTAGENS:
• Propensas a overfitting
• Instáveis (pequenas mudanças nos dados)
• Fronteiras de decisão retilíneas`,
            keyPoints: [
                "Gini e Entropy: critérios de divisão comuns",
                "max_depth controla complexidade (regularização)",
                "Não precisa escalonar dados",
                "Muito interpretáveis (exportar regras)",
                "Overfitting é problema principal"
            ],
            example: `from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# Treinar árvore
tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
tree.fit(X_train, y_train)

# Visualizar (se features nomeadas)
plt.figure(figsize=(20, 10))
plot_tree(tree, 
          feature_names=feature_names,
          class_names=['No', 'Yes'],
          filled=True,
          rounded=True)
plt.show()

# Extrair regras de decisão
rules = export_text(tree, feature_names=feature_names)
print(rules)

# Feature importance
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': tree.feature_importances_
}).sort_values('importance', ascending=False)
print(importances)

# Comparar complexidades
for depth in [2, 4, 6, 8, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_acc = tree.score(X_train, y_train)
    test_acc = tree.score(X_test, y_test)
    n_leaves = tree.get_n_leaves()
    print(f"Depth {depth}: Train={train_acc:.3f}, Test={test_acc:.3f}, Leaves={n_leaves}")`,
            realCase: {
                title: "Diagnóstico Médico Explicável",
                description: "Hospitais usam árvores de decisão para triagem porque médicos podem seguir e explicar as decisões. Regulamentações de saúde frequentemente exigem modelos interpretáveis.",
                impact: "Confiança médica + conformidade regulatória = adoção clínica"
            }
        },
        {
            title: "Random Forests",
            concept: `Random Forest combina muitas árvores via bagging:

🌲 ALGORITMO:
1. Criar N amostras bootstrap (com reposição)
2. Treinar árvore em cada amostra
3. Em cada nó, considerar apenas √features aleatórias
4. Agregar previsões (votação ou média)

📊 COMPONENTES:

BAGGING (Bootstrap Aggregating):
• Amostrar com reposição
• ~63% dos dados em cada árvore
• Reduz variância

FEATURE RANDOMNESS:
• Cada split considera subconjunto de features
• Decorrelaciona as árvores
• Torna ensemble mais robusto

💡 VANTAGENS:
• Muito menos overfitting que árvore única
• Robusto a outliers
• Feature importance agregada
• Out-of-bag (OOB) error como validação

⚙️ HIPERPARÂMETROS:
• n_estimators: número de árvores
• max_features: features por split
• max_depth: profundidade das árvores
• min_samples_split/leaf: regularização`,
            keyPoints: [
                "Ensemble de árvores via bootstrap sampling",
                "Feature randomness decorrelaciona árvores",
                "OOB error: validação gratuita sem split",
                "Mais árvores = melhor (até certo ponto)",
                "Menos interpretável que árvore única"
            ],
            example: `from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',  # √n_features
    max_depth=10,
    min_samples_split=5,
    oob_score=True,  # usar OOB para validação
    random_state=42,
    n_jobs=-1  # paralelizar
)
rf.fit(X_train, y_train)

# OOB Score (validação gratuita!)
print(f"OOB Score: {rf.oob_score_:.3f}")
print(f"Test Score: {rf.score(X_test, y_test):.3f}")

# Feature Importance
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importances['feature'][:15], importances['importance'][:15])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances')
plt.gca().invert_yaxis()

# Encontrar n_estimators ótimo
errors = []
for n in range(10, 200, 10):
    rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)
    errors.append(1 - rf.oob_score_)

plt.plot(range(10, 200, 10), errors)
plt.xlabel('n_estimators')
plt.ylabel('OOB Error')`,
            realCase: {
                title: "Detecção de Fraude em Transações",
                description: "Bancos usam Random Forests para detectar fraude por sua robustez e capacidade de lidar com features mistas (categóricas + numéricas) sem pré-processamento extenso.",
                impact: "Alta precisão + feature importance = detecção confiável"
            }
        },
        {
            title: "Gradient Boosting",
            concept: `Boosting treina modelos sequencialmente para corrigir erros:

📊 ALGORITMO:
1. Treinar modelo inicial (previsão simples)
2. Calcular resíduos (erros)
3. Treinar próximo modelo nos resíduos
4. Adicionar ao ensemble com learning rate
5. Repetir até N modelos

💡 DIFERENÇA DE BAGGING:
• Bagging: modelos paralelos, independentes
• Boosting: modelos sequenciais, corrigem erros

📈 IMPLEMENTAÇÕES:

GRADIENT BOOSTING (sklearn):
• Implementação básica
• Relativamente lento

XGBOOST:
• Regularização L1/L2
• Tratamento de missing values
• Paralelizado, muito rápido

LIGHTGBM:
• Crescimento leaf-wise
• Ainda mais rápido
• Ótimo para grandes datasets

CATBOOST:
• Excelente para categóricas
• Menos overfitting

⚙️ HIPERPARÂMETROS CHAVE:
• n_estimators: número de árvores
• learning_rate: contribuição de cada árvore
• max_depth: profundidade (menor que RF)`,
            keyPoints: [
                "Boosting: modelos sequenciais corrigindo erros",
                "Learning rate × n_estimators: trade-off",
                "XGBoost/LightGBM: estado da arte para tabulares",
                "max_depth geralmente menor que Random Forest",
                "Early stopping previne overfitting"
            ],
            example: `from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb

# Gradient Boosting (sklearn)
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=False
)
print(f"Best iteration: {xgb_model.best_iteration}")

# LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(10)]
)

# Comparar
models = {'GradientBoosting': gb, 'XGBoost': xgb_model, 'LightGBM': lgb_model}
for name, model in models.items():
    print(f"{name}: {model.score(X_test, y_test):.3f}")`,
            realCase: {
                title: "Kaggle Competitions",
                description: "XGBoost e LightGBM dominam competições Kaggle para dados tabulares. O XGBoost foi usado por 17 dos 29 vencedores em 2015, estabelecendo-se como padrão da indústria.",
                impact: "Gradient boosting é estado da arte para dados tabulares"
            }
        }
    ]
};

// Lesson 9: Support Vector Machines
STUDY_CONTENT[9] = {
    name: "Lesson 9: Building Support Vector Machines",
    icon: "📐",
    weight: "Focus Areas: SVM Classification, Kernel Trick, SVM Regression",
    topics: [
        {
            title: "SVM for Classification",
            concept: `SVM encontra hiperplano que maximiza margem entre classes:

📊 CONCEITOS FUNDAMENTAIS:

HIPERPLANO:
• Fronteira de decisão que separa classes
• Em 2D: linha, em 3D: plano, em nD: hiperplano

MARGEM:
• Distância entre hiperplano e pontos mais próximos
• SVM maximiza esta margem
• Maior margem = melhor generalização

VETORES DE SUPORTE:
• Pontos mais próximos do hiperplano
• Definem a margem
• Únicos pontos que importam para decisão

📐 TIPOS:

HARD MARGIN:
• Dados perfeitamente separáveis
• Sem violações permitidas
• Raramente possível em dados reais

SOFT MARGIN:
• Permite algumas violações
• Parâmetro C controla trade-off
• C alto: menos violações, risco de overfit
• C baixo: mais violações, mais generalização

⚙️ HIPERPARÂMETROS:
• C: penalidade por violações
• kernel: tipo de kernel
• gamma: parâmetro para kernels RBF`,
            keyPoints: [
                "SVM maximiza margem entre classes",
                "Vetores de suporte definem a fronteira",
                "C alto: fit rígido, C baixo: mais tolerante",
                "Funciona bem em alta dimensionalidade",
                "ESCALONAMENTO É OBRIGATÓRIO"
            ],
            example: `from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# OBRIGATÓRIO: escalonar dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM Linear
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)
print(f"Linear SVM: {svm_linear.score(X_test_scaled, y_test):.3f}")
print(f"Vetores de suporte: {svm_linear.n_support_}")

# Grid Search para hiperparâmetros
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 1]
}

svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
print(f"Melhores parâmetros: {grid_search.best_params_}")
print(f"Melhor score: {grid_search.best_score_:.3f}")

# Visualizar fronteira (2D)
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X_train_scaled[:, :2], y_train, clf=svm_linear)`,
            realCase: {
                title: "Classificação de Texto e Spam",
                description: "SVMs foram estado da arte para classificação de texto antes de deep learning. Funcionam bem em alta dimensionalidade (milhares de features TF-IDF) onde outros algoritmos falham.",
                impact: "SpamAssassin e filtros de email usaram SVM por décadas"
            }
        },
        {
            title: "Kernel Trick",
            concept: `Kernels permitem SVM capturar padrões não-lineares:

📊 O PROBLEMA:
• Dados frequentemente não são linearmente separáveis
• Projetar para dimensão maior pode tornar separáveis
• Mas computação explícita é cara

💡 O TRUQUE:
• Kernels computam produto interno no espaço maior
• SEM calcular coordenadas explicitamente
• K(x, y) = φ(x) · φ(y)

📐 KERNELS COMUNS:

LINEAR:
• K(x, y) = xᵀy
• Para dados linearmente separáveis

RBF (Radial Basis Function):
• K(x, y) = exp(-γ||x-y||²)
• Projeta para dimensão INFINITA
• Mais usado, funciona para maioria dos casos

POLYNOMIAL:
• K(x, y) = (γxᵀy + r)^d
• Captura interações polinomiais
• d = grau do polinômio

SIGMOID:
• K(x, y) = tanh(γxᵀy + r)
• Similar a rede neural

⚙️ GAMMA (para RBF):
• Alto: considera apenas vizinhos muito próximos
• Baixo: considera vizinhos distantes
• Controla "alcance" do kernel`,
            keyPoints: [
                "Kernel trick evita computação explícita em alta dimensão",
                "RBF é kernel padrão, funciona para maioria dos casos",
                "gamma controla alcance de influência",
                "Polynomial captura interações de features",
                "Escolha de kernel via cross-validation"
            ],
            example: `from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Comparar kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
results = {}

for kernel in kernels:
    svm = SVC(kernel=kernel, C=1.0)
    svm.fit(X_train_scaled, y_train)
    acc = svm.score(X_test_scaled, y_test)
    results[kernel] = acc
    print(f"{kernel}: {acc:.3f}")

# Explorar gamma para RBF
gammas = [0.001, 0.01, 0.1, 1, 10]
for gamma in gammas:
    svm = SVC(kernel='rbf', gamma=gamma, C=1)
    svm.fit(X_train_scaled, y_train)
    train_acc = svm.score(X_train_scaled, y_train)
    test_acc = svm.score(X_test_scaled, y_test)
    print(f"gamma={gamma}: Train={train_acc:.3f}, Test={test_acc:.3f}")

# Visualizar efeito de gamma (dados 2D sintéticos)
from sklearn.datasets import make_circles
X_circle, y_circle = make_circles(n_samples=200, noise=0.1, factor=0.3)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, gamma in zip(axes, [0.1, 1, 10]):
    svm = SVC(kernel='rbf', gamma=gamma)
    svm.fit(X_circle, y_circle)
    ax.scatter(X_circle[:, 0], X_circle[:, 1], c=y_circle)
    ax.set_title(f'gamma = {gamma}')`,
            realCase: {
                title: "Reconhecimento de Dígitos MNIST",
                description: "Antes de CNNs dominarem, SVMs com kernel RBF alcançavam ~98% de precisão no MNIST. A capacidade de projetar para dimensões infinitas permitia separar dígitos complexos.",
                impact: "SVM foi benchmark para classificação de imagens por anos"
            }
        },
        {
            title: "SVM for Regression (SVR)",
            concept: `SVM também pode fazer regressão:

📊 DIFERENÇA CONCEITUAL:
• Classificação: maximiza margem entre classes
• Regressão: cria "tubo" de tolerância ε

📐 EPSILON-INSENSITIVE:
• Erros menores que ε são ignorados
• Apenas erros maiores que ε são penalizados
• Cria "tubo" ao redor da função

⚙️ PARÂMETROS:

EPSILON (ε):
• Largura do tubo de tolerância
• Maior ε = mais tolerância a erros pequenos
• Controla sparsidade dos vetores de suporte

C:
• Penalidade por erros fora do tubo
• Maior C = menos tolerância
• Trade-off entre fit e generalização

KERNEL:
• Mesmos kernels de classificação
• RBF mais comum para não-linear

💡 QUANDO USAR:
• Dados com outliers (robusto)
• Problemas não-lineares
• Quando sparsidade é desejada`,
            keyPoints: [
                "SVR cria tubo de tolerância ao redor da função",
                "Erros dentro de ε são ignorados",
                "C e ε controlam trade-off bias-variance",
                "Robusto a outliers (devido ao ε-tube)",
                "Kernels funcionam igual à classificação"
            ],
            example: `from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Escalonar (OBRIGATÓRIO para SVR também)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
X_test_s = scaler_X.transform(X_test)

# SVR básico
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_train_s, y_train_s)

# Previsões (desfazer escala)
y_pred_s = svr.predict(X_test_s)
y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# Grid Search para SVR
param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 0.1, 1]
}

svr = SVR(kernel='rbf')
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_s, y_train_s)

print(f"Melhores parâmetros: {grid_search.best_params_}")
print(f"Melhor RMSE: {np.sqrt(-grid_search.best_score_):.3f}")

# Comparar com Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"Linear Regression R²: {lr.score(X_test, y_test):.3f}")`,
            realCase: {
                title: "Previsão de Demanda de Energia",
                description: "Utilities usam SVR para prever demanda de energia, onde robustez a outliers (picos anormais) é crucial. O ε-tube ignora variações normais, focando em tendências.",
                impact: "Previsões robustas para planejamento de capacidade energética"
            }
        }
    ]
};

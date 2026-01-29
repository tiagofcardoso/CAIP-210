// CAIP-210 Study Content - Lessons 4-6
// Regression, Forecasting, Classification Models

// Lesson 4: Building Linear Regression Models
STUDY_CONTENT[4] = {
    name: "Lesson 4: Building Linear Regression Models",
    icon: "📈",
    weight: "Focus Areas: Linear Algebra, Regularization, Gradient Descent",
    topics: [
        {
            title: "Linear Regression Fundamentals",
            concept: `Regressão Linear modela a relação entre variáveis independentes (features) e uma variável dependente contínua (target).

📐 EQUAÇÃO BÁSICA:
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

Onde:
• ŷ = valor previsto
• β₀ = intercepto (bias)
• β₁...βₙ = coeficientes (pesos)
• x₁...xₙ = features

📊 PRESSUPOSTOS:
1. Relação linear entre X e Y
2. Independência dos erros
3. Homocedasticidade (variância constante)
4. Normalidade dos resíduos
5. Ausência de multicolinearidade

🎯 OBJETIVO:
Minimizar a Soma dos Quadrados dos Resíduos (RSS):
RSS = Σ(yᵢ - ŷᵢ)²

EQUAÇÃO NORMAL (solução fechada):
β = (XᵀX)⁻¹Xᵀy`,
            keyPoints: [
                "Regressão linear assume relação linear entre features e target",
                "Coeficientes indicam quanto Y muda para cada unidade de X",
                "Equação normal funciona bem para datasets pequenos",
                "Multicolinearidade pode distorcer coeficientes",
                "R² mede proporção da variância explicada"
            ],
            example: `import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Dados de exemplo: prever preço de casa
X = np.array([[1500, 3], [1800, 4], [2400, 4], [3000, 5], [3500, 5]])
y = np.array([300000, 350000, 450000, 550000, 600000])

# Treinar modelo
model = LinearRegression()
model.fit(X, y)

# Coeficientes
print(f"Intercepto: {model.intercept_:.0f}")
print(f"Coef. área: {model.coef_[0]:.0f}/sqft")
print(f"Coef. quartos: {model.coef_[1]:.0f}/quarto")

# Previsão
nova_casa = [[2000, 4]]
preco_previsto = model.predict(nova_casa)
print(f"Preço previsto: {preco_previsto[0]:.0f}")

# Métricas
y_pred = model.predict(X)
print(f"R²: {r2_score(y, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.0f}")`,
            realCase: {
                title: "Zillow Zestimate",
                description: "O Zillow usa modelos de regressão com centenas de features (área, localização, características, vendas próximas) para estimar valores de imóveis.",
                impact: "Estima valores de 100+ milhões de propriedades nos EUA com erro médio de ~2%"
            }
        },
        {
            title: "Regularização: Ridge e Lasso",
            concept: `Regularização previne overfitting adicionando penalidade aos coeficientes:

🔷 RIDGE REGRESSION (L2):
• Adiciona penalidade λΣβⱼ² à função de custo
• Reduz coeficientes mas NUNCA zera
• Bom quando todas as features são relevantes
• Parâmetro α controla força da regularização

🔶 LASSO REGRESSION (L1):
• Adiciona penalidade λΣ|βⱼ| à função de custo
• PODE zerar coeficientes (seleção de features automática)
• Bom quando muitas features são irrelevantes
• Produz modelos mais interpretáveis

🔷🔶 ELASTIC NET:
• Combina L1 e L2: λ₁Σ|βⱼ| + λ₂Σβⱼ²
• Parâmetro l1_ratio controla proporção
• Útil quando features são correlacionadas

📊 ESCOLHENDO α (regularização):
• α muito baixo → overfitting
• α muito alto → underfitting
• Use cross-validation para encontrar α ótimo`,
            keyPoints: [
                "Regularização adiciona viés para reduzir variância",
                "Ridge: todos os coeficientes encolhem mas não zeram",
                "Lasso: pode zerar coeficientes → seleção de features",
                "Elastic Net: combinação de L1 e L2",
                "Cross-validation essencial para escolher α"
            ],
            example: `from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
import numpy as np

# Comparar regularizações
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"{name}: R² = {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# Encontrar melhor alpha via CV
from sklearn.linear_model import RidgeCV, LassoCV

ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])
ridge_cv.fit(X, y)
print(f"Melhor alpha Ridge: {ridge_cv.alpha_}")

lasso_cv = LassoCV(alphas=[0.001, 0.01, 0.1, 1], cv=5)
lasso_cv.fit(X, y)
print(f"Melhor alpha Lasso: {lasso_cv.alpha_}")

# Lasso zera coeficientes irrelevantes
print(f"Coeficientes Lasso: {lasso_cv.coef_}")`,
            realCase: {
                title: "Seleção de Genes em Bioinformática",
                description: "Pesquisadores usam Lasso para identificar quais genes (de milhares) são relevantes para prever doenças.",
                impact: "Reduziu análises de 20.000 genes para dezenas de genes relevantes"
            }
        },
        {
            title: "Gradient Descent",
            concept: `Gradient Descent é um método iterativo para minimizar funções de custo:

🔄 ALGORITMO:
1. Inicializar pesos aleatoriamente
2. Calcular gradiente da função de custo
3. Atualizar pesos: w = w - α × ∇J(w)
4. Repetir até convergência

📊 VARIANTES:

BATCH GRADIENT DESCENT (BGD):
• Usa TODO o dataset para calcular gradiente
• Convergência estável mas lenta
• Memória: precisa de todo dataset na RAM

STOCHASTIC GRADIENT DESCENT (SGD):
• Usa UM exemplo por iteração
• Convergência rápida mas ruidosa
• Pode escapar de mínimos locais

MINI-BATCH GRADIENT DESCENT:
• Usa LOTE de exemplos (32, 64, 128...)
• Equilíbrio entre velocidade e estabilidade
• Mais usado na prática

⚙️ HIPERPARÂMETROS:
• Learning rate (α): tamanho do passo
• Epochs: passagens pelo dataset
• Batch size: exemplos por iteração`,
            keyPoints: [
                "Gradient descent encontra mínimos iterativamente",
                "Learning rate muito alto → oscilação, muito baixo → lentidão",
                "Batch GD: estável mas lento para grandes datasets",
                "SGD: rápido mas ruidoso, pode escapar mínimos locais",
                "Mini-batch: padrão para deep learning"
            ],
            example: `import numpy as np

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    
    for epoch in range(epochs):
        # Previsão: ŷ = Xw + b
        y_pred = np.dot(X, weights) + bias
        
        # Gradientes
        dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
        db = (1/n_samples) * np.sum(y_pred - y)
        
        # Atualizar pesos
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        if epoch % 100 == 0:
            mse = np.mean((y_pred - y)**2)
            print(f"Epoch {epoch}: MSE = {mse:.4f}")
    
    return weights, bias

# Usar SGDRegressor do sklearn
from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor(
    loss='squared_error',
    learning_rate='adaptive',
    eta0=0.01,
    max_iter=1000,
    early_stopping=True
)
sgd.fit(X_scaled, y)`,
            realCase: {
                title: "Treinamento de GPT",
                description: "Modelos de linguagem como GPT usam variantes de SGD (como Adam) para ajustar bilhões de parâmetros.",
                impact: "Adam optimizer é o padrão para treinar redes neurais modernas"
            }
        }
    ]
};

// Lesson 5: Building Forecasting Models
STUDY_CONTENT[5] = {
    name: "Lesson 5: Building Forecasting Models",
    icon: "📅",
    weight: "Focus Areas: Time Series, ARIMA, Multivariate Forecasting",
    topics: [
        {
            title: "Time Series Fundamentals",
            concept: `Séries temporais são sequências de dados ordenados no tempo:

📊 COMPONENTES:
• TENDÊNCIA: Direção geral ao longo do tempo (alta/baixa)
• SAZONALIDADE: Padrões que repetem em intervalos fixos
• CICLO: Flutuações de longo prazo (não fixas)
• RUÍDO: Variação aleatória residual

🔍 ANÁLISE EXPLORATÓRIA:
1. Plotar série ao longo do tempo
2. Identificar tendência visual
3. Detectar padrões sazonais
4. Verificar outliers e mudanças

📈 ESTACIONARIEDADE:
Uma série é estacionária quando:
• Média constante ao longo do tempo
• Variância constante
• Autocovariância não depende do tempo

Testes: ADF (Augmented Dickey-Fuller), KPSS

🔧 TRANSFORMAÇÕES:
• Diferenciação: yₜ' = yₜ - yₜ₋₁
• Log: estabiliza variância
• Decomposição: separa componentes`,
            keyPoints: [
                "Séries temporais têm ordem temporal significativa",
                "Componentes: tendência, sazonalidade, ciclo, ruído",
                "Estacionariedade é requisito para muitos modelos",
                "Diferenciação remove tendência",
                "Decomposição ajuda a entender estrutura"
            ],
            example: `import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Carregar dados de vendas mensais
dates = pd.date_range('2020-01-01', periods=36, freq='M')
sales = [100+i*2+10*np.sin(i/2)+np.random.randn()*5 for i in range(36)]
ts = pd.Series(sales, index=dates)

# Decomposição
decomposition = seasonal_decompose(ts, model='additive', period=12)

# Teste de estacionariedade (ADF)
result = adfuller(ts)
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')
if result[1] < 0.05:
    print("Série é ESTACIONÁRIA")
else:
    print("Série NÃO ESTACIONÁRIA - aplicar diferenciação")
    
# Diferenciação
ts_diff = ts.diff().dropna()`,
            realCase: {
                title: "Previsão de Demanda na Amazon",
                description: "Amazon usa modelos de séries temporais para prever demanda por milhões de produtos. Sazonalidade (Black Friday, Natal) e tendências são críticas.",
                impact: "Previsões precisas economizam bilhões em custos de estoque e envio"
            }
        },
        {
            title: "ARIMA Models",
            concept: `ARIMA combina três componentes para previsão:

📊 ARIMA(p, d, q):

AR (AutoRegressive) - p:
• Usa valores passados para prever
• yₜ = c + φ₁yₜ₋₁ + φ₂yₜ₋₂ + ... + εₜ
• p = número de lags

I (Integrated) - d:
• Ordem de diferenciação
• d=1: uma diferenciação
• Torna série estacionária

MA (Moving Average) - q:
• Usa erros passados
• yₜ = c + εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂
• q = número de lags de erro

📈 SARIMA para sazonalidade:
SARIMA(p,d,q)(P,D,Q,s)
• s = período sazonal (12 para mensal)
• P, D, Q = componentes sazonais

🔍 ESCOLHENDO PARÂMETROS:
• ACF plot → determina q
• PACF plot → determina p
• Auto ARIMA: busca automática`,
            keyPoints: [
                "AR: previsão baseada em valores passados",
                "I: diferenciação para estacionariedade",
                "MA: previsão baseada em erros passados",
                "SARIMA adiciona componentes sazonais",
                "ACF e PACF ajudam a identificar p e q"
            ],
            example: `from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm

# ARIMA manual
model = ARIMA(ts, order=(1, 1, 1))
fitted = model.fit()
print(fitted.summary())

# Previsão
forecast = fitted.forecast(steps=6)
print("Próximos 6 meses:", forecast.values)

# Auto ARIMA - encontra melhor modelo automaticamente
auto_model = pm.auto_arima(
    ts,
    seasonal=True, m=12,
    stepwise=True,
    suppress_warnings=True,
    trace=True
)
print(f"Melhor modelo: {auto_model.order} x {auto_model.seasonal_order}")

# SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12))
sarima_fit = sarima.fit(disp=False)
sarima_forecast = sarima_fit.forecast(steps=12)`,
            realCase: {
                title: "Previsão de Vendas no Walmart",
                description: "Walmart usa modelos SARIMA e variantes para prever vendas semanais por loja.",
                impact: "Top solutions combinaram ARIMA com gradient boosting para capturar padrões sazonais"
            }
        },
        {
            title: "Multivariate Time Series",
            concept: `Quando múltiplas séries temporais se influenciam mutuamente:

📊 VAR (Vector AutoRegression):
• Estende AR para múltiplas variáveis
• Cada variável depende de seus lags E lags das outras
• Captura interdependências

Exemplo: PIB e Taxa de Juros se influenciam mutuamente

📈 EXOGENOUS VARIABLES (SARIMAX):
• Variáveis externas que afetam a série
• Não são previstas, são fornecidas
• Ex: temperatura → vendas de sorvete

🔍 COINTEGRAÇÃO:
• Séries não estacionárias que se movem juntas
• Relação de longo prazo estável
• Importante para economia/finanças

⚙️ ABORDAGEM PRÁTICA:
1. Testar estacionariedade de cada série
2. Verificar causalidade de Granger
3. Selecionar ordem do VAR (AIC/BIC)
4. Validar com forecasting out-of-sample`,
            keyPoints: [
                "VAR modela múltiplas séries interdependentes",
                "Cada série depende de lags próprios e das outras",
                "SARIMAX adiciona variáveis exógenas ao ARIMA",
                "Granger causality testa se uma série prevê outra",
                "Cointegração indica relação de longo prazo"
            ],
            example: `from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

# Dados multivariados
data = pd.DataFrame({
    'sales': sales_series,
    'marketing': marketing_series,
    'temperature': temp_series
})

# Teste de causalidade de Granger
granger_test = grangercausalitytests(
    data[['sales', 'marketing']], 
    maxlag=4, 
    verbose=True
)

# Modelo VAR
model = VAR(data[['sales', 'marketing']])

# Selecionar ordem ótima
for i in range(1, 11):
    result = model.fit(i)
    print(f'Lag {i}: AIC={result.aic:.2f}, BIC={result.bic:.2f}')

# Ajustar VAR
var_result = model.fit(maxlags=4, ic='aic')

# SARIMAX com variável exógena
sarimax = SARIMAX(
    sales_series,
    exog=marketing_series,
    order=(1,1,1),
    seasonal_order=(1,1,1,12)
)
sarimax_fit = sarimax.fit()`,
            realCase: {
                title: "Previsão Econômica do Federal Reserve",
                description: "O Federal Reserve usa modelos VAR para prever PIB, inflação e desemprego simultaneamente.",
                impact: "Modelos VAR são fundamentais para política macroeconômica global"
            }
        }
    ]
};

// Lesson 6: Classification Models
STUDY_CONTENT[6] = {
    name: "Lesson 6: Classification with Logistic Regression & k-NN",
    icon: "🎯",
    weight: "Focus Areas: Logistic Regression, k-NN, Multi-class, Evaluation",
    topics: [
        {
            title: "Logistic Regression",
            concept: `Apesar do nome, Logistic Regression é para CLASSIFICAÇÃO:

📊 FUNÇÃO SIGMOID:
σ(z) = 1 / (1 + e⁻ᶻ)
• Converte qualquer valor para [0, 1]
• Interpretado como probabilidade

📐 MODELO:
P(y=1|x) = σ(β₀ + β₁x₁ + β₂x₂ + ...)

🎯 DECISÃO:
• Se P(y=1) > threshold → classe 1
• Threshold padrão = 0.5
• Pode ajustar para precision/recall

📈 TREINAMENTO:
• Usa Maximum Likelihood Estimation
• Otimiza Log-Loss (Cross-Entropy)
• Log-Loss = -Σ[y·log(p) + (1-y)·log(1-p)]

💡 INTERPRETAÇÃO:
• Coeficientes = log odds ratio
• exp(β) = quanto odds multiplicam para +1 unidade
• Altamente interpretável!

⚠️ LIMITAÇÕES:
• Assume relação linear no log-odds
• Não captura interações automaticamente
• Sensível a outliers`,
            keyPoints: [
                "Sigmoid transforma output em probabilidade [0,1]",
                "Otimiza log-loss, não MSE",
                "Coeficientes interpretáveis como log odds",
                "Threshold ajustável baseado no problema",
                "Base para redes neurais (neurônio sigmoid)"
            ],
            example: `from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# Dados de exemplo: prever se cliente compra
X = np.array([[25, 30000], [35, 50000], [45, 80000], 
              [20, 20000], [50, 100000], [30, 40000]])
y = np.array([0, 0, 1, 0, 1, 0])

# Treinar
model = LogisticRegression()
model.fit(X, y)

# Probabilidades
probs = model.predict_proba(X)
print("Probabilidades de compra:", probs[:, 1])

# Coeficientes (interpretação)
print(f"Coef. idade: {model.coef_[0][0]:.4f}")
print(f"Coef. renda: {model.coef_[0][1]:.8f}")
print(f"Odds ratio renda: {np.exp(model.coef_[0][1]*10000):.2f}x por +10k")

# Ajustar threshold para mais recall
threshold = 0.3
y_pred_custom = (probs[:, 1] >= threshold).astype(int)

# Métricas
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y, probs[:, 1])`,
            realCase: {
                title: "Credit Scoring em Bancos",
                description: "Bancos usam logistic regression para credit scoring por sua interpretabilidade. Reguladores exigem decisões de crédito explicáveis.",
                impact: "Decisões de crédito transparentes e auditáveis, exigidas por regulamentação"
            }
        },
        {
            title: "k-Nearest Neighbors (k-NN)",
            concept: `k-NN classifica baseado nos vizinhos mais próximos:

📊 ALGORITMO:
1. Calcular distância até todos os pontos de treino
2. Selecionar k vizinhos mais próximos
3. Votar pela classe mais frequente

📐 MÉTRICAS DE DISTÂNCIA:
• Euclidiana: √Σ(xᵢ - yᵢ)²
• Manhattan: Σ|xᵢ - yᵢ|
• Minkowski: generalização

⚙️ ESCOLHENDO k:
• k pequeno: mais sensível a ruído
• k grande: mais suave mas perde detalhes
• Usar CV para encontrar k ótimo
• k ímpar evita empates

⚠️ CARACTERÍSTICAS:
• Não-paramétrico (sem modelo fixo)
• "Lazy learner": não treina, só memoriza
• Sensível à escala → NORMALIZAR!
• Lento para grandes datasets
• Maldição da dimensionalidade

💡 QUANDO USAR:
• Datasets pequenos/médios
• Fronteiras de decisão não lineares
• Como baseline simples`,
            keyPoints: [
                "Classifica pelo voto dos k vizinhos mais próximos",
                "ESCALONAMENTO É OBRIGATÓRIO",
                "k pequeno = overfit, k grande = underfit",
                "Lazy learner: lento na previsão",
                "Sofre com maldição da dimensionalidade"
            ],
            example: `from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# IMPORTANTE: Escalonar features!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encontrar melhor k
k_range = range(1, 21)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Melhor k
best_k = k_range[np.argmax(cv_scores)]
print(f"Melhor k: {best_k}")

# Modelo final
knn = KNeighborsClassifier(
    n_neighbors=best_k,
    weights='distance',
    metric='euclidean'
)
knn.fit(X_scaled, y)

# Previsão
novo_cliente = scaler.transform([[40, 60000]])
print(f"Classe: {knn.predict(novo_cliente)}")`,
            realCase: {
                title: "Sistemas de Recomendação Colaborativa",
                description: "Netflix usou variantes de k-NN em seu sistema de recomendação original. O Netflix Prize foi vencido usando ensemble incluindo k-NN.",
                impact: "k-NN colaborativo foi fundação para sistemas de recomendação modernos"
            }
        },
        {
            title: "Multi-class Classification",
            concept: `Quando há 3 ou mais classes para prever:

📊 ESTRATÉGIAS:

ONE-VS-REST (OvR):
• Treina N classificadores binários
• Cada um: "classe i vs todas as outras"
• Prevê classe com maior confiança
• Mais comum, eficiente

ONE-VS-ONE (OvO):
• Treina N(N-1)/2 classificadores
• Cada par de classes
• Votação para classe final
• Melhor para SVMs

MULTINOMIAL (Softmax):
• Um modelo com N outputs
• Softmax: eᶻⁱ / Σeᶻʲ
• Probabilidades somam 1
• Usado em redes neurais

📈 MÉTRICAS MULTI-CLASS:

Macro Average:
• Média simples por classe
• Trata classes igualmente

Weighted Average:
• Média ponderada por suporte
• Considera desbalanceamento

Micro Average:
• Agregado global
• Igual a accuracy global`,
            keyPoints: [
                "OvR: N classificadores, um por classe",
                "OvO: N(N-1)/2 classificadores, cada par",
                "Softmax: probabilidades multi-classe",
                "Macro: média igual por classe",
                "Weighted: considera proporção de classes"
            ],
            example: `from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# Dados multi-classe (3 classes)
X_multi = np.random.randn(150, 4)
y_multi = np.repeat([0, 1, 2], 50)

# Opção 1: Multinomial nativo
lr_multi = LogisticRegression(multi_class='multinomial', max_iter=1000)
lr_multi.fit(X_multi, y_multi)
print("Probabilidades softmax:", lr_multi.predict_proba(X_multi[:1]))

# Opção 2: One-vs-Rest explícito
ovr = OneVsRestClassifier(LogisticRegression())
ovr.fit(X_multi, y_multi)

# Opção 3: One-vs-One explícito
ovo = OneVsOneClassifier(LogisticRegression())
ovo.fit(X_multi, y_multi)

# Métricas multi-classe
y_pred = lr_multi.predict(X_multi)
print(classification_report(y_multi, y_pred, 
                            target_names=['Classe 0', 'Classe 1', 'Classe 2']))

# Confusion matrix multi-classe
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_multi, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['C0', 'C1', 'C2']).plot()`,
            realCase: {
                title: "Classificação de Imagens ImageNet",
                description: "ImageNet tem 1000 classes de objetos. Modelos como ResNet usam softmax final com 1000 outputs.",
                impact: "Softmax multi-classe é padrão para classificação de imagens"
            }
        },
        {
            title: "Classification Metrics Deep Dive",
            concept: `Métricas detalhadas para avaliar classificadores:

📊 CONFUSION MATRIX:
                Predicted
                Neg    Pos
Actual  Neg     TN     FP    ← Specificity = TN/(TN+FP)
        Pos     FN     TP    ← Recall = TP/(TP+FN)
                ↓      ↓
             NPV   Precision

📈 CURVAS:

ROC CURVE:
• Eixo X: False Positive Rate (1 - Specificity)
• Eixo Y: True Positive Rate (Recall)
• AUC: área sob a curva
• AUC = 0.5: random, AUC = 1: perfeito

PR CURVE (Precision-Recall):
• Eixo X: Recall
• Eixo Y: Precision
• Melhor para dados desbalanceados
• AP: Average Precision

🎯 QUANDO USAR CADA:
• Accuracy: dados balanceados
• Precision: custo alto de FP (spam filter)
• Recall: custo alto de FN (diagnóstico médico)
• F1: equilíbrio precision-recall
• AUC-ROC: comparar modelos geralmente
• AUC-PR: dados muito desbalanceados`,
            keyPoints: [
                "ROC-AUC bom para comparar modelos geralmente",
                "PR-AUC melhor para dados desbalanceados",
                "Threshold afeta precision-recall tradeoff",
                "F1 é média harmônica de precision e recall",
                "Escolha de métrica depende do custo de erros"
            ],
            example: `from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                               average_precision_score, f1_score)
import matplotlib.pyplot as plt

# Obter probabilidades
y_proba = model.predict_proba(X_test)[:, 1]

# ROC Curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# PR Curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'PR (AP = {ap:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

# Encontrar threshold ótimo para F1
f1_scores = []
for thresh in thresholds_pr:
    y_pred_temp = (y_proba >= thresh).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_temp))
    
best_thresh = thresholds_pr[np.argmax(f1_scores)]
print(f"Threshold ótimo para F1: {best_thresh:.3f}")`,
            realCase: {
                title: "Detecção de Fraude em Cartões",
                description: "Com apenas 0.1% de transações fraudulentas, accuracy é inútil. Bancos otimizam para recall alto enquanto mantêm precision aceitável.",
                impact: "AUC-PR e recall são métricas principais, não accuracy"
            }
        }
    ]
};

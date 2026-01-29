// CAIP-210 Study Content - Lessons 10-12
// Neural Networks and MLOps

// Lesson 10: Building Artificial Neural Networks
STUDY_CONTENT[10] = {
    name: "Lesson 10: Artificial Neural Networks",
    icon: "🧠",
    weight: "Focus Areas: MLP, CNN, RNN, Deep Learning",
    topics: [
        {
            title: "Multi-Layer Perceptrons (MLP)",
            concept: `MLPs são redes neurais feedforward com camadas densamente conectadas:

📊 ARQUITETURA:
• Camada de entrada: recebe features
• Camadas ocultas: transformações não-lineares
• Camada de saída: previsões finais

🔗 CONEXÕES:
• Cada neurônio conectado a todos do layer seguinte
• Pesos (weights) multiplicam inputs
• Bias adicionado em cada neurônio

⚡ ATIVAÇÕES:

ReLU (Rectified Linear Unit):
• f(x) = max(0, x)
• Mais usada em camadas ocultas
• Resolve vanishing gradient

SIGMOID:
• f(x) = 1 / (1 + e⁻ˣ)
• Output [0, 1]
• Classificação binária

SOFTMAX:
• Converte para probabilidades (soma = 1)
• Classificação multi-classe

🔄 TREINAMENTO:
• Forward pass: calcular output
• Loss: comparar com target
• Backpropagation: calcular gradientes
• Update: ajustar pesos

📐 REGULARIZAÇÃO:
• Dropout: "desliga" neurônios aleatoriamente
• L1/L2: penalidade nos pesos
• Early stopping: parar antes de overfit`,
            keyPoints: [
                "MLP = camadas densas (fully connected)",
                "ReLU é ativação padrão para camadas ocultas",
                "Backpropagation calcula gradientes eficientemente",
                "Dropout é regularização mais efetiva em NNs",
                "Mais camadas/neurônios = mais capacidade (e overfit)"
            ],
            example: `import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Arquitetura MLP
model = Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # classificação binária
])

# Compilar
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Resumo da arquitetura
model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Treinar
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Plotar learning curves
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()`,
            realCase: {
                title: "Sistemas de Recomendação do Spotify",
                description: "Spotify usa MLPs em seu sistema de recomendação, combinando features de usuários e músicas para prever preferências. Deep learning captura padrões complexos de gosto musical.",
                impact: "Discover Weekly usa embeddings de músicas treinados via MLPs"
            }
        },
        {
            title: "Convolutional Neural Networks (CNN)",
            concept: `CNNs são especializadas em processar dados com estrutura espacial (imagens):

📊 CAMADAS PRINCIPAIS:

CONVOLUTIONAL:
• Aplica filtros (kernels) à imagem
• Detecta features locais (bordas, texturas)
• Parâmetros compartilhados (eficiente)
• Output: feature map

POOLING:
• Reduz dimensionalidade espacial
• Max pooling: pega valor máximo
• Average pooling: pega média
• Torna representação mais robusta

FLATTEN:
• Achata feature maps para vetor
• Conecta a camadas densas

📐 HIPERPARÂMETROS:
• Número de filtros: quantas features detectar
• Kernel size: tamanho do filtro (3x3, 5x5)
• Stride: passo entre aplicações
• Padding: preservar dimensões ('same')

🏗️ ARQUITETURAS FAMOSAS:
• LeNet: pioneira para dígitos
• AlexNet: breakthrough no ImageNet
• VGG: camadas 3x3 profundas
• ResNet: conexões residuais
• EfficientNet: estado da arte`,
            keyPoints: [
                "Convoluções detectam features espaciais locais",
                "Pooling reduz dimensionalidade e aumenta robustez",
                "Filtros iniciais: bordas, texturas; finais: objetos",
                "Transfer learning: usar redes pré-treinadas",
                "Data augmentation crucial para evitar overfit"
            ],
            example: `from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, 
                                       Dense, Dropout, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Arquitetura CNN
model = Sequential([
    # Bloco 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    # Bloco 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    # Bloco 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    # Classificador
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation (MUITO importante para imagens)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Transfer Learning com VGG16
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # congelar pesos

x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)
model_transfer = Model(inputs=base_model.input, outputs=output)`,
            realCase: {
                title: "Diagnóstico de Câncer de Pele",
                description: "Pesquisadores de Stanford treinaram CNNs para detectar câncer de pele com precisão comparável a dermatologistas. O modelo analisa imagens de lesões e classifica como benignas ou malignas.",
                impact: "CNNs atingiram nível de especialista em diagnóstico dermatológico"
            }
        },
        {
            title: "Recurrent Neural Networks (RNN)",
            concept: `RNNs processam sequências mantendo memória de passos anteriores:

📊 PROBLEMA COM SEQUÊNCIAS:
• Texto, áudio, séries temporais
• Ordem dos dados importa
• Contexto de passos anteriores é relevante

🔄 ARQUITETURA RNN:
• Estado oculto h_t mantém memória
• h_t = f(W_x × x_t + W_h × h_{t-1} + b)
• Mesmo peso W compartilhado entre passos
• Output pode ser a cada passo ou no final

⚠️ PROBLEMA:
• Vanishing/exploding gradients
• Difícil aprender dependências longas

🔷 LSTM (Long Short-Term Memory):
• Célula de memória para longo prazo
• Portões controlam fluxo de informação:
  - Forget gate: o que esquecer
  - Input gate: o que adicionar
  - Output gate: o que output

🔶 GRU (Gated Recurrent Unit):
• Versão simplificada do LSTM
• Menos parâmetros
• Performance similar

📐 APLICAÇÕES:
• NLP: tradução, sentiment analysis
• Séries temporais: previsão
• Geração de texto/música`,
            keyPoints: [
                "RNNs mantêm estado oculto entre passos",
                "LSTM resolve vanishing gradient com células de memória",
                "GRU é alternativa mais simples ao LSTM",
                "Bidirectional: processa sequência em ambas direções",
                "Transformers estão substituindo RNNs em NLP"
            ],
            example: `from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Bidirectional

# Para NLP: texto → classes
model_nlp = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Para Séries Temporais
# Formato input: (samples, timesteps, features)
model_ts = Sequential([
    LSTM(50, return_sequences=True, input_shape=(n_timesteps, n_features)),
    LSTM(50),
    Dense(1)  # prever próximo valor
])

model_ts.compile(optimizer='adam', loss='mse')

# Preparar dados para LSTM (windowing)
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps)])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

n_steps = 10
X, y = create_sequences(time_series_data, n_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

# GRU alternativa
model_gru = Sequential([
    GRU(50, input_shape=(n_timesteps, n_features)),
    Dense(1)
])`,
            realCase: {
                title: "Google Translate Neural Machine Translation",
                description: "Google usou arquiteturas sequence-to-sequence com LSTMs para tradução automática. O sistema codifica a frase fonte em um vetor e decodifica para o idioma alvo.",
                impact: "Reduziu erros de tradução em 60% comparado a sistemas baseados em regras"
            }
        }
    ]
};

// Lesson 11: Operationalizing ML Models
STUDY_CONTENT[11] = {
    name: "Lesson 11: Operationalizing ML Models",
    icon: "🚀",
    weight: "Focus Areas: Deployment, MLOps, Model Integration",
    topics: [
        {
            title: "Model Deployment",
            concept: `Deployar modelo = torná-lo disponível para produção:

📊 FORMAS DE DEPLOY:

BATCH PREDICTION:
• Processa dados em lotes periódicos
• Ex: previsões noturnas
• Mais simples, menos latência crítica

REAL-TIME/ONLINE:
• Previsões instantâneas via API
• Latência baixa é crucial
• Requer infraestrutura robusta

EDGE DEPLOYMENT:
• Modelo roda no dispositivo
• Sem conexão com servidor
• Ex: apps mobile, IoT

📐 FORMATOS DE MODELO:

PICKLE/JOBLIB:
• Serialização Python nativa
• Fácil mas dependente de versão

ONNX:
• Formato interoperável
• Funciona entre frameworks

TENSORFLOW SAVEDMODEL:
• Formato TensorFlow nativo
• Inclui grafo completo

TORCHSCRIPT:
• Formato PyTorch otimizado
• Para produção

🛠️ FERRAMENTAS:
• Flask/FastAPI: APIs simples
• Docker: containerização
• Kubernetes: orquestração
• MLflow: lifecycle management`,
            keyPoints: [
                "Batch: offline, latência não crítica",
                "Real-time: API, latência baixa",
                "Edge: no dispositivo, sem servidor",
                "Docker containeriza modelo + dependências",
                "APIs REST são padrão para servir modelos"
            ],
            example: `# Salvar modelo
import joblib
joblib.dump(model, 'model.pkl')

# FastAPI para servir modelo
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('model.pkl')

class PredictionRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict([request.features])
    probability = model.predict_proba([request.features])
    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0].max())
    }

# Rodar: uvicorn app:app --host 0.0.0.0 --port 8000

# Dockerfile
"""
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# Testar API
import requests
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [1.5, 2.3, 4.1, 0.8]}
)
print(response.json())`,
            realCase: {
                title: "Uber ML Platform Michelangelo",
                description: "Uber construiu a plataforma Michelangelo para gerenciar todo o lifecycle de ML: feature engineering, treinamento, deploy e monitoramento. Suporta milhares de modelos em produção.",
                impact: "Democratizou ML na Uber, permitindo que não-especialistas deployem modelos"
            }
        },
        {
            title: "MLOps Fundamentals",
            concept: `MLOps = DevOps aplicado a Machine Learning:

📊 PILARES:

CI/CD PARA ML:
• CI: testar código, dados e modelos
• CD: deploy automatizado
• CT (Continuous Training): retreinar modelos

VERSIONAMENTO:
• Código: Git
• Dados: DVC, Delta Lake
• Modelos: MLflow, Weights & Biases
• Experimentos: logs de hiperparâmetros

MONITORAMENTO:
• Performance do modelo (accuracy, latency)
• Data drift: mudança na distribuição dos dados
• Concept drift: mudança na relação input-output

🔄 PIPELINE TÍPICO:
1. Feature Store → features consistentes
2. Training Pipeline → treinar modelo
3. Model Registry → armazenar versões
4. Serving → API para previsões
5. Monitoring → observar performance

📐 FERRAMENTAS:
• MLflow: experiment tracking, registry
• Kubeflow: pipelines em Kubernetes
• Airflow: orquestração de workflows
• Great Expectations: validação de dados
• Evidently: monitoramento de drift`,
            keyPoints: [
                "MLOps = automação do lifecycle de ML",
                "Versionar código, dados E modelos",
                "Continuous Training: retreinar periodicamente",
                "Monitorar drift: dados e performance",
                "Feature Stores garantem consistência"
            ],
            example: `# MLflow para tracking de experimentos
import mlflow
import mlflow.sklearn

# Iniciar experimento
mlflow.set_experiment("churn_prediction")

with mlflow.start_run():
    # Logar parâmetros
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Treinar modelo
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Logar métricas
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Logar modelo
    mlflow.sklearn.log_model(model, "model")
    
    # Registrar no Model Registry
    mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/model",
        "ChurnModel"
    )

# Carregar modelo do registry
model = mlflow.pyfunc.load_model("models:/ChurnModel/Production")

# DVC para versionamento de dados
"""
dvc init
dvc add data/training_data.csv
git add data/training_data.csv.dvc
git commit -m "Add training data"
dvc push
"""

# Great Expectations para validação
import great_expectations as ge

df_ge = ge.from_pandas(df)
df_ge.expect_column_values_to_be_between("age", min_value=0, max_value=120)
df_ge.expect_column_values_to_not_be_null("customer_id")`,
            realCase: {
                title: "Netflix Model Lifecycle",
                description: "Netflix retreina modelos de recomendação diariamente com novos dados de visualização. MLOps automatiza todo o processo: coleta de dados, treinamento, validação, e deploy gradual (canary).",
                impact: "Modelos sempre atualizados com comportamento recente dos usuários"
            }
        },
        {
            title: "Model Monitoring & Maintenance",
            concept: `Modelos em produção requerem monitoramento contínuo:

📊 O QUE MONITORAR:

PERFORMANCE:
• Accuracy, F1, RMSE ao longo do tempo
• Comparar com baseline/threshold
• Alertar quando degradar

DATA DRIFT:
• Distribuição dos inputs mudou?
• Estatísticas: média, variância, distribuição
• Testes: KS, Chi-squared, PSI

CONCEPT DRIFT:
• Relação input-output mudou?
• Mesmo input → outputs diferentes?
• Harder to detect

OPERATIONAL:
• Latência de previsão
• Throughput (requests/segundo)
• Erros e exceções

🔄 ESTRATÉGIAS DE RETREINAMENTO:

SCHEDULED:
• Retreinar periodicamente (daily, weekly)
• Simples mas pode ser desnecessário

TRIGGERED:
• Retreinar quando drift detectado
• Mais eficiente
• Requer bom monitoramento

ONLINE LEARNING:
• Atualizar modelo continuamente
• Para dados em streaming
• Mais complexo

⚠️ ALERT FATIGUE:
• Balancear sensibilidade de alertas
• False positives cansam o time
• Priorizar alertas críticos`,
            keyPoints: [
                "Monitorar performance, data drift, concept drift",
                "Data drift: distribuição de inputs mudou",
                "Concept drift: relação input-output mudou",
                "Retreinamento scheduled ou triggered",
                "Alertas bem calibrados evitam fatigue"
            ],
            example: `# Monitoramento com Evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# Comparar dados de referência vs. atual
reference_data = df_train
current_data = df_production

# Relatório de Data Drift
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=reference_data, current_data=current_data)
drift_report.save_html("drift_report.html")

# Verificar drift programaticamente
drift_result = drift_report.as_dict()
if drift_result['metrics'][0]['result']['dataset_drift']:
    print("ALERTA: Data drift detectado!")
    # Trigger retreinamento
    
# Performance ao longo do tempo
from datetime import datetime, timedelta
import pandas as pd

# Simular log de previsões
predictions_log = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
    'prediction': np.random.binomial(1, 0.7, 1000),
    'actual': np.random.binomial(1, 0.7, 1000)
})

# Calcular accuracy por dia
predictions_log['date'] = predictions_log['timestamp'].dt.date
daily_accuracy = predictions_log.groupby('date').apply(
    lambda x: (x['prediction'] == x['actual']).mean()
)

# Alertar se accuracy cair abaixo de threshold
threshold = 0.65
if daily_accuracy.iloc[-1] < threshold:
    print(f"ALERTA: Accuracy caiu para {daily_accuracy.iloc[-1]:.2%}")

# PSI (Population Stability Index) para drift
def calculate_psi(expected, actual, bins=10):
    expected_percents = np.histogram(expected, bins)[0] / len(expected)
    actual_percents = np.histogram(actual, bins)[0] / len(actual)
    psi_values = (expected_percents - actual_percents) * np.log(expected_percents / actual_percents)
    return np.sum(psi_values)

psi = calculate_psi(reference_data['feature'], current_data['feature'])
if psi > 0.25:
    print(f"ALERTA: PSI = {psi:.3f} indica drift significativo")`,
            realCase: {
                title: "Monitoramento de Modelos de Fraude",
                description: "Bancos monitoram modelos de fraude continuamente porque fraudadores adaptam táticas. Quando taxa de detecção cai ou falsos positivos sobem, o modelo é retreinado com novos padrões.",
                impact: "Adaptação rápida a novas técnicas de fraude protege milhões em transações"
            }
        }
    ]
};

// Lesson 12: Maintaining ML Operations
STUDY_CONTENT[12] = {
    name: "Lesson 12: Maintaining ML Operations",
    icon: "🔧",
    weight: "Focus Areas: Security, Production Maintenance, Best Practices",
    topics: [
        {
            title: "Securing ML Pipelines",
            concept: `Segurança é crítica em sistemas de ML:

🔒 ÁREAS DE RISCO:

DADOS:
• Dados sensíveis (PII) precisam proteção
• Anonimização, pseudonimização
• Controle de acesso granular
• Criptografia em repouso e trânsito

MODELOS:
• Modelos são IP (propriedade intelectual)
• Ataques de extração de modelo
• Backdoors em supply chain

PREVISÕES:
• Outputs podem revelar dados de treino
• Membership inference attacks
• Adversarial examples

📐 PRÁTICAS DE SEGURANÇA:

AUTHENTICATION:
• Verificar identidade de usuários/sistemas
• API keys, OAuth, JWT

AUTHORIZATION:
• Controlar o que cada identidade pode fazer
• Principle of least privilege

AUDIT LOGGING:
• Registrar acessos e operações
• Detectar uso indevido
• Compliance

PRIVACY BY DESIGN:
• Considerar privacidade desde o início
• Data minimization
• Differential privacy`,
            keyPoints: [
                "Proteger dados, modelos E previsões",
                "PII requer tratamento especial (GDPR, LGPD)",
                "Principle of least privilege para acessos",
                "Audit logs para compliance e detecção",
                "Adversarial attacks: modelos podem ser enganados"
            ],
            example: `# Anonimização de dados
import hashlib
from faker import Faker

def anonymize_pii(df):
    fake = Faker()
    
    # Hash de IDs (irreversível)
    df['customer_id_hash'] = df['customer_id'].apply(
        lambda x: hashlib.sha256(str(x).encode()).hexdigest()
    )
    
    # Remover colunas originais
    df = df.drop(['customer_id', 'name', 'email', 'phone'], axis=1)
    
    # Generalização de idade (k-anonimidade)
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                             labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    df = df.drop('age', axis=1)
    
    return df

# Controle de acesso em API
from functools import wraps
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key not in VALID_API_KEYS:
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated

# Audit logging
import logging
from datetime import datetime

audit_logger = logging.getLogger('audit')
audit_logger.setLevel(logging.INFO)

def log_prediction(user_id, input_data, prediction):
    audit_logger.info({
        'timestamp': datetime.utcnow().isoformat(),
        'user_id': user_id,
        'action': 'prediction',
        'input_hash': hashlib.md5(str(input_data).encode()).hexdigest(),
        'prediction': prediction
    })`,
            realCase: {
                title: "GDPR e Direito ao Esquecimento",
                description: "Sob GDPR, usuários podem solicitar exclusão de seus dados. Se um modelo foi treinado com esses dados, pode ser necessário retreinar sem eles (machine unlearning) para compliance.",
                impact: "Empresas precisam rastrear proveniência de dados em modelos"
            }
        },
        {
            title: "Models in Production",
            concept: `Manter modelos em produção requer práticas específicas:

📊 DEPLOYMENT STRATEGIES:

CANARY RELEASE:
• Deploy para pequena % de tráfego
• Monitorar métricas
• Rollout gradual se ok

BLUE-GREEN:
• Dois ambientes: atual (blue) e novo (green)
• Switch instantâneo
• Rollback fácil

A/B TESTING:
• Comparar modelos lado a lado
• Dividir tráfego entre versões
• Estatisticamente significante

SHADOW MODE:
• Novo modelo roda em paralelo
• Não afeta usuários
• Compara outputs

🔄 ROLLBACK:
• Sempre ter versão anterior pronta
• Automatizar rollback em caso de falha
• Definir critérios de rollback

📐 SCALING:

HORIZONTAL:
• Mais instâncias do modelo
• Load balancer distribui
• Kubernetes autoscaling

VERTICAL:
• Máquina mais potente
• Limitado por hardware

CACHING:
• Cache de previsões frequentes
• Reduz latência e custo`,
            keyPoints: [
                "Canary release: deploy gradual com monitoramento",
                "A/B testing: comparar modelos estatisticamente",
                "Shadow mode: testar sem afetar produção",
                "Rollback deve ser automatizado e rápido",
                "Autoscaling adapta capacidade à demanda"
            ],
            example: `# Kubernetes deployment com canary
"""
# Deployment principal (90% do tráfego)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-v1
spec:
  replicas: 9
  selector:
    matchLabels:
      app: ml-model
      version: v1
  template:
    metadata:
      labels:
        app: ml-model
        version: v1
    spec:
      containers:
      - name: model
        image: ml-model:v1
---
# Canary (10% do tráfego)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-v2
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ml-model
      version: v2
  template:
    metadata:
      labels:
        app: ml-model
        version: v2
    spec:
      containers:
      - name: model
        image: ml-model:v2
"""

# A/B Testing em Python
import numpy as np

def ab_test_model(request, model_a, model_b, test_ratio=0.1):
    """Roteia requests entre modelos para A/B test"""
    if np.random.random() < test_ratio:
        # Grupo B (novo modelo)
        prediction = model_b.predict(request.features)
        log_ab_test('B', request.id, prediction)
    else:
        # Grupo A (modelo atual)
        prediction = model_a.predict(request.features)
        log_ab_test('A', request.id, prediction)
    return prediction

# Analisar resultados de A/B test
from scipy import stats

def analyze_ab_results(group_a_conversions, group_a_total,
                       group_b_conversions, group_b_total):
    rate_a = group_a_conversions / group_a_total
    rate_b = group_b_conversions / group_b_total
    
    # Chi-squared test
    contingency = [[group_a_conversions, group_a_total - group_a_conversions],
                   [group_b_conversions, group_b_total - group_b_conversions]]
    chi2, p_value = stats.chi2_contingency(contingency)[:2]
    
    print(f"Rate A: {rate_a:.2%}, Rate B: {rate_b:.2%}")
    print(f"Lift: {(rate_b - rate_a) / rate_a:.2%}")
    print(f"p-value: {p_value:.4f}")
    print(f"Significativo: {p_value < 0.05}")`,
            realCase: {
                title: "Facebook Continuous Deployment",
                description: "Facebook deploya mudanças milhares de vezes por dia usando canary releases. Modelos de ML seguem o mesmo processo: deploy para 1% → 10% → 50% → 100% com monitoramento automático.",
                impact: "Detecção rápida de problemas antes de afetar todos os usuários"
            }
        },
        {
            title: "Best Practices Summary",
            concept: `Resumo das melhores práticas para ML em produção:

📊 DESENVOLVIMENTO:

✅ Versionar código, dados E modelos
✅ Reprodutibilidade: seeds, versões fixas
✅ Documentar decisões e razões
✅ Code review para ML code
✅ Testes unitários + integração

📐 TRAINING:

✅ Never train on test data
✅ Stratified splits para desbalanceados
✅ Cross-validation para robustez
✅ Early stopping para evitar overfit
✅ Logar todos os experimentos

🚀 DEPLOYMENT:

✅ Containerizar (Docker)
✅ Canary/gradual releases
✅ Feature flags para rollback
✅ Health checks automatizados
✅ Documentar API (OpenAPI)

📈 MONITORING:

✅ Dashboard de métricas chave
✅ Alertas para degradação
✅ Detectar data/concept drift
✅ Audit logs para compliance
✅ SLAs definidos e monitorados

🔒 SEGURANÇA:

✅ Criptografia de dados sensíveis
✅ Least privilege access
✅ Input validation e sanitization
✅ Model versioning seguro
✅ Incident response plan`,
            keyPoints: [
                "Reprodutibilidade é fundamental: versionar tudo",
                "Automatizar o máximo possível (CI/CD/CT)",
                "Monitorar proativamente, não reativamente",
                "Segurança desde o design, não como addon",
                "Documentação é parte do entregável"
            ],
            example: `# Checklist de produção
production_checklist = {
    'data': {
        'versioned': True,
        'validated': True,
        'pii_handled': True,
        'lineage_documented': True
    },
    'model': {
        'versioned': True,
        'registered': True,
        'metrics_logged': True,
        'reproducible': True
    },
    'deployment': {
        'containerized': True,
        'health_check': True,
        'rollback_ready': True,
        'scaled': True
    },
    'monitoring': {
        'performance_dashboard': True,
        'drift_detection': True,
        'alerts_configured': True,
        'logging_enabled': True
    },
    'security': {
        'auth_required': True,
        'data_encrypted': True,
        'audit_logging': True,
        'access_controlled': True
    }
}

# Verificar checklist
for category, items in production_checklist.items():
    missing = [k for k, v in items.items() if not v]
    if missing:
        print(f"⚠️ {category}: faltando {missing}")
    else:
        print(f"✅ {category}: completo")

# Template de documentação
'''
# Model Card: [Nome do Modelo]

## Overview
- **Purpose**: [O que o modelo faz]
- **Owner**: [Equipe/pessoa responsável]
- **Version**: [Versão atual]

## Training
- **Data**: [Fonte, período, tamanho]
- **Algorithm**: [Algoritmo usado]
- **Hyperparameters**: [Principais hiperparâmetros]

## Performance
- **Metrics**: [Accuracy, F1, etc.]
- **Fairness**: [Análise de viés]
- **Limitations**: [Onde o modelo falha]

## Usage
- **Input**: [Formato esperado]
- **Output**: [Formato de saída]
- **API**: [Endpoint e documentação]

## Monitoring
- **SLAs**: [Latência, uptime]
- **Alerts**: [Condições de alerta]
- **Retraining**: [Frequência/trigger]
'''`,
            realCase: {
                title: "Google ML Best Practices",
                description: "Google publicou seu paper 'Rules of ML' com 43 regras práticas aprendidas em anos de ML em produção. Regra #1: 'Don't be afraid to launch without machine learning' - as vezes uma heurística simples é melhor.",
                impact: "Guia referência para engenheiros de ML em todo o mundo"
            }
        }
    ]
};

// CAIP-210 Advanced Questions - Extra Set 3 (English Version)
// Based on grok-CAIP-210.txt advanced topics
// Manually curated for quality and completeness

const questionsExtra3_en = [
    // Domain 1: AI & ML Fundamentals
    {
        id: 129,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "A global logistics company faces unpredictable demand fluctuations due to geopolitical events and climate change. Which integrated approach would be most appropriate to maximize business value while mitigating ethical risks?",
        options: [
            "Formulate as simple regression, ignoring ethics to focus on accuracy",
            "Use Design of Experiments (DOE) for controllable variables, integrating bias analysis and transparency as performance metrics",
            "Apply unsupervised clustering without explicit formulation",
            "Prioritize reinforcement learning without considering external stakeholders"
        ],
        correct: 1,
        explanation: "ML problem formulation should integrate DOE for independent/dependent variables, while ethics (bias, transparency) must be considered from the start. This preserves integrity and aligns with stakeholders."
    },
    {
        id: 130,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "What is the most critical impact of randomness and uncertainty in a stochastic ML model when solving problems like public health crisis prediction?",
        options: [
            "Guarantees perfect predictions for individual events",
            "Allows capturing general patterns, but requires mitigation via ensemble methods to reduce variance",
            "Eliminates the need for DOE experimentation",
            "Makes all models deterministic with sufficient data"
        ],
        correct: 1,
        explanation: "Stochastic models capture general patterns, but randomness causes variance. Ensemble methods (like Random Forest, Gradient Boosting) mitigate this variance, especially in uncertain scenarios."
    },

    // Domain 2: Data Preparation
    {
        id: 131,
        domain: 2,
        domainName: "Data Preparation",
        question: "A healthcare company collects wearable data with 30% missing values and outliers from faulty sensors. Which integrated strategy minimizes bias while preserving statistical integrity?",
        options: [
            "Exclude all data with missing values for simplicity",
            "Use median imputation for outliers, Box-Cox for normalization, and embedding for unstructured data, with ethical verification for PII",
            "Apply deduplication without transformation",
            "Ignore unstructured data, focusing only on numerical data"
        ],
        correct: 1,
        explanation: "Robust preparation includes: median imputation (robust to outliers), Box-Cox for skewed distributions, embedding for unstructured data, and ethical verification to protect personally identifiable information (PII)."
    },
    {
        id: 132,
        domain: 2,
        domainName: "Data Preparation",
        question: "What is the most critical challenge when working with unstructured audio data in a voice fraud detection system?",
        options: [
            "Low amplitude ignores sampling rate",
            "Need for Fourier transformation and MFCCs for feature extraction, dealing with noise and periodicity",
            "Direct conversion to text without preprocessing",
            "Ignoring spectrograms for speed"
        ],
        correct: 1,
        explanation: "Audio data requires Fourier transformation for frequency analysis and MFCCs (Mel-Frequency Cepstral Coefficients) for feature extraction. Noise and periodicity significantly affect quality."
    },
    {
        id: 133,
        domain: 2,
        domainName: "Data Preparation",
        question: "In a dataset with 50% unstructured text from reviews, which preprocessing sequence is most effective for sentiment analysis?",
        options: [
            "Tokenization → Stemming → Bag of words → Stop words removal",
            "Stop words removal → Tokenization → Lemmatization → Embedding",
            "Direct embedding without tokenization",
            "Deduplication → Normalization → Binning"
        ],
        correct: 1,
        explanation: "The correct sequence for text preprocessing is: remove stop words, tokenize, apply lemmatization (better than stemming for preserving meaning), then create embeddings for dense vector representation."
    },
    {
        id: 134,
        domain: 2,
        domainName: "Data Preparation",
        question: "What is the impact of the curse of dimensionality on large datasets?",
        options: [
            "Automatically increases performance",
            "Reduces ability to learn patterns, requiring dimensionality reduction",
            "Eliminates need for feature selection",
            "Makes all features relevant"
        ],
        correct: 1,
        explanation: "The curse of dimensionality occurs when many features relative to samples reduce the model's ability to learn useful patterns. Techniques like PCA, feature selection, or regularization are necessary."
    },

    // Domain 3: Training & Tuning
    {
        id: 135,
        domain: 3,
        domainName: "Training & Tuning",
        question: "A cancer detection model has 98% accuracy on training but drops to 70% on validation due to imbalance and noise. Which strategy resolves overfitting and improves generalization?",
        options: [
            "Increase epochs without cross-validation",
            "Use stratified k-fold cross-validation, learning curves for bias-variance, and regularization",
            "Ignore metrics like F1, focusing only on accuracy",
            "Reduce data for simplicity"
        ],
        correct: 1,
        explanation: "Stratified k-fold maintains class proportions, learning curves diagnose bias-variance tradeoff, and regularization (L1/L2) prevents overfitting. Accuracy alone is inadequate for imbalanced data."
    },
    {
        id: 136,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is the role of learning curves in diagnosing irreducible error?",
        options: [
            "Show when more data doesn't reduce error",
            "Ignore bias",
            "Always indicate underfitting",
            "Replace cross-validation"
        ],
        correct: 0,
        explanation: "Learning curves show how training and validation error evolve with more data. When both curves converge to a plateau, it indicates irreducible error (inherent to the problem)."
    },
    {
        id: 137,
        domain: 3,
        domainName: "Training & Tuning",
        question: "In a black box model, how to evaluate performance beyond accuracy?",
        options: [
            "Only with AUC",
            "Use Precision-Recall Curve for imbalance, F1 for tradeoff, and explainability tools",
            "Ignore variance",
            "Focus only on training time"
        ],
        correct: 1,
        explanation: "For complete evaluation: PRC for imbalanced data, F1 score to balance precision-recall, and explainability tools (SHAP, LIME) to understand model decisions."
    },
    {
        id: 138,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Scenario: Iterative training with overfitting. How to optimize hyperparameters efficiently?",
        options: [
            "Exhaustive grid search",
            "Bayesian optimization for efficiency in large spaces",
            "Randomized search without distribution",
            "Manual tuning"
        ],
        correct: 1,
        explanation: "Bayesian optimization uses probabilities to intelligently explore the hyperparameter space, being much more efficient than grid search in large spaces."
    },
    {
        id: 139,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Which metric is best for imbalanced classification?",
        options: [
            "Accuracy",
            "F1 score",
            "MSE",
            "R²"
        ],
        correct: 1,
        explanation: "F1 score is the harmonic mean of precision and recall, ideal for imbalanced data. Accuracy can be misleading (e.g., 99% accuracy by always predicting the majority class)."
    },

    // Domain 3: Advanced Algorithms
    {
        id: 140,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Real estate pricing model with high multicollinearity and extreme outliers. Which approach is most robust for obtaining interpretable coefficients?",
        options: [
            "Simple linear regression without regularization",
            "Ridge regression (L2) combined with VIF analysis and selective removal of collinear variables",
            "Lasso regression (L1) without prior analysis",
            "Batch Gradient Descent without regularization"
        ],
        correct: 1,
        explanation: "Ridge (L2) reduces multicollinearity impact without zeroing coefficients (maintains interpretability). VIF (Variance Inflation Factor) helps identify and explicitly treat collinearity."
    },
    {
        id: 141,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Which statement is FALSE about regularization in linear regression?",
        options: [
            "Ridge penalizes the sum of squared coefficients",
            "Lasso can zero coefficients and perform automatic variable selection",
            "Elastic Net never zeros coefficients when λ is small",
            "Ridge keeps all coefficients non-zero (except in extreme cases)"
        ],
        correct: 2,
        explanation: "Elastic Net CAN zero coefficients (inherits Lasso property), but in a more controlled way than pure Lasso. It combines L1 and L2 to balance feature selection and stability."
    },
    {
        id: 142,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Sales forecasting with weekly seasonality, holidays, and promotions. Which modeling pipeline is most appropriate?",
        options: [
            "Simple ARIMA without differencing",
            "SARIMA with seasonal components + exogenous regressors (SARIMAX)",
            "VAR without stationarity verification",
            "ARIMA with order (0,0,0)"
        ],
        correct: 1,
        explanation: "SARIMAX handles seasonality (S component), trend (via differencing I), and exogenous variables (X) like promotions and holidays — ideal for multivariate forecasting with seasonality."
    },
    {
        id: 143,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Which statistical test verifies stationarity in time series before applying ARIMA?",
        options: [
            "Augmented Dickey-Fuller (ADF) test",
            "Shapiro-Wilk test",
            "Levene test",
            "Chi-square test"
        ],
        correct: 0,
        explanation: "ADF (Augmented Dickey-Fuller) tests for unit root presence, indicating non-stationarity. It's the standard test before applying ARIMA to time series."
    },
    {
        id: 144,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Credit risk classification with 3% defaulters and 200 features. Which combination offers best tradeoff between interpretability and performance?",
        options: [
            "Logistic Regression + SMOTE + L1 regularization",
            "k-NN with k=1 + no balancing",
            "Logistic Regression without regularization + random undersampling",
            "k-NN with k=50 + no feature selection"
        ],
        correct: 0,
        explanation: "Logistic Regression offers interpretability via coefficients; L1 (Lasso) performs automatic feature selection; SMOTE (Synthetic Minority Over-sampling) handles imbalance while preserving information."
    },
    {
        id: 145,
        domain: 3,
        domainName: "Training & Tuning",
        question: "In binary classification, moving the threshold from 0.5 to 0.7 generally causes:",
        options: [
            "Increase in recall and decrease in precision",
            "Increase in precision and decrease in recall",
            "Simultaneous increase in both",
            "Simultaneous decrease in both"
        ],
        correct: 1,
        explanation: "Higher threshold (0.7) makes the model more conservative → classifies positive only with high confidence → fewer false positives → higher precision, but lower recall (misses true positives)."
    },
    {
        id: 146,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Which linkage is most robust to outliers in hierarchical clustering?",
        options: [
            "Single linkage",
            "Complete linkage",
            "Average linkage",
            "Ward linkage"
        ],
        correct: 1,
        explanation: "Complete linkage considers maximum distance between points of different clusters, being less sensitive to outliers than single linkage (which uses minimum distance and can create chains)."
    },
    {
        id: 147,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Churn model with 1.2M rows and 300 features with high variance. Which ensemble offers best balance between performance and training time?",
        options: [
            "Single deep CART tree",
            "Random Forest with 500 trees + max_features = sqrt(n)",
            "Gradient Boosting with learning rate 0.01 and 2000 estimators",
            "Bagging with shallow trees without feature subsampling"
        ],
        correct: 1,
        explanation: "Random Forest reduces variance via bagging + random feature selection, is scalable for large datasets, and provides feature importance. Gradient Boosting would be too slow with 2000 estimators."
    },
    {
        id: 148,
        domain: 3,
        domainName: "Training & Tuning",
        question: "In SVM for regression (SVR), the ε (epsilon) parameter controls:",
        options: [
            "The width of the tolerance margin",
            "The penalty for margin violations",
            "The degree of polynomial kernel",
            "The radius of RBF kernel"
        ],
        correct: 0,
        explanation: "In SVR, ε defines the acceptable error range without penalty (epsilon tube). Predictions within ±ε of the true value are not penalized, allowing control over model sensitivity."
    },
    {
        id: 149,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Defect detection in high-resolution industrial images. Which architecture is most suitable for capturing fine textures with computational efficiency?",
        options: [
            "MLP with 5 dense layers",
            "CNN with multiple convolutional layers + batch normalization + global average pooling",
            "RNN with LSTM for sequential pixels",
            "Transformer without convolution"
        ],
        correct: 1,
        explanation: "CNN is gold standard for computer vision: convolutional layers capture local and hierarchical patterns, batch normalization stabilizes training, and global average pooling reduces parameters while maintaining spatial information."
    },

    // Domain 4: MLOps & Production
    {
        id: 150,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is the main advantage of using Docker + Kubernetes for ML model deployment compared to bare-metal servers?",
        options: [
            "Lower prediction latency",
            "Portability, horizontal scalability, and dependency isolation",
            "Lower memory consumption",
            "Higher model interpretability"
        ],
        correct: 1,
        explanation: "Docker provides containerization (dependency isolation and portability), while Kubernetes offers orchestration (automatic horizontal scaling, self-healing, load balancing) — essential for modern MLOps."
    }
];

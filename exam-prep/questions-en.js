// CAIP-210 Exam Questions Database - English Version
// Based on CertNexus Certified AI Practitioner official course material

const questions_en = [
    // ===== DOMAIN 1: AI & ML FUNDAMENTALS (26%) =====
    {
        id: 1,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "What is the main difference between machine learning and traditional programming?",
        options: [
            "Machine learning makes predictions based on data without explicit instructions",
            "Traditional programming is slower",
            "Machine learning requires more hardware",
            "Machine learning only works with Big Data"
        ],
        correct: 0,
        explanation: "Machine learning differs from traditional programming because computers make predictions and decisions based on datasets, without explicit instructions provided by humans. This allows automating decision-making processes faster and more efficiently."
    },
    {
        id: 2,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "What is Deep Learning in relation to Machine Learning?",
        options: [
            "A data visualization method",
            "A form of cloud storage",
            "A subset of ML that uses complex artificial neural networks",
            "A database optimization technique"
        ],
        correct: 2,
        explanation: "Deep Learning is a subset of machine learning that involves the use of complex artificial neural networks. These networks are even more effective at solving complex problems."
    },
    {
        id: 3,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "In the DIK (Data, Information, Knowledge) hierarchy, what transforms data into information?",
        options: [
            "Data backup",
            "File compression",
            "Database storage",
            "Aggregation, organization, and interpretation of data"
        ],
        correct: 3,
        explanation: "Raw data generally has little context. When aggregated, organized, and interpreted, it becomes useful information for business decisions."
    },
    {
        id: 4,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "What type of learning uses labeled data with known outcomes?",
        options: [
            "Reinforcement Learning",
            "Unsupervised Learning",
            "Supervised Learning",
            "Transfer Learning"
        ],
        correct: 2,
        explanation: "Supervised Learning uses labeled data (with known outcomes) to train the model. The model learns to make predictions based on this historical data."
    },
    {
        id: 5,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "What type of learning is used to find hidden patterns in unlabeled data?",
        options: [
            "Supervised Learning",
            "Semi-supervised Learning",
            "Reinforcement Learning",
            "Unsupervised Learning"
        ],
        correct: 3,
        explanation: "Unsupervised Learning makes inferences from datasets that have no labels. It's used to discover hidden patterns (like customer segmentation) without prior guidance."
    },
    {
        id: 6,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "In Semisupervised Learning, what is the advantage of using a small amount of labeled data?",
        options: [
            "Greater accuracy than supervised learning",
            "Does not require any data",
            "Lower cost and faster labeling compared to labeling the entire dataset",
            "Does not need validation"
        ],
        correct: 2,
        explanation: "Semi-supervised learning reduces costs and time associated with labeling large datasets. A good model can be created with just a small set of labeled examples combined with unlabeled data."
    },
    {
        id: 7,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "What are the main ethical concerns related to AI/ML (PATHS)?",
        options: [
            "Prediction, Automation, Technology, Hosting, Security",
            "Programming, Analysis, Training, Hosting, Services",
            "Performance, Accuracy, Testing, Hardware, Software",
            "Privacy, Accountability, Transparency, Harm, Fairness"
        ],
        correct: 3,
        explanation: "The main ethical risks in AI/ML include: Privacy (personal data protection), Accountability (responsibility for decisions), Transparency/Explainability (ability to understand decisions), Fairness (fair treatment without discrimination), and Safety/Security (harm minimization)."
    },
    {
        id: 8,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "What does 'stochastic' mean in the context of machine learning models?",
        options: [
            "Individual samples are random, but the set follows a general pattern",
            "Models are deterministic and always produce the same result",
            "Models require constant human supervision",
            "Models cannot learn from data"
        ],
        correct: 0,
        explanation: "Stochastic models recognize that individual samples are inherently random, but the dataset follows general patterns that allow making useful estimates."
    },
    {
        id: 9,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "In ML problem formulation, what does 'Task' represent in the Task-Experience-Performance framework?",
        options: [
            "The hardware needed for processing",
            "The model evaluation metric",
            "What the solution should accomplish (e.g., predict house price)",
            "The dataset used for training"
        ],
        correct: 2,
        explanation: "In the TEP framework, Task defines what the solution should accomplish (e.g., 'Predict the sale price of a house'), Experience defines which dataset will be used for learning, and Performance defines how to evaluate performance."
    },
    {
        id: 10,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "When is it NOT recommended to use AI/ML to solve a problem?",
        options: [
            "When the problem can be solved with simpler traditional programming logic",
            "When there are non-obvious patterns in the data",
            "When complex decisions need to be made",
            "When large volumes of data are available"
        ],
        correct: 0,
        explanation: "AI/ML can be expensive, time-consuming, and risky. If the problem can be solved with simpler traditional programming (e.g., rule-based ticket routing), AI/ML may not be justifiable."
    },
    {
        id: 11,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "What are independent variables (input/predictor variables) in Design of Experiments?",
        options: [
            "Variables that result from other calculations",
            "Variables you can directly change to see their impact",
            "Variables that are always constant",
            "Variables you cannot control"
        ],
        correct: 1,
        explanation: "Independent variables are those you can directly change in the experiment. Dependent variables (output/response) are those that change indirectly as a result."
    },
    {
        id: 12,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Which stakeholder is responsible for providing insight into tools, technologies, and resources needed for the project?",
        options: [
            "Governments",
            "Customers/End Users",
            "Team Members (practitioners)",
            "Sponsors/Champions"
        ],
        correct: 2,
        explanation: "Team Members are practitioners who work directly on project development and can provide insights about the tools, technologies, and resources needed for success."
    },

    // ===== DOMAIN 2: DATA PREPARATION (20%) =====
    {
        id: 13,
        domain: 2,
        domainName: "Data Preparation",
        question: "Which imputation method finds similar records in the same dataset to fill missing values?",
        options: [
            "Regression Imputation",
            "Hot-deck Imputation",
            "Cold-deck Imputation",
            "Mean/Mode Imputation"
        ],
        correct: 1,
        explanation: "Hot-deck imputation finds records in the same sample that have similar values in other features, and copies the missing value from one of these similar records."
    },
    {
        id: 14,
        domain: 2,
        domainName: "Data Preparation",
        question: "What is the correct formula for NORMALIZATION (min-max scaling)?",
        options: [
            "x' = log(x)",
            "x' = (x - min) / (max - min)",
            "x' = x / max",
            "x' = (x - μ) / σ"
        ],
        correct: 1,
        explanation: "Normalization transforms values to the [0, 1] range using the formula: x' = (x - min) / (max - min), where min and max are the minimum and maximum values of the feature."
    },
    {
        id: 15,
        domain: 2,
        domainName: "Data Preparation",
        question: "What is the correct formula for STANDARDIZATION (z-score)?",
        options: [
            "x' = (x - μ) / σ",
            "x' = log10(x)",
            "x' = x^(1/3)",
            "x' = (x - min) / (max - min)"
        ],
        correct: 0,
        explanation: "Standardization calculates the z-score: x' = (x - μ) / σ, where μ is the mean and σ is the standard deviation. This centers the data at 0 with standard deviation 1."
    },
    {
        id: 16,
        domain: 2,
        domainName: "Data Preparation",
        question: "When is feature scaling LESS important?",
        options: [
            "When using Decision Trees and Random Forests",
            "When using Support Vector Machines (SVM)",
            "When using neural networks",
            "When using k-Nearest Neighbor (k-NN)"
        ],
        correct: 0,
        explanation: "Tree-based algorithms (Decision Trees, Random Forests) do not require features to be scaled. Distance-based algorithms (k-NN, SVM) require scaling."
    },
    {
        id: 17,
        domain: 2,
        domainName: "Data Preparation",
        question: "Which encoding method is most appropriate when categories have NO natural order or ranking?",
        options: [
            "Hash Encoding",
            "One-hot Encoding",
            "Target Encoding",
            "Label Encoding (Ordinal Encoding)"
        ],
        correct: 1,
        explanation: "One-hot encoding creates dummy columns for each class, assigning 1 or 0. This prevents the algorithm from interpreting an order/ranking between categories."
    },
    {
        id: 18,
        domain: 2,
        domainName: "Data Preparation",
        question: "What is 'discretization' of a continuous variable?",
        options: [
            "Converting a string variable to a number",
            "Removing duplicate values",
            "Calculating the mean of the variable",
            "Converting a continuous variable into discrete intervals (bins)"
        ],
        correct: 3,
        explanation: "Discretization (or data binning) is the process of converting a continuous variable into discrete intervals. For example, transforming exact age into age groups (18-24, 25-34, etc.)."
    },
    {
        id: 19,
        domain: 2,
        domainName: "Data Preparation",
        question: "What is the 'curse of dimensionality'?",
        options: [
            "Having too little data to train a model",
            "The difficulty of processing data in real-time",
            "The reduction in model's ability to learn when there are too many features relative to samples",
            "The high cost of data storage"
        ],
        correct: 2,
        explanation: "The curse of dimensionality occurs when adding more features (without increasing samples) starts to reduce the model's ability to learn useful patterns."
    },
    {
        id: 20,
        domain: 2,
        domainName: "Data Preparation",
        question: "Which dimensionality reduction algorithm selects features that contribute the most linear variance in the data?",
        options: [
            "k-Means",
            "PCA (Principal Component Analysis)",
            "t-SNE",
            "Random Forest"
        ],
        correct: 1,
        explanation: "PCA projects high-dimensional data into a lower-dimensional space, selecting features that contribute the most linear variance."
    },
    {
        id: 21,
        domain: 2,
        domainName: "Data Preparation",
        question: "Which transformation helps reduce positive skewness in non-normally distributed data?",
        options: [
            "Standardization",
            "Log transformation",
            "One-hot encoding",
            "Target encoding"
        ],
        correct: 1,
        explanation: "Log transformation helps reduce positive skewness in non-normally distributed datasets, bringing them closer to a normal distribution."
    },
    {
        id: 22,
        domain: 2,
        domainName: "Data Preparation",
        question: "Which pandas function is used to identify missing values in a DataFrame?",
        options: [
            "df.duplicated()",
            "df.fillna()",
            "df.dropna()",
            "df.isna() or df.isnull()"
        ],
        correct: 3,
        explanation: "pandas.DataFrame.isna() returns a DataFrame of booleans indicating which values are formatted as missing type (None, NaN)."
    },
    {
        id: 23,
        domain: 2,
        domainName: "Data Preparation",
        question: "When a column has more than 70% missing values, what is the recommended approach?",
        options: [
            "Convert to 'unknown' category",
            "Use mean imputation for all values",
            "Duplicate values from other columns",
            "Drop (remove) the entire column"
        ],
        correct: 3,
        explanation: "When a column has a high percentage of missing values (like 70% or more), it's recommended to remove (drop) the entire column, as imputation may introduce too much noise."
    },
    {
        id: 24,
        domain: 2,
        domainName: "Data Preparation",
        question: "What is Feature Selection vs Feature Extraction in dimensionality reduction?",
        options: [
            "Selection chooses a subset of original features; Extraction derives new features by combining originals",
            "Selection removes outliers; Extraction removes duplicates",
            "Selection is manual; Extraction is automatic",
            "They are the same thing, just different names"
        ],
        correct: 0,
        explanation: "Feature Selection selects a subset of original features (excluding redundant/irrelevant ones). Feature Extraction derives new features by combining multiple correlated features into one."
    },

    // ===== DOMAIN 3: TRAINING & TUNING (24%) =====
    {
        id: 25,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is OVERFITTING in machine learning?",
        options: [
            "When the model fits too closely to training data and performs poorly on new data",
            "When the model takes too long to train",
            "When the model is too simple to capture patterns in the data",
            "When there's not enough data for training"
        ],
        correct: 0,
        explanation: "Overfitting occurs when the model learns the training data too well (including noise) and fails to generalize to new, unseen data."
    },
    {
        id: 26,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is UNDERFITTING in machine learning?",
        options: [
            "When the model trains too quickly",
            "When the model is too complex",
            "When there's too much data for training",
            "When the model is too simple to capture underlying patterns in the data"
        ],
        correct: 3,
        explanation: "Underfitting occurs when the model is too simple (high bias) and cannot capture the underlying patterns in the data, resulting in poor performance on both training and test data."
    },
    {
        id: 27,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What does the R² (R-squared) metric represent in regression?",
        options: [
            "The proportion of variance in the dependent variable explained by the model",
            "The number of iterations needed to converge",
            "The mean absolute error of the model",
            "The learning rate of the model"
        ],
        correct: 0,
        explanation: "R² (coefficient of determination) measures the proportion of variance in the dependent variable that is explained by the independent variables in the model. Values closer to 1 indicate a better fit."
    },
    {
        id: 28,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Which metric is most appropriate when it's crucial to minimize FALSE NEGATIVES?",
        options: [
            "Recall (Sensitivity)",
            "Accuracy",
            "Precision",
            "Specificity"
        ],
        correct: 0,
        explanation: "Recall (Sensitivity) measures the proportion of actual positives correctly identified. It's crucial when false negatives are dangerous (e.g., not detecting a serious disease)."
    },
    {
        id: 29,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Which metric is most appropriate when it's crucial to minimize FALSE POSITIVES?",
        options: [
            "Sensitivity",
            "Recall",
            "Precision",
            "F1-Score"
        ],
        correct: 2,
        explanation: "Precision measures the proportion of positive predictions that are correct. It's crucial when false positives are costly (e.g., classifying legitimate email as spam)."
    },
    {
        id: 30,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is Cross-Validation and what is it used for?",
        options: [
            "A technique that divides data into multiple folds to evaluate the model more robustly",
            "A technique to remove outliers",
            "A technique to visualize data",
            "A technique to collect more data"
        ],
        correct: 0,
        explanation: "Cross-validation divides data into multiple folds, using each fold as test while the others are used for training. This provides a more robust evaluation of model performance."
    },
    {
        id: 31,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is Regularization (L1/Lasso, L2/Ridge) in regression models?",
        options: [
            "A technique that adds a penalty to prevent overfitting",
            "A technique to speed up training",
            "A technique to increase model complexity",
            "A technique to increase training data"
        ],
        correct: 0,
        explanation: "Regularization adds a penalty (regularization term) to the cost function to reduce model complexity and prevent overfitting. L1 (Lasso) can zero out coefficients; L2 (Ridge) reduces them."
    },
    {
        id: 32,
        domain: 3,
        domainName: "Training & Tuning",
        question: "In k-Nearest Neighbors (k-NN), what happens when K is too small?",
        options: [
            "The model becomes too sensitive to noise (overfitting)",
            "The model becomes too generalist (underfitting)",
            "Training time increases significantly",
            "The model stops working"
        ],
        correct: 0,
        explanation: "With small K, the model considers few neighbors, making it very sensitive to individual points (including noise), resulting in overfitting."
    },
    {
        id: 33,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is a Confusion Matrix?",
        options: [
            "A hyperparameter table",
            "A data transformation matrix",
            "A table showing True Positives, False Positives, True Negatives, and False Negatives",
            "A matrix showing correlation between features"
        ],
        correct: 2,
        explanation: "The Confusion Matrix is a table that summarizes the performance of a classification model, showing TP (true positives), FP (false positives), TN (true negatives), and FN (false negatives)."
    },
    {
        id: 34,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Which classification algorithm creates decision boundaries that maximize the margin between classes?",
        options: [
            "k-Nearest Neighbors",
            "Logistic Regression",
            "Support Vector Machines (SVM)",
            "Naive Bayes"
        ],
        correct: 2,
        explanation: "SVMs find the hyperplane that maximizes the margin (distance) between classes, making them robust for classification."
    },
    {
        id: 35,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What are 'support vectors' in an SVM?",
        options: [
            "The most important features",
            "The data points closest to the decision boundary that define the hyperplane",
            "The cluster centroids",
            "All points in the dataset"
        ],
        correct: 1,
        explanation: "Support vectors are the data points closest to the decision boundary (hyperplane). They are critical because they define the position and orientation of the separating hyperplane."
    },
    {
        id: 36,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is the 'kernel trick' in SVMs?",
        options: [
            "A feature selection technique",
            "A technique to speed up training",
            "A regularization technique",
            "A technique that allows finding non-linear boundaries by mapping data to higher dimensions"
        ],
        correct: 3,
        explanation: "The kernel trick allows SVMs to find non-linear decision boundaries by implicitly mapping data to higher-dimensional spaces where they can be linearly separable."
    },
    {
        id: 37,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is the main advantage of Random Forests over a single Decision Tree?",
        options: [
            "It's faster to train",
            "Reduces overfitting by combining multiple trees (ensemble)",
            "Doesn't require numerical data",
            "Uses less memory"
        ],
        correct: 1,
        explanation: "Random Forests combine multiple decision trees (ensemble), each trained on different subsets of data. This reduces variance and overfitting compared to a single tree."
    },
    {
        id: 38,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is k-Means Clustering?",
        options: [
            "A clustering algorithm that partitions data into K clusters based on centroids",
            "A dimensionality reduction algorithm",
            "A supervised classification algorithm",
            "A regression algorithm"
        ],
        correct: 0,
        explanation: "k-Means is an unsupervised clustering algorithm that partitions n observations into K clusters, where each observation belongs to the cluster with the nearest centroid."
    },
    {
        id: 39,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is the 'Elbow Method' used to determine?",
        options: [
            "The ideal learning rate",
            "The optimal number of clusters (K) in k-Means",
            "The best ML algorithm to use",
            "The number of epochs for training"
        ],
        correct: 1,
        explanation: "The Elbow Method plots inertia (sum of distances to centroids) vs number of clusters. The point where the curve forms an 'elbow' indicates the optimal number of clusters."
    },
    {
        id: 40,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What are Multi-Layer Perceptrons (MLPs)?",
        options: [
            "A type of clustering algorithm",
            "A type of regularization",
            "Neural networks with multiple connected layers (feedforward)",
            "A feature engineering method"
        ],
        correct: 2,
        explanation: "MLPs are artificial neural networks with multiple layers (input, hidden, output) connected in a feedforward manner. They are the foundation for deep learning."
    },
    {
        id: 41,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Which type of neural network is most appropriate for IMAGE processing?",
        options: [
            "Multi-Layer Perceptrons (MLP)",
            "Convolutional Neural Networks (CNN)",
            "Autoencoders",
            "Recurrent Neural Networks (RNN)"
        ],
        correct: 1,
        explanation: "CNNs are designed for grid data processing (like images). They use convolution layers to detect local and hierarchical patterns."
    },
    {
        id: 42,
        domain: 3,
        domainName: "Training & Tuning",
        question: "Which type of neural network is most appropriate for SEQUENTIAL data (like text or time series)?",
        options: [
            "Autoencoders",
            "Generative Adversarial Networks (GAN)",
            "Convolutional Neural Networks (CNN)",
            "Recurrent Neural Networks (RNN)"
        ],
        correct: 3,
        explanation: "RNNs are designed to process sequential data, maintaining 'memory' of previous inputs. They are ideal for NLP, translation, and time series."
    },
    {
        id: 43,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is Gradient Descent?",
        options: [
            "A feature selection technique",
            "A cross-validation method",
            "An optimization algorithm that adjusts parameters to minimize the cost function",
            "A type of regularization"
        ],
        correct: 2,
        explanation: "Gradient Descent is an iterative optimization algorithm that adjusts model parameters in the direction opposite to the gradient of the cost function, seeking to minimize it."
    },
    {
        id: 44,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is the activation function in neural networks?",
        options: [
            "A function that initializes weights",
            "A function that introduces non-linearity, allowing the network to learn complex patterns",
            "A function that determines batch size",
            "A function that calculates model loss"
        ],
        correct: 1,
        explanation: "Activation functions (like ReLU, Sigmoid, Tanh) introduce non-linearity in the neural network, allowing it to learn complex relationships beyond linear transformations."
    },

    // ===== DOMAIN 4: MLOps & PRODUCTION (30%) =====
    {
        id: 45,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is MLOps?",
        options: [
            "A deep learning framework",
            "The practice of combining Machine Learning with DevOps to automate the ML lifecycle",
            "A feature engineering technique",
            "A type of machine learning algorithm"
        ],
        correct: 1,
        explanation: "MLOps is the practice of applying DevOps principles to machine learning, automating the development, deployment, and maintenance of models in production."
    },
    {
        id: 46,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is 'model drift' or 'concept drift'?",
        options: [
            "When data or relationships change over time, degrading model performance",
            "When the model code is modified",
            "When the model is transferred to another server",
            "When the model learns too fast"
        ],
        correct: 0,
        explanation: "Model/concept drift occurs when patterns in data change over time, causing a model trained on old data to lose effectiveness on new data."
    },
    {
        id: 47,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is an 'ML Pipeline'?",
        options: [
            "A visualization tool",
            "Specific hardware for ML",
            "An automated sequence of steps from data preparation to model deployment",
            "A type of neural network"
        ],
        correct: 2,
        explanation: "An ML Pipeline is an automated and reproducible sequence of steps that includes data collection, preprocessing, training, evaluation, and model deployment."
    },
    {
        id: 48,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is 'model versioning' and why is it important?",
        options: [
            "Updating model documentation",
            "Giving different names to models",
            "Creating model backups",
            "Tracking different versions of models, data, and code for reproducibility"
        ],
        correct: 3,
        explanation: "Model versioning tracks different versions of models, datasets, and code. It's crucial for reproducibility, rollback in case of problems, and auditing."
    },
    {
        id: 49,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is 'A/B Testing' in the context of model deployment?",
        options: [
            "Testing accuracy and precision separately",
            "Testing the model on two different datasets",
            "Splitting data into train and test",
            "Comparing two models/versions by serving different user groups simultaneously"
        ],
        correct: 3,
        explanation: "A/B Testing in ML context means serving two different versions of a model to different user groups, comparing their performance in production."
    },
    {
        id: 50,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is 'Canary Deployment'?",
        options: [
            "Deployment to multiple servers simultaneously",
            "Automatic deployment without tests",
            "Gradual deployment of new model to a small percentage of users before full rollout",
            "Deployment in development environment"
        ],
        correct: 2,
        explanation: "Canary deployment releases the new model to a small percentage of traffic first, allowing detection of problems before impacting all users."
    },
    {
        id: 51,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is the importance of monitoring models in production?",
        options: [
            "Only to measure infrastructure costs",
            "Only for error logging",
            "To measure training time",
            "To detect performance degradation, drift, and ensure the model continues meeting requirements"
        ],
        correct: 3,
        explanation: "Continuous monitoring is essential to detect performance degradation, data/concept drift, anomalies, and ensure the model continues meeting business requirements."
    },
    {
        id: 52,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is 'model retraining' and when should it be done?",
        options: [
            "Documenting the existing model",
            "Training the model only once",
            "Testing the model before deployment",
            "Periodically updating the model with new data to maintain performance"
        ],
        correct: 3,
        explanation: "Model retraining is the process of updating the model with more recent data. It should be done periodically or when monitoring metrics indicate degradation (drift)."
    },
    {
        id: 53,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What are 'Feature Stores' in the context of MLOps?",
        options: [
            "Traditional databases",
            "Online feature shops",
            "Centralized repositories for storing, managing, and serving features for ML models",
            "Algorithm libraries"
        ],
        correct: 2,
        explanation: "Feature Stores are centralized repositories that store, manage, and serve computed features for model training and inference, ensuring consistency."
    },
    {
        id: 54,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is an important security consideration for ML pipelines?",
        options: [
            "Use only public data",
            "Don't use encryption for speed",
            "Protect sensitive data, control access, and ensure model integrity",
            "Keep all models public"
        ],
        correct: 2,
        explanation: "Security in ML pipelines includes: protecting sensitive data, access control, encryption, model integrity (preventing adversarial attacks), and auditing."
    },
    {
        id: 55,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is 'model explainability' and why is it important in production?",
        options: [
            "The model's inference speed",
            "The ability to explain how and why a model made a specific decision",
            "The model size in megabytes",
            "Documenting model code"
        ],
        correct: 1,
        explanation: "Model explainability is the ability to understand and explain the model's decisions. It's crucial for regulatory compliance, debugging, user trust, and bias identification."
    },
    {
        id: 56,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is 'batch inference' vs 'real-time inference'?",
        options: [
            "Batch processes multiple predictions at once; real-time processes individual predictions immediately",
            "Batch is for training; real-time is for testing",
            "Batch is more accurate; real-time is less accurate",
            "They are the same thing"
        ],
        correct: 0,
        explanation: "Batch inference processes large volumes of data at once (e.g., overnight). Real-time inference processes individual predictions immediately when requested (e.g., recommendations)."
    },
    {
        id: 57,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is CI/CD in the context of MLOps?",
        options: [
            "Customer Interface / Customer Development",
            "Continuous Integration / Continuous Deployment - automation of build, test, and deploy",
            "Code Inspection / Code Debugging",
            "Continuous Intelligence / Continuous Data"
        ],
        correct: 1,
        explanation: "CI/CD (Continuous Integration / Continuous Deployment) automates the build, test, and deployment process for code and models, ensuring faster and more reliable deliveries."
    },
    {
        id: 58,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is 'data lineage' and why is it important?",
        options: [
            "Data format",
            "The size of data",
            "Data quality",
            "Tracking the origin, transformations, and movement of data through the pipeline"
        ],
        correct: 3,
        explanation: "Data lineage tracks where data came from, how it was transformed, and where it went. It's important for debugging, auditing, compliance, and reproducibility."
    },
    {
        id: 59,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is a recommended practice for model rollback in production?",
        options: [
            "Only rollback manually",
            "Delete old versions immediately",
            "Never rollback, always move forward",
            "Keep previous versions available and have an automated rollback process"
        ],
        correct: 3,
        explanation: "It's important to keep previous model versions and have an automated rollback process to quickly revert if the new model has problems."
    },
    {
        id: 60,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is 'shadow deployment' (shadow mode)?",
        options: [
            "Deploying with hidden features",
            "Running the new model in parallel with current one, without affecting users, to compare results",
            "Deploying to backup servers",
            "Deploying the model only at night"
        ],
        correct: 1,
        explanation: "Shadow deployment runs the new model in parallel with the current model, processing the same requests, but without returning its results to users. This allows comparing performance in real production without risks."
    }
];

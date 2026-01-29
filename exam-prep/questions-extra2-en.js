// Additional CAIP-210 Questions - Set 3 - English Version
// Based on CertNexus Certified AI Practitioner official course material

const questionsExtra2_en = [
    // More Domain 1 questions
    {
        id: 101,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "What is Computer Vision and what techniques does it include?",
        options: [
            "An AI area that processes visual data (images/videos), including object recognition, motion detection, and image generation",
            "A data visualization technique",
            "A visual debugging technique",
            "A type of computer monitor"
        ],
        correct: 0,
        explanation: "Computer Vision is an umbrella term for techniques that process visual data: object recognition, object classification, image generation, motion detection, trajectory estimation, video tracking, and navigation."
    },
    {
        id: 102,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "What is NLP (Natural Language Processing)?",
        options: [
            "A visualization tool",
            "An AI area where computers work with human languages using ML techniques",
            "A programming language",
            "A type of database"
        ],
        correct: 1,
        explanation: "NLP is the general term for tasks where computers work with human languages using ML. It includes: speech recognition, text analysis, natural language understanding/generation/translation."
    },
    {
        id: 103,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "When should you NOT use AI/ML to solve a problem?",
        options: [
            "When the problem is complex",
            "When the problem can be solved with simple traditional logic or there's insufficient data",
            "When budget is available",
            "When lots of data is available"
        ],
        correct: 1,
        explanation: "AI/ML can be expensive, time-consuming, and risky. If the problem can be solved with simple traditional programming, or if there's insufficient data, AI/ML may not be justifiable."
    },
    {
        id: 104,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "What do 'labels' represent in supervised learning?",
        options: [
            "Data types",
            "Feature names",
            "Column names",
            "The known correct answers (ground truth) used to train the model"
        ],
        correct: 3,
        explanation: "Labels are the known correct answers (ground truth) in a supervised learning dataset. The model learns by comparing its predictions with these labels during training."
    },

    // More Domain 2 questions
    {
        id: 105,
        domain: 2,
        domainName: "Data Preparation",
        question: "What is 'Cold-deck Imputation' for missing values?",
        options: [
            "Leaving missing values as they are",
            "Using values from an external similar dataset to fill missing values",
            "Removing missing values",
            "Using the mean from the same dataset"
        ],
        correct: 1,
        explanation: "Cold-deck imputation uses values from an external similar dataset to fill missing values, unlike hot-deck which uses values from the same dataset."
    },
    {
        id: 106,
        domain: 2,
        domainName: "Data Preparation",
        question: "What is 'Regression Imputation' for missing values?",
        options: [
            "Using a regression model trained on other features to predict the missing value",
            "Replacing with zero",
            "Removing rows with missing values",
            "Using the median"
        ],
        correct: 0,
        explanation: "Regression imputation trains a regression model using other features to predict the missing value. It's more sophisticated than mean/mode imputation but may introduce bias."
    },
    {
        id: 107,
        domain: 2,
        domainName: "Data Preparation",
        question: "Which transformation is used for data with negative skewness?",
        options: [
            "Standardization",
            "Exponential or square transformation",
            "One-hot encoding",
            "Log transformation"
        ],
        correct: 1,
        explanation: "For negative skewness (left tail), exponential or square transformations help. Log transformation is used for positive skewness."
    },
    {
        id: 108,
        domain: 2,
        domainName: "Data Preparation",
        question: "What is 'Box-Cox Transformation'?",
        options: [
            "A parametric transformation that automatically finds the best power transformation to normalize data",
            "A visualization technique",
            "An encoding technique",
            "A feature selection method"
        ],
        correct: 0,
        explanation: "Box-Cox is a parametric transformation that automatically finds the best lambda parameter to transform skewed data into a more normal distribution."
    },
    {
        id: 109,
        domain: 2,
        domainName: "Data Preparation",
        question: "What is 't-SNE' used for?",
        options: [
            "Model training",
            "Category encoding",
            "Dimensionality reduction for visualization, preserving local data structure",
            "Missing value imputation"
        ],
        correct: 2,
        explanation: "t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique primarily used for visualizing high-dimensional data in 2D or 3D."
    },
    {
        id: 110,
        domain: 2,
        domainName: "Data Preparation",
        question: "What is 'Target Encoding' for categorical variables?",
        options: [
            "Replacing categories with the mean of the target for that category",
            "Removing rare categories",
            "Sorting categories alphabetically",
            "Creating dummy columns"
        ],
        correct: 0,
        explanation: "Target encoding replaces each category with the mean of the target variable for that category. Useful for high cardinality, but can cause data leakage if not done correctly."
    },

    // More Domain 3 questions
    {
        id: 111,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is 'Stratified K-Fold Cross Validation'?",
        options: [
            "Validation without folds",
            "Simple cross-validation",
            "Validation only on test set",
            "Cross-validation that maintains the same class proportions in each fold"
        ],
        correct: 3,
        explanation: "Stratified K-Fold ensures that each fold maintains approximately the same class proportions as the original dataset. Important for imbalanced datasets."
    },
    {
        id: 112,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is 'Grid Search' for hyperparameter tuning?",
        options: [
            "Exhaustive search that tests all possible combinations of specified hyperparameters",
            "Random parameter search",
            "Automatic optimization",
            "Manual search"
        ],
        correct: 0,
        explanation: "Grid Search exhaustively tests all combinations of hyperparameters in a specified grid. It's complete but can be computationally expensive."
    },
    {
        id: 113,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is 'Random Search' for hyperparameter tuning?",
        options: [
            "No search, uses defaults",
            "Random sampling of hyperparameter combinations, generally more efficient than grid search",
            "Exhaustive search",
            "Manual search"
        ],
        correct: 1,
        explanation: "Random Search randomly samples hyperparameter combinations. Studies show it's often more efficient than grid search for finding good hyperparameters."
    },
    {
        id: 114,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is the 'AUC-ROC' metric?",
        options: [
            "Accuracy Under Curve",
            "Average User Count",
            "Area under the ROC curve, measures model's ability to distinguish between classes",
            "Automated Utility Check"
        ],
        correct: 2,
        explanation: "AUC-ROC (Area Under the ROC Curve) measures the classifier's ability to distinguish between classes. Value of 0.5 = random, 1.0 = perfect. Useful for imbalanced data."
    },
    {
        id: 115,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is 'Early Stopping' in neural network training?",
        options: [
            "Stopping when data runs out",
            "Stopping after fixed time",
            "Stopping training when validation set performance stops improving, preventing overfitting",
            "Never stopping training"
        ],
        correct: 2,
        explanation: "Early stopping monitors validation set performance and stops training when it stops improving (or starts worsening), preventing overfitting."
    },
    {
        id: 116,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is 'Learning Rate' in gradient descent?",
        options: [
            "Computer speed",
            "Number of epochs",
            "Batch size",
            "The step size taken in the gradient direction at each iteration"
        ],
        correct: 3,
        explanation: "Learning rate determines the step size in the direction opposite to the gradient. Too high may not converge, too low may be slow and get stuck in local minima."
    },
    {
        id: 117,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is 'Ensemble Learning'?",
        options: [
            "Using a single strong model",
            "Training on multiple datasets",
            "Combining multiple models to achieve better performance than any individual model",
            "Using multiple GPUs"
        ],
        correct: 2,
        explanation: "Ensemble learning combines predictions from multiple models (bagging, boosting, stacking) to achieve better generalization and performance than individual models."
    },
    {
        id: 118,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is 'Bagging' (Bootstrap Aggregating)?",
        options: [
            "Training multiple models on bootstrap samples of data and aggregating their predictions",
            "Artificially augmenting data",
            "Training models sequentially",
            "Reducing features"
        ],
        correct: 0,
        explanation: "Bagging trains multiple models independently on different bootstrap samples (with replacement) of data, and aggregates their predictions (vote/average). Random Forest uses bagging."
    },
    {
        id: 119,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is 'Boosting'?",
        options: [
            "Training models in parallel",
            "Compressing models",
            "Training models sequentially, each focusing on the previous one's errors",
            "Increasing training speed"
        ],
        correct: 2,
        explanation: "Boosting trains models sequentially, where each model tries to correct the previous one's errors. Examples: AdaBoost, Gradient Boosting, XGBoost."
    },
    {
        id: 120,
        domain: 3,
        domainName: "Training & Tuning",
        question: "What is 'LSTM' in neural networks?",
        options: [
            "A type of convolutional network",
            "An optimizer",
            "A regularization technique",
            "Long Short-Term Memory - a type of RNN that solves the vanishing gradient problem for long sequences"
        ],
        correct: 3,
        explanation: "LSTM (Long Short-Term Memory) is a type of RNN with memory cells and gates that allow learning long-term dependencies, solving the vanishing gradient problem."
    },

    // More Domain 4 questions
    {
        id: 121,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is 'Infrastructure as Code (IaC)' in MLOps?",
        options: [
            "Machine learning code",
            "Managing and provisioning infrastructure through versioned code (Terraform, CloudFormation)",
            "Test scripts",
            "Code that runs on servers"
        ],
        correct: 1,
        explanation: "IaC allows managing and provisioning all infrastructure (servers, networks, storage) through versioned configuration files, ensuring reproducibility and automation."
    },
    {
        id: 122,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is 'model throughput'?",
        options: [
            "Model accuracy",
            "Training time",
            "The number of requests/predictions the model can process per unit of time",
            "Model size"
        ],
        correct: 2,
        explanation: "Throughput measures how many requests the model can process per second/minute. Important for high-volume systems where many predictions are needed simultaneously."
    },
    {
        id: 123,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is 'model observability'?",
        options: [
            "Monitoring only errors",
            "Ability to understand the model's internal state and behavior in production through metrics and logs",
            "Seeing model predictions",
            "Viewing model code"
        ],
        correct: 1,
        explanation: "Observability is the ability to understand the system's internal state through external outputs. Includes metrics (latency, throughput), logs, and traces for debugging."
    },
    {
        id: 124,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is 'model quantization'?",
        options: [
            "Increasing model complexity",
            "Dividing model into parts",
            "Training with more data",
            "Reducing numerical precision of weights (e.g., float32 to int8) to decrease size and increase speed"
        ],
        correct: 3,
        explanation: "Quantization reduces the numerical precision of weights and activations (e.g., from 32-bit float to 8-bit int), decreasing model size and accelerating inference, with minimal accuracy loss."
    },
    {
        id: 125,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is 'model pruning'?",
        options: [
            "Removing connections/neurons with weights close to zero to reduce model size",
            "Adding more layers",
            "Training longer",
            "Increasing learning rate"
        ],
        correct: 0,
        explanation: "Pruning removes weights, neurons, or layers that contribute little to prediction (generally with values close to zero), reducing model size without losing much accuracy."
    },
    {
        id: 126,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is 'knowledge distillation' in model compression?",
        options: [
            "Documenting model knowledge",
            "Extracting features from model",
            "Training a smaller model (student) to mimic a larger model (teacher)",
            "Training a model from scratch"
        ],
        correct: 2,
        explanation: "Knowledge distillation trains a smaller model (student) to reproduce the behavior of a larger, more complex model (teacher), transferring 'knowledge' to a more efficient model."
    },
    {
        id: 127,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What are 'SLAs' (Service Level Agreements) for models in production?",
        options: [
            "Automation scripts",
            "System logs",
            "ML algorithms",
            "Formal agreements about expected service levels (latency, uptime, accuracy)"
        ],
        correct: 3,
        explanation: "SLAs are formal agreements that define expected service levels: maximum response time, minimum availability, acceptable error rate, etc. Crucial for production applications."
    },
    {
        id: 128,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is 'edge deployment' for ML models?",
        options: [
            "Deployment in test environment",
            "Gradual deployment",
            "Deployment in datacenters",
            "Deploying models on local devices (smartphones, IoT) instead of cloud servers"
        ],
        correct: 3,
        explanation: "Edge deployment runs the model directly on local devices (edge devices), reducing latency, network costs, and enabling offline operation. Requires optimized models."
    }
];

// CAIP-210 Questions - Extra Set 4 (English Version)
// Domain 1 (AI & ML Fundamentals) and Domain 4 (MLOps & Production)
// Created to balance question distribution

const questionsExtra4_en = [
    // ========================================
    // DOMAIN 1: AI & ML FUNDAMENTALS (10 questions)
    // ========================================

    {
        id: 151,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "A retail company wants to predict product demand to optimize inventory. Which type of ML problem is most appropriate and why?",
        options: [
            "Classification, because it needs to categorize products into high or low demand",
            "Regression, because demand is a continuous numerical value that needs to be predicted",
            "Clustering, because it needs to group similar products",
            "Reinforcement learning, because it needs to learn from customer feedback"
        ],
        correct: 1,
        explanation: "Demand forecasting is a regression problem because the goal is to predict a continuous numerical value (product quantity). Classification would be used if the goal was only to categorize as 'high' or 'low' demand."
    },
    {
        id: 152,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "What is the main difference between supervised and unsupervised learning?",
        options: [
            "Supervised uses more data than unsupervised",
            "Supervised requires labeled data while unsupervised discovers patterns without labels",
            "Unsupervised is always more accurate",
            "Supervised only works with numerical data"
        ],
        correct: 1,
        explanation: "The fundamental difference is that supervised learning requires labeled data (examples with known answers) for training, while unsupervised learning discovers patterns and structures in data without pre-defined labels."
    },
    {
        id: 153,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "A fintech wants to detect fraudulent transactions in real-time. Which ethical consideration is MOST critical in this scenario?",
        options: [
            "Maximize company profit above all",
            "Ensure the model doesn't discriminate against specific customer groups and is transparent in decisions",
            "Use as much personal data as possible for better accuracy",
            "Keep the model completely secret for security"
        ],
        correct: 1,
        explanation: "In financial applications, it's critical to avoid discriminatory bias (fairness) and maintain transparency in decisions affecting customers. This includes not discriminating by race, gender, location, etc., and allowing decisions to be explainable and contestable."
    },
    {
        id: 154,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Which business metric would be most appropriate to evaluate a product recommendation system in e-commerce?",
        options: [
            "Only model accuracy",
            "Conversion rate and revenue per user increase",
            "Only model training time",
            "Number of features used"
        ],
        correct: 1,
        explanation: "Business metrics like conversion rate and revenue per user are more relevant than isolated technical metrics. A model with high technical accuracy may not generate business value if it doesn't increase sales or engagement."
    },
    {
        id: 155,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "In an ML project for medical diagnosis, what is the biggest ethical risk to mitigate?",
        options: [
            "High computational cost",
            "Bias in training data that can lead to incorrect diagnoses for underrepresented groups",
            "Slow inference time",
            "Code complexity"
        ],
        correct: 1,
        explanation: "In healthcare, data bias can result in incorrect diagnoses or inadequate treatments for underrepresented groups (e.g., ethnic minorities, women). This can have serious consequences for patient health and perpetuate inequalities."
    },
    {
        id: 156,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Which type of machine learning is most suitable for training an agent that learns to play chess through trial and error?",
        options: [
            "Supervised learning with historical game dataset",
            "Unsupervised learning to discover patterns",
            "Reinforcement learning where the agent receives rewards for good moves",
            "Semi-supervised learning"
        ],
        correct: 2,
        explanation: "Reinforcement learning is ideal for scenarios where an agent learns through interaction with an environment, receiving rewards (wins) or penalties (losses). The agent learns the optimal policy through trial and error."
    },
    {
        id: 157,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "A startup wants to use ML to automate resume screening. What is the main bias risk to consider?",
        options: [
            "The model might be too slow",
            "The model might perpetuate historical hiring biases, discriminating against candidates by gender, race, or age",
            "The model might use too much memory",
            "The model might need too much data"
        ],
        correct: 1,
        explanation: "If historical hiring data contains biases (e.g., preference for certain gender or race), the model will learn and perpetuate these biases. It's essential to audit data and model to ensure fairness and compliance with anti-discrimination laws."
    },
    {
        id: 158,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "What is the difference between binary classification and multiclass classification?",
        options: [
            "Binary uses only 2 features, multiclass uses more",
            "Binary predicts between 2 classes (e.g., spam/not-spam), multiclass predicts between 3+ classes (e.g., flower type)",
            "Multiclass is always more accurate",
            "Binary only works with numerical data"
        ],
        correct: 1,
        explanation: "Binary classification predicts between two possible classes (yes/no, true/false), while multiclass classification predicts between three or more classes (e.g., classifying animal type: cat, dog, bird, etc.)."
    },
    {
        id: 159,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "When formulating an ML problem, what is the most important first step?",
        options: [
            "Choose the most complex algorithm available",
            "Clearly define the business objective and how success will be measured",
            "Collect as much data as possible",
            "Implement the model immediately"
        ],
        correct: 1,
        explanation: "Correct problem formulation starts with clearly defining the business objective and success metrics. This guides all subsequent decisions about data, features, algorithms, and evaluation."
    },
    {
        id: 160,
        domain: 1,
        domainName: "AI & ML Fundamentals",
        question: "Which scenario is a classic example of unsupervised learning?",
        options: [
            "Predict house prices based on characteristics",
            "Classify emails as spam or not-spam",
            "Segment customers into groups with similar behaviors without pre-defined categories",
            "Predict whether a patient has a disease or not"
        ],
        correct: 2,
        explanation: "Customer segmentation (clustering) is unsupervised learning because there are no pre-defined categories. The algorithm discovers natural groups in the data based on similarities. The other examples are supervised (regression or classification)."
    },

    // ========================================
    // DOMAIN 4: MLOps & PRODUCTION (15 questions)
    // ========================================

    {
        id: 161,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Which deployment strategy allows testing a new model with a small percentage of users before full rollout?",
        options: [
            "Big bang deployment",
            "Canary deployment, where the new model is gradually exposed to an increasing percentage of traffic",
            "Rollback deployment",
            "Batch deployment"
        ],
        correct: 1,
        explanation: "Canary deployment exposes the new model to a small percentage of traffic initially (e.g., 5%), monitors performance, and gradually increases if everything is working well. This minimizes risk of production failures."
    },
    {
        id: 162,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is model drift and why is it important to monitor it?",
        options: [
            "Syntax error in model code",
            "Model performance degradation over time due to changes in production data",
            "Increase in inference time",
            "Memory problem on server"
        ],
        correct: 1,
        explanation: "Model drift occurs when the distribution of production data changes relative to training data, causing performance degradation. Monitoring drift is essential to know when to retrain the model."
    },
    {
        id: 163,
        domain: 4,
        domainName: "MLOps & Production",
        question: "In a CI/CD pipeline for ML, which step is ESSENTIAL before deploying to production?",
        options: [
            "Only check if code compiles",
            "Run automated tests including model performance validation on test data",
            "Deploy directly without tests",
            "Only check if model trains without errors"
        ],
        correct: 1,
        explanation: "CI/CD for ML must include automated tests that validate not only code, but also model performance (accuracy, latency, etc.) on test data before allowing production deployment."
    },
    {
        id: 164,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is the main advantage of using a model registry in MLOps?",
        options: [
            "Only store models",
            "Centralized model versioning, metadata, and traceability of which models are in production",
            "Train models faster",
            "Reduce infrastructure cost"
        ],
        correct: 1,
        explanation: "Model registry provides centralized versioning, metadata storage (metrics, hyperparameters), lineage traceability, and control of which versions are in staging/production. Essential for governance and reproducibility."
    },
    {
        id: 165,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Which metric is most important to monitor in production to detect data drift?",
        options: [
            "Only CPU usage",
            "Statistical distribution of input features compared to training data",
            "Only response time",
            "Number of requests"
        ],
        correct: 1,
        explanation: "Data drift is detected by comparing the statistical distribution of input features in production with training data distribution. Tests like Kolmogorov-Smirnov or chi-square can identify significant changes."
    },
    {
        id: 166,
        domain: 4,
        domainName: "MLOps & Production",
        question: "In a Blue-Green deployment strategy, what is the main benefit?",
        options: [
            "Use fewer servers",
            "Allow instant rollback to previous version in case of problems",
            "Train models faster",
            "Reduce storage cost"
        ],
        correct: 1,
        explanation: "Blue-Green deployment maintains two complete environment versions (blue = current, green = new). Traffic is redirected to green after validation. If there are problems, rollback to blue is instant, minimizing downtime."
    },
    {
        id: 167,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Which tool is essential for tracking ML experiments (hyperparameters, metrics, artifacts)?",
        options: [
            "Only Excel spreadsheets",
            "Experiment tracking platform like MLflow, Weights & Biases, or Neptune",
            "Only text logs",
            "No need to track experiments"
        ],
        correct: 1,
        explanation: "Experiment tracking tools like MLflow, W&B, or Neptune allow tracking hyperparameters, metrics, code, data, and artifacts in a structured way, facilitating experiment comparison and reproducibility."
    },
    {
        id: 168,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is the purpose of a feature store in MLOps architecture?",
        options: [
            "Only store raw data",
            "Centralize, version, and serve processed features for training and inference, ensuring consistency",
            "Only backup models",
            "Replace traditional databases"
        ],
        correct: 1,
        explanation: "Feature store centralizes engineered features, ensures consistency between training and inference (avoiding training-serving skew), enables feature reuse, and provides versioning and lineage."
    },
    {
        id: 169,
        domain: 4,
        domainName: "MLOps & Production",
        question: "When deploying a model to production, which security consideration is CRITICAL?",
        options: [
            "Only use HTTPS",
            "Implement authentication, authorization, data encryption, and protection against adversarial attacks",
            "No need to worry about security",
            "Only use firewall"
        ],
        correct: 1,
        explanation: "ML security includes: API authentication/authorization, data encryption in transit and at rest, protection against adversarial attacks (malicious inputs), and compliance with regulations (GDPR, LGPD)."
    },
    {
        id: 170,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is the difference between batch inference and real-time inference?",
        options: [
            "No difference",
            "Batch processes multiple predictions at once (e.g., daily), real-time responds to individual requests immediately",
            "Batch is always more accurate",
            "Real-time uses fewer resources"
        ],
        correct: 1,
        explanation: "Batch inference processes large data volumes periodically (e.g., daily demand forecasts). Real-time inference responds to individual requests with low latency (e.g., fraud detection in transactions). Each has latency, throughput, and cost trade-offs."
    },
    {
        id: 171,
        domain: 4,
        domainName: "MLOps & Production",
        question: "Which SLA (Service Level Agreement) metric is most critical for an ML model in production?",
        options: [
            "Only server uptime",
            "Combination of latency (response time), throughput (requests/second), and model accuracy",
            "Only infrastructure cost",
            "Only number of features"
        ],
        correct: 1,
        explanation: "ML SLA should include: latency (e.g., p95 < 100ms), throughput (e.g., 1000 req/s), uptime (e.g., 99.9%), and model performance (e.g., accuracy > 90%). All are critical to ensure service quality."
    },
    {
        id: 172,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is A/B testing in the context of ML model deployment?",
        options: [
            "Test two algorithms during training",
            "Expose different model versions to different user groups and compare business performance",
            "Test only in development environment",
            "Not applicable to ML"
        ],
        correct: 1,
        explanation: "A/B testing exposes different model versions (A = control, B = new) to random user groups in production, allowing statistically rigorous comparison of business metrics (conversion, revenue) before deciding which version to keep."
    },
    {
        id: 173,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is the purpose of continuous model monitoring in production?",
        options: [
            "Only check if server is online",
            "Detect performance degradation, drift, anomalies, and trigger retraining when necessary",
            "Only save logs",
            "No need to monitor after deployment"
        ],
        correct: 1,
        explanation: "Continuous monitoring detects: model drift (data changes), concept drift (feature-target relationship changes), performance degradation, anomalies, and triggers alerts or automatic retraining. Essential to maintain quality over time."
    },
    {
        id: 174,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is the best practice for retraining models in production?",
        options: [
            "Never retrain",
            "Retrain periodically (schedule-based) or when drift/degradation is detected (trigger-based)",
            "Only retrain when model completely fails",
            "Retrain randomly"
        ],
        correct: 1,
        explanation: "Retraining can be: schedule-based (e.g., monthly) for data with seasonality, or trigger-based (when drift/degradation is detected). The choice depends on data nature and application criticality."
    },
    {
        id: 175,
        domain: 4,
        domainName: "MLOps & Production",
        question: "What is the role of observability in ML systems in production?",
        options: [
            "Only collect logs",
            "Provide complete system visibility through logs, metrics, traces, and dashboards for debugging and optimization",
            "Only monitor CPU",
            "Not necessary in ML"
        ],
        correct: 1,
        explanation: "ML observability includes: structured logs, model and infrastructure metrics, distributed tracing, dashboards, and alerts. Enables understanding system behavior, debugging problems, and optimizing performance."
    }
];

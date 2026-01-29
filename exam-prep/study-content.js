// CAIP-210 Complete Study Content - Based on official CertNexus eBook
// Reorganized to match the 12 lessons from the course

const STUDY_CONTENT = {
    // LESSON 1: Solving Business Problems Using AI and ML
    1: {
        name: "Lesson 1: Solving Business Problems Using AI and ML",
        icon: "💼",
        weight: "Focus Areas: Business Context, Problem Formulation",
        topics: [
            {
                title: "AI and Machine Learning Overview",
                concept: `Machine Learning is a discipline of Artificial Intelligence (AI) where computers make predictions and decisions based on data, without explicit human instructions.

Key differences from traditional programming:
• Traditional: Programmer writes explicit rules
• ML: System LEARNS patterns from data

The Data Hierarchy (DIK):
📊 DATA → Raw facts without context
📈 INFORMATION → Organized, contextualized data  
💡 KNOWLEDGE → Actionable intelligence
🧠 WISDOM → Experience-based decision making

Machine Learning builds on itself over time through experience, becoming more effective at solving problems. Deep Learning is a subset using complex artificial neural networks.

An Algorithm is a set of rules for solving problems - a mathematical formula that takes input and produces output. Algorithms support "learning" by updating beliefs about data over time.`,
                keyPoints: [
                    "ML makes predictions without explicit programming",
                    "Deep Learning uses complex artificial neural networks",
                    "DIK hierarchy transforms raw data into knowledge",
                    "Algorithms are mathematical formulas that learn from data",
                    "Data Science encompasses ML but also includes broader analysis"
                ],
                example: `# Traditional Programming vs Machine Learning

# TRADITIONAL - Explicit rules
def is_spam_traditional(email):
    spam_words = ["free", "winner", "click here"]
    for word in spam_words:
        if word in email.lower():
            return True
    return False

# MACHINE LEARNING - Learn from examples
from sklearn.naive_bayes import MultinomialNB

# Model learns patterns from labeled examples
model = MultinomialNB()
model.fit(X_train_vectorized, y_train_labels)
prediction = model.predict(new_email_vectorized)`,
                realCase: {
                    title: "DIK Hierarchy in Healthcare",
                    description: "Hospital sensors collect thousands of DATA points (heart rate, blood pressure). These become INFORMATION when combined ('Patient X vitals trending abnormally'). This becomes KNOWLEDGE when AI predicts 'Patient likely to experience cardiac event in 6 hours - alert medical team'.",
                    impact: "Early warning systems have reduced cardiac arrests by 30-40% in hospitals using AI monitoring"
                }
            },
            {
                title: "AI/ML Applications by Sector",
                concept: `AI and ML solve different types of problems across sectors:

🏢 COMMERCIAL PROBLEMS:
• Poor sales growth → Lead scoring, automated prospecting
• Low customer retention → Personalized recommendations
• Inventory management → Demand forecasting
• Quality control → Visual inspection with computer vision
• Employee turnover → Attrition prediction

🏛️ GOVERNMENTAL PROBLEMS:
• Public health crises → Outbreak prediction, vaccine logistics
• Security breaches → Anomaly detection, intrusion prevention
• Economic planning → Recession forecasting, policy simulation
• Tax fraud → Anomaly detection in filings

🌍 PUBLIC INTEREST PROBLEMS:
• Education → Personalized learning paths
• Crime → Predictive policing, resource allocation
• Climate change → Impact prediction, mitigation strategies

🔬 RESEARCH PROBLEMS:
• Peer review → Automated literature assessment
• Bias detection → Meta-analysis of studies
• Research gaps → Identifying understudied areas`,
                keyPoints: [
                    "Commercial: Focus on revenue, efficiency, customer value",
                    "Government: Focus on public safety, policy, resource optimization",
                    "Public interest: Focus on societal benefits",
                    "Research: Focus on accelerating knowledge discovery",
                    "Same ML techniques apply across different domains"
                ],
                example: `# Example: Customer Churn Prediction (Commercial)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Features that might predict churn
features = ['tenure', 'monthly_charges', 'total_charges',
            'contract_type', 'payment_method', 'support_tickets']

# Train model to identify at-risk customers
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train[features], y_train)

# Predict which customers will churn
at_risk = model.predict_proba(X_current[features])[:, 1]
high_risk_customers = customers[at_risk > 0.7]`,
                realCase: {
                    title: "Predictive Maintenance at Rolls-Royce",
                    description: "Rolls-Royce uses AI to analyze sensor data from aircraft engines in real-time. The system predicts component failures before they occur, allowing proactive maintenance scheduling.",
                    impact: "Reduced unplanned maintenance events by 25%, saving airlines millions in downtime costs"
                }
            },
            {
                title: "Stakeholders in ML Projects",
                concept: `Project Stakeholder = Person with vested interest in project outcome or actively involved in its work.

👥 KEY STAKEHOLDERS:

• CUSTOMERS/END USERS
  - Use the product/service generated
  - Provide payment and usability feedback
  
• SPONSORS/CHAMPIONS
  - Provide finances and management support
  - Control overall project direction
  
• PROJECT MANAGERS
  - Manage timelines and resources
  - Coordinate team activities
  
• TEAM MEMBERS
  - Data scientists, ML engineers, developers
  - Provide technical expertise
  
• BUSINESS PARTNERS
  - May sell product or provide resources
  - Bring external perspective
  
• GOVERNMENT
  - Provide regulatory frameworks
  - Relevant for large-scope projects

📋 KEY RESPONSIBILITIES:
• Investigate user requirements feasibility
• Consult with domain experts
• Participate in AI/ML community
• Communicate results to audiences
• Address ethical risks`,
                keyPoints: [
                    "Stakeholders have competing interests and priorities",
                    "Communication strategy essential for project success",
                    "Requirements must be gathered early and validated",
                    "Technical feasibility must align with business needs",
                    "Ethical risks require stakeholder awareness"
                ],
                example: `# Stakeholder Communication Matrix

stakeholders = {
    "Executives": {
        "interest": "ROI, strategic alignment",
        "frequency": "Monthly summaries",
        "format": "Dashboard, KPIs"
    },
    "Data Science Team": {
        "interest": "Technical accuracy, model performance",
        "frequency": "Daily/weekly standups",
        "format": "Notebooks, code reviews"
    },
    "End Users": {
        "interest": "Usability, reliability",
        "frequency": "Beta testing, feedback loops",
        "format": "Surveys, interviews"
    },
    "Legal/Compliance": {
        "interest": "Regulatory compliance, risk",
        "frequency": "Milestone reviews",
        "format": "Documentation, audits"
    }
}`,
                realCase: {
                    title: "IBM Watson Health Stakeholder Misalignment",
                    description: "Watson for Oncology struggled partly due to stakeholder misalignment. Doctors didn't trust AI recommendations, training data was limited, and business expectations were unrealistic about what AI could deliver.",
                    impact: "Project was discontinued at multiple hospitals, highlighting importance of stakeholder alignment"
                }
            },
            {
                title: "Ethical Risks in AI/ML",
                concept: `AI/ML projects carry inherent ethical risks that must be managed:

🔒 PRIVACY
• Protects sensitive data from unauthorized access
• Personal data use must be transparent and consented
• GDPR, HIPAA and other regulations apply

⚖️ ACCOUNTABILITY
• Who is responsible when AI makes wrong decisions?
• Challenge: machines make consequential decisions
• Need clear ownership and audit trails

🔍 TRANSPARENCY & EXPLAINABILITY
• Can people understand how AI decides?
• "Black box" models are difficult to explain
• Explainable AI (XAI) increasingly important

🤝 FAIRNESS & NON-DISCRIMINATION
• AI can perpetuate or amplify existing biases
• Training data may contain historical discrimination
• Must test for disparate impact across groups

🛡️ SAFETY & SECURITY
• Minimize chance of physical harm
• Protect against adversarial attacks
• Ensure reliability in critical systems`,
                keyPoints: [
                    "Privacy: How personal data is collected and used",
                    "Accountability: Clear responsibility for AI decisions",
                    "Transparency: Explainable decision-making processes",
                    "Fairness: Equal treatment across demographic groups",
                    "Safety: Minimize potential for harm"
                ],
                example: `# Fairness Check Example
from sklearn.metrics import confusion_matrix

def check_fairness(y_true, y_pred, sensitive_attribute):
    """Check if model treats groups fairly"""
    groups = sensitive_attribute.unique()
    results = {}
    
    for group in groups:
        mask = sensitive_attribute == group
        tn, fp, fn, tp = confusion_matrix(
            y_true[mask], y_pred[mask]
        ).ravel()
        
        # Calculate acceptance rate
        results[group] = {
            'acceptance_rate': (tp + fp) / len(y_pred[mask]),
            'false_positive_rate': fp / (fp + tn),
            'false_negative_rate': fn / (fn + tp)
        }
    
    # Check for disparate impact
    rates = [r['acceptance_rate'] for r in results.values()]
    disparate_impact = min(rates) / max(rates)
    print(f"Disparate Impact Ratio: {disparate_impact:.2f}")
    print("(Should be > 0.8 for fairness)")
    
    return results`,
                realCase: {
                    title: "Amazon AI Recruiting Tool Bias",
                    description: "Amazon built an AI recruiting tool that systematically penalized resumes containing words like 'women's' (as in 'women's chess club captain'). The system learned from 10 years of hiring data, which was predominantly male.",
                    impact: "Project was scrapped. Now a case study in how historical bias in training data leads to discriminatory AI systems."
                }
            },
            {
                title: "Formulating ML Problems",
                concept: `Converting business problems into well-defined ML problems:

📋 PROBLEM FORMULATION PROCESS:

1. IDENTIFY BUSINESS OBJECTIVE
   • What outcome does the business want?
   • How will success be measured?

2. DEFINE ML TASK TYPE
   • Classification: Categorize into classes
   • Regression: Predict continuous value
   • Clustering: Find natural groupings
   • Ranking: Order items by relevance

3. DETERMINE TARGET VARIABLE
   • What exactly will the model predict?
   • Is this measurable and available in data?

4. IDENTIFY FEATURES
   • What inputs will help predict the target?
   • Are these available at prediction time?

5. DEFINE SUCCESS CRITERIA
   • What accuracy/performance is acceptable?
   • What are the costs of different errors?

6. CONSIDER CONSTRAINTS
   • Latency requirements
   • Interpretability needs
   • Data privacy restrictions`,
                keyPoints: [
                    "Start with business objective, not ML technique",
                    "Target variable must be measurable and available",
                    "Features must be available at prediction time",
                    "Different error types have different business costs",
                    "Constraints shape which solutions are viable"
                ],
                example: `# Problem Formulation Example

business_problem = "Too many customers are canceling subscriptions"

# Step 1: Define ML problem
ml_problem = {
    "type": "Binary Classification",
    "target": "Will customer cancel in next 30 days? (0/1)",
    "features": [
        "tenure_months",
        "monthly_charges",
        "support_tickets_last_90_days",
        "usage_decline_pct",
        "contract_type",
        "payment_method"
    ],
    "success_metric": "Recall > 0.80 (catch 80% of churners)",
    "constraints": [
        "Predictions needed daily for 1M customers",
        "Model must be explainable to customer success team",
        "Cannot use protected attributes (age, gender)"
    ]
}

# Step 2: Evaluate tradeoffs
# High recall = catch more churners but more false positives
# High precision = fewer false alarms but miss some churners
# Business decision: better to contact non-churner than miss churner`,
                realCase: {
                    title: "Netflix Problem Formulation",
                    description: "Netflix doesn't just predict 'will user like this movie?' They formulated the problem as 'what should we show in each row position on the homepage to maximize engagement?' This reframing led to personalized row ordering.",
                    impact: "Users find content faster, reducing churn and increasing viewing time"
                }
            },
            {
                title: "Approaches to Machine Learning",
                concept: `Three main learning paradigms:

🎓 SUPERVISED LEARNING
• Training data includes labels (correct answers)
• Model learns to predict labels for new data
• Types: Classification, Regression
• Examples: Spam detection, price prediction

🔍 UNSUPERVISED LEARNING
• Training data has NO labels
• Model discovers patterns/structure
• Types: Clustering, Dimensionality Reduction
• Examples: Customer segmentation, anomaly detection

🔄 SEMI-SUPERVISED LEARNING
• Small amount of labeled data
• Large amount of unlabeled data
• Combines benefits of both approaches
• Useful when labeling is expensive

🎮 REINFORCEMENT LEARNING
• Agent learns through trial and error
• Receives rewards/penalties for actions
• Learns optimal strategy over time
• Examples: Game AI, robotics, autonomous vehicles

Each approach has different data requirements, computational costs, and suitable use cases.`,
                keyPoints: [
                    "Supervised: Need labeled data, most common in business",
                    "Unsupervised: Discover hidden patterns without labels",
                    "Semi-supervised: Leverage small labeled + large unlabeled data",
                    "Reinforcement: Learn through interaction with environment",
                    "Choice depends on data availability and problem type"
                ],
                example: `# Supervised vs Unsupervised Example

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# SUPERVISED - We have labels
# Predict if transaction is fraud (labeled data)
X_train = transactions[['amount', 'time', 'location_score']]
y_train = transactions['is_fraud']  # 0 or 1 labels

supervised_model = RandomForestClassifier()
supervised_model.fit(X_train, y_train)
predictions = supervised_model.predict(X_new)

# UNSUPERVISED - No labels
# Group customers into segments (no predefined groups)
X_customers = customers[['recency', 'frequency', 'monetary']]

unsupervised_model = KMeans(n_clusters=4)
unsupervised_model.fit(X_customers)
segments = unsupervised_model.labels_  # Discovered clusters`,
                realCase: {
                    title: "DeepMind AlphaGo - Reinforcement Learning",
                    description: "AlphaGo used reinforcement learning to master the game of Go. It played millions of games against itself, learning strategies through reward signals for winning games.",
                    impact: "Defeated world champion Lee Sedol in 2016, previously thought impossible for AI"
                }
            }
        ]
    },

    // LESSON 2: Preparing Data
    2: {
        name: "Lesson 2: Preparing Data",
        icon: "🔧",
        weight: "Focus Areas: Collection, Transformation, Feature Engineering",
        topics: [
            {
                title: "Data Collection",
                concept: `Understanding data sources and quality considerations:

📊 DATA SOURCES:
• Internal databases (CRM, ERP, logs)
• External APIs (weather, economic, social)
• Web scraping (with legal considerations)
• Third-party data providers
• IoT sensors and devices
• User-generated content

📋 DATA QUALITY DIMENSIONS:
• COMPLETENESS: Are all required values present?
• ACCURACY: Are values correct?
• CONSISTENCY: Do values agree across sources?
• TIMELINESS: Is data sufficiently current?
• VALIDITY: Do values follow business rules?
• UNIQUENESS: Are there duplicates?

⚠️ COMMON ISSUES:
• Missing values (nulls, blanks)
• Duplicate records
• Incorrect data types
• Inconsistent formatting
• Outliers and anomalies
• Stale or outdated data`,
                keyPoints: [
                    "Data quality impacts model quality significantly",
                    "Multiple sources require integration and reconciliation",
                    "Legal and privacy constraints affect data availability",
                    "Data profiling should precede any modeling",
                    "Document data lineage for reproducibility"
                ],
                example: `import pandas as pd
import numpy as np

def data_quality_report(df):
    """Generate comprehensive data quality report"""
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1e6
    }
    
    column_stats = []
    for col in df.columns:
        stats = {
            'column': col,
            'dtype': str(df[col].dtype),
            'missing_count': df[col].isna().sum(),
            'missing_pct': df[col].isna().mean() * 100,
            'unique_values': df[col].nunique(),
            'sample_values': df[col].dropna().head(3).tolist()
        }
        column_stats.append(stats)
    
    report['columns'] = pd.DataFrame(column_stats)
    return report

# Usage
quality = data_quality_report(df)
print(f"Duplicate rows: {quality['duplicate_rows']}")
print(quality['columns'][['column', 'missing_pct', 'unique_values']])`,
                realCase: {
                    title: "NASA Mars Climate Orbiter Failure",
                    description: "The Mars Climate Orbiter was lost due to a unit conversion error - one team used metric, another used imperial. This $327M lesson shows how data consistency issues can have catastrophic consequences.",
                    impact: "Led to mandatory unit verification processes and highlighted importance of data validation"
                }
            },
            {
                title: "Handling Missing Data",
                concept: `Missing data must be addressed before training models:

❌ DELETION METHODS:
• Listwise deletion: Remove entire row if ANY value missing
• Pairwise deletion: Use available data for each analysis
• Column deletion: Remove column if >70% missing

📊 IMPUTATION METHODS:

SIMPLE IMPUTATION:
• Mean: Replace with column average (numeric)
• Median: Replace with middle value (robust to outliers)
• Mode: Replace with most frequent (categorical)
• Constant: Replace with specific value

ADVANCED IMPUTATION:
• Hot-deck: Copy from similar record in dataset
• Cold-deck: Copy from external source
• Regression: Predict missing values using other features
• KNN: Use k-nearest neighbors to impute
• Multiple imputation: Create multiple datasets with different imputations

⚠️ CONSIDERATIONS:
• Understand WHY data is missing (MCAR, MAR, MNAR)
• Simple imputation can distort distributions
• Advanced methods preserve relationships better
• Document imputation decisions for reproducibility`,
                keyPoints: [
                    "Never ignore missing data - it will affect model",
                    "Mean imputation reduces variance artificially",
                    "Hot-deck maintains natural data variability",
                    "Multiple imputation provides uncertainty estimates",
                    "Choice depends on missing data mechanism"
                ],
                example: `import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

# Sample data with missing values
df = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 50, np.nan],
    'income': [50000, np.nan, 60000, 75000, np.nan, 90000],
    'city': ['NYC', 'LA', 'NYC', np.nan, 'LA', 'NYC']
})

# 1. Simple Imputation - Mean for numeric
num_imputer = SimpleImputer(strategy='mean')
df['age_imputed'] = num_imputer.fit_transform(df[['age']])

# 2. Mode imputation for categorical
df['city_imputed'] = df['city'].fillna(df['city'].mode()[0])

# 3. KNN Imputation - Uses similar rows
knn_imputer = KNNImputer(n_neighbors=3)
df[['age_knn', 'income_knn']] = knn_imputer.fit_transform(
    df[['age', 'income']]
)

# 4. Check remaining missing
print(df.isna().sum())`,
                realCase: {
                    title: "Medical Records Imputation",
                    description: "Healthcare datasets often have missing values due to incomplete medical histories. Researchers use multiple imputation with chained equations (MICE) to handle missing lab values while accounting for uncertainty.",
                    impact: "Enables analysis of incomplete patient data while maintaining statistical validity"
                }
            },
            {
                title: "Data Transformation",
                concept: `Transforming data to improve model performance:

📏 FEATURE SCALING:

NORMALIZATION (Min-Max):
• Scales to [0, 1] range
• x' = (x - min) / (max - min)
• Sensitive to outliers
• Good for: neural networks, distance-based algorithms

STANDARDIZATION (Z-Score):
• Scales to mean=0, std=1
• x' = (x - μ) / σ
• Less sensitive to outliers
• Good for: linear models, SVM

📊 DATA ENCODING:

For Categorical Variables:
• Label Encoding: 0, 1, 2... (ordinal only!)
• One-Hot Encoding: Binary columns per category
• Binary Encoding: Binary representation
• Target Encoding: Mean of target per category

📈 DISTRIBUTION TRANSFORMS:
• Log transform: For right-skewed data
• Square root: For count data
• Box-Cox: Automatic power transform
• Quantile: To uniform or normal distribution`,
                keyPoints: [
                    "k-NN, SVM, neural networks REQUIRE scaling",
                    "Tree-based methods do NOT require scaling",
                    "Use training set parameters on test data",
                    "Label encoding implies order - use carefully",
                    "One-hot can create many columns (curse of dimensionality)"
                ],
                example: `from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                       LabelEncoder, OneHotEncoder)
import pandas as pd

# SCALING
scaler_std = StandardScaler()
scaler_mm = MinMaxScaler()

X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)  # Use same params!

X_train_mm = scaler_mm.fit_transform(X_train)
X_test_mm = scaler_mm.transform(X_test)

# ENCODING
# Ordinal: Use mapping
size_map = {'S': 0, 'M': 1, 'L': 2, 'XL': 3}
df['size_encoded'] = df['size'].map(size_map)

# Nominal: One-Hot
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True)

# Or with sklearn
encoder = OneHotEncoder(sparse=False, drop='first')
color_encoded = encoder.fit_transform(df[['color']])`,
                realCase: {
                    title: "Recommendation System Scaling Issue",
                    description: "An e-commerce site had features 'price' (0-10000) and 'rating' (1-5) for product similarity. Without scaling, price dominated similarity calculations, recommending products by price rather than preference.",
                    impact: "After proper scaling, recommendation relevance improved 35%"
                }
            },
            {
                title: "Feature Engineering",
                concept: `Creating new features to improve model performance:

🔧 FEATURE ENGINEERING TECHNIQUES:

📅 TEMPORAL FEATURES:
• Day of week, month, quarter, year
• Is weekend, is holiday
• Time since last event
• Rolling statistics (moving average)
• Lag features (previous values)

📊 AGGREGATION FEATURES:
• Count, sum, mean, median by group
• Min, max, range, std
• Percentiles, mode

🔢 MATHEMATICAL TRANSFORMS:
• Interactions: A × B, A / B
• Polynomial: A², A³
• Binning: Age groups, income brackets

🔤 TEXT FEATURES:
• Word count, character count
• Sentiment scores
• TF-IDF vectors
• Embeddings

🎯 DOMAIN-SPECIFIC:
• RFM (Recency, Frequency, Monetary)
• Customer lifetime value
• Industry-specific ratios`,
                keyPoints: [
                    "Good features often matter more than algorithm choice",
                    "Domain knowledge is crucial for effective features",
                    "Avoid data leakage - don't use future information",
                    "Test feature importance before deploying",
                    "Document feature creation logic"
                ],
                example: `import pandas as pd
import numpy as np

def create_ecommerce_features(df):
    """Create features for customer churn prediction"""
    
    # Temporal features
    df['day_of_week'] = df['purchase_date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['purchase_date'].dt.month
    
    # RFM Features (per customer)
    customer_features = df.groupby('customer_id').agg({
        'purchase_date': lambda x: (pd.Timestamp.now() - x.max()).days,  # Recency
        'order_id': 'nunique',  # Frequency
        'amount': ['sum', 'mean', 'std']  # Monetary
    }).reset_index()
    
    customer_features.columns = ['customer_id', 'recency', 'frequency',
                                  'monetary_sum', 'monetary_mean', 'monetary_std']
    
    # Interaction features
    customer_features['avg_order_value'] = (
        customer_features['monetary_sum'] / customer_features['frequency']
    )
    
    # Trend feature (is spending increasing?)
    # ... additional logic
    
    return customer_features`,
                realCase: {
                    title: "Airbnb Feature Engineering",
                    description: "Airbnb engineers created thousands of features for search ranking: listing characteristics, host behavior patterns, seasonal trends, local events, review sentiment. Feature engineering contributed more to performance than model complexity.",
                    impact: "80% of ranking improvements came from better features, not better algorithms"
                }
            },
            {
                title: "Working with Unstructured Data",
                concept: `Processing images, text, and audio for ML:

🖼️ IMAGE DATA:
• Represented as pixel arrays (height × width × channels)
• Preprocessing: resize, normalize, augment
• Techniques: convolutions, pooling, transfer learning
• Common formats: RGB (3 channels), grayscale (1 channel)

📝 TEXT DATA:
• Tokenization: Split into words/subwords
• Cleaning: Remove punctuation, lowercase, stop words
• Stemming: Reduce to word stems
• Lemmatization: Reduce to dictionary form
• Encoding: Bag of Words, TF-IDF, Embeddings

🎵 AUDIO DATA:
• Represented as waveforms (amplitude over time)
• Sampling: Convert continuous to discrete
• Features: MFCCs, spectrograms
• Preprocessing: Normalize, segment

📊 DATA AUGMENTATION:
• Create variations to increase training data
• Images: flip, rotate, crop, color adjust
• Text: synonym replacement, back translation
• Audio: time stretch, pitch shift`,
                keyPoints: [
                    "Unstructured data requires specialized preprocessing",
                    "Transfer learning leverages pre-trained models",
                    "Text embeddings capture semantic meaning",
                    "Augmentation increases effective dataset size",
                    "Domain knowledge guides preprocessing choices"
                ],
                example: `# TEXT PROCESSING
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(documents)

# IMAGE PROCESSING
from tensorflow.keras.preprocessing.image import ImageDataGenerator

augmentor = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.1
)`,
                realCase: {
                    title: "GPT and Transformer Embeddings",
                    description: "Modern NLP uses transformer-based embeddings that capture contextual meaning. The same word gets different embeddings based on surrounding context, enabling much better understanding of language.",
                    impact: "Transformers revolutionized NLP, powering ChatGPT, BERT, and most modern language AI"
                }
            }
        ]
    },

    // Continuing with more lessons...
    3: {
        name: "Lesson 3: Training & Evaluating Models",
        icon: "⚙️",
        weight: "Focus Areas: Training Process, Evaluation Metrics, Tuning",
        topics: [
            {
                title: "The Training Process",
                concept: `How machine learning models learn from data:

🎯 TRAINING FUNDAMENTALS:
• Model learns by minimizing a loss/cost function
• Parameters (weights) are adjusted iteratively
• Goal: find parameters that minimize error on training data

📊 TRAIN/TEST SPLIT:
• Training set: Data used to fit the model (typically 70-80%)
• Test set: Data used to evaluate generalization (20-30%)
• Validation set: For hyperparameter tuning (optional split)

🔄 CROSS-VALIDATION:
• More robust than single split
• K-fold: Divide data into K parts, train K times
• Each fold serves as test set once
• Average performance across folds

⚠️ CRITICAL RULES:
• NEVER train on test data
• NEVER leak future data into past (time series)
• Use stratification for imbalanced classes
• Keep test set truly held out until final evaluation`,
                keyPoints: [
                    "Training minimizes error on training data",
                    "Test set evaluates generalization to new data",
                    "Cross-validation provides robust performance estimates",
                    "Data leakage invalidates model assessments",
                    "Stratification maintains class proportions"
                ],
                example: `from sklearn.model_selection import (train_test_split, 
                                                  cross_val_score,
                                                  StratifiedKFold)

# Simple train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# K-Fold Cross-Validation
model = RandomForestClassifier()

# Basic 5-fold CV
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# Stratified K-Fold (maintains class proportions)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')

# Time series: Never shuffle!
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)`,
                realCase: {
                    title: "Kaggle Competition Data Leakage",
                    description: "Many Kaggle competitors have built models that performed amazingly in validation but failed on the private leaderboard. The cause? Subtle data leakage where information from the test period leaked into training features.",
                    impact: "Learning to detect and prevent leakage is a crucial skill for any ML practitioner"
                }
            },
            {
                title: "Overfitting and Underfitting",
                concept: `The fundamental tradeoff in machine learning:

📉 UNDERFITTING (High Bias):
• Model is too simple
• Cannot capture patterns in data
• High error on BOTH training and test
• Signs: Low training score, low test score
• Fix: More features, more complex model, less regularization

📈 OVERFITTING (High Variance):
• Model is too complex
• Memorizes training data including noise
• Low error on training, HIGH error on test
• Signs: High training score, low test score
• Fix: More data, regularization, simpler model, dropout, early stopping

✅ GOOD FIT:
• Model generalizes well to new data
• Reasonable error on both training and test
• Captures signal without memorizing noise

📊 BIAS-VARIANCE TRADEOFF:
• Bias: Error from oversimplified assumptions
• Variance: Sensitivity to training data fluctuations
• Goal: Balance bias and variance for minimum total error`,
                keyPoints: [
                    "Underfitting: Train bad + Test bad = Model too simple",
                    "Overfitting: Train good + Test bad = Model too complex",
                    "Learning curves help diagnose fit issues",
                    "Regularization trades training accuracy for generalization",
                    "More data usually helps with overfitting"
                ],
                example: `import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y, title):
    """Diagnose over/underfitting with learning curves"""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), 
             label='Training Score')
    plt.plot(train_sizes, val_scores.mean(axis=1), 
             label='Validation Score')
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    
    # Interpretation:
    # If both curves are low: UNDERFITTING
    # If training high, validation low: OVERFITTING
    # If both converge high: GOOD FIT
    
    return plt

# Detect overfitting
from sklearn.tree import DecisionTreeClassifier

# Overfit model (no constraints)
overfit = DecisionTreeClassifier(max_depth=None)

# Regularized model
regularized = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)`,
                realCase: {
                    title: "Titanic Competition Overfitting",
                    description: "Beginners on Kaggle's Titanic competition often create extremely complex models that perfectly classify the training passengers but fail on the test set. Simple models with 5-10 well-engineered features often outperform complex ensembles.",
                    impact: "Demonstrates that feature quality often matters more than model complexity"
                }
            },
            {
                title: "Classification Metrics",
                concept: `Evaluating classification model performance:

📊 CONFUSION MATRIX:
                    Predicted
                    Neg    Pos
Actual  Neg         TN     FP
        Pos         FN     TP

📈 KEY METRICS:

ACCURACY = (TP + TN) / Total
• Percentage of correct predictions
• Misleading for imbalanced data!

PRECISION = TP / (TP + FP)
• Of predicted positives, how many are correct?
• Important when FP is costly (spam filter)

RECALL (Sensitivity) = TP / (TP + FN)
• Of actual positives, how many did we catch?
• Important when FN is costly (cancer detection)

F1 SCORE = 2 × (Precision × Recall) / (Precision + Recall)
• Harmonic mean of precision and recall
• Balanced measure when both matter

SPECIFICITY = TN / (TN + FP)
• Of actual negatives, how many identified correctly?

AUC-ROC
• Area under Receiver Operating Characteristic curve
• Measures ranking ability across thresholds`,
                keyPoints: [
                    "Accuracy fails for imbalanced classes (99% negative → 99% accuracy by predicting all negative)",
                    "Precision: Use when false positives are costly",
                    "Recall: Use when false negatives are dangerous",
                    "F1: Use when you need to balance precision and recall",
                    "AUC-ROC: Good for comparing models overall"
                ],
                example: `from sklearn.metrics import (accuracy_score, precision_score,
                               recall_score, f1_score, 
                               confusion_matrix, classification_report,
                               roc_auc_score, roc_curve)

y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

# All metrics at once
print(classification_report(y_true, y_pred))

# Individual metrics
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.2f}")
print(f"Precision: {precision_score(y_true, y_pred):.2f}")
print(f"Recall:    {recall_score(y_true, y_pred):.2f}")
print(f"F1:        {f1_score(y_true, y_pred):.2f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
# [[TN, FP],
#  [FN, TP]]

# AUC-ROC (need probabilities)
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)`,
                realCase: {
                    title: "COVID-19 Test Sensitivity vs Specificity",
                    description: "COVID tests prioritized recall (sensitivity) over precision. A test that catches 95% of positive cases but has some false positives was better for public health than a precise test that missed infections.",
                    impact: "Emphasized recall > 95% even if it meant lower specificity"
                }
            },
            {
                title: "Regression Metrics",
                concept: `Evaluating regression model performance:

📊 KEY METRICS:

MAE (Mean Absolute Error)
• Average of |predicted - actual|
• Same units as target variable
• Less sensitive to outliers
• Formula: MAE = Σ|yᵢ - ŷᵢ| / n

MSE (Mean Squared Error)
• Average of (predicted - actual)²
• Penalizes large errors more heavily
• Common loss function
• Formula: MSE = Σ(yᵢ - ŷᵢ)² / n

RMSE (Root Mean Squared Error)
• Square root of MSE
• Same units as target variable
• Popular and interpretable
• Formula: RMSE = √MSE

R² (Coefficient of Determination)
• Proportion of variance explained
• Range: -∞ to 1 (1 is perfect)
• R² = 0 means model = baseline (mean)
• Formula: 1 - (SS_res / SS_tot)

MAPE (Mean Absolute Percentage Error)
• Percentage error
• Scale-independent
• Undefined when y = 0`,
                keyPoints: [
                    "MAE: Average error in original units, robust to outliers",
                    "MSE/RMSE: Penalizes large errors, commonly used loss",
                    "R²: Interpretable as 'variance explained', compare to baseline",
                    "MAPE: Good for comparing across different scales",
                    "Choose metric based on business impact of errors"
                ],
                example: `from sklearn.metrics import (mean_absolute_error, 
                               mean_squared_error, r2_score)
import numpy as np

y_true = [100, 150, 200, 250, 300]
y_pred = [110, 145, 195, 260, 290]

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(f"MAE:  {mae:.2f}")   # 10.00 (average $10 error)
print(f"RMSE: {rmse:.2f}")  # 10.95 (penalizes larger errors)
print(f"R²:   {r2:.3f}")    # 0.99 (explains 99% of variance)

# MAPE (manual)
mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / 
                       np.array(y_true))) * 100
print(f"MAPE: {mape:.1f}%")  # Average % error`,
                realCase: {
                    title: "Housing Price Prediction Metrics",
                    description: "For Zillow's Zestimate, they use multiple metrics: MAE for typical error magnitude, percentage within 5%/10%/20% of actual price for customer communication, and tracking outliers separately.",
                    impact: "Multiple metrics provide different perspectives on model performance"
                }
            },
            {
                title: "Hyperparameter Tuning",
                concept: `Optimizing model configuration for best performance:

🔧 HYPERPARAMETERS vs PARAMETERS:
• Parameters: Learned during training (weights, coefficients)
• Hyperparameters: Set BEFORE training (learning rate, tree depth)

🔍 TUNING METHODS:

GRID SEARCH:
• Try all combinations of specified values
• Exhaustive but expensive
• Good for small search spaces

RANDOM SEARCH:
• Random sampling from distributions
• More efficient than grid for large spaces
• Often finds good solutions faster

BAYESIAN OPTIMIZATION:
• Uses past results to guide search
• More intelligent sampling
• Tools: Optuna, HyperOpt, Scikit-Optimize

SUCCESSIVE HALVING / HYPERBAND:
• Early stopping for bad configurations
• Very efficient for expensive models

🎯 BEST PRACTICES:
• Start with wide ranges, then narrow
• Use cross-validation within tuning
• Log all experiments
• Set random seeds for reproducibility`,
                keyPoints: [
                    "Hyperparameters control model learning behavior",
                    "Grid search is thorough but computationally expensive",
                    "Random search often more efficient for large spaces",
                    "Bayesian methods learn which regions work best",
                    "Always use validation set/CV, never test set for tuning"
                ],
                example: `from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import optuna

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_leaf': [1, 2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(), 
    param_grid, 
    cv=5, 
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")

# Random Search (often better for large spaces)
from scipy.stats import randint, uniform
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 50),
    'min_samples_leaf': randint(1, 20)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_dist,
    n_iter=50,  # Number of random samples
    cv=5,
    scoring='f1'
)`,
                realCase: {
                    title: "Google AutoML",
                    description: "Google's AutoML uses neural architecture search and hyperparameter optimization at massive scale. They've found that automated search can discover model architectures that outperform human-designed ones.",
                    impact: "AutoML has democratized access to well-tuned models"
                }
            }
        ]
    },

    // Add more lessons as needed - this shows the structure
    // Lessons 4-12 would follow similar pattern with topics from the eBook
};

// Export for use in app
if (typeof module !== 'undefined' && module.exports) {
    module.exports = STUDY_CONTENT;
}

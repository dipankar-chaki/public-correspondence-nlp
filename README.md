![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)

# 🤖 NLP Public Correspondence Classification

This project focuses on classifying public correspondence messages using natural language processing (NLP) techniques. The goal is to accurately categorize messages into different types (e.g., customer inquiries, company responses) using both traditional machine learning models and modern transformer-based approaches.

---

## 🎯 Project Goals

- **Message Classification**: Develop robust models to accurately classify public correspondence messages into predefined categories (customer inquiries, company responses, etc.)

- **Model Comparison**: Compare and evaluate the performance of traditional ML models (Logistic Regression, Random Forest, SVM) against modern transformer-based approaches (BERT)

- **Feature Engineering**: Create and analyze effective text-based features that improve classification accuracy

- **Text Preprocessing**: Implement comprehensive text cleaning and normalization techniques to handle social media text effectively

- **Model Optimization**: Fine-tune models to achieve optimal performance while maintaining computational efficiency

- **Performance Analysis**: Conduct detailed analysis of model performance across different message types and scenarios

- **Scalability**: Ensure the solution can handle large-scale datasets efficiently

- **Reproducibility**: Create a well-documented, modular pipeline that can be easily reproduced and extended

### ✨ Key Features

- 📝 Text preprocessing and cleaning
- 🔧 Feature engineering (text length, word count, character count)
- 📊 TF-IDF vectorization
- 🤖 Multiple model implementations (Traditional ML and BERT)
- 📈 Model performance comparison
- 💾 Model serialization and saving

### 📊 Dataset

The dataset consists of public correspondence messages with the following key features:

- `tweet_id`: Unique identifier for each message
- `author_id`: Identifier of the message author
- `inbound`: Boolean indicating if the message is inbound (from a customer) or outbound (from a company)
- `created_at`: Timestamp of the message
- `text`: Raw message text
- `response_tweet_id`: ID of the response message (if any)
- `in_response_to_tweet_id`: ID of the message this is responding to (if any)
- `message_type`: Classification label (e.g., "customer_complaint_or_inquiry", "company_response")

#### 📈 Dataset Statistics:

- Total messages: ~1.25M
- Class distribution:
  - Customer complaints/inquiries: ~63%
  - Company responses: ~37%

### 📁 Project Structure

```
nlp-public-correspondence/
├── data/
│   └── twcs/                  # Twitter Customer Service dataset
│       ├── twcs.csv           # Raw dataset
│       └── twcs_features.csv  # Feature-engineered dataset
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Initial data analysis
│   ├── 02_cleaning_and_labels.ipynb   # Data cleaning and labeling
│   ├── 03_text_preprocessing.ipynb    # Text preprocessing steps
│   ├── 04_feature_engineering.ipynb   # Feature extraction and engineering
│   └── 05_model_training.ipynb        # Model training and evaluation
├── models/                    # Saved model files
│   ├── bert_model/           # BERT model and tokenizer
│   ├── logistic_regression_model.joblib
│   ├── random_forest_model.joblib
│   └── svm_approx_rbf_model.joblib
├── outputs/                   # Model outputs and visualizations
│   ├── model_comparison.png   # Performance comparison plot
│   └── model_scores.csv       # Detailed model metrics
├── scripts/
│   └── utils.py              # Utility functions
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

---

## 🚀 Environment Setup

### ✅ Prerequisites

- Python 3.11 or higher
- Git
- Conda (recommended) or pip

### 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/dipankar-chaki/nlp-public-correspondence.git
   cd nlp-public-correspondence
   ```

2. Create and activate a virtual environment:

   ```bash
   # Using conda (recommended)
   conda create -n nlp-env python=3.11
   conda activate nlp-env

   # Alternative: using venv
   python -m venv nlp-env
   source nlp-env/bin/activate  # On Unix/macOS
   # or
   .\nlp-env\Scripts\activate  # On Windows
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

### 💻 Hardware Requirements

- Minimum: 8GB RAM
- Recommended: 16GB+ RAM
- GPU: Optional but recommended for BERT training
  - CUDA-compatible GPU for faster training
  - CPU-only training is supported but slower

---

## 📚 Usage

The project is organized as a series of Jupyter notebooks that walk through the entire pipeline:

### 📊 1. Data Exploration (`01_data_exploration.ipynb`)

- Load and explore the dataset
- Analyze basic statistics and distributions
- Visualize message type distribution

### 🧹 2. Data Cleaning and Labeling (`02_cleaning_and_labels.ipynb`)

- Clean the dataset
- Assign message types based on message properties
- Remove irrelevant messages
- Handle missing values

### 📝 3. Text Preprocessing (`03_text_preprocessing.ipynb`)

- Remove URLs, mentions, hashtags
- Clean and normalize text
- Apply spaCy for advanced text processing
- Implement custom cleaning functions

### 🔧 4. Feature Engineering (`04_feature_engineering.ipynb`)

- Extract text-based features
- Create additional features like text length, word count
- Analyze feature distributions
- Generate feature correlation analysis

### 🤖 5. Model Training (`05_model_training.ipynb`)

- Train traditional ML models (Logistic Regression, Random Forest, SVM)
- Train and fine-tune BERT model
- Evaluate model performance
- Save trained models
- Generate performance visualizations

---

## 🧠 Model Details

### 📊 Traditional Machine Learning Models

1. **Logistic Regression**

   - Solver: 'saga'
   - Max iterations: 2000
   - Multi-class: 'multinomial'

2. **Random Forest**

   - Estimators: 100
   - Criterion: 'gini'
   - Max depth: None

3. **SVM (Approximate RBF)**
   - Kernel: RBF (approximated)
   - Components: 300
   - Loss: 'hinge'

### 🤖 BERT Model

- Base model: `bert-base-uncased`
- Training epochs: 2
- Batch size: 16 (training), 32 (evaluation)
- Learning rate: 2e-5
- Max sequence length: 128
- Optimizer: AdamW

---

## 📈 Model Performance

The project implements several models with the following performance metrics:

| Model               | Accuracy | F1 Score | Training Time |
| ------------------- | -------- | -------- | ------------- |
| BERT                | 0.992    | 0.992    | ~8.5 hours    |
| Random Forest       | 0.958    | 0.958    | ~5 minutes    |
| Logistic Regression | 0.949    | 0.949    | ~2 minutes    |
| SVM (approx RBF)    | 0.628    | 0.502    | ~15 minutes   |

Note: Training times are approximate and may vary based on hardware.

---

## 📦 Dependencies

Key dependencies include:

- pandas >= 2.2.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- spacy >= 3.7.0
- transformers == 4.37.2
- torch >= 2.0.0
- accelerate <= 0.25.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- plotly >= 5.18.0
- jupyter >= 1.0.0
- joblib >= 1.3.0

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### 📝 Development Guidelines

- Follow PEP 8 style guide for Python code
- Add docstrings to new functions and classes
- Include tests for new features
- Update documentation as needed
- Use meaningful commit messages

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- The Twitter Customer Service dataset for providing the training data
- Hugging Face for the transformers library and BERT implementation
- The spaCy team for NLP tools and models
- The scikit-learn team for machine learning implementations

---

## 👨‍💻 Author

Dipankar Chaki
PhD in Computer Science | ML & AI Researcher

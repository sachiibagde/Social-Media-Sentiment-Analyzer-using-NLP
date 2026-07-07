# 📊 Social Media Sentiment Analyzer Using NLP

A Machine Learning and Deep Learning based web application that analyzes social media text and classifies it into **Positive**, **Negative**, or **Neutral** sentiments. The project compares multiple NLP models to identify the best-performing algorithm for sentiment classification.

---

## 🚀 Features

- Analyze sentiment of user-entered text
- Real-time sentiment prediction
- Compare multiple ML and DL models
- Interactive Flask web interface
- Confidence score for predictions
- Model performance comparison
- Confusion Matrix and Evaluation Metrics
- Multilingual support using XLM-RoBERTa

---

## 🎯 Objectives

- Automate sentiment classification of social media text
- Compare traditional Machine Learning and Deep Learning models
- Evaluate models using Accuracy, Precision, Recall, and F1-Score
- Identify the best-performing model for deployment

---

## 🧠 Models Used

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- LSTM
- XLM-RoBERTa

---

## 🛠️ Technologies Used

- Python
- Flask
- Scikit-learn
- TensorFlow / Keras
- Transformers (Hugging Face)
- NLTK
- Pandas
- NumPy
- Matplotlib
- HTML
- CSS
- JavaScript

---

## 📂 Dataset

**Dataset:** Twitter Entity Sentiment Analysis

**Source:** Kaggle

**Classes Used**
- Positive
- Negative
- Neutral

---

## 🔄 Project Workflow

1. Collect Dataset
2. Data Preprocessing
3. Text Cleaning
4. Tokenization
5. Stopword Removal
6. Negation Handling
7. TF-IDF Vectorization
8. Train Multiple Models
9. Evaluate Performance
10. Compare Models
11. Deploy using Flask
12. Predict Sentiment

---

## 📁 Project Structure

```
Social-Media-Sentiment-Analyzer/
│
├── preprocess.py
├── sentiment_analysis.py
├── train_logreg.py
├── train_svm.py
├── train_random_forest.py
├── train_lstm.py
├── xlm_sentiment.py
├── ensemble_models.py
├── explain.py
├── flask_app.py
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── svm_model.pkl
│   ├── random_forest_model.pkl
│   ├── vectorizer.pkl
│   └── tokenizer.pkl
│
├── templates/
├── static/
├── dataset/
├── requirements.txt
└── README.md
```

---

## 📈 Model Performance

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 84.9% |
| SVM | 73.4% |
| **Random Forest** ⭐ | **91.3%** |
| LSTM | 26.2% |
| XLM-RoBERTa | 56.3% |

**Best Model:** Random Forest

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve
- AUC Score

---

## 💻 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/social-media-sentiment-analyzer.git
```

### Navigate to Project

```bash
cd social-media-sentiment-analyzer
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Flask Application

```bash
python flask_app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000/
```

---

## 📷 User Interface

The application provides an easy-to-use interface where users can:

- Enter text
- Analyze sentiment
- View confidence scores
- Compare sentiment probabilities

---

## 🎯 Applications

- Product Review Analysis
- Customer Feedback Analysis
- Brand Reputation Monitoring
- Social Media Monitoring
- Market Research
- Business Intelligence
- Public Opinion Analysis

---

## 🔮 Future Scope

- Emotion Detection
- Sarcasm Detection
- Live Twitter API Integration
- Batch Sentiment Analysis
- Cloud Deployment
- Dashboard & Analytics
- Speech-to-Text Sentiment Analysis

---

## 👨‍💻 Team Members

- **Sachi Bagade**
- **Bharti Zankar**
- **Rushikesh Shirsath**

**Guide:** Prof. B. N. Babar

Department of Artificial Intelligence and Data Science

---

## 📚 References

- Kaggle - Twitter Entity Sentiment Analysis Dataset
- Scikit-learn Documentation
- TensorFlow Documentation
- Hugging Face Transformers
- NLTK Documentation

---

## 📜 License

This project is developed for educational and research purposes.

---

⭐ If you found this project helpful, don't forget to **Star** the repository!

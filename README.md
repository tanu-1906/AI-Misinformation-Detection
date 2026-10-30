# 🔍 AI-Based Social Media Misinformation Detection System

## 📌 About the Project

The AI-Based Social Media Misinformation Detection System is a Machine Learning-based application designed to analyze news articles and social media posts and classify them as potentially misleading or likely reliable.

The system uses Natural Language Processing (NLP) and Machine Learning techniques to analyze text and provide classification results with confidence scores.

## 🎯 Objectives

- Identify potentially misleading news and social media content.
- Analyze text using NLP techniques.
- Classify content using Machine Learning.
- Display prediction results and confidence scores.
- Help users become more aware of online misinformation.

## 🛠️ Technologies Used

- Python
- Machine Learning
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Scikit-learn
- Pandas
- Streamlit
- Supabase
- PostgreSQL

## ⚙️ Key Features

- User Login and Registration
- Google Authentication
- News and Social Media Text Analysis
- Fake and Real News Classification
- Confidence Score Display
- Interactive Streamlit Dashboard

## 🤖 Machine Learning Approach

The system uses TF-IDF (Term Frequency–Inverse Document Frequency) to convert text into numerical features.

A trained Machine Learning classification model analyzes the extracted features and predicts whether the submitted content belongs to the potentially misleading or likely reliable category.

## 📊 Model Performance

- Training Dataset: 38,646 news records
- Test Dataset: 7,730 records
- Model Accuracy: 98.29%

*Accuracy is based on the project's current test dataset and may not represent performance on real-world social media content.*

## 🖥️ Application Interface

Users can enter news headlines, articles or social media posts and receive a classification result with a confidence score.

## 🚀 Installation and Setup

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**2. Open the project folder**

```bash
cd AI-Misinformation-Detector
```

**3. Install dependencies**

```bash
pip install pandas scikit-learn streamlit supabase python-dotenv
```

**4. Configure environment variables**

Create a `.env` file and add your Supabase credentials:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

Never upload your actual credentials to GitHub.

**5. Run the application**

```bash
python -m streamlit run app.py
```

## 📁 Project Files

- `app.py` – Streamlit application
- `train_model.py` – Machine Learning model training
- `misinformation_model.pkl` – Trained classification model
- `tfidf_vectorizer.pkl` – TF-IDF vectorizer
- `.env` – Supabase configuration (not uploaded)

## 🔮 Future Scope

- Integration of advanced NLP and Deep Learning models.
- Support for multiple regional languages.
- Real-time misinformation analysis.
- Improved detection of misleading content and manipulated claims.

## ⚠️ Disclaimer

This system is an academic project. Its predictions are not a guarantee that news is true or false. Users should independently verify information through reliable sources.

## 👩‍💻 Project Information

**Project:** AI-Based Social Media Misinformation Detection System

**Domain:** Artificial Intelligence, Machine Learning and NLP

**Developed as:** Academic Engineering Project

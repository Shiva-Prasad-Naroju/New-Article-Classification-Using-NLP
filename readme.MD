# 📰 News Topic Classifier & Text Analyzer

This project is a powerful News Topic Classifier that automatically categorizes news articles into one of four topics:
🌎 World    

🏅 Sports    

💼 Business    

🔬 Science/Technology

Built using Natural Language Processing (NLP) techniques and a Logistic Regression classifier (achieving 91% accuracy), the project features a modern, interactive web app built and deployed with Streamlit Community Cloud.

## ✨ Features:
Automatic Topic Classification using Logistic Regression and TF-IDF

Interactive Web App: User-friendly, visually appealing, and easy to use

Model Confidence Visualization: See prediction probabilities for each topic

Text Analytics: Get word count, unique words, top keywords, and more

Preprocessing Transparency: View how your text is cleaned before prediction

Sample Inputs: Try out the app with real-world news examples

## 🌐 Live Demo:
🖥️ Try the App on Streamlit Cloud!

## 🏗️ Project Structure:
text
.
├── app.py                  # Streamlit web app
├── log_reg_model.pkl       # Trained Logistic Regression model
├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
├── requirements.txt        # Project dependencies
├── README.md               # This file
└── News_Classification.ipynb # Jupyter notebook

## 📊 Model Performance:
Classifier: Logistic Regression

Feature Extraction: TF-IDF

Test Accuracy: 91%

Classes: World, Sports, Business, Science/Technology

🛠️ Getting Started:

1. Clone the Repository
bash
git clone https://github.com/Shiva-Prasad-Naroju/New-Article-Classification-Using-NLP.git
cd news-topic-classifier
2. Install Dependencies
bash
pip install -r requirements.txt
3. Run the App Locally
bash
streamlit run app.py
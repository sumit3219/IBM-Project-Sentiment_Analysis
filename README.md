# 📝 Sentiment Analysis of Product Reviews using Naive Bayes

## 📌 Project Overview
This project performs sentiment analysis on product reviews (tweets) using Natural Language Processing (NLP) and machine learning. It classifies the reviews as **Positive** or **Negative** using the **Naive Bayes classifier**, trained on a labeled dataset.

## ⚙ Technologies Used
- Python  
- scikit-learn  
- pandas  
- numpy  
- matplotlib  
- CountVectorizer (for text vectorization)  
- Multinomial Naive Bayes (for classification)


## 📁 Dataset
The dataset is taken from:  
**[Twitter Sentiment Analysis Dataset](https://github.com/dD2405/Twitter_Sentiment_Analysis)**  

It contains labeled tweets with sentiment values:
- `0 = Negative`
- `1 = Positive`


## 🚀 How It Works

1. **Data Cleaning**: Drops missing values and maps numeric sentiment to text  
2. **Train-Test Split**: 80% data for training, 20% for testing  
3. **Vectorization**: Text transformed using CountVectorizer  
4. **Model Training**: Trained using MultinomialNB  
5. **Prediction & Evaluation**:  
   - Accuracy score  
   - Confusion matrix  
   - Classification report  


## 📊 Results
- Model achieved **~85% accuracy**
- Evaluation using confusion matrix and classification report
- Results visualized using matplotlib


## 🖼 Example Output

Accuracy: 85%

Classification Report:

              precision    recall  f1-score   support
              
    Negative       0.84      0.87      0.85       500
    Positive       0.86      0.83      0.84       500


## ✅ Requirements

Install required libraries using:

pip install pandas numpy matplotlib scikit-learn



## 📂 How to Run

python sentiment_analysis.py

Make sure you are connected to the internet, as the dataset is loaded via a URL.


## Developed by

**Sumit Mishra**  
Student at United College of Engineering and Research, Naini, Prayagraj

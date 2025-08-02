import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load dataset
url = 'https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv'
data = pd.read_csv(url)

# Keep only useful columns
data = data[['tweet', 'label']]
data.columns = ['Review', 'Sentiment']

# Map to string labels
data['Sentiment'] = data['Sentiment'].map({0: 'Negative', 1: 'Positive'})

# Drop missing just in case
data.dropna(inplace=True)

# Split
X = data['Review']
y = data['Sentiment']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Vectorize
cv = CountVectorizer(stop_words='english')
X_train = cv.fit_transform(x_train)
X_test = cv.transform(x_test)  # ✅ fixed here

# Model
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Prediction
y_pred = nb.predict(X_test)

# Evaluation
print("Acc:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["Positive", "Negative"])

fig, ax = plt.subplots(figsize=(5, 5))
cax = ax.matshow(cm, cmap='Blues')
plt.title("Confusin Matrix")  # typo still there for realism
plt.colorbar(cax)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Pos', 'Neg'])
ax.set_yticklabels(['Pos', 'Neg'])
plt.xlabel("Predcted")  # another typo
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm[0])):
        ax.text(j, i, str(cm[i][j]), va='center', ha='center')

plt.tight_layout()
plt.show()

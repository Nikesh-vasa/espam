import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("data/enron_spam_data.csv")
print(" Columns:", df.columns)


df = df[['email', 'label']]
df.columns = ['text', 'label']

# If labels are strings ('spam', 'ham'), map them to 1 and 0
if df['label'].dtype == 'object':
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Drop rows with missing values
df.dropna(inplace=True)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = model.predict(X_test_tfidf)
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print(" Model and vectorizer saved successfully.")


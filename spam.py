import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# STEP 1: Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

print("Data loaded")

# STEP 2: Keep useful columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# STEP 3: Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# STEP 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# STEP 5: Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# STEP 6: Train model
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model trained")

# STEP 7: Predict
y_pred = model.predict(X_test)

# STEP 8: Evaluate
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# STEP 9: Test custom message
msg = ["Congratulations! You won a free iPhone"]
msg_vec = vectorizer.transform(msg)
print("\nCustom Test:", model.predict(msg_vec))
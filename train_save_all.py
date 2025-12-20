import pandas as pd
import pickle
import time
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Ensure models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

print("--- 1. Loading Data ---")
# Load data
try:
    df = pd.read_csv('data/Data_finish.csv')
    df = df.dropna(subset=['tweet_ok'])
    X = df['tweet_ok']
    y = df['class']
except FileNotFoundError:
    print("Error: data/Data_finish.csv not found. Please ensure the data exists.")
    exit()

print(f"Total rows: {len(df)}")

# Split Data (80/20 as per notebook)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- 2. Vectorizing (TF-IDF) ---")
# Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Save Vectorizer
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("Vectorizer saved.")

# Define Models with Best Params from Notebook
models = {
    'Naive Bayes': MultinomialNB(alpha=1.0),
    'Logistic Regression': LogisticRegression(C=10, solver='liblinear', max_iter=1000),
    'SVM (LinearSVC)': LinearSVC(C=1, dual=False),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
}

metrics_data = []

print("--- 3. Training & Saving Models ---")

for name, model in models.items():
    print(f"Training {name}...")
    start_time = time.time()
    
    # Train
    model.fit(X_train_tfidf, y_train)
    
    # Measure time
    train_time = time.time() - start_time
    
    # Predict
    y_pred = model.predict(X_test_tfidf)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    
    print(f"  > Accuracy: {acc:.4f} | Time: {train_time:.2f}s")
    
    # Save Model
    # Create a safe filename
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    file_path = f'models/{safe_name}.pkl'
    
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
        
    metrics_data.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'F1-Score': round(f1, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'Time (s)': round(train_time, 2),
        'Filename': f'{safe_name}.pkl'
    })

# Save Metrics for Streamlit to use
with open('models/metrics.json', 'w') as f:
    json.dump(metrics_data, f, indent=4)

print("--- Done! All models and metrics saved. ---")

import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 🔹 Step 1: Set and Create Local NLTK Data Directory
NLTK_PATH = os.path.join(os.getcwd(), "nltk_data")  # Define NLTK path inside project
nltk.data.path.append(NLTK_PATH)  # Ensure NLTK uses this path

# Ensure the NLTK directory exists
if not os.path.exists(NLTK_PATH):
    os.makedirs(NLTK_PATH)

# 🔹 Step 2: Download NLTK Data to Local Folder
nltk.download('stopwords', download_dir=NLTK_PATH)
nltk.download('punkt_tab', download_dir=NLTK_PATH)

# 🔹 Step 3: Define Dataset Path (Ensure This is Correct)
DATASET_PATH = "data/aclImdb"

# 🔹 Step 4: Load IMDb Data from Local Folders
def load_imdb_data(data_dir):
    """
    Loads IMDb dataset from the given directory.
    Assumes 'pos/' and 'neg/' subdirectories exist inside 'train/'.
    """
    data = []
    labels = []

    for label in ["pos", "neg"]:
        dir_path = os.path.join(data_dir, "train", label)  # Adjusting path
        if not os.path.exists(dir_path):
            print(f"❌ Directory not found: {dir_path}")
            continue

        for file_name in os.listdir(dir_path):
            with open(os.path.join(dir_path, file_name), "r", encoding="utf-8") as f:
                text = f.read().strip()
                data.append(text)
                labels.append(1 if label == "pos" else 0)

    return pd.DataFrame({"review": data, "sentiment": labels})

# Load Training Data
train_data = load_imdb_data(DATASET_PATH)

if train_data.empty:
    print("❌ No data found. Check dataset path.")
    exit()

print("✅ IMDb dataset loaded successfully!")

# 🔹 Step 5: Text Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize words
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(tokens)

train_data['cleaned_review'] = train_data['review'].apply(preprocess_text)

# 🔹 Step 6: Convert Text to Vectors
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(train_data['cleaned_review'])
y = train_data['sentiment']

# Split dataset into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Step 7: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Ensure models directory exists
if not os.path.exists("models"):
    os.makedirs("models")

# 🔹 Step 8: Save Model and Vectorizer
pickle.dump(model, open("models/sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("✅ Model training complete. Saved in 'models/' folder.")

import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer
import pandas as pd
import re


#Loading the dataset
file_path = "Datasets/amazon_review_small.txt"
df = pd.read_csv(file_path, sep=",", quotechar='"', engine="python", header=None, names=["rating","title", "review"])

print("Print first 2 rows of the dataset:")
print(df.head(2))

#Preprocessing the dataset
#Fill missing values with empty string
df["title"] = df["title"].fillna("")

#Creating a copy of dataset and Combining title and review into a single text column
df_work = df.copy()
df_work["text"] = df_work["title"] + " " + df_work["review"]
#Selecting only the text and rating columns for further processing
df_work = df_work[["text", "rating"]]
df_work["text"][0]

# Function to minimally normalize text for BERT tokenization
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Keep punctuation/case information for the BERT tokenizer
    # Normalize whitespace only
    text = re.sub(r"\s+", " ", text).strip()
    return text

#Apply cleaning function to the text column
df_work["text"] = df_work["text"].apply(clean_text)

# Separate features (X) and target (y)
X = df_work["text"]
y = df_work["rating"]

# First split: 80% train, 20% test
from sklearn.model_selection import train_test_split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Second split (on training data): 80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=42
)

#Tokenization using BERT tokenizer
#Bert tokenizer is a pre-trained tokenizer that converts text into tokens that can be fed into a BERT model. It handles tasks like lowercasing, punctuation removal, and splitting words into subwords as needed by the BERT architecture.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(texts):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128  # start with 128 (not too big)
    )
    
X_train_enc = tokenize_function(X_train.tolist())
X_val_enc = tokenize_function(X_val.tolist())
X_test_enc = tokenize_function(X_test.tolist())

print("Tokenization complete. Sample tokenized input for first training example:")
print("Original text:", X_train.iloc[0])
print("Tokenized input IDs:", X_train_enc["input_ids"][0])
print("Tokenized attention mask:", X_train_enc["attention_mask"][0])

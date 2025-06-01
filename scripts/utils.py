# Common imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.preprocessing import LabelEncoder
import joblib

# Text cleaning function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)        # remove mentions and hashtags
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)              # remove numbers
    text = re.sub(r"\s+", " ", text).strip()     # remove extra whitespace
    return text

# Load dataset
def load_tweets_csv(path="../data/twcs/twcs.csv"):
    return pd.read_csv(path)

# Basic plot config
def set_plot_style():
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 5)

# Feature engineering: text length, word count, etc.
def get_text_features(df, text_col):
    df = df.copy()
    df[f'{text_col}_length'] = df[text_col].apply(lambda x: len(str(x)))
    df[f'{text_col}_word_count'] = df[text_col].apply(lambda x: len(str(x).split()))
    df[f'{text_col}_char_count'] = df[text_col].apply(lambda x: len(str(x).replace(' ', '')))
    return df

# Label encoding/decoding
def encode_labels(labels):
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    return encoded, encoder

def decode_labels(encoded_labels, encoder):
    return encoder.inverse_transform(encoded_labels)

# Model save/load
def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

# 📌 Improved Twitter Sentiment & Emotion Analysis Pipeline

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline
from bertopic import BERTopic
import joblib

# 1️⃣ Load Data
emotion_df = pd.read_csv('data/emotions_train.csv')
sentiment_df = pd.read_csv('data/training_senti.csv')
tweet_df = pd.read_pickle('data/twitter_final_extract_cadmv.p')

print("Emotion:", emotion_df.shape)
print("Sentiment:", sentiment_df.shape)
print("Tweets:", tweet_df.shape)

# 2️⃣ Clean text
def clean_tweet(text):
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"@\\w+", "", text)
    text = re.sub(r"#\\w+", "", text)
    text = re.sub(r"\\d+", "", text)
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text.strip()

emotion_df['clean_text'] = emotion_df['content'].apply(clean_tweet)
sentiment_df['clean_text'] = sentiment_df['tweet'].apply(clean_tweet)
tweet_df['clean_text'] = tweet_df['text'].apply(clean_tweet)

# 3️⃣ Sentiment model with transformers
sentiment_df['label'] = sentiment_df['polarity'].apply(lambda x: 1 if x == 4 else 0)
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Predict sentiment
sentiment_preds = []
for text in tweet_df['clean_text']:
    result = sentiment_model(text[:512])
    sentiment_preds.append(1 if result[0]['label'] == 'POSITIVE' else 0)

tweet_df['predicted_sentiment'] = sentiment_preds

# 4️⃣ Emotion detection with zero-shot
emotion_labels = ["joy", "anger", "sadness", "fear", "surprise", "love"]
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

emotion_preds = []
for text in tweet_df['clean_text']:
    result = classifier(text[:512], candidate_labels=emotion_labels)
    emotion_preds.append(result['labels'][0])

tweet_df['predicted_emotion'] = emotion_preds

# 5️⃣ Topic extraction with BERTopic
topic_model = BERTopic(language="english")
topics, probs = topic_model.fit_transform(tweet_df['clean_text'])
tweet_df['topic'] = topics

# Save results
tweet_df.to_csv('data/final_analysis_with_predictions.csv', index=False)
topic_model.save('bertopic_model')

print("✅ Pipeline complete. Results saved.")

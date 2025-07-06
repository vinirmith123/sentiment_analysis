# 🚗 DMV Twitter Sentiment & Emotion Analysis

This project identifies and analyzes the **sentiments**, **emotions**, and **topics** of tweets addressed to the **California Department of Motor Vehicles (DMV)**.

The goal is to help the DMV understand public opinion about services such as licenses, registrations, and license plates by analyzing real Twitter data.

---

## 📦 Project Structure

├── app.py # 🚀 Streamlit dashboard for final analytics
├── train_test.ipynb # 🧠 Notebook for training & testing models
├── sentiment.py # ⚙️ Script version of the improved NLP pipeline
├── data/
│ ├── emotions_train.csv
│ ├── training_senti.csv
│ ├── twitter_final_extract_cadmv.p
│ ├── final_analysis_with_predictions.csv
├── bertopic_model/ # 📚 Saved topic model
├── requirements.txt # 📌 Python dependencies
└── README.md # 📄 This file


---

## 🗂️ Datasets

**`data/` folder includes:**

- `emotions_train.csv` — Labeled tweets with emotion classes  
- `training_senti.csv` — Labeled tweets with sentiment polarity (0 = negative, 4 = positive)  
- `twitter_final_extract_cadmv.p` — Raw tweets to analyze (addressed to @CA_DMV)  
- `final_analysis_with_predictions.csv` — Output with predicted sentiments, emotions, and topics

---

## 🧠 How It Works

**1️⃣ Data Cleaning**  
- Removes URLs, mentions, hashtags, numbers, punctuation  
- Converts all text to lowercase

**2️⃣ Sentiment Prediction**  
- Uses a **transformer-based DistilBERT** model (`distilbert-base-uncased-finetuned-sst-2-english`)  
- Predicts positive or negative sentiment for each tweet

**3️⃣ Emotion Classification**  
- Uses a **zero-shot transformer** (`facebook/bart-large-mnli`)  
- Predicts the most likely emotion from: `joy`, `anger`, `sadness`, `fear`, `surprise`, `love`

**4️⃣ Topic Extraction**  
- Uses **BERTopic** for clustering and keyword extraction

**5️⃣ Results**  
- All predictions are saved to `final_analysis_with_predictions.csv` for visualization

---

## 📊 Key Analytics

✅ Most Discussed Topics  
✅ Average polarity (sentiment) per topic  
✅ Top 3 sentiment split for each topic  
✅ Overall average sentiment of the most popular tweets (likes + retweets)  
✅ Top emotions for the selected topic

---

## 🧮 Error Metrics

- Validation metrics (accuracy, F1, confusion matrix) are calculated in `train_test.ipynb` using the labeled training datasets.

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt

Python app.py or app1.py


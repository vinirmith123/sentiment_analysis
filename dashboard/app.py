import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from bertopic import BERTopic

# ✅ Streamlit config
st.set_page_config(page_title="DMV Tweet Sentiment Dashboard", layout="wide")
st.title("Twitter Sentiment Dashboard (Improved)")

# ✅ Load data
df = pd.read_csv("data/final_analysis_with_predictions.csv")

# ✅ Sidebar
all_topics = df['topic'].dropna().unique().tolist()
selected_topic = st.sidebar.selectbox("📌 Select a Topic", ["All"] + list(map(str, all_topics)))

filtered_df = df if selected_topic == "All" else df[df['topic'] == int(selected_topic)]

# ✅ Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Most Discussed Topics")
    topic_counts = df['topic'].value_counts().head(10).reset_index()
    topic_counts.columns = ['Topic', 'Count']
    fig = px.bar(topic_counts, x='Count', y='Topic', orientation='h', color='Topic')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("😊 Sentiment Distribution")
    sentiment_counts = filtered_df['predicted_sentiment'].value_counts().rename({0: 'Negative', 1: 'Positive'})
    fig2 = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Breakdown")
    st.plotly_chart(fig2, use_container_width=True)

# ✅ Top Emotions
st.subheader(f"😶 Top Emotions for Selected Topic")
top_emotions = filtered_df['predicted_emotion'].value_counts().head(5).reset_index()
top_emotions.columns = ['Emotion', 'Count']
st.bar_chart(top_emotions.set_index('Emotion'))

# ✅ Average Polarity per Topic
st.subheader("📈 Average Sentiment per Topic")
avg_sentiment = df.groupby('topic')['predicted_sentiment'].mean().reset_index().sort_values(by='predicted_sentiment', ascending=False)
avg_sentiment.columns = ['Topic', 'Average Sentiment']
st.dataframe(avg_sentiment)

# ✅ Top 3 Sentiment Split for Selected Topic
st.subheader(f"🔍 Sentiment Split for Topic: {selected_topic}")
sentiment_split = filtered_df['predicted_sentiment'].value_counts(normalize=True).rename({0: 'Negative', 1: 'Positive'}).reset_index()
sentiment_split.columns = ['Sentiment', 'Proportion']
sentiment_split['Proportion'] = sentiment_split['Proportion'] * 100
fig3 = px.bar(sentiment_split, x='Sentiment', y='Proportion', color='Sentiment', text_auto='.2f')
st.plotly_chart(fig3, use_container_width=True)

# ✅ Most Popular Tweets
st.subheader("🔥 Most Popular Tweets (Top 5 by Likes + Retweets)")
popular = filtered_df.copy()
popular['popularity'] = popular['favorite_count'] + popular['retweets_count']
top_popular = popular.sort_values(by='popularity', ascending=False).head(5)

avg_popular_sentiment = top_popular['predicted_sentiment'].mean()
st.metric("⭐ Average Sentiment of Top 5 Tweets", f"{avg_popular_sentiment:.2f}")

for _, row in top_popular.iterrows():
    st.markdown(f"**📝 Tweet:** {row['text']}")
    st.markdown(
        f"💬 Emotion: `{row['predicted_emotion']}` | ❤️ Likes: `{row['favorite_count']}` | "
        f"🔁 Retweets: `{row['retweets_count']}` | 🧠 Sentiment: `{'Positive' if row['predicted_sentiment'] == 1 else 'Negative'}`"
    )
    st.markdown("---")

st.success("✅ Dashboard Loaded!")

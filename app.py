import streamlit as st
import requests
import torch
from transformers import pipeline
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque, Counter
import re

# Load emotion classifier
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

# Clean tweets
def clean_tweet(text):
    return re.sub(r"http\S+|@\S+|#\S+", "", text).strip()

# Get tweets
def get_recent_tweets(username, bearer_token, max_results=10):
    headers = {"Authorization": f"Bearer {bearer_token}"}
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        'query': f'from:{username}',  # ✅ No filters
        'tweet.fields': 'created_at,text',
        'max_results': max_results
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        st.error(f"Twitter API Error: {response.status_code}")
        return []
    return response.json().get("data", [])

# Pattern logic
def predict_next_emotion(emotion_list):
    if all(e == "neutral" for e in emotion_list):
        return "neutral"
    count = Counter(emotion_list)
    if count.get("sadness", 0) + count.get("anger", 0) >= 3:
        return "disgust"
    if count.get("joy", 0) >= 3:
        return "joy"
    if len(emotion_list) >= 2 and emotion_list[-1] != emotion_list[-2]:
        return "fear"
    return emotion_list[-1]

# Spotify links
emotion_to_music = {
    "joy": "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",
    "sadness": "https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1",
    "anger": "https://open.spotify.com/playlist/37i9dQZF1DWY3PJWG3ogmJ",
    "fear": "https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO",
    "love": "https://open.spotify.com/playlist/37i9dQZF1DWVY4eLfA3XFQ",
    "disgust": "https://open.spotify.com/playlist/37i9dQZF1DX6taq20FeuKj",
    "neutral": "https://open.spotify.com/playlist/37i9dQZF1DX4WYpdgoIcn6"
}

# UI layout
st.title("🎧 AI DJ: Real-Time Mood Analyzer")
username = st.text_input("Twitter Handle", "elonmusk")
bearer_token = st.text_input("Bearer Token", type="password")

if st.button("Analyze Now"):
    with st.spinner("Analyzing..."):
        tweets = get_recent_tweets(username, bearer_token)
        emotion_memory = deque(maxlen=5)
        rows = []

        for tw in tweets:
            c = clean_tweet(tw['text'])
            if len(c) < 5:
                continue
            res = emotion_classifier(c)[0]
            emotion_memory.append(res['label'])
            rows.append({
                "Time": tw["created_at"],
                "Tweet": c,
                "Emotion": res["label"],
                "Conf": round(res["score"], 2)
            })

        df = pd.DataFrame(rows)

        if df.empty:
            st.warning("No tweets found or valid for analysis. Try a different user or token.")
        else:
            df["Time"] = pd.to_datetime(df["Time"])
            st.subheader("🧠 Analyzed Tweets")
            st.dataframe(df)

            st.subheader("📊 Emotion Distribution")
            st.pyplot(df["Emotion"].value_counts().plot.pie(autopct="%1.1f%%", figsize=(6, 6)).figure)

            st.subheader("📈 Emotion Over Time")
            st.line_chart(df.groupby([pd.Grouper(key="Time", freq="D"), "Emotion"]).size().unstack().fillna(0))

            mood = predict_next_emotion(emotion_memory)
            st.success(f"🎵 Recommended Mood: **{mood.upper()}**")
            st.markdown(f"[🎧 Listen on Spotify]({emotion_to_music[mood]})", unsafe_allow_html=True)

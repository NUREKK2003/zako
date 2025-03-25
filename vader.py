from transformers import pipeline
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
df = pd.read_csv('data/reddit.csv',sep=';')

for row in df.itertuples(index=False):
    print(row.selftext)
    text = row.selftext
    score = analyzer.polarity_scores(text)

    print(score)
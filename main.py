from transformers import pipeline
import pandas as pd

classifier = pipeline("sentiment-analysis")
df = pd.read_csv('data/reddit.csv',sep=';')

for row in df.itertuples(index=False):
    print("---------------")
    print(row.selftext)
    text = row.selftext
    result = classifier(text)
    print(result)
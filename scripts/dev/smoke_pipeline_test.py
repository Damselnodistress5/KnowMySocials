import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
from src import preprocessing, embedding, sentiment, summarization, trend_prediction

file = Path('data/sample_reviews.csv')
df = pd.read_csv(file)
print('columns', df.columns.tolist())
print('rows', len(df))
df_pre = preprocessing.preprocess_data(df)
print('preprocessed cols', df_pre.columns.tolist())
embeddings = embedding.generate_sentence_embeddings(df_pre['preprocessed_text'].astype(str).tolist())
print('emb shape', len(embeddings), type(embeddings[0]) if embeddings else None)
sent_df = sentiment.analyze_sentiment_batch(df_pre)
print('sentiment rows', len(sent_df), sent_df.columns.tolist())
summary_df = summarization.generate_summaries_batch(df_pre, use_parallel=False)
print('summaries rows', len(summary_df), summary_df.columns.tolist())
trend_df = trend_prediction.analyze_trends_batch(df_pre)
print('trend rows', len(trend_df), trend_df.columns.tolist())

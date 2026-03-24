import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os

print("--- CINEMATCH SEARCH TEST ---")

# Load model
print("Loading model (all-mpnet-base-v2)...")
model = SentenceTransformer('all-mpnet-base-v2')

# Load data
print("Loading enriched dataset...")
df = pd.read_csv("../data/enriched_tmdb_with_wiki.csv")
df['detailed_plot'] = df['detailed_plot'].fillna("")

# Perform search
query = "A man trapped alone on Mars trying to survive"
print(f"\nSEARCHING FOR: '{query}'")

# Encode query and plots
query_embedding = model.encode(query, convert_to_tensor=True)
movie_embeddings = model.encode(df['detailed_plot'].tolist()[:500], convert_to_tensor=True) # Limit to 500 for speed in this test

# Compute scores
scores = util.cos_sim(query_embedding, movie_embeddings)[0]
df_subset = df.iloc[:500].copy()
df_subset['match_percent'] = (scores.cpu().numpy() * 100).round(2)

# Sort and show top 5
top_5 = df_subset.sort_values('match_percent', ascending=False).head(5)

print("\nTOP 5 RESULTS:")
print("-" * 50)
for i, (idx, row) in enumerate(top_5.iterrows(), 1):
    print(f"{i}. {row['original_title']} ({int(row['release_date'][:4]) if pd.notnull(row['release_date']) else 'N/A'})")
    print(f"   Match: {row['match_percent']}%")
    print(f"   Plot Snippet: {row['detailed_plot'][:100]}...")
    print("-" * 50)

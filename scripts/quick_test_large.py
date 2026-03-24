import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
import time

print("="*60)
print("CINEMATCH - QUICK TEST (MAXIMUM ACCURACY MODE)")
print("="*60)

# Load models
print("\n[1/3] Loading models...")
print("  - sentence-t5-large (1.1GB)...")
bi_encoder = SentenceTransformer('sentence-t5-large')
print("  - cross-encoder-L-12 (280MB)...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
print("✓ Models loaded!")

# Load data
print("\n[2/3] Loading dataset subset (50 movies)...")
df = pd.read_csv("../data/enriched_tmdb_with_wiki.csv")
df['detailed_plot'] = df['detailed_plot'].fillna("")
df_subset = df.head(50).copy()
print(f"✓ Loaded {len(df_subset)} movies")

# Perform search
query = "A man trapped alone on Mars trying to survive"
print(f"\n[3/3] Searching for: '{query}'")

start_time = time.time()

# STAGE 1: Bi-encoder retrieval
print("  - Stage 1: Bi-encoder search...")
query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
movie_embeddings = bi_encoder.encode(df_subset['detailed_plot'].tolist(), convert_to_tensor=True, show_progress_bar=False)
bi_scores = util.cos_sim(query_embedding, movie_embeddings)[0]

# Get top 30 candidates
top_30_indices = bi_scores.topk(30).indices.cpu().numpy()
candidates = df_subset.iloc[top_30_indices].copy()
candidates['bi_score'] = bi_scores[top_30_indices].cpu().numpy()

# STAGE 2: Cross-encoder re-ranking
print("  - Stage 2: Cross-encoder re-ranking...")
pairs = [[query, row['detailed_plot'][:1000]] for _, row in candidates.iterrows()]
cross_scores = cross_encoder.predict(pairs)

# Add scores and sort
candidates['match_score'] = cross_scores
candidates['match_percent'] = (cross_scores * 100).clip(0, 100).round(2)
top_5 = candidates.sort_values('match_score', ascending=False).head(5)

elapsed = time.time() - start_time
print(f"✓ Search completed in {elapsed:.2f}s")

print("\nTOP 5 RESULTS (MAXIMUM ACCURACY):")
print("-" * 60)
for i, (idx, row) in enumerate(top_5.iterrows(), 1):
    print(f"{i}. {row['original_title']} ({int(row['release_date'][:4]) if pd.notnull(row['release_date']) else 'N/A'})")
    print(f"   MATCH SCORE: {row['match_percent']}%")
    print(f"   Bi-encoder: {row['bi_score']:.3f} | Cross-encoder: {row['match_score']:.3f}")
    print(f"   Plot Snippet: {row['detailed_plot'][:120]}...")
    print("-" * 60)

print("\nExpected: 'The Martian' should be #1 with 90%+ match.")
print("="*60)

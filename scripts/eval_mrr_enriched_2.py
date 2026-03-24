"""
MRR Evaluation: Enriched Dataset - Model 2 (MiniLM - CHUNKED)
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

TEST_QUERIES = [
    ("A man trapped alone on Mars trying to survive", "The Martian", 5),
    ("A cowboy toy who feels replaced by a space ranger toy", "Toy Story", 3),
    ("Animated movie about emotions inside a girl's head", "Inside Out", 3),
    ("Wizard boy fights dark lord at magic school", "Harry Potter and the Philosopher's Stone", 5),
    ("A robot left alone on Earth falls in love", "WALL·E", 5),
    ("Dream thieves plant ideas in people's minds", "Inception", 3),
    ("Blue aliens fight humans on a distant planet", "Avatar", 5),
    ("A fish gets lost and his father searches for him", "Finding Nemo", 3),
    ("Time travelers prevent robot apocalypse", "The Terminator", 5),
    ("A man relives the same day over and over", "Groundhog Day", 5),
    ("Superhero team fights alien invasion in New York", "The Avengers", 5),
    ("A writer creates a fake family to adopt children", "Despicable Me", 5),
    ("Talking toys come to life when humans aren't around", "Toy Story", 3),
    ("A young lion prince reclaims his kingdom", "The Lion King", 3),
    ("Dinosaurs brought back to life in a theme park", "Jurassic Park", 3),
    ("A billionaire becomes a bat-themed vigilante", "Batman Begins", 5),
    ("Prisoners plan an elaborate escape from a maximum security prison", "The Shawshank Redemption", 5),
    ("A computer hacker discovers reality is a simulation", "The Matrix", 3),
    ("A clownfish and blue tang fish search across the ocean", "Finding Nemo", 3),
    ("An ogre and donkey rescue a princess from a dragon", "Shrek", 5),
]

print("LOADING ENRICHED DATASET")
df_enriched = pd.read_csv("../data/enriched_tmdb_with_wiki.csv")
if 'soup' in df_enriched.columns:
    df_enriched['combined_text'] = df_enriched['soup'].fillna("")
elif 'detailed_plot' in df_enriched.columns:
    df_enriched['combined_text'] = df_enriched['detailed_plot'].fillna("") + " . " + df_enriched['genres_display'].fillna("")
else:
    df_enriched['combined_text'] = df_enriched['overview'].fillna("")

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = df_enriched['combined_text'].tolist()

chunk_size = 1000
embeddings_list = []
for i in range(0, len(texts), chunk_size):
    chunk = texts[i:i + chunk_size]
    print(f"  Encoding chunk {i//chunk_size + 1} of {len(texts)//chunk_size + 1}...")
    chunk_embeddings = model.encode(chunk, convert_to_tensor=True, show_progress_bar=False)
    embeddings_list.append(chunk_embeddings)

movie_embeddings = torch.cat(embeddings_list, dim=0)

k_values = [1, 3, 5, 10]
precision_at_k = {k: [] for k in k_values}
reciprocal_ranks = []

for query, expected_title, _ in TEST_QUERIES:
    query_vec = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_vec, movie_embeddings)[0]
    top_indices = scores.topk(20).indices.cpu().numpy()
    top_titles = df_enriched.iloc[top_indices]['original_title'].values
    
    for k in k_values:
        precision_at_k[k].append(1.0 if expected_title in top_titles[:k] else 0.0)
    
    if expected_title in top_titles:
        rank = list(top_titles).index(expected_title) + 1
        reciprocal_ranks.append(1.0 / rank)
    else:
        reciprocal_ranks.append(0.0)

res = {'dataset': 'Enriched', 'model': 'all-MiniLM-L6-v2', 'MRR': np.mean(reciprocal_ranks)}
for k in k_values:
    res[f'P@{k}'] = np.mean(precision_at_k[k]) * 100

pd.DataFrame([res]).to_csv('results_enriched_2.csv', index=False)
print("Saved: results_enriched_2.csv")

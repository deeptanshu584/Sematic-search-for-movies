"""
MRR Evaluation: Base TMDB Dataset Only
"""

import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util

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

MODELS = {
    "all-mpnet-base-v2": "all-mpnet-base-v2",
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "e5-small-v2": "intfloat/e5-small-v2",
}

print("LOADING BASE TMDB DATASET")
df_base = pd.read_csv("../data/tmdb_5000_movies.csv")
df_base['overview'] = df_base['overview'].fillna("")

def parse_genres(genre_str):
    try:
        return [g['name'] for g in json.loads(genre_str)]
    except:
        return []

df_base['genres_display'] = df_base['genres'].apply(parse_genres).apply(lambda x: ", ".join(x) if x else "Unknown")
df_base['combined_text'] = df_base['original_title'].fillna("") + " . " + df_base['overview'] + " . " + df_base['genres_display']

def evaluate_dataset_model(df, model_name, model_path, dataset_label):
    print(f"Testing: {dataset_label} + {model_name}")
    model = SentenceTransformer(model_path)
    texts = df['combined_text'].tolist()
    if "e5" in model_path:
        texts = ["passage: " + t for t in texts]
    movie_embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    
    k_values = [1, 3, 5, 10]
    precision_at_k = {k: [] for k in k_values}
    reciprocal_ranks = []
    
    for query, expected_title, _ in TEST_QUERIES:
        query_text = ("query: " + query) if "e5" in model_path else query
        query_vec = model.encode(query_text, convert_to_tensor=True)
        scores = util.cos_sim(query_vec, movie_embeddings)[0]
        top_indices = scores.topk(20).indices.cpu().numpy()
        top_titles = df.iloc[top_indices]['original_title'].values
        
        for k in k_values:
            precision_at_k[k].append(1.0 if expected_title in top_titles[:k] else 0.0)
        
        if expected_title in top_titles:
            rank = list(top_titles).index(expected_title) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    results = {'dataset': dataset_label, 'model': model_name, 'MRR': np.mean(reciprocal_ranks)}
    for k in k_values:
        results[f'P@{k}'] = np.mean(precision_at_k[k]) * 100
    return results

all_results = []
for model_name, model_path in MODELS.items():
    all_results.append(evaluate_dataset_model(df_base, model_name, model_path, "Base TMDB"))

pd.DataFrame(all_results).to_csv('results_base.csv', index=False)
print("Saved: results_base.csv")

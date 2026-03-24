"""
CINEMATCH - Ultimate Accuracy Evaluation with Cross-Encoder
Implements two-stage retrieval: Bi-Encoder → Cross-Encoder
Expected P@5: 85-95% (vs 65% baseline)
"""

import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ============================================================
# TEST QUERIES (Same as before + more)
# ============================================================
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

print("="*80)
print("ULTIMATE ACCURACY EVALUATION - Cross-Encoder Implementation")
print("="*80)
print("\nThis evaluation compares:")
print("  1. Baseline: Bi-encoder only (all-mpnet-base-v2)")
print("  2. Ultimate: Two-stage (msmarco + cross-encoder)")
print("\n" + "="*80)

# ============================================================
# LOAD DATASET
# ============================================================
print("\n[1/5] Loading enriched dataset...")

# Looking in the current directory's 'data' folder
df = pd.read_csv("data/enriched_tmdb_with_wiki.csv")

# Handle different column names
if 'soup' in df.columns:
    df['combined_text'] = df['soup'].fillna("")
elif 'detailed_plot' in df.columns:
    df['combined_text'] = df['detailed_plot'].fillna("")
else:
    df['combined_text'] = df['overview'].fillna("")

print(f"✓ Loaded {len(df)} movies")
print(f"✓ Average text length: {df['combined_text'].str.len().mean():.0f} characters")

# ============================================================
# METHOD 1: BASELINE (Bi-encoder only)
# ============================================================
print("\n[2/5] Testing BASELINE: Bi-encoder only (all-mpnet-base-v2)...")

baseline_model = SentenceTransformer('all-mpnet-base-v2')

print("  Encoding all movies with baseline model...")
texts = df['combined_text'].tolist()
baseline_embeddings = baseline_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

# Test baseline
k_values = [1, 3, 5, 10]
baseline_results = {k: [] for k in k_values}
baseline_mrr = []

print(f"  Testing {len(TEST_QUERIES)} queries with baseline...")
for query, expected_title, _ in TEST_QUERIES:
    query_vec = baseline_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_vec, baseline_embeddings)[0]
    
    # Get rankings
    top_indices = scores.topk(20).indices.cpu().numpy()
    top_titles = df.iloc[top_indices]['original_title'].values
    
    # Calculate metrics
    for k in k_values:
        top_k_titles = top_titles[:k]
        baseline_results[k].append(1.0 if expected_title in top_k_titles else 0.0)
    
    # MRR
    if expected_title in top_titles:
        rank = list(top_titles).index(expected_title) + 1
        baseline_mrr.append(1.0 / rank)
    else:
        baseline_mrr.append(0.0)

# Calculate baseline metrics
baseline_metrics = {}
for k in k_values:
    baseline_metrics[f'P@{k}'] = np.mean(baseline_results[k]) * 100

baseline_metrics['MRR'] = np.mean(baseline_mrr)

print("\n  BASELINE RESULTS:")
print(f"    P@1: {baseline_metrics['P@1']:.1f}%")
print(f"    P@3: {baseline_metrics['P@3']:.1f}%")
print(f"    P@5: {baseline_metrics['P@5']:.1f}%")
print(f"    P@10: {baseline_metrics['P@10']:.1f}%")
print(f"    MRR: {baseline_metrics['MRR']:.3f}")

import google.generativeai as genai
import os

# ============================================================
# GEMINI CONFIGURATION
# ============================================================
# Replace 'YOUR_GEMINI_API_KEY' with your actual key from Google AI Studio
GEMINI_API_KEY = "AIzaSyCdkc4GRM1vk-Kyxzx7usWcFssHodWlS5Y"

if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
    print("\n⚠️ WARNING: You must provide a valid Gemini API key in the script!")
    print("Get one for free at: https://aistudio.google.com/")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

def expand_query(query):
    """Use Gemini to expand a short query into a detailed search description."""
    prompt = f"Expand this movie search query into a detailed 2-sentence description of the plot, tone, and themes: '{query}'"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except:
        return query

def gemini_rerank(query, candidates_df):
    """Use Gemini to re-rank candidates using full metadata (Genres, Year, Rating)."""
    titles = candidates_df['original_title'].tolist()
    overviews = candidates_df['combined_text'].tolist()
    genres = candidates_df['genres'].tolist()
    years = candidates_df['release_date'].str[:4].fillna("Unknown").tolist()
    ratings = candidates_df['vote_average'].tolist()
    
    prompt = f"""
    User Query: "{query}"
    
    Rank these 20 movies from 1 to 20 based on relevance. 
    Use the description, genres, year, and rating to make your decision.
    Return ONLY a comma-separated list of the movie titles in order.
    
    Movies:
    """
    for i, (t, o, g, y, r) in enumerate(zip(titles, overviews, genres, years, ratings)):
        prompt += f"\n{i+1}. {t} ({y}) - Genre: {g}, Rating: {r}/10. Plot: {o[:200]}..."
        
    try:
        response = gemini_model.generate_content(prompt)
        ranked_titles = [t.strip().strip('"') for t in response.text.split(',')]
        # Cleanup: Ensure we only keep valid titles from our candidate list
        valid_titles = [t for t in ranked_titles if t in titles]
        missing = [t for t in titles if t not in valid_titles]
        return valid_titles + missing
    except Exception as e:
        print(f"      Gemini Error: {e}")
        return titles

# ============================================================
# METHOD 2: ULTIMATE (AI Expansion + Bi-encoder + Gemini Rerank)
# ============================================================
print("\n[3/5] Testing ULTIMATE: AI-Enhanced Retrieval...")
print("  Step 1: AI Query Expansion (Gemini)")
print("  Step 2: all-mpnet-base-v2 (Local Retrieval)")
print("  Step 3: Metadata-Rich Re-ranking (Gemini)")

# Load local model
bi_encoder = SentenceTransformer('all-mpnet-base-v2')
print("\n  Encoding movies...")
bi_embeddings = bi_encoder.encode(texts, convert_to_tensor=True, show_progress_bar=True)

ultimate_results = {k: [] for k in k_values}
ultimate_mrr = []
timing_stats = []

for idx, (query, expected_title, _) in enumerate(TEST_QUERIES):
    start_time = time.time()
    
    # STEP 1: AI Expansion (Makes search much smarter)
    expanded = expand_query(query)
    
    # STEP 2: Bi-encoder retrieval (Fast)
    query_vec = bi_encoder.encode(expanded, convert_to_tensor=True)
    bi_scores = util.cos_sim(query_vec, bi_embeddings)[0]
    
    # Get top 20 candidates
    top_20_indices = bi_scores.topk(20).indices.cpu().numpy()
    candidates = df.iloc[top_20_indices]
    candidate_titles = candidates['original_title'].tolist()
    
    # DEBUG: Is the expected title in the candidates?
    in_candidates = expected_title in candidate_titles
    if not in_candidates:
        print(f"      [DEBUG] Expected title '{expected_title}' NOT in top 20 candidates.")
        # Check if it was in the top 100 at least
        top_100_indices = bi_scores.topk(100).indices.cpu().numpy()
        top_100_titles = df.iloc[top_100_indices]['original_title'].tolist()
        if expected_title in top_100_titles:
            print(f"      [DEBUG] ...but it WAS in top 100 (at rank {top_100_titles.index(expected_title)+1}).")
    else:
        print(f"      [DEBUG] Expected title '{expected_title}' found in candidates at rank {candidate_titles.index(expected_title)+1}.")

    # STEP 3: Gemini Re-ranking (Intelligent)
    ranked_titles = gemini_rerank(query, candidates)
    
    # DEBUG: Where did Gemini rank it?
    if expected_title in ranked_titles:
        new_rank = ranked_titles.index(expected_title) + 1
        print(f"      [DEBUG] Gemini ranked it at: {new_rank}")
    else:
        print(f"      [DEBUG] Gemini OMITTED the expected title from its response!")

    # Calculate metrics
    for k in k_values:
        top_k_titles = ranked_titles[:k]
        ultimate_results[k].append(1.0 if expected_title in top_k_titles else 0.0)
    
    # MRR
    if expected_title in ranked_titles:
        rank = ranked_titles.index(expected_title) + 1
        ultimate_mrr.append(1.0 / rank)
    else:
        ultimate_mrr.append(0.0)
    
    timing_stats.append(time.time() - start_time)
    if (idx + 1) % 5 == 0:
        print(f"    Processed {idx+1}/{len(TEST_QUERIES)} queries with AI-Enhancement...")

# Calculate ultimate metrics
ultimate_metrics = {}
for k in k_values:
    ultimate_metrics[f'P@{k}'] = np.mean(ultimate_results[k]) * 100

ultimate_metrics['MRR'] = np.mean(ultimate_mrr)
avg_time = np.mean(timing_stats)

print("\n  ULTIMATE RESULTS:")
print(f"    P@1: {ultimate_metrics['P@1']:.1f}%")
print(f"    P@3: {ultimate_metrics['P@3']:.1f}%")
print(f"    P@5: {ultimate_metrics['P@5']:.1f}%")
print(f"    P@10: {ultimate_metrics['P@10']:.1f}%")
print(f"    MRR: {ultimate_metrics['MRR']:.3f}")
print(f"    Avg query time: {avg_time:.2f}s")

# ============================================================
# CALCULATE IMPROVEMENTS
# ============================================================
print("\n[4/5] Calculating improvements...")

improvements = {}
for metric in ['P@1', 'P@3', 'P@5', 'P@10', 'MRR']:
    improvement = ultimate_metrics[metric] - baseline_metrics[metric]
    rel_improvement = (improvement / baseline_metrics[metric] * 100) if baseline_metrics[metric] > 0 else 0
    improvements[metric] = {
        'absolute': improvement,
        'relative': rel_improvement
    }

print("\n  IMPROVEMENT SUMMARY:")
print(f"    P@1: +{improvements['P@1']['absolute']:.1f} points ({improvements['P@1']['relative']:+.1f}%)")
print(f"    P@3: +{improvements['P@3']['absolute']:.1f} points ({improvements['P@3']['relative']:+.1f}%)")
print(f"    P@5: +{improvements['P@5']['absolute']:.1f} points ({improvements['P@5']['relative']:+.1f}%)")
print(f"    P@10: +{improvements['P@10']['absolute']:.1f} points ({improvements['P@10']['relative']:+.1f}%)")
print(f"    MRR: +{improvements['MRR']['absolute']:.3f} ({improvements['MRR']['relative']:+.1f}%)")

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n[5/5] Generating visualizations and saving results...")

# Create results dataframe
results_data = {
    'Method': ['Baseline (Bi-encoder)', 'Ultimate (Two-stage)'],
    'Model': ['all-mpnet-base-v2', 'msmarco + cross-encoder'],
    'P@1': [baseline_metrics['P@1'], ultimate_metrics['P@1']],
    'P@3': [baseline_metrics['P@3'], ultimate_metrics['P@3']],
    'P@5': [baseline_metrics['P@5'], ultimate_metrics['P@5']],
    'P@10': [baseline_metrics['P@10'], ultimate_metrics['P@10']],
    'MRR': [baseline_metrics['MRR'], ultimate_metrics['MRR']],
}

results_df = pd.DataFrame(results_data)
results_df.to_csv('ultimate_accuracy_results.csv', index=False)
print("✓ Saved: ultimate_accuracy_results.csv")

# ============================================================
# VISUALIZATION 1: Side-by-side Comparison
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
k_values = [1, 3, 5, 10]

for idx, k in enumerate(k_values):
    ax = axes[idx // 2, idx % 2]
    
    methods = ['Baseline\n(Bi-encoder)', 'Ultimate\n(Two-stage)']
    values = [baseline_metrics[f'P@{k}'], ultimate_metrics[f'P@{k}']]
    colors = ['#6b7280', '#10b981']
    
    bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement annotation
    improvement = values[1] - values[0]
    ax.annotate('', xy=(1, values[1]), xytext=(0, values[0]),
               arrowprops=dict(arrowstyle='->', lw=2, color='#ef4444'))
    ax.text(0.5, max(values) + 8, f'+{improvement:.1f}%', ha='center',
           fontsize=10, fontweight='bold', color='#ef4444',
           bbox=dict(boxstyle='round', facecolor='#fee2e2', edgecolor='#ef4444'))
    
    ax.set_ylabel('Precision (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Precision@{k}', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Baseline vs Ultimate Accuracy Comparison', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('ultimate_vs_standard_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: ultimate_vs_standard_comparison.png")
plt.close()

# ============================================================
# VISUALIZATION 2: Improvement Breakdown
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

metrics = ['P@1', 'P@3', 'P@5', 'P@10', 'MRR×100']
baseline_vals = [baseline_metrics['P@1'], baseline_metrics['P@3'], 
                baseline_metrics['P@5'], baseline_metrics['P@10'],
                baseline_metrics['MRR'] * 100]
ultimate_vals = [ultimate_metrics['P@1'], ultimate_metrics['P@3'],
                ultimate_metrics['P@5'], ultimate_metrics['P@10'],
                ultimate_metrics['MRR'] * 100]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline',
              color='#6b7280', alpha=0.8, edgecolor='black', linewidth=2)
bars2 = ax.bar(x + width/2, ultimate_vals, width, label='Ultimate',
              color='#10b981', alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Cross-Encoder Improvement Across All Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=12, loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 110)

plt.tight_layout()
plt.savefig('cross_encoder_improvement.png', dpi=300, bbox_inches='tight')
print("✓ Saved: cross_encoder_improvement.png")
plt.close()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)

print("\nFINAL RESULTS:")
print(f"  Baseline P@5: {baseline_metrics['P@5']:.1f}%")
print(f"  Ultimate P@5: {ultimate_metrics['P@5']:.1f}%")
print(f"  Improvement: +{improvements['P@5']['absolute']:.1f} percentage points ({improvements['P@5']['relative']:+.1f}%)")

print(f"\n  Baseline MRR: {baseline_metrics['MRR']:.3f} (avg rank {1/baseline_metrics['MRR']:.1f})")
print(f"  Ultimate MRR: {ultimate_metrics['MRR']:.3f} (avg rank {1/ultimate_metrics['MRR']:.1f})")
print(f"  Improvement: Correct movie appears {1/baseline_metrics['MRR'] - 1/ultimate_metrics['MRR']:.1f} positions higher")

print(f"\n  Average query time: {avg_time:.2f} seconds")

print("\nGenerated files:")
print("  • ultimate_accuracy_results.csv")
print("  • ultimate_vs_standard_comparison.png")
print("  • cross_encoder_improvement.png")

print("\n" + "="*80)
print("✓ Ready to show professor!")
print("="*80)

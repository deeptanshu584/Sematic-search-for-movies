"""
Comprehensive Evaluation: Base TMDB vs Enriched Dataset
Compares semantic search performance across datasets and models
"""

import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ============================================================
# TEST QUERIES
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

# ============================================================
# MODELS TO TEST
# ============================================================
MODELS = {
    "all-mpnet-base-v2": "all-mpnet-base-v2",
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "e5-small-v2": "intfloat/e5-small-v2",
}

# ============================================================
# LOAD BASE TMDB DATASET
# ============================================================
print("="*80)
print("LOADING BASE TMDB DATASET")
print("="*80)

df_base = pd.read_csv("../data/tmdb_5000_movies.csv")
df_base['overview'] = df_base['overview'].fillna("")

def parse_genres(genre_str):
    try:
        return [g['name'] for g in json.loads(genre_str)]
    except:
        return []

df_base['genre_list'] = df_base['genres'].apply(parse_genres)
df_base['genres_display'] = df_base['genre_list'].apply(lambda x: ", ".join(x) if x else "Unknown")

# Simple combined text for base dataset
df_base['combined_text'] = (
    df_base['original_title'].fillna("") + " . " +
    df_base['overview'] + " . " +
    df_base['genres_display']
)

print(f"* Loaded {len(df_base)} movies")
print(f"* Average text length: {df_base['combined_text'].str.len().mean():.0f} characters")

# ============================================================
# LOAD ENRICHED DATASET
# ============================================================
print("\n" + "="*80)
print("LOADING ENRICHED DATASET (TMDB + Wikipedia)")
print("="*80)

try:
    df_enriched = pd.read_csv("../data/enriched_tmdb_with_wiki.csv")
    
    # Handle different column names from different enrichment scripts
    if 'soup' in df_enriched.columns:
        df_enriched['combined_text'] = df_enriched['soup'].fillna("")
    elif 'detailed_plot' in df_enriched.columns:
        df_enriched['combined_text'] = (
            df_enriched['detailed_plot'].fillna("") + " . " + 
            df_enriched['genres_display'].fillna("")
        )
    else:
        df_enriched['combined_text'] = df_enriched['overview'].fillna("")
    
    print(f"* Loaded {len(df_enriched)} movies")
    print(f"* Average text length: {df_enriched['combined_text'].str.len().mean():.0f} characters")
    
    enriched_available = True
    
except FileNotFoundError:
    print("⚠ Enriched dataset not found!")
    print("  Make sure enriched_tmdb_with_wiki.csv is in data/ folder")
    enriched_available = False
    df_enriched = None
    exit(1)

# ============================================================
# EVALUATION FUNCTION
# ============================================================
def evaluate_dataset_model(df, model_name, model_path, dataset_label):
    """
    Evaluate a specific dataset + model combination
    Returns precision@k metrics
    """
    print(f"\n  Testing: {dataset_label} + {model_name}")
    
    # Load model
    model = SentenceTransformer(model_path)
    
    # Encode all movies
    texts = df['combined_text'].tolist()
    if "e5" in model_path:
        texts = ["passage: " + t for t in texts]
    
    print(f"    Encoding {len(texts)} movies...")
    movie_embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    
    # Test queries
    k_values = [1, 3, 5, 10]
    precision_at_k = {k: [] for k in k_values}
    
    print(f"    Testing {len(TEST_QUERIES)} queries...")
    for query, expected_title, _ in TEST_QUERIES:
        query_text = ("query: " + query) if "e5" in model_path else query
        query_vec = model.encode(query_text, convert_to_tensor=True)
        scores = util.cos_sim(query_vec, movie_embeddings)[0]
        
        for k in k_values:
            top_indices = scores.topk(k).indices.cpu().numpy()
            top_titles = df.iloc[top_indices]['original_title'].values
            precision_at_k[k].append(1.0 if expected_title in top_titles else 0.0)
    
    # Calculate metrics
    results = {
        'dataset': dataset_label,
        'model': model_name,
    }
    
    for k in k_values:
        precision = np.mean(precision_at_k[k]) * 100
        results[f'P@{k}'] = precision
    
    print(f"    * P@1: {results['P@1']:.1f}%  |  P@3: {results['P@3']:.1f}%  |  P@5: {results['P@5']:.1f}%  |  P@10: {results['P@10']:.1f}%")
    
    return results

# ============================================================
# RUN ALL EVALUATIONS
# ============================================================
print("\n" + "="*80)
print("RUNNING COMPREHENSIVE EVALUATION")
print("="*80)
print("\nThis will test 6 configurations (3 models × 2 datasets)")
print("Estimated time: 15-45 minutes (first run downloads models)")
print("="*80)

all_results = []

# Evaluate Base TMDB
print("\n[1/2] EVALUATING BASE TMDB DATASET")
print("-" * 80)
for model_name, model_path in MODELS.items():
    result = evaluate_dataset_model(df_base, model_name, model_path, "Base TMDB")
    all_results.append(result)

# Evaluate Enriched Dataset
print("\n[2/2] EVALUATING ENRICHED DATASET")
print("-" * 80)
for model_name, model_path in MODELS.items():
    result = evaluate_dataset_model(df_enriched, model_name, model_path, "Enriched")
    all_results.append(result)

# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================
results_df = pd.DataFrame(all_results)
print("\n" + "="*80)
print("COMPLETE RESULTS TABLE")
print("="*80)
print(results_df.to_string(index=False))

# Save to CSV
results_df.to_csv('evaluation_results.csv', index=False)
print("\n* Saved: evaluation_results.csv")

# ============================================================
# VISUALIZATION 1: PRECISION@K COMPARISON
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
k_values = [1, 3, 5, 10]

for idx, k in enumerate(k_values):
    ax = axes[idx // 2, idx % 2]
    
    # Prepare data
    base_data = results_df[results_df['dataset'] == 'Base TMDB']
    enriched_data = results_df[results_df['dataset'] == 'Enriched']
    
    x = np.arange(len(MODELS))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, base_data[f'P@{k}'].values, width, 
                   label='Base TMDB', color='#6b7280', alpha=0.8, edgecolor='white', linewidth=2)
    bars2 = ax.bar(x + width/2, enriched_data[f'P@{k}'].values, width,
                   label='Enriched', color='#f43f5e', alpha=0.8, edgecolor='white', linewidth=2)
    
    ax.set_ylabel('Precision (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Precision@{k}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS.keys(), rotation=15, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 110)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Base TMDB vs Enriched Dataset: Precision Comparison', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('dataset_comparison_precision.png', dpi=300, bbox_inches='tight')
print("* Saved: dataset_comparison_precision.png")

# ============================================================
# VISUALIZATION 2: IMPROVEMENT HEATMAP
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate improvement percentages
improvement_matrix = []
for model_name in MODELS.keys():
    base_row = results_df[(results_df['dataset'] == 'Base TMDB') & (results_df['model'] == model_name)]
    enrich_row = results_df[(results_df['dataset'] == 'Enriched') & (results_df['model'] == model_name)]
    
    improvements = []
    for k in [1, 3, 5, 10]:
        base_val = base_row[f'P@{k}'].values[0]
        enrich_val = enrich_row[f'P@{k}'].values[0]
        improvement = enrich_val - base_val
        improvements.append(improvement)
    
    improvement_matrix.append(improvements)

# Plot heatmap
im = ax.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=30)

ax.set_xticks(np.arange(4))
ax.set_yticks(np.arange(len(MODELS)))
ax.set_xticklabels(['P@1', 'P@3', 'P@5', 'P@10'], fontsize=11, fontweight='bold')
ax.set_yticklabels(MODELS.keys(), fontsize=11)

# Add text annotations
for i in range(len(MODELS)):
    for j in range(4):
        value = improvement_matrix[i][j]
        color = 'white' if abs(value) > 10 else 'black'
        text = ax.text(j, i, f'{value:+.1f}%', ha="center", va="center", 
                      color=color, fontweight='bold', fontsize=12)

ax.set_title('Improvement: Enriched vs Base TMDB (percentage points)', 
             fontsize=14, fontweight='bold', pad=20)
cbar = fig.colorbar(im, ax=ax, label='Improvement (%)')
cbar.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig('dataset_improvement_heatmap.png', dpi=300, bbox_inches='tight')
print("* Saved: dataset_improvement_heatmap.png")

# ============================================================
# VISUALIZATION 3: BEST CONFIGURATION
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Find best configurations
best_base = results_df[results_df['dataset'] == 'Base TMDB'].loc[results_df[results_df['dataset'] == 'Base TMDB']['P@5'].idxmax()]
best_enriched = results_df[results_df['dataset'] == 'Enriched'].loc[results_df[results_df['dataset'] == 'Enriched']['P@5'].idxmax()]

configs = ['Base TMDB\n(Best Model)', 'Enriched\n(Best Model)']
p1_vals = [best_base['P@1'], best_enriched['P@1']]
p3_vals = [best_base['P@3'], best_enriched['P@3']]
p5_vals = [best_base['P@5'], best_enriched['P@5']]
p10_vals = [best_base['P@10'], best_enriched['P@10']]

x = np.arange(len(configs))
width = 0.2

bars1 = ax.bar(x - 1.5*width, p1_vals, width, label='P@1', color='#ef4444', edgecolor='white', linewidth=2)
bars2 = ax.bar(x - 0.5*width, p3_vals, width, label='P@3', color='#f59e0b', edgecolor='white', linewidth=2)
bars3 = ax.bar(x + 0.5*width, p5_vals, width, label='P@5', color='#10b981', edgecolor='white', linewidth=2)
bars4 = ax.bar(x + 1.5*width, p10_vals, width, label='P@10', color='#3b82f6', edgecolor='white', linewidth=2)

# Add value labels
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
ax.set_title('Best Configuration Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 110)

# Add model names
ax.text(0, -15, f"Model: {best_base['model']}", ha='center', fontsize=10, color='#666')
ax.text(1, -15, f"Model: {best_enriched['model']}", ha='center', fontsize=10, color='#666')

plt.tight_layout()
plt.savefig('best_configuration.png', dpi=300, bbox_inches='tight')
print("* Saved: best_configuration.png")

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for k in [1, 3, 5, 10]:
    base_avg = results_df[results_df['dataset'] == 'Base TMDB'][f'P@{k}'].mean()
    enrich_avg = results_df[results_df['dataset'] == 'Enriched'][f'P@{k}'].mean()
    improvement = enrich_avg - base_avg
    
    print(f"\nPrecision@{k}:")
    print(f"  Base TMDB average:    {base_avg:.1f}%")
    print(f"  Enriched average:     {enrich_avg:.1f}%")
    print(f"  Average improvement:  {improvement:+.1f} percentage points")

print("\n" + "-"*80)
print("BEST OVERALL CONFIGURATION:")
best_config = results_df.loc[results_df['P@5'].idxmax()]
print(f"  Dataset: {best_config['dataset']}")
print(f"  Model: {best_config['model']}")
print(f"  P@1: {best_config['P@1']:.1f}%")
print(f"  P@3: {best_config['P@3']:.1f}%")
print(f"  P@5: {best_config['P@5']:.1f}%")
print(f"  P@10: {best_config['P@10']:.1f}%")

print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  - dataset_comparison_precision.png")
print("  - dataset_improvement_heatmap.png")
print("  - best_configuration.png")
print("  - evaluation_results.csv")
print("="*80)

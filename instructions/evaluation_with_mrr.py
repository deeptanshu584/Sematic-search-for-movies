"""
Comprehensive Evaluation with MRR: Base TMDB vs Enriched Dataset
Includes Precision@K AND Mean Reciprocal Rank (MRR) metrics
"""

import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns

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

df_base['combined_text'] = (
    df_base['original_title'].fillna("") + " . " +
    df_base['overview'] + " . " +
    df_base['genres_display']
)

print(f"✓ Loaded {len(df_base)} movies")
print(f"✓ Average text length: {df_base['combined_text'].str.len().mean():.0f} characters")

# ============================================================
# LOAD ENRICHED DATASET
# ============================================================
print("\n" + "="*80)
print("LOADING ENRICHED DATASET (TMDB + Wikipedia)")
print("="*80)

df_enriched = pd.read_csv("../data/enriched_tmdb_with_wiki.csv")

if 'soup' in df_enriched.columns:
    df_enriched['combined_text'] = df_enriched['soup'].fillna("")
elif 'detailed_plot' in df_enriched.columns:
    df_enriched['combined_text'] = (
        df_enriched['detailed_plot'].fillna("") + " . " + 
        df_enriched['genres_display'].fillna("")
    )
else:
    df_enriched['combined_text'] = df_enriched['overview'].fillna("")

print(f"✓ Loaded {len(df_enriched)} movies")
print(f"✓ Average text length: {df_enriched['combined_text'].str.len().mean():.0f} characters")

# ============================================================
# EVALUATION FUNCTION WITH MRR
# ============================================================
def evaluate_dataset_model(df, model_name, model_path, dataset_label):
    """
    Evaluate a specific dataset + model combination
    Returns precision@k AND MRR metrics
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
    reciprocal_ranks = []
    
    print(f"    Testing {len(TEST_QUERIES)} queries...")
    for query, expected_title, _ in TEST_QUERIES:
        query_text = ("query: " + query) if "e5" in model_path else query
        query_vec = model.encode(query_text, convert_to_tensor=True)
        scores = util.cos_sim(query_vec, movie_embeddings)[0]
        
        # Get top results for this query
        top_indices = scores.topk(20).indices.cpu().numpy()  # Get top 20 for MRR calculation
        top_titles = df.iloc[top_indices]['original_title'].values
        
        # Calculate Precision@K
        for k in k_values:
            top_k_titles = top_titles[:k]
            precision_at_k[k].append(1.0 if expected_title in top_k_titles else 0.0)
        
        # Calculate Reciprocal Rank
        # Find the rank of the correct movie (1-indexed)
        if expected_title in top_titles:
            rank = list(top_titles).index(expected_title) + 1
            reciprocal_rank = 1.0 / rank
        else:
            # If not found in top 20, give it a very low score
            reciprocal_rank = 0.0
        
        reciprocal_ranks.append(reciprocal_rank)
    
    # Calculate metrics
    results = {
        'dataset': dataset_label,
        'model': model_name,
        'MRR': np.mean(reciprocal_ranks),  # NEW: Mean Reciprocal Rank
    }
    
    for k in k_values:
        precision = np.mean(precision_at_k[k]) * 100
        results[f'P@{k}'] = precision
    
    print(f"    ✓ MRR: {results['MRR']:.3f}  |  P@1: {results['P@1']:.1f}%  |  P@3: {results['P@3']:.1f}%  |  P@5: {results['P@5']:.1f}%  |  P@10: {results['P@10']:.1f}%")
    
    return results

# ============================================================
# RUN ALL EVALUATIONS
# ============================================================
print("\n" + "="*80)
print("RUNNING COMPREHENSIVE EVALUATION WITH MRR")
print("="*80)
print("\nThis will test 6 configurations (3 models × 2 datasets)")
print("Now includes Mean Reciprocal Rank (MRR) metric!")
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

# Reorder columns to put MRR first after model info
column_order = ['dataset', 'model', 'MRR', 'P@1', 'P@3', 'P@5', 'P@10']
results_df = results_df[column_order]

print("\n" + "="*80)
print("COMPLETE RESULTS TABLE (WITH MRR)")
print("="*80)
print(results_df.to_string(index=False))

# Save to CSV
results_df.to_csv('evaluation_results_with_mrr.csv', index=False)
print("\n✓ Saved: evaluation_results_with_mrr.csv")

# ============================================================
# VISUALIZATION 1: PRECISION@K COMPARISON (SAME AS BEFORE)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
k_values = [1, 3, 5, 10]

for idx, k in enumerate(k_values):
    ax = axes[idx // 2, idx % 2]
    
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
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Base TMDB vs Enriched Dataset: Precision Comparison', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('dataset_comparison_precision.png', dpi=300, bbox_inches='tight')
print("✓ Saved: dataset_comparison_precision.png")

# ============================================================
# VISUALIZATION 2: IMPROVEMENT HEATMAP (SAME AS BEFORE)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

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

im = ax.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=30)

ax.set_xticks(np.arange(4))
ax.set_yticks(np.arange(len(MODELS)))
ax.set_xticklabels(['P@1', 'P@3', 'P@5', 'P@10'], fontsize=11, fontweight='bold')
ax.set_yticklabels(MODELS.keys(), fontsize=11)

for i in range(len(MODELS)):
    for j in range(4):
        value = improvement_matrix[i][j]
        color = 'white' if abs(value) > 10 else 'black'
        ax.text(j, i, f'{value:+.1f}%', ha="center", va="center", 
                color=color, fontweight='bold', fontsize=12)

ax.set_title('Improvement: Enriched vs Base TMDB (percentage points)', 
             fontsize=14, fontweight='bold', pad=20)
cbar = fig.colorbar(im, ax=ax, label='Improvement (%)')
plt.tight_layout()
plt.savefig('dataset_improvement_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: dataset_improvement_heatmap.png")

# ============================================================
# VISUALIZATION 3: BEST CONFIGURATION (SAME AS BEFORE)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

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

ax.text(0, -15, f"Model: {best_base['model']}", ha='center', fontsize=10, color='#666')
ax.text(1, -15, f"Model: {best_enriched['model']}", ha='center', fontsize=10, color='#666')

plt.tight_layout()
plt.savefig('best_configuration.png', dpi=300, bbox_inches='tight')
print("✓ Saved: best_configuration.png")

# ============================================================
# VISUALIZATION 4: MRR COMPARISON (NEW!)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

base_data = results_df[results_df['dataset'] == 'Base TMDB']
enriched_data = results_df[results_df['dataset'] == 'Enriched']

x = np.arange(len(MODELS))
width = 0.35

bars1 = ax.bar(x - width/2, base_data['MRR'].values, width,
               label='Base TMDB', color='#6b7280', alpha=0.8, edgecolor='white', linewidth=2)
bars2 = ax.bar(x + width/2, enriched_data['MRR'].values, width,
               label='Enriched', color='#f43f5e', alpha=0.8, edgecolor='white', linewidth=2)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add rank interpretation labels
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(y=0.33, color='red', linestyle='--', alpha=0.3, linewidth=1)

ax.text(len(MODELS) - 0.3, 1.02, 'Rank 1 (Perfect)', fontsize=9, color='green', alpha=0.7)
ax.text(len(MODELS) - 0.3, 0.52, 'Rank 2 (Good)', fontsize=9, color='orange', alpha=0.7)
ax.text(len(MODELS) - 0.3, 0.35, 'Rank 3 (Okay)', fontsize=9, color='red', alpha=0.7)

ax.set_ylabel('Mean Reciprocal Rank (MRR)', fontsize=12, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_title('Mean Reciprocal Rank: Base TMDB vs Enriched Dataset\nHigher = Better (Correct movie appears at earlier rank)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(MODELS.keys(), rotation=0, fontsize=11)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('mrr_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: mrr_comparison.png")

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS (INCLUDING MRR)")
print("="*80)

# MRR Summary
print("\n" + "="*80)
print("MEAN RECIPROCAL RANK (MRR):")
print("="*80)
base_mrr_avg = results_df[results_df['dataset'] == 'Base TMDB']['MRR'].mean()
enrich_mrr_avg = results_df[results_df['dataset'] == 'Enriched']['MRR'].mean()
mrr_improvement = enrich_mrr_avg - base_mrr_avg

print(f"\nBase TMDB average MRR:    {base_mrr_avg:.3f} (correct movie at avg rank {1/base_mrr_avg:.1f})")
print(f"Enriched average MRR:     {enrich_mrr_avg:.3f} (correct movie at avg rank {1/enrich_mrr_avg:.1f})")
print(f"Average improvement:      {mrr_improvement:+.3f} ({mrr_improvement/base_mrr_avg*100:+.1f}% relative)")
print(f"Rank improvement:         Correct movie appears {1/base_mrr_avg - 1/enrich_mrr_avg:.1f} positions higher")

# Precision@K Summary
for k in [1, 3, 5, 10]:
    base_avg = results_df[results_df['dataset'] == 'Base TMDB'][f'P@{k}'].mean()
    enrich_avg = results_df[results_df['dataset'] == 'Enriched'][f'P@{k}'].mean()
    improvement = enrich_avg - base_avg
    
    print(f"\nPrecision@{k}:")
    print(f"  Base TMDB average:    {base_avg:.1f}%")
    print(f"  Enriched average:     {enrich_avg:.1f}%")
    print(f"  Average improvement:  {improvement:+.1f} percentage points")

print("\n" + "-"*80)
print("BEST OVERALL CONFIGURATION (by P@5):")
best_config = results_df.loc[results_df['P@5'].idxmax()]
print(f"  Dataset: {best_config['dataset']}")
print(f"  Model: {best_config['model']}")
print(f"  MRR: {best_config['MRR']:.3f} (avg rank {1/best_config['MRR']:.1f})")
print(f"  P@1: {best_config['P@1']:.1f}%")
print(f"  P@3: {best_config['P@3']:.1f}%")
print(f"  P@5: {best_config['P@5']:.1f}%")
print(f"  P@10: {best_config['P@10']:.1f}%")

print("\n" + "="*80)
print("EVALUATION COMPLETE WITH MRR!")
print("="*80)
print("\nGenerated files:")
print("  - dataset_comparison_precision.png")
print("  - dataset_improvement_heatmap.png")
print("  - best_configuration.png")
print("  - mrr_comparison.png (NEW!)")
print("  - evaluation_results_with_mrr.csv (NEW!)")
print("\nInclude all of these in your project report!")
print("="*80)

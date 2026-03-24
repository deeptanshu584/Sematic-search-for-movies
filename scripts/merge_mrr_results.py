"""
Merge MRR results and generate visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

MODELS = ["all-mpnet-base-v2", "all-MiniLM-L6-v2", "e5-small-v2"]

# Load results
df_base = pd.read_csv('results_base.csv')
df_e1 = pd.read_csv('results_enriched_1.csv')
df_e2 = pd.read_csv('results_enriched_2.csv')
df_e3 = pd.read_csv('results_enriched_3.csv')

df_enriched = pd.concat([df_e1, df_e2, df_e3], ignore_index=True)
results_df = pd.concat([df_base, df_enriched], ignore_index=True)

# Reorder columns
column_order = ['dataset', 'model', 'MRR', 'P@1', 'P@3', 'P@5', 'P@10']
results_df = results_df[column_order]

print("\nCOMPLETE RESULTS TABLE")
print(results_df.to_string(index=False))

# Save
results_df.to_csv('evaluation_results_with_mrr.csv', index=False)

# VISUALIZATION: MRR COMPARISON
fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(MODELS))
width = 0.35

base_mrr = results_df[results_df['dataset'] == 'Base TMDB']['MRR'].values
enriched_mrr = results_df[results_df['dataset'] == 'Enriched']['MRR'].values

bars1 = ax.bar(x - width/2, base_mrr, width, label='Base TMDB', color='#6b7280', alpha=0.8)
bars2 = ax.bar(x + width/2, enriched_mrr, width, label='Enriched', color='#f43f5e', alpha=0.8)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Mean Reciprocal Rank (MRR)')
ax.set_title('Mean Reciprocal Rank: Base TMDB vs Enriched Dataset')
ax.set_xticks(x)
ax.set_xticklabels(MODELS)
ax.legend()
ax.set_ylim(0, 1.1)
plt.savefig('mrr_comparison.png', dpi=300)

print("\nSaved: mrr_comparison.png and evaluation_results_with_mrr.csv")

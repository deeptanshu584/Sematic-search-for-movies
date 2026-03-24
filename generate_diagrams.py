"""
CINEMATCH - Diagram Generation Script
Generates all visual diagrams for presentation and report
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
import os
os.makedirs('../results/diagrams', exist_ok=True)

print("="*60)
print("GENERATING CINEMATCH PROJECT DIAGRAMS")
print("="*60)

# ============================================================
# DIAGRAM 1: SYSTEM ARCHITECTURE
# ============================================================
print("\n[1/6] Creating System Architecture Diagram...")

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11.5, 'CINEMATCH SYSTEM ARCHITECTURE', 
        ha='center', fontsize=18, fontweight='bold')

# Layer 1: Data Layer
data_box = FancyBboxPatch((0.5, 9), 9, 1.8, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='#3b82f6', facecolor='#dbeafe', linewidth=2)
ax.add_patch(data_box)
ax.text(5, 10.5, 'DATA LAYER', ha='center', fontsize=14, fontweight='bold')
ax.text(2, 9.8, 'TMDB 5000\n4,803 movies\n305 chars avg', ha='center', fontsize=9)
ax.text(5, 9.8, 'Wikipedia Plots\n74.5% match\n2,843 chars avg', ha='center', fontsize=9)
ax.text(8, 9.8, 'Enriched Dataset\n4,900 movies\n+831% detail', ha='center', fontsize=9, color='green', fontweight='bold')

# Arrow 1
arrow1 = FancyArrowPatch((5, 9), (5, 7.8), arrowstyle='->', 
                        mutation_scale=30, linewidth=2, color='#ef4444')
ax.add_patch(arrow1)

# Layer 2: AI/ML Layer
ai_box = FancyBboxPatch((0.5, 6), 9, 1.8,
                        boxstyle="round,pad=0.1",
                        edgecolor='#10b981', facecolor='#d1fae5', linewidth=2)
ax.add_patch(ai_box)
ax.text(5, 7.5, 'AI/ML LAYER', ha='center', fontsize=14, fontweight='bold')
ax.text(5, 7.0, 'Sentence Transformer (all-mpnet-base-v2)', ha='center', fontsize=10)
ax.text(5, 6.5, 'Text → 768-dimensional Vector Embeddings', ha='center', fontsize=9)
ax.text(5, 6.2, '"The Martian soup..." → [0.23, -0.45, 0.67, ..., 0.12]', 
        ha='center', fontsize=8, family='monospace', style='italic')

# Arrow 2
arrow2 = FancyArrowPatch((5, 6), (5, 4.8), arrowstyle='->', 
                        mutation_scale=30, linewidth=2, color='#ef4444')
ax.add_patch(arrow2)

# Layer 3: Search Engine Layer
search_box = FancyBboxPatch((0.5, 3), 9, 1.8,
                           boxstyle="round,pad=0.1",
                           edgecolor='#f59e0b', facecolor='#fef3c7', linewidth=2)
ax.add_patch(search_box)
ax.text(5, 4.5, 'SEARCH ENGINE LAYER', ha='center', fontsize=14, fontweight='bold')
ax.text(5, 4.0, 'Cosine Similarity + Hybrid Scoring', ha='center', fontsize=10)
ax.text(5, 3.5, '70% Semantic + 20% TF-IDF + 10% Keyword/Genre Boost', ha='center', fontsize=9)

# Arrow 3
arrow3 = FancyArrowPatch((5, 3), (5, 1.8), arrowstyle='->', 
                        mutation_scale=30, linewidth=2, color='#ef4444')
ax.add_patch(arrow3)

# Layer 4: Application Layer
app_box = FancyBboxPatch((0.5, 0), 9, 1.8,
                        boxstyle="round,pad=0.1",
                        edgecolor='#8b5cf6', facecolor='#ede9fe', linewidth=2)
ax.add_patch(app_box)
ax.text(5, 1.5, 'APPLICATION LAYER (Streamlit)', ha='center', fontsize=14, fontweight='bold')
ax.text(5, 1.0, 'Search UI + Filters + 3×3 Grid + Explainability', ha='center', fontsize=10)
ax.text(5, 0.5, 'Real-time Results in <0.5 seconds', ha='center', fontsize=9, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/diagrams/system_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: system_architecture.png")
plt.close()

# ============================================================
# DIAGRAM 2: SEMANTIC VS KEYWORD SEARCH
# ============================================================
print("[2/6] Creating Semantic vs Keyword Comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Keyword Search (Traditional)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('KEYWORD SEARCH (Traditional)', fontsize=14, fontweight='bold', color='#ef4444')

box1 = FancyBboxPatch((0.5, 7.5), 9, 1.5, boxstyle="round,pad=0.1",
                      edgecolor='#ef4444', facecolor='#fee2e2', linewidth=2)
ax1.add_patch(box1)
ax1.text(5, 8.5, 'Query: "astronaut stranded on mars"', ha='center', fontsize=11, fontweight='bold')

ax1.text(5, 6.5, '❌ Searches for EXACT words:', ha='center', fontsize=10)
ax1.text(5, 6.0, '"astronaut" AND "stranded" AND "mars"', ha='center', fontsize=9, family='monospace')

ax1.text(5, 5.0, '❌ MISSES these valid matches:', ha='center', fontsize=10)
ax1.text(5, 4.4, '• "cosmonaut isolated on red planet"', ha='left', fontsize=9)
ax1.text(5, 3.9, '• "spaceman alone on Mars"', ha='left', fontsize=9)
ax1.text(5, 3.4, '• "space traveler abandoned"', ha='left', fontsize=9)

ax1.text(5, 2.0, 'Result: Limited matches', ha='center', fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='#fecaca'))
ax1.text(5, 1.0, 'Accuracy (P@3): 45%', ha='center', fontsize=10, fontweight='bold')

# Semantic Search (Your Project)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('SEMANTIC SEARCH (CineMatch)', fontsize=14, fontweight='bold', color='#10b981')

box2 = FancyBboxPatch((0.5, 7.5), 9, 1.5, boxstyle="round,pad=0.1",
                      edgecolor='#10b981', facecolor='#d1fae5', linewidth=2)
ax2.add_patch(box2)
ax2.text(5, 8.5, 'Query: "astronaut stranded on mars"', ha='center', fontsize=11, fontweight='bold')

ax2.text(5, 6.5, '✅ Understands MEANING:', ha='center', fontsize=10)
ax2.text(5, 6.0, '"person alone on another planet"', ha='center', fontsize=9, style='italic')

ax2.text(5, 5.0, '✅ FINDS all these matches:', ha='center', fontsize=10)
ax2.text(5, 4.4, '• "cosmonaut isolated on red planet" ✓', ha='left', fontsize=9)
ax2.text(5, 3.9, '• "spaceman alone on Mars" ✓', ha='left', fontsize=9)
ax2.text(5, 3.4, '• "space traveler abandoned" ✓', ha='left', fontsize=9)

ax2.text(5, 2.0, 'Result: Comprehensive matches', ha='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='#bbf7d0'))
ax2.text(5, 1.0, 'Accuracy (P@3): 60% (+33%!)', ha='center', fontsize=10, fontweight='bold', color='green')

plt.tight_layout()
plt.savefig('../results/diagrams/semantic_vs_keyword.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: semantic_vs_keyword.png")
plt.close()

# ============================================================
# DIAGRAM 3: VECTOR SPACE VISUALIZATION
# ============================================================
print("[3/6] Creating Vector Space Visualization...")

fig, ax = plt.subplots(figsize=(12, 10))

# Sci-Fi cluster
scifi_movies = [
    ('The Martian', 7.5, 8.0),
    ('Interstellar', 7.8, 8.5),
    ('Gravity', 7.0, 7.8),
    ('Ad Astra', 8.0, 7.5),
    ('Arrival', 7.3, 8.3)
]
scifi_x = [m[1] for m in scifi_movies]
scifi_y = [m[2] for m in scifi_movies]
ax.scatter(scifi_x, scifi_y, s=400, c='#3b82f6', alpha=0.6, edgecolors='black', linewidth=2, label='Sci-Fi Movies')

# Animation cluster
anim_movies = [
    ('Toy Story', 3.0, 3.2),
    ('Finding Nemo', 3.5, 2.8),
    ('Inside Out', 2.8, 3.5),
    ('WALL-E', 3.2, 3.0)
]
anim_x = [m[1] for m in anim_movies]
anim_y = [m[2] for m in anim_movies]
ax.scatter(anim_x, anim_y, s=400, c='#10b981', alpha=0.6, edgecolors='black', linewidth=2, label='Animation Movies')

# Drama cluster
drama_movies = [
    ('Shawshank', 5.0, 5.5),
    ('Inception', 6.0, 6.5),
    ('The Godfather', 5.5, 5.0)
]
drama_x = [m[1] for m in drama_movies]
drama_y = [m[2] for m in drama_movies]
ax.scatter(drama_x, drama_y, s=400, c='#f59e0b', alpha=0.6, edgecolors='black', linewidth=2, label='Drama Movies')

# Query
query_x, query_y = 7.6, 8.1
ax.scatter([query_x], [query_y], s=600, c='#ef4444', marker='*', 
          edgecolors='black', linewidth=2, label='Your Query', zorder=10)

# Add labels
for movie, x, y in scifi_movies:
    ax.annotate(movie, (x, y), fontsize=9, ha='center', va='bottom', fontweight='bold')

for movie, x, y in anim_movies:
    ax.annotate(movie, (x, y), fontsize=9, ha='center', va='bottom', fontweight='bold')

for movie, x, y in drama_movies:
    ax.annotate(movie, (x, y), fontsize=9, ha='center', va='bottom', fontweight='bold')

ax.annotate('Query:\n"astronaut on mars"', (query_x, query_y), 
           fontsize=11, ha='center', va='bottom', color='#ef4444', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#fee2e2', edgecolor='#ef4444'))

# Draw circle showing semantic similarity
circle = plt.Circle((query_x, query_y), 1.2, color='#ef4444', fill=False, linestyle='--', linewidth=2, alpha=0.5)
ax.add_patch(circle)
ax.text(query_x + 1.5, query_y + 1.0, 'High Similarity\nRegion', fontsize=9, style='italic', color='#ef4444')

ax.set_xlabel('Dimension 1 (Simplified)', fontsize=12, fontweight='bold')
ax.set_ylabel('Dimension 2 (Simplified)', fontsize=12, fontweight='bold')
ax.set_title('Vector Space Visualization\n(Actual: 768 dimensions, shown here in 2D)', 
            fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 10)
ax.set_ylim(1, 10)

plt.tight_layout()
plt.savefig('../results/diagrams/vector_space_visualization.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: vector_space_visualization.png")
plt.close()

# ============================================================
# DIAGRAM 4: DATASET ENRICHMENT IMPACT
# ============================================================
print("[4/6] Creating Dataset Enrichment Impact Diagram...")

fig, ax = plt.subplots(figsize=(12, 8))

categories = ['Plot\nLength', 'Precision@1', 'Precision@3', 'Precision@5', 'MRR']
base_values = [305, 35, 45, 55, 0.42]
enriched_values = [2843, 40, 60, 65, 0.52]

# Normalize plot length to fit on same scale
base_values_display = [305/30, 35, 45, 55, 0.42*100]
enriched_values_display = [2843/30, 40, 60, 65, 0.52*100]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, base_values_display, width, label='Base TMDB',
              color='#6b7280', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, enriched_values_display, width, label='Enriched',
              color='#10b981', alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (b1, b2) in enumerate(zip(bars1, bars2)):
    height1 = b1.get_height()
    height2 = b2.get_height()
    
    if i == 0:
        ax.text(b1.get_x() + b1.get_width()/2., height1 + 2,
               f'{base_values[i]} chars', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(b2.get_x() + b2.get_width()/2., height2 + 2,
               f'{enriched_values[i]} chars', ha='center', va='bottom', fontsize=9, fontweight='bold', color='green')
    elif i == 4:
        ax.text(b1.get_x() + b1.get_width()/2., height1 + 2,
               f'{base_values[i]:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(b2.get_x() + b2.get_width()/2., height2 + 2,
               f'{enriched_values[i]:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='green')
    else:
        ax.text(b1.get_x() + b1.get_width()/2., height1 + 2,
               f'{base_values[i]:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(b2.get_x() + b2.get_width()/2., height2 + 2,
               f'{enriched_values[i]:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='green')

# Add improvement arrows
improvements = ['+831%', '+5pts', '+15pts', '+10pts', '+0.10']
for i, imp in enumerate(improvements):
    ax.annotate('', xy=(i + width/2, enriched_values_display[i] + 1), 
               xytext=(i - width/2, base_values_display[i] + 1),
               arrowprops=dict(arrowstyle='->', lw=2, color='#ef4444'))
    ax.text(i, max(base_values_display[i], enriched_values_display[i]) + 15,
           imp, ha='center', fontsize=10, fontweight='bold', color='#ef4444',
           bbox=dict(boxstyle='round', facecolor='#fee2e2', edgecolor='#ef4444'))

ax.set_ylabel('Value (scaled)', fontsize=12, fontweight='bold')
ax.set_title('Dataset Enrichment Impact', 
            fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=12, loc='upper left')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../results/diagrams/dataset_enrichment_impact.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: dataset_enrichment_impact.png")
plt.close()

# ============================================================
# DIAGRAM 5: EVALUATION METRICS
# ============================================================
print("[5/6] Creating Evaluation Metrics Explanation...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# P@K
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 12)
ax1.axis('off')
ax1.set_title('PRECISION@K', fontsize=14, fontweight='bold')

query_box = FancyBboxPatch((0.5, 10), 9, 1, boxstyle="round,pad=0.1",
                          edgecolor='#3b82f6', facecolor='#dbeafe', linewidth=2)
ax1.add_patch(query_box)
ax1.text(5, 10.5, 'Query: "Wizard boy magic school"', ha='center', fontsize=11, fontweight='bold')

ax1.text(1, 9, 'Top 5 Results:', ha='left', fontsize=11, fontweight='bold')

results = [
    ('1. Harry Potter ✅', '#10b981'),
    ('2. Percy Jackson', '#6b7280'),
    ('3. Fantastic Beasts', '#6b7280'),
    ('4. The Magicians', '#6b7280'),
    ('5. Narnia', '#6b7280')
]

y_pos = 8.2
for result, color in results:
    box = FancyBboxPatch((1, y_pos-0.3), 8, 0.5, boxstyle="round,pad=0.05",
                        edgecolor=color, facecolor='white', linewidth=2)
    ax1.add_patch(box)
    ax1.text(1.5, y_pos, result, ha='left', fontsize=10, color=color, fontweight='bold')
    y_pos -= 0.7

ax1.text(1, 4, 'Metrics:', ha='left', fontsize=11, fontweight='bold')
ax1.text(1.5, 3.3, 'P@1 = 100%', ha='left', fontsize=10)
ax1.text(1.5, 2.8, 'P@3 = 100%', ha='left', fontsize=10)
ax1.text(1.5, 2.3, 'P@5 = 100%', ha='left', fontsize=10)

# MRR
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 12)
ax2.axis('off')
ax2.set_title('MEAN RECIPROCAL RANK', fontsize=14, fontweight='bold')

query_box2 = FancyBboxPatch((0.5, 10), 9, 1, boxstyle="round,pad=0.1",
                           edgecolor='#3b82f6', facecolor='#dbeafe', linewidth=2)
ax2.add_patch(query_box2)
ax2.text(5, 10.5, 'MRR = 1 / rank', ha='center', fontsize=11, fontweight='bold')

examples = [
    ('Rank 1: MRR = 1.00', '#10b981'),
    ('Rank 2: MRR = 0.50', '#f59e0b'),
    ('Rank 3: MRR = 0.33', '#ef4444')
]

y_pos = 8.5
for example, color in examples:
    box = FancyBboxPatch((1, y_pos-0.3), 8, 0.5, boxstyle="round,pad=0.05",
                        edgecolor=color, facecolor='white', linewidth=2)
    ax2.add_patch(box)
    ax2.text(1.5, y_pos, example, ha='left', fontsize=10, fontweight='bold', color=color)
    y_pos -= 0.9

ax2.text(1, 4.5, 'Your Results:', ha='left', fontsize=11, fontweight='bold')
ax2.text(1.5, 3.8, 'Base: MRR = 0.42', ha='left', fontsize=10)
ax2.text(1.5, 2.5, 'Enriched: MRR = 0.52', ha='left', fontsize=10, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/diagrams/evaluation_metrics_explained.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: evaluation_metrics_explained.png")
plt.close()

# ============================================================
# DIAGRAM 6: HYBRID SCORING
# ============================================================
print("[6/6] Creating Hybrid Scoring Breakdown...")

fig, ax = plt.subplots(figsize=(10, 8))

components = ['Semantic\nSimilarity', 'TF-IDF\nKeyword', 'Genre\nBoost']
percentages = [70, 20, 10]
colors = ['#3b82f6', '#10b981', '#f59e0b']

left = 0
for i, (component, percentage, color) in enumerate(zip(components, percentages, colors)):
    ax.barh(0, percentage, left=left, height=0.8, color=color, 
           edgecolor='black', linewidth=2, alpha=0.8)
    
    ax.text(left + percentage/2, 0, f'{percentage}%\n{component}', 
           ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    left += percentage

ax.set_title('HYBRID SCORING APPROACH', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(0, 100)
ax.set_ylim(-1, 1)
ax.set_xlabel('Contribution (%)', fontsize=12, fontweight='bold')
ax.set_yticks([])
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../results/diagrams/hybrid_scoring_breakdown.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: hybrid_scoring_breakdown.png")
plt.close()

print("\n" + "="*60)
print("ALL DIAGRAMS GENERATED!")
print("="*60)
print("\nCheck results/diagrams/ folder")

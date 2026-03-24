import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Set style
sns.set(style="whitegrid")

# 1. Load Data
df = pd.read_csv("../data/tmdb_5000_movies.csv")

# 2. Missing Values Analysis
plt.figure(figsize=(10, 6))
sns.barplot(x=df.isnull().sum().index, y=df.isnull().sum().values, hue=df.isnull().sum().index, palette="viridis", legend=False)
plt.xticks(rotation=45)
plt.title("Missing Values per Column")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../results/missing_values.png")
plt.close()

# 3. Genre Distribution
def parse_genres(genre_str):
    try:
        genres = json.loads(genre_str)
        return [g['name'] for g in genres]
    except:
        return []

all_genres = []
for genres in df['genres'].apply(parse_genres):
    all_genres.extend(genres)

genre_counts = pd.Series(all_genres).value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.index, y=genre_counts.values, hue=genre_counts.index, palette="magma", legend=False)
plt.xticks(rotation=45)
plt.title("Distribution of Movie Genres")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../results/genre_distribution.png")
plt.close()

# 4. Rating Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['vote_average'], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating (out of 10)")
plt.tight_layout()
plt.savefig("../results/rating_distribution.png")
plt.close()

# 5. Year Distribution
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
plt.figure(figsize=(12, 6))
sns.histplot(df['year'].dropna(), bins=50, kde=True, color="salmon")
plt.title("Distribution of Movies over the Years")
plt.xlabel("Release Year")
plt.tight_layout()
plt.savefig("../results/year_distribution.png")
plt.close()

# 6. Overview Length Analysis
df['overview_len'] = df['overview'].fillna("").str.len()
plt.figure(figsize=(10, 6))
sns.histplot(df['overview_len'], bins=50, kde=True, color="teal")
plt.title("Distribution of Plot Overview Lengths")
plt.xlabel("Length (Characters)")
plt.tight_layout()
plt.savefig("../results/overview_length.png")
plt.close()

print("All visualizations generated successfully!")

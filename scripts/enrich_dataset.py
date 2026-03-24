import pandas as pd
import json
import ast

def enrich_data():
    print("Loading datasets...")
    # Load TMDB (our base)
    tmdb = pd.read_csv("../data/tmdb_5000_movies.csv")
    credits = pd.read_csv("../data/tmdb_5000_credits.csv")
    
    # Load Wikipedia (our plot source)
    wiki = pd.read_csv("../data/wiki_movie_plots_deduped.csv")
    
    # 1. Merge TMDB and Credits first
    df = tmdb.merge(credits, on='title')
    
    # 2. Prepare TMDB for matching
    # Extract year from release_date (format: 2009-12-10)
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
    
    # 3. Clean Wikipedia titles and years for better matching
    wiki['Title_Clean'] = wiki['Title'].str.lower().str.strip()
    wiki['Release Year'] = wiki['Release Year'].astype(int)
    
    df['Title_Clean'] = df['title'].str.lower().str.strip()

    print("Matching Wikipedia plots to TMDB movies...")
    # 4. Merge with Wikipedia plots based on Title and Year
    # We use a left join so we keep all TMDB movies even if no Wiki plot exists
    df_enriched = df.merge(
        wiki[['Title_Clean', 'Release Year', 'Plot']], 
        left_on=['Title_Clean', 'release_year'], 
        right_on=['Title_Clean', 'Release Year'], 
        how='left'
    )

    # 5. Fallback: Use TMDB Overview if Wiki Plot is missing
    df_enriched['detailed_plot'] = df_enriched['Plot'].fillna(df_enriched['overview']).fillna("")
    
    print(f"Enrichment Complete. Found Wikipedia plots for {df_enriched['Plot'].notna().sum()} out of {len(df)} movies.")

    # 6. Re-build the search 'soup' with the detailed plot
    def get_list(x):
        try: return [i['name'] for i in json.loads(x)]
        except: return []

    def get_director(x):
        try:
            for i in json.loads(x):
                if i['job'] == 'Director': return [i['name']]
            return []
        except: return []

    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        return []

    df_enriched['genres_clean'] = df_enriched['genres'].apply(get_list).apply(clean_data)
    df_enriched['keywords_clean'] = df_enriched['keywords'].apply(get_list).apply(clean_data)
    df_enriched['cast_clean'] = df_enriched['cast'].apply(lambda x: [i['name'] for i in json.loads(x)][:3] if isinstance(x, str) else []).apply(clean_data)
    df_enriched['director_clean'] = df_enriched['crew'].apply(get_director).apply(clean_data)

    df_enriched['soup'] = (
        df_enriched['detailed_plot'] + ' ' + 
        df_enriched['genres_clean'].apply(lambda x: ' '.join(x)) + ' ' + 
        df_enriched['keywords_clean'].apply(lambda x: ' '.join(x)) + ' ' + 
        df_enriched['cast_clean'].apply(lambda x: ' '.join(x)) + ' ' + 
        df_enriched['director_clean'].apply(lambda x: ' '.join(x))
    )

    # Save to a NEW file to keep your progress safe
    output_file = "../data/enriched_tmdb_with_wiki.csv"
    df_enriched.to_csv(output_file, index=False)
    print(f"Saved enriched dataset to {output_file}")

if __name__ == "__main__":
    enrich_data()

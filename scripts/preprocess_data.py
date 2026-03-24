import pandas as pd
import json

def load_and_preprocess():
    movies = pd.read_csv("../data/tmdb_5000_movies.csv")
    credits = pd.read_csv("../data/tmdb_5000_credits.csv")
    
    # Merge datasets
    df = movies.merge(credits, on='title')
    
    # Helper to parse JSON columns
    def get_list(x):
        try:
            return [i['name'] for i in json.loads(x)]
        except:
            return []

    # Get Director from crew
    def get_director(x):
        try:
            for i in json.loads(x):
                if i['job'] == 'Director':
                    return [i['name']]
            return []
        except:
            return []

    # Extract genres, keywords, cast (top 3), and director
    df['genres_list'] = df['genres'].apply(get_list)
    df['keywords_list'] = df['keywords'].apply(get_list)
    df['cast_list'] = df['cast'].apply(lambda x: [i['name'] for i in json.loads(x)][:3] if isinstance(x, str) else [])
    df['director_list'] = df['crew'].apply(get_director)

    # Clean names (remove spaces)
    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''

    df['genres_clean'] = df['genres_list'].apply(clean_data)
    df['keywords_clean'] = df['keywords_list'].apply(clean_data)
    df['cast_clean'] = df['cast_list'].apply(clean_data)
    df['director_clean'] = df['director_list'].apply(clean_data)

    # Create 'tags' column - combining all into a "soup"
    df['soup'] = (
        df['overview'].fillna('') + ' ' + 
        df['genres_clean'].apply(lambda x: ' '.join(x)) + ' ' + 
        df['keywords_clean'].apply(lambda x: ' '.join(x)) + ' ' + 
        df['cast_clean'].apply(lambda x: ' '.join(x)) + ' ' + 
        df['director_clean'].apply(lambda x: ' '.join(x))
    )
    
    return df

if __name__ == "__main__":
    df = load_and_preprocess()
    print("Columns in processed dataframe:", df.columns)
    print("Sample 'soup' for first movie:")
    print(df['soup'].iloc[0])
    df.to_csv("../data/processed_movies.csv", index=False)

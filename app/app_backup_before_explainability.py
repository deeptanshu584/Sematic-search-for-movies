import streamlit as st
import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util
import torch

# ============================================================
# 1. PAGE CONFIG & CUSTOM STYLING
# ============================================================
st.set_page_config(
    page_title="CineMatch - AI Movie Search",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for a cinematic, dark-themed UI
st.markdown("""
<style>
    /* Dark cinematic background */
    .stApp {
        background-color: #0f1117;
        color: #e8e8e8;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title styling */
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e94560, #c23152);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        padding-top: 20px;
    }

    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }

    /* Search input */
    .stTextInput input {
        background-color: #1a1d26 !important;
        border: 2px solid #2a2d36 !important;
        border-radius: 12px !important;
        color: #fff !important;
        padding: 14px 18px !important;
        font-size: 1rem !important;
        transition: border-color 0.3s;
    }

    /* Movie card container */
    .movie-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
        min-height: 280px;
        display: flex;
        flex-direction: column;
    }

    .movie-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(239, 68, 68, 0.5);
        transform: translateY(-2px);
    }

    .movie-emoji {
        font-size: 2.2rem;
        text-align: center;
        margin-bottom: 10px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 10px;
    }

    .movie-title {
        font-size: 16px;
        font-weight: 700;
        color: #f43f5e;
        margin-bottom: 4px;
        line-height: 1.2;
        text-align: center;
        min-height: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .movie-meta {
        font-size: 11px;
        color: #9ca3af;
        margin-bottom: 4px;
        text-align: center;
    }

    .movie-genres {
        font-size: 10px;
        color: #6b7280;
        margin-bottom: 8px;
        font-style: italic;
        text-align: center;
    }

    .movie-plot {
        font-size: 12px;
        color: #d1d5db;
        line-height: 1.4;
        margin-top: 8px;
        overflow: hidden;
    }

    .match-bar-container {
        position: relative;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
        height: 20px;
        margin: 5px 0;
        overflow: hidden;
    }

    .match-bar {
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        border-radius: 4px;
        transition: width 0.6s ease;
    }

    .match-label {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 10px;
        font-weight: 700;
        color: white;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
        z-index: 1;
    }

    /* Result count */
    .result-count {
        color: #666;
        font-size: 0.88rem;
        margin: 10px 0 18px;
        text-align: center;
    }

    /* Sidebar */
    .stSidebar {
        background-color: #12141c !important;
        border-right: 1px solid #2a2d36;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. LOAD MODEL & DATA
# ============================================================
@st.cache_resource
def load_model():
    """Load the Sentence Transformer model (cached so it loads only once)."""
    return SentenceTransformer('all-mpnet-base-v2')

@st.cache_data
def load_data():
    """Load and clean the movie dataset."""
    df = pd.read_csv("../data/enriched_tmdb_with_wiki.csv")
    df['detailed_plot'] = df['detailed_plot'].fillna("")

    # --- Parse genres from JSON string to a readable list ---
    def parse_genres(genre_str):
        try:
            genres = json.loads(genre_str)
            return [g['name'] for g in genres]
        except:
            return []

    df['genre_list'] = df['genres'].apply(parse_genres)
    df['genres_display'] = df['genre_list'].apply(lambda x: ", ".join(x) if x else "Unknown")

    # --- Extract release year ---
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)

    # --- Clean vote_average (rating) ---
    df['rating'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)

    return df

model = load_model()
df = load_data()


# ============================================================
# 3. ENCODE ALL MOVIE OVERVIEWS INTO VECTORS
# ============================================================
@st.cache_data
def compute_embeddings(overviews: list):
    """Encode all movie overviews into sentence embeddings."""
    return model.encode(overviews, convert_to_tensor=True, show_progress_bar=False)

movie_embeddings = compute_embeddings(df['detailed_plot'].tolist())


# ============================================================
# 4. GET ALL UNIQUE GENRES (for filter dropdown)
# ============================================================
all_genres = sorted(set(g for genres in df['genre_list'] for g in genres))


# ============================================================
# 6. UI LAYOUT
# ============================================================

# --- Header ---
st.markdown('<div class="main-title">🎬 CineMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered movie search • Just describe what you\'re looking for</div>', unsafe_allow_html=True)

# --- Search Input ---
query = st.text_input(
    "",
    placeholder="e.g. A cowboy toy who feels replaced by a space ranger toy ...",
    key="search_query"
)

# --- Filters in Sidebar ---
st.sidebar.markdown("### 🎛️ Filters")

# Genre filter
selected_genres = st.sidebar.multiselect(
    "Genre",
    options=all_genres,
    placeholder="Select genres..."
)

# Year range filter
min_year = int(df['year'][df['year'] > 0].min())
max_year = int(df['year'].max())
year_range = st.sidebar.slider(
    "Release Year",
    min_value=min_year,
    max_value=max_year,
    value=(1990, max_year),
    step=1
)

# Minimum rating filter
min_rating = st.sidebar.slider(
    "Minimum Rating",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.5
)

# Number of results
num_results = st.sidebar.selectbox(
    "Number of Results",
    options=[3, 6, 9, 12, 15],
    index=2  # default = 9 (3x3 grid)
)

# Minimum match threshold
min_match = st.sidebar.slider(
    "Minimum Match %",
    min_value=10,
    max_value=80,
    value=25,
    step=5
)


# ============================================================
# 7. SEARCH LOGIC
# ============================================================
if query.strip():
    with st.spinner("🔍 Searching movies..."):

        # --- Encode the user query ---
        query_embedding = model.encode(query, convert_to_tensor=True)

        # --- Compute cosine similarity with all movies ---
        scores = util.cos_sim(query_embedding, movie_embeddings)[0]

        # --- Add scores to dataframe ---
        df['match_score'] = scores.cpu().numpy()
        df['match_percent'] = (df['match_score'] * 100).round(2)

        # --- Apply Filters ---
        filtered = df.copy()

        # Genre filter
        if selected_genres:
            filtered = filtered[filtered['genre_list'].apply(
                lambda genres: any(g in genres for g in selected_genres)
            )]

        # Year range filter
        filtered = filtered[
            (filtered['year'] >= year_range[0]) &
            (filtered['year'] <= year_range[1])
        ]

        # Minimum rating filter
        filtered = filtered[filtered['rating'] >= min_rating]

        # Minimum match % filter
        filtered = filtered[filtered['match_percent'] >= min_match]

        # --- Sort by match score descending and take top N ---
        top_movies = filtered.sort_values('match_percent', ascending=False).head(num_results)

    # --- Display result count ---
    st.markdown(
        f'<div class="result-count">Showing <b>{len(top_movies)}</b> result(s) for "<i>{query}</i>"</div>',
        unsafe_allow_html=True
    )

    # --- Display Movies in 3x3 Grid ---
    if len(top_movies) > 0:
        # Genre to emoji mapping
        genre_icons = {
            "Action": "💥", "Comedy": "😂", "Drama": "🎭", "Sci-Fi": "🚀", "Science Fiction": "🚀",
            "Horror": "👻", "Romance": "💕", "Thriller": "🔪", "Adventure": "🗺️",
            "Animation": "🎨", "Fantasy": "🧙", "Crime": "🕵️", "Documentary": "📽️",
            "Family": "👨‍👩‍👧", "Mystery": "🔍", "War": "⚔️", "History": "📜",
            "Music": "🎵", "Western": "🤠", "Horror": "👻"
        }

        # Create rows of 3 movies each
        for i in range(0, len(top_movies), 3):
            cols = st.columns(3)
            
            # Get up to 3 movies for this row
            row_movies = top_movies.iloc[i:i+3]
            
            for col_idx, (idx, movie) in enumerate(row_movies.iterrows()):
                with cols[col_idx]:
                    # Pick icon
                    first_genre = movie['genre_list'][0] if movie['genre_list'] else "Unknown"
                    icon = genre_icons.get(first_genre, "🎬")
                    
                    # Movie card container
                    st.markdown(f'''
                        <div class="movie-card">
                            <div class="movie-emoji">{icon}</div>
                            <div class="movie-title">{movie["original_title"]}</div>
                            <div class="movie-meta">📅 {int(movie["year"]) if movie["year"] > 0 else "N/A"} • ⭐ {movie["rating"]:.1f}/10</div>
                            <div class="movie-genres">🎭 {movie["genres_display"]}</div>
                    ''', unsafe_allow_html=True)
                    
                    # Match score bar
                    match_pct = movie['match_percent']
                    bar_color = '#10b981' if match_pct >= 70 else '#f59e0b' if match_pct >= 50 else '#ef4444'
                    st.markdown(f'''
                        <div class="match-bar-container">
                            <div class="match-bar" style="width: {match_pct}%; background: {bar_color};"></div>
                            <div class="match-label">{match_pct:.1f}% Match</div>
                        </div>
                    ''', unsafe_allow_html=True)
                    
                    # Plot overview (truncated)
                    plot = movie['detailed_plot']
                    plot_preview = plot[:120] + '...' if len(plot) > 120 else plot
                    st.markdown(f'<div class="movie-plot">{plot_preview}</div>', 
                               unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No movies found matching your criteria. Try adjusting the filters.")

else:
    # --- Empty state when no query ---
    st.markdown('''
        <div style="text-align: center; padding: 60px 20px; color: #6b7280;">
            <div style="font-size: 48px; margin-bottom: 20px;">🎬</div>
            <div style="font-size: 18px; margin-bottom: 10px;">Start by describing a movie</div>
            <div style="font-size: 14px; color: #9ca3af;">
                Try: "A wizard boy at magic school" or "Toys that come alive"
            </div>
        </div>
    ''', unsafe_allow_html=True)

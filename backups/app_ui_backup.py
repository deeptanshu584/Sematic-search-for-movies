import streamlit as st
import pandas as pd
import json
import ast
from sentence_transformers import SentenceTransformer, CrossEncoder, util
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
    .stTextInput input:focus {
        border-color: #e94560 !important;
        box-shadow: 0 0 15px rgba(233, 69, 96, 0.3) !important;
    }

    /* Movie card container */
    .movie-card {
        background: linear-gradient(145deg, #1a1d26, #161821);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 18px;
        border: 1px solid #2a2d36;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s, border-color 0.3s;
    }
    .movie-card:hover {
        transform: translateY(-3px);
        border-color: #e94560;
    }

    /* Match badge */
    .match-badge {
        display: inline-block;
        background: linear-gradient(135deg, #e94560, #c23152);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 5px 14px;
        border-radius: 20px;
        margin-bottom: 8px;
    }

    .match-badge-low {
        background: linear-gradient(135deg, #e6a817, #d4890e);
    }

    .match-badge-vlow {
        background: linear-gradient(135deg, #5a7a8a, #4a6a7a);
    }

    /* Movie title */
    .movie-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fff;
        margin: 6px 0 4px;
    }

    /* Movie meta info (year, genre, rating) */
    .movie-meta {
        color: #aaa;
        font-size: 0.9rem;
        margin-bottom: 10px;
    }
    .movie-meta span {
        margin-right: 14px;
    }
    .star {
        color: #e6a817;
    }

    /* Movie overview text */
    .movie-overview {
        color: #bbb;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Filters section */
    .filter-label {
        color: #e94560;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }

    /* Selectbox & slider styling */
    .stSelectbox select, .stMultiselect input {
        background-color: #1a1d26 !important;
        color: #fff !important;
        border-color: #2a2d36 !important;
    }

    /* Result count */
    .result-count {
        color: #666;
        font-size: 0.88rem;
        margin: 10px 0 18px;
        text-align: center;
    }

    /* No results message */
    .no-results {
        text-align: center;
        color: #666;
        font-size: 1.05rem;
        margin-top: 60px;
    }

    /* Poster image */
    .movie-poster {
        border-radius: 10px;
        width: 100%;
        max-width: 150px;
    }

    /* Divider */
    .stDivider {
        border-top-color: #2a2d36 !important;
    }

    /* Sidebar */
    .stSidebar {
        background-color: #12141c !important;
        border-right: 1px solid #2a2d36;
    }

    /* Buttons */
    .stButton button {
        background-color: #e94560 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: background-color 0.3s !important;
    }
    .stButton button:hover {
        background-color: #c23152 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. LOAD MODELS & DATA
# ============================================================
@st.cache_resource
def load_models():
    """Load the Sentence Transformer models."""
    # Bi-Encoder: Fast retrieval
    bi_encoder = SentenceTransformer('all-mpnet-base-v2')
    # Cross-Encoder: Precise re-ranking
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return bi_encoder, cross_encoder

@st.cache_data
def load_data():
    """Load the enriched movie dataset with Wikipedia plots."""
    try:
        df = pd.read_csv("enriched_tmdb_with_wiki.csv")
    except FileNotFoundError:
        # Fallback if enrichment hasn't run
        df = pd.read_csv("processed_movies.csv")
        
    df['overview'] = df['overview'].fillna("")
    df['soup'] = df['soup'].fillna("")
    
    # Genres display for UI
    def parse_genres(genre_str):
        if pd.isna(genre_str) or genre_str == "":
            return []
        try:
            # Try parsing the string into a Python object
            if isinstance(genre_str, str) and "[" in genre_str:
                data = ast.literal_eval(genre_str)
            else:
                data = genre_str
            
            # If we have a list, extract the names if they are in dicts
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    return [g.get('name', '') for g in data]
                return [str(i) for i in data]
            return []
        except:
            return []

    # Check which column to use for genres
    genre_col = 'genres_list' if 'genres_list' in df.columns else 'genres'
    df['genre_list'] = df[genre_col].apply(parse_genres)
    df['genres_display'] = df['genre_list'].apply(lambda x: ", ".join(x) if x else "Unknown")

    # --- Extract release year ---
    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)

    # --- Clean vote_average (rating) ---
    df['rating'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)

    return df

bi_encoder, cross_encoder = load_models()
df = load_data()


# ============================================================
# 3. ENCODE ALL MOVIE SOUP INTO VECTORS
# ============================================================
@st.cache_data
def compute_embeddings(soups: list):
    """Encode the movie 'soup' (enriched metadata) into sentence embeddings."""
    return bi_encoder.encode(soups, convert_to_tensor=True, show_progress_bar=False)

movie_embeddings = compute_embeddings(df['soup'].tolist())


# ============================================================
# 4. GET ALL UNIQUE GENRES (for filter dropdown)
# ============================================================
all_genres = sorted(set(g for genres in df['genre_list'] for g in genres))


# ============================================================
# 6. UI LAYOUT
# ============================================================

# --- Header ---
st.markdown('<div class="main-title">🎬 CineMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Describe a movie in your own words — AI finds the best matches</div>', unsafe_allow_html=True)

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
    options=[3, 5, 10, 15, 20],
    index=2  # default = 10
)

# Minimum match threshold
min_match = st.sidebar.slider(
    "Minimum Match %",
    min_value=5,
    max_value=80,
    value=15,
    step=5
)


# ============================================================
# 7. SEARCH LOGIC
# ============================================================
if query.strip():
    with st.spinner("🔍 Searching movies..."):

        # --- Phase 1: Bi-Encoder Retrieval (Fast) ---
        query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, movie_embeddings)[0]
        
        # Add initial scores
        df['initial_score'] = scores.cpu().numpy()
        
        # --- Apply Initial Filters (to narrow down for re-ranking) ---
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

        # Take top 100 for re-ranking
        top_candidates = filtered.sort_values('initial_score', ascending=False).head(100)
        
        if len(top_candidates) > 0:
            # --- Phase 2: Cross-Encoder Re-ranking (Slow but Accurate) ---
            sentence_pairs = [[query, soup] for soup in top_candidates['soup'].tolist()]
            cross_scores = cross_encoder.predict(sentence_pairs)
            
            # Map scores to 0-100% roughly for display
            # Cross-encoder output for ms-marco is logits, we can normalize
            top_candidates['match_score'] = cross_scores
            # Simple normalization for display purposes
            min_s, max_s = top_candidates['match_score'].min(), top_candidates['match_score'].max()
            if max_s != min_s:
                top_candidates['match_percent'] = ((top_candidates['match_score'] - min_s) / (max_s - min_s) * 70 + 30).round(2)
            else:
                top_candidates['match_percent'] = 50.0

            # Filter by match threshold
            top_candidates = top_candidates[top_candidates['match_percent'] >= min_match]
            
            # Final sort
            top_movies = top_candidates.sort_values('match_score', ascending=False).head(num_results)
        else:
            top_movies = pd.DataFrame()

    # --- Display result count ---
    st.markdown(
        f'<div class="result-count">Showing <b>{len(top_movies)}</b> result(s) for "<i>{query}</i>"</div>',
        unsafe_allow_html=True
    )

    # --- Display Results ---
    if len(top_movies) == 0:
        st.markdown('<div class="no-results">😕 No movies matched your filters. Try relaxing the filters or changing your query.</div>', unsafe_allow_html=True)
    else:
        for _, row in top_movies.iterrows():
            match_pct = row['match_percent']

            # Decide badge color based on match strength
            if match_pct >= 60:
                badge_class = "match-badge"
            elif match_pct >= 40:
                badge_class = "match-badge match-badge-low"
            else:
                badge_class = "match-badge match-badge-vlow"

            # --- Two-column layout: Genre Icon Card | Info ---
            col1, col2 = st.columns([1, 4])

            with col1:
                # Genre icon card instead of poster
                genre_icons = {"Action": "💥", "Comedy": "😂", "Drama": "🎭", "Sci-Fi": "🚀",
                               "Horror": "👻", "Romance": "💕", "Thriller": "🔪", "Adventure": "🗺️",
                               "Animation": "🎨", "Fantasy": "🧙", "Crime": "🕵️", "Documentary": "📽️",
                               "Family": "👨‍👩‍👧", "Mystery": "🔍", "War": "⚔️", "History": "📜",
                               "Music": "🎵", "Western": "🤠", "Science Fiction": "🚀"}
                first_genre = row['genre_list'][0] if row['genre_list'] else "Unknown"
                icon = genre_icons.get(first_genre, "🎬")
                st.markdown(
                    f"""<div style="
                        background: linear-gradient(145deg, #1a1d26, #161821);
                        border: 1px solid #2a2d36;
                        border-radius: 14px;
                        text-align: center;
                        padding: 30px 10px;
                        font-size: 2.8rem;
                    ">{icon}<br>
                    <span style="font-size: 0.75rem; color: #888;">{first_genre}</span>
                    </div>""",
                    unsafe_allow_html=True
                )

            with col2:
                # Match badge
                st.markdown(f'<span class="{badge_class}">{match_pct}% Match</span>', unsafe_allow_html=True)

                # Title
                st.markdown(f'<div class="movie-title">{row["original_title"]}</div>', unsafe_allow_html=True)

                # Meta: Year | Genres | Rating
                star_html = f'<span class="star">★</span> {row["rating"]}/10'
                st.markdown(
                    f'<div class="movie-meta">'
                    f'<span>📅 {int(row["year"])}</span>'
                    f'<span>🎭 {row["genres_display"]}</span>'
                    f'<span>{star_html}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Overview
                st.markdown(f'<div class="movie-overview">{row["overview"]}</div>', unsafe_allow_html=True)

            st.divider()

else:
    # --- Show when no query is entered ---
    st.markdown("""
    <div style="text-align:center; margin-top:60px; color:#555;">
        <p style="font-size:1.1rem;">👆 Type a movie description above to get started!</p>
        <p style="font-size:0.9rem; margin-top:10px;">
            Examples:<br>
            <i>"A man trapped on Mars tries to survive alone"</i><br>
            <i>"An animated movie about emotions inside a girl's head"</i><br>
            <i>"A heist movie where the team has to steal from a casino"</i>
        </p>
    </div>
    """, unsafe_allow_html=True)

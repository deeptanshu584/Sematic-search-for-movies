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

# Custom CSS for a modern, cinematic UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');

    /* Global styles */
    .stApp {
        background-color: #05070a;
        color: #e8e8e8;
        font-family: 'Montserrat', sans-serif;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #05070a;
    }
    ::-webkit-scrollbar-thumb {
        background: #e94560;
        border-radius: 10px;
    }

    /* Glassmorphism Title Section */
    .header-container {
        text-align: center;
        padding: 40px 20px;
        background: radial-gradient(circle at top, rgba(233, 69, 96, 0.1) 0%, rgba(5, 7, 10, 0) 70%);
        margin-bottom: 20px;
    }

    .main-title {
        font-size: 4rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #ffffff 0%, #e94560 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    .subtitle {
        color: #888;
        font-size: 1.2rem;
        font-weight: 400;
    }

    /* Search Box styling */
    .stTextInput div div input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        color: #fff !important;
        padding: 20px 25px !important;
        font-size: 1.1rem !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .stTextInput div div input:focus {
        border-color: #e94560 !important;
        box-shadow: 0 0 20px rgba(233, 69, 96, 0.2) !important;
        background-color: rgba(255, 255, 255, 0.08) !important;
    }

    /* Movie Card Grid Styling */
    .movie-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 24px;
        height: 100%;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
    }

    .movie-card:hover {
        transform: translateY(-10px);
        background: rgba(255, 255, 255, 0.07);
        border-color: rgba(233, 69, 96, 0.5);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
    }

    /* Match Badge */
    .match-badge {
        position: absolute;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #e94560, #93002d);
        padding: 4px 12px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 700;
        color: white;
        box-shadow: 0 4px 10px rgba(233, 69, 96, 0.3);
    }

    .genre-icon-box {
        font-size: 3rem;
        margin-bottom: 15px;
        background: rgba(255, 255, 255, 0.05);
        width: 80px;
        height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 18px;
        margin-left: auto;
        margin-right: auto;
    }

    .movie-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 8px;
        text-align: center;
        line-height: 1.2;
    }

    .movie-meta {
        font-size: 0.85rem;
        color: #e94560;
        font-weight: 600;
        margin-bottom: 12px;
        text-align: center;
    }

    .movie-overview {
        font-size: 0.9rem;
        color: #bbb;
        line-height: 1.5;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 4;
        -webkit-box-orient: vertical;
        text-align: center;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background-color: rgba(5, 7, 10, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Buttons */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #e94560 0%, #93002d 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(233, 69, 96, 0.4) !important;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. LOAD MODELS & DATA
# ============================================================
@st.cache_resource
def load_models():
    """Load the Sentence Transformer models."""
    bi_encoder = SentenceTransformer('sentence-t5-large')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return bi_encoder, cross_encoder

@st.cache_data
def load_data():
    """Load the enriched movie dataset."""
    try:
        df = pd.read_csv("enriched_tmdb_with_wiki.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("processed_movies.csv")
        except:
            df = pd.read_csv("tmdb_5000_movies.csv")
            df['soup'] = df['overview'].fillna("")
        
    df['overview'] = df['overview'].fillna("")
    df['soup'] = df['soup'].fillna("")
    
    def parse_genres(genre_str):
        if pd.isna(genre_str) or genre_str == "":
            return []
        try:
            if isinstance(genre_str, str) and "[" in genre_str:
                data = ast.literal_eval(genre_str)
            else:
                data = genre_str
            
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    return [g.get('name', '') for g in data]
                return [str(i) for i in data]
            return []
        except:
            return []

    genre_col = 'genres_list' if 'genres_list' in df.columns else 'genres'
    df['genre_list'] = df[genre_col].apply(parse_genres)
    df['genres_display'] = df['genre_list'].apply(lambda x: ", ".join(x) if x else "Unknown")

    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)

    df['rating'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)
    return df

bi_encoder, cross_encoder = load_models()
df = load_data()


# ============================================================
# 3. ENCODE ALL MOVIE SOUP
# ============================================================
@st.cache_data
def compute_embeddings(soups: list):
    return bi_encoder.encode(soups, convert_to_tensor=True, show_progress_bar=False)

movie_embeddings = compute_embeddings(df['soup'].tolist())


# ============================================================
# 4. PREPARE UI
# ============================================================
all_genres = sorted(set(g for genres in df['genre_list'] for g in genres))

# --- Header Section ---
st.markdown("""
<div class="header-container">
    <div class="main-title">CINEMATCH</div>
    <div class="subtitle">Discover your next favorite movie with AI Intelligence</div>
</div>
""", unsafe_allow_html=True)

# --- Search Section ---
col_s1, col_s2, col_s3 = st.columns([1, 4, 1])
with col_s2:
    query = st.text_input(
        "",
        placeholder="Describe a movie plot, a feeling, or a specific combination of actors...",
        key="search_query"
    )

# --- Sidebar Filters ---
st.sidebar.markdown("<h2 style='color: #e94560; font-size: 1.5rem;'>REFINE SEARCH</h2>", unsafe_allow_html=True)

selected_genres = st.sidebar.multiselect("GENRES", options=all_genres)

min_year = int(df['year'][df['year'] > 0].min())
year_range = st.sidebar.slider("YEAR RANGE", min_year, 2024, (1990, 2024))

min_rating = st.sidebar.slider("MIN RATING", 0.0, 10.0, 0.0, 0.5)

num_results = st.sidebar.selectbox("RESULT COUNT", [6, 9, 12, 18, 24], index=1)

min_match = st.sidebar.slider("MIN MATCH %", 5, 80, 15)


# ============================================================
# 5. SEARCH & GRID DISPLAY
# ============================================================
if query.strip():
    with st.spinner("🧠 Deep Neural Search in progress..."):
        # Phase 1: Fast Retrieval
        query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, movie_embeddings)[0]
        df['initial_score'] = scores.cpu().numpy()
        
        filtered = df.copy()
        if selected_genres:
            filtered = filtered[filtered['genre_list'].apply(lambda g: any(i in g for i in selected_genres))]
        filtered = filtered[(filtered['year'] >= year_range[0]) & (filtered['year'] <= year_range[1])]
        filtered = filtered[filtered['rating'] >= min_rating]

        top_candidates = filtered.sort_values('initial_score', ascending=False).head(100)
        
        if not top_candidates.empty:
            # Phase 2: Precise Re-ranking
            sentence_pairs = [[query, soup] for soup in top_candidates['soup'].tolist()]
            cross_scores = cross_encoder.predict(sentence_pairs)
            top_candidates['match_score'] = cross_scores
            
            min_s, max_s = top_candidates['match_score'].min(), top_candidates['match_score'].max()
            if max_s != min_s:
                top_candidates['match_percent'] = ((top_candidates['match_score'] - min_s) / (max_s - min_s) * 70 + 30).round(1)
            else:
                top_candidates['match_percent'] = 50.0

            top_candidates = top_candidates[top_candidates['match_percent'] >= min_match]
            top_movies = top_candidates.sort_values('match_score', ascending=False).head(num_results)
        else:
            top_movies = pd.DataFrame()

    # --- Result Grid Display ---
    if top_movies.empty:
        st.markdown("<div style='text-align:center; padding: 100px; color: #555;'>No movies found. Try adjusting your filters.</div>", unsafe_allow_html=True)
    else:
        # Create a grid of 3 columns
        for i in range(0, len(top_movies), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(top_movies):
                    row = top_movies.iloc[i + j]
                    
                    genre_icons = {"Action": "💥", "Comedy": "😂", "Drama": "🎭", "Sci-Fi": "🚀", "Science Fiction": "🚀",
                                   "Horror": "👻", "Romance": "💕", "Thriller": "🔪", "Adventure": "🗺️",
                                   "Animation": "🎨", "Fantasy": "🧙", "Crime": "🕵️", "Family": "👨‍👩‍👧", 
                                   "Mystery": "🔍", "War": "⚔️", "History": "📜", "Music": "🎵", "Western": "🤠"}
                    first_genre = row['genre_list'][0] if row['genre_list'] else "Unknown"
                    icon = genre_icons.get(first_genre, "🎬")
                    
                    with cols[j]:
                        st.markdown(f"""
                        <div class="movie-card">
                            <div class="match-badge">{row['match_percent']}% MATCH</div>
                            <div class="genre-icon-box">{icon}</div>
                            <div class="movie-title">{row['original_title']}</div>
                            <div class="movie-meta">{int(row['year'])} • {first_genre} • ★ {row['rating']}</div>
                            <div class="movie-overview">{row['overview']}</div>
                        </div>
                        """, unsafe_allow_html=True)
            st.write("") # Padding between rows

else:
    # --- Landing Page ---
    st.markdown("""
    <div style="text-align:center; margin-top:100px; opacity: 0.5;">
        <p style="font-size:1.5rem; letter-spacing: 2px;">DESCRIBE THE VIBE TO BEGIN</p>
        <p>Examples: "Mind-bending space heist" or "Heartwarming animated dog movie"</p>
    </div>
    """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import torch
import psutil
import gc

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
def load_models():
    """Load both bi-encoder and cross-encoder models - MAXIMUM ACCURACY MODE"""
    import time
    
    st.info("🚀 Loading MAXIMUM ACCURACY models (sentence-t5-large)...")
    st.warning("⚠️ First time: This will download ~1.1GB and take 5-10 minutes")
    
    start = time.time()
    
    # Load sentence-t5-large (large model)
    with st.spinner("Loading bi-encoder (sentence-t5-large - ~1.1GB)..."):
        bi_encoder = SentenceTransformer('sentence-t5-large')
    
    # Load better cross-encoder
    with st.spinner("Loading cross-encoder (L-12 variant)..."):
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    
    elapsed = time.time() - start
    
    # Check memory
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024**3)
    
    st.success(f"✓ Models loaded in {elapsed:.1f}s | RAM usage: {used_gb:.1f}GB")
    
    return bi_encoder, cross_encoder

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

def explain_match(query, movie_title, movie_plot, movie_genres, match_score):
    """
    Generate explanation for why a movie matched the query.
    Returns a dictionary with different explanation components.
    """
    explanation = {
        'keyword_matches': [],
        'semantic_score': match_score,
        'genre_matches': [],
        'confidence': 'High' if match_score > 0.7 else 'Medium' if match_score > 0.5 else 'Low'
    }
    
    # 1. Keyword Matching
    stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
                  'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had',
                  'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
                  'might', 'must', 'can', 'about', 'from', 'into', 'through', 'during',
                  'before', 'after', 'above', 'below', 'between', 'under', 'again',
                  'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                  'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
                  'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                  'too', 'very', 'that', 'this', 'these', 'those', 'who', 'which'}
    
    query_words = [w.lower() for w in query.split() if w.lower() not in stop_words and len(w) > 2]
    plot_lower = movie_plot.lower()
    title_lower = movie_title.lower()
    
    for word in query_words:
        plot_count = plot_lower.count(word)
        title_count = title_lower.count(word)
        
        if plot_count > 0 or title_count > 0:
            explanation['keyword_matches'].append({
                'word': word,
                'in_plot': plot_count,
                'in_title': title_count
            })
    
    # 2. Genre Matching
    all_genre_keywords = {
        'action': ['action', 'fight', 'battle', 'war', 'combat', 'martial'],
        'comedy': ['funny', 'comedy', 'laugh', 'humor', 'hilarious', 'joke'],
        'drama': ['drama', 'emotional', 'serious', 'tragic'],
        'horror': ['horror', 'scary', 'frightening', 'terror', 'ghost', 'monster'],
        'romance': ['romance', 'love', 'romantic', 'relationship'],
        'sci-fi': ['space', 'alien', 'future', 'robot', 'technology', 'science'],
        'fantasy': ['magic', 'wizard', 'fantasy', 'dragon', 'kingdom', 'enchanted'],
        'thriller': ['thriller', 'suspense', 'mystery', 'detective'],
        'animation': ['animated', 'animation', 'cartoon'],
        'adventure': ['adventure', 'quest', 'journey', 'explore']
    }
    
    query_lower = query.lower()
    detected_genres = []
    
    for genre, keywords in all_genre_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                detected_genres.append(genre)
                break
    
    movie_genre_list = [g.lower() for g in movie_genres]
    
    for detected in detected_genres:
        if any(detected in mg for mg in movie_genre_list):
            explanation['genre_matches'].append(detected)
    
    return explanation

bi_model, cross_model = load_models()
df = load_data()


# ============================================================
# 3. ENCODE ALL MOVIE OVERVIEWS INTO VECTORS
# ============================================================
@st.cache_data
def compute_embeddings(texts: list, _model):
    """Encode all movie texts in batches to save memory"""
    import torch
    
    batch_size = 16  # Smaller batches for large model
    embeddings = []
    
    # Show progress
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = _model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        embeddings.append(batch_embeddings)
        
        # Update progress
        progress = (i + batch_size) / len(texts)
        progress_bar.progress(min(progress, 1.0))
        progress_text.text(f"Encoding movies: {min(i+batch_size, len(texts))}/{len(texts)}")
        
        # Clean memory every 10 batches
        if (i // batch_size) % 10 == 0:
            gc.collect()
    
    progress_text.empty()
    progress_bar.empty()
    
    return torch.cat(embeddings, dim=0)

movie_embeddings = compute_embeddings(df['detailed_plot'].tolist(), bi_model)


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

# --- System Status ---
st.sidebar.markdown("---")
st.sidebar.subheader("💻 System Status")

mem = psutil.virtual_memory()
ram_used = mem.used / (1024**3)
ram_total = mem.total / (1024**3)
ram_percent = mem.percent

st.sidebar.metric(
    "RAM Usage",
    f"{ram_used:.1f}GB / {ram_total:.1f}GB",
    f"{ram_percent:.1f}%"
)

if ram_percent > 80:
    st.sidebar.error("⚠️ High RAM usage! Close other apps.")
elif ram_percent > 60:
    st.sidebar.warning("RAM usage elevated")
else:
    st.sidebar.success("RAM usage normal")


# ============================================================
# 7. SEARCH LOGIC
# ============================================================
if query.strip():
    with st.spinner("🔍 Searching movies... (Ultimate Accuracy Mode)"):
        
        # ========================================
        # STAGE 1: BI-ENCODER FAST RETRIEVAL
        # ========================================
        
        # Encode the user query with bi-encoder
        query_embedding = bi_model.encode(query, convert_to_tensor=True)
        
        # Compute cosine similarity with all movies
        scores = util.cos_sim(query_embedding, movie_embeddings)[0]
        
        # Add initial scores to dataframe
        df['bi_encoder_score'] = scores.cpu().numpy()
        df['match_percent_initial'] = (df['bi_encoder_score'] * 100).round(2)
        
        # Apply filters FIRST (before cross-encoder)
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
        
        # Minimum match % filter (use initial bi-encoder score)
        filtered = filtered[filtered['match_percent_initial'] >= min_match]
        
        # Get top 50 candidates from bi-encoder
        top_50_candidates = filtered.sort_values('bi_encoder_score', ascending=False).head(50)
        
        # ========================================
        # STAGE 2: CROSS-ENCODER RE-RANKING
        # ========================================
        
        if len(top_50_candidates) > 0:
            # CRITICAL: Truncate to 1000 chars for cross-encoder
            pairs = [[query, row['detailed_plot'][:1000]] for _, row in top_50_candidates.iterrows()]
            
            # Get precise relevance scores from cross-encoder
            cross_scores = cross_model.predict(pairs)
            
            # Add cross-encoder scores to the candidates
            top_50_candidates = top_50_candidates.copy()
            top_50_candidates['cross_encoder_score'] = cross_scores
            top_50_candidates['match_score'] = cross_scores  # Use cross-encoder as final score
            top_50_candidates['match_percent'] = (cross_scores * 100).clip(0, 100).round(2)
            
            # Sort by cross-encoder scores and take top N
            top_movies = top_50_candidates.sort_values('cross_encoder_score', ascending=False).head(num_results)
        else:
            # No candidates after filtering
            top_movies = pd.DataFrame()
        
        # Generate explanations (keep existing code)
        if len(top_movies) > 0:
            top_movies['explanation'] = top_movies.apply(
                lambda row: explain_match(
                    query, 
                    row['original_title'], 
                    row['detailed_plot'], 
                    row['genre_list'],
                    row['match_score']
                ),
                axis=1
            )
        
        # Clean up memory
        gc.collect()

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
                    
                    # --- EXPLANATION SECTION (NEW!) ---
                    explanation = movie['explanation']
                    
                    # Expandable "Why this matched?" section
                    with st.expander("🔍 Why this matched?", expanded=False):
                        
                        # Confidence level
                        confidence = explanation['confidence']
                        conf_color = '#10b981' if confidence == 'High' else '#f59e0b' if confidence == 'Medium' else '#ef4444'
                        st.markdown(f'**Confidence:** <span style="color: {conf_color}; font-weight: bold;">{confidence}</span>', 
                                   unsafe_allow_html=True)
                        
                        # Semantic similarity score
                        semantic_pct = explanation['semantic_score'] * 100
                        st.markdown(f'**Plot Similarity:** {semantic_pct:.1f}% (AI understanding of the story)')
                        
                        # Keyword matches
                        if explanation['keyword_matches']:
                            st.markdown('**Matching Keywords:**')
                            keyword_display = []
                            for kw in explanation['keyword_matches'][:5]:  # Show top 5
                                word = kw['word']
                                count = kw['in_plot'] + kw['in_title']
                                keyword_display.append(f"• *{word}* ({count}x)")
                            st.markdown('<br>'.join(keyword_display), unsafe_allow_html=True)
                        else:
                            st.markdown('**Matching Keywords:** None (pure semantic match)')
                        
                        # Genre matches
                        if explanation['genre_matches']:
                            genre_str = ', '.join(explanation['genre_matches'])
                            st.markdown(f'**Genre Match:** {genre_str.title()}')
                        
                        # Visual breakdown
                        st.markdown('**Score Breakdown:**')
                        semantic_pct = explanation['semantic_score'] * 100
                        kw_count = len(explanation['keyword_matches'])
                        genre_count = len(explanation['genre_matches'])
                        
                        # Custom HTML for horizontal progress bars
                        st.markdown(f"""
                        <div style="display: flex; flex-direction: column; gap: 10px; margin-top: 10px;">
                            <!-- Semantic Bar -->
                            <div style="display: flex; align-items: center;">
                                <div style="width: 70px; font-size: 11px; color: #9ca3af;">Semantic</div>
                                <div style="flex-grow: 1; background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px; margin: 0 10px; overflow: hidden;">
                                    <div style="width: {semantic_pct}%; background: #3b82f6; height: 100%; border-radius: 4px;"></div>
                                </div>
                                <div style="width: 40px; font-size: 11px; color: #e8e8e8; text-align: right;">{semantic_pct:.0f}%</div>
                            </div>
                            <!-- Keywords Bar -->
                            <div style="display: flex; align-items: center;">
                                <div style="width: 70px; font-size: 11px; color: #9ca3af;">Keywords</div>
                                <div style="flex-grow: 1; background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px; margin: 0 10px; overflow: hidden;">
                                    <div style="width: {min(kw_count * 20, 100)}%; background: #10b981; height: 100%; border-radius: 4px;"></div>
                                </div>
                                <div style="width: 40px; font-size: 11px; color: #e8e8e8; text-align: right;">{kw_count}</div>
                            </div>
                            <!-- Genre Bar -->
                            <div style="display: flex; align-items: center;">
                                <div style="width: 70px; font-size: 11px; color: #9ca3af;">Genres</div>
                                <div style="flex-grow: 1; background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px; margin: 0 10px; overflow: hidden;">
                                    <div style="width: {min(genre_count * 33, 100)}%; background: #f59e0b; height: 100%; border-radius: 4px;"></div>
                                </div>
                                <div style="width: 40px; font-size: 11px; color: #e8e8e8; text-align: right;">{genre_count}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
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

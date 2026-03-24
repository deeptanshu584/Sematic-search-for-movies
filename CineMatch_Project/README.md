# 🎬 CineMatch: AI-Powered Semantic Movie Search

CineMatch is a next-generation movie recommendation engine that uses Deep Learning to understand the "vibe" and context of your search, rather than just matching keywords.

## 🚀 Features
- **Semantic Search:** Describe a plot, a feeling, or a scene in your own words.
- **Enriched Data:** Uses detailed Wikipedia plots merged with TMDB metadata (Cast, Crew, Keywords).
- **Two-Stage Retrieval:** Uses a Bi-Encoder for speed and a Cross-Encoder for extreme precision.
- **Modern UI:** Premium Glassmorphic interface with a responsive 3-column grid.

---

## 🛠️ How to Run

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your computer.

### 2. Setup
Open your terminal or command prompt inside this folder and install the required AI libraries:
```bash
pip install -r requirements.txt
```

### 3. Launch the App
Run the following command to start the web interface:
```bash
streamlit run app.py
```

### 4. First-Run Note
The first time you search, the application will:
1. Download the pre-trained AI models (approx. 1GB).
2. Process the movie library to create "Embeddings."
**This may take 5–10 minutes depending on your CPU.** Every search after this will be near-instant!

---

## 🧠 Technology Stack
- **Frontend:** Streamlit (Python-based Web Framework)
- **AI Models:** 
  - `all-mpnet-base-v2` (Bi-Encoder for retrieval)
  - `ms-marco-MiniLM-L-6-v2` (Cross-Encoder for re-ranking)
- **Dataset:** TMDB 5000 + Wikipedia Movie Plots (35,000+ entries)

---
*Created as a project to demonstrate Modern Natural Language Processing (NLP) techniques.*

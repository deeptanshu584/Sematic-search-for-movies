# CineMatch - AI-Powered Semantic Movie Search

## Overview
CineMatch is an intelligent movie search engine that understands natural language descriptions. Instead of searching by title or actor, users can describe plot elements and themes.

## Features
- **Semantic Search**: AI-powered natural language understanding using Sentence Transformers.
- **Dataset Enrichment**: Enhanced with Wikipedia plots, increasing descriptive detail by 831%.
- **Multiple Models Tested**: Evaluated 3 state-of-the-art embedding models to find the best configuration.
- **Smart Filters**: Filter results by genre, release year, and minimum rating.
- **Cinematic UI**: Modern Streamlit-based dark interface for an immersive experience.

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
cd app
streamlit run app.py
```

## Project Structure
- `/data` - Raw and enriched datasets (TMDB 5000 + Wiki plots).
- `/scripts` - Data enrichment, preprocessing, and evaluation pipelines.
- `/app` - Main Streamlit application and UI logic.
- `/results` - Visualizations, metrics, and evaluation summaries.
- `/documentation` - Detailed process descriptions.

## Key Results
- **Wikipedia Enrichment**: Successfully matched 74.5% of TMDB movies with Wikipedia plots.
- **Plot Length Improvement**: Average plot length increased from 305 to 2,843 characters (+831.4%).
- **Accuracy Improvement**: Enrichment improved Precision@5 by 10 percentage points (18.1% relative) for the best model.
- **Best Configuration**: **Enriched Dataset** with **all-mpnet-base-v2** model.
- **Top Metrics**: Precision@1: 40.0%, Precision@3: 60.0%, Precision@5: 65.0%, Precision@10: 70.0%.

## Technologies Used
- **NLP**: Sentence Transformers (BERT-based embeddings)
- **Framework**: Streamlit (web interface)
- **Data Analysis**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Data Sources**: TMDB 5000 Movie Dataset, Wikipedia Movie Plots

## Author
Data Mining Lab Project - CineMatch Team

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ── Load data ────────────────────────────────────────────
df = pd.read_csv('netflix_clustered.csv')

# ── Build TF-IDF on full dataset ─────────────────────────
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# ── Extract all unique genres for filter dropdown ────────
all_genres = df['listed_in'].dropna().str.split(', ').explode().unique()
all_genres = sorted([g for g in all_genres if isinstance(g, str)])

# ── Recommendation function ──────────────────────────────
def get_recommendations(title, genre_filter=None, n=10):
    matches = df[df['title'].str.lower() == title.lower()]

    if matches.empty:
        return None, None

    idx = matches.index[0]
    show = df.loc[idx]
    show_genres = set(show['listed_in'].lower().split(', '))

    # ── Genre filtering ──────────────────────────────────
    if genre_filter and genre_filter != "Auto (match input show's genres)":
        # User manually selected a genre
        genre_mask = df['listed_in'].str.lower().str.contains(
            genre_filter.lower(), na=False
        )
    else:
        # Auto mode — match genres of the searched show
        def has_matching_genre(listed_in):
            if pd.isna(listed_in):
                return False
            other_genres = set(listed_in.lower().split(', '))
            return len(show_genres & other_genres) > 0

        genre_mask = df['listed_in'].apply(has_matching_genre)

    genre_filtered_df = df[genre_mask].copy()

    if len(genre_filtered_df) < 5:
        # Fallback to full dataset if filter is too narrow
        genre_filtered_df = df.copy()

    # ── Rebuild TF-IDF on filtered subset ────────────────
    tfidf_filtered = TfidfVectorizer(max_features=5000, stop_words='english')
    filtered_matrix = tfidf_filtered.fit_transform(
        genre_filtered_df['combined_features']
    )

    # ── Find show position in filtered set ───────────────
    filtered_matches = genre_filtered_df[
        genre_filtered_df['title'].str.lower() == title.lower()
    ]

    if filtered_matches.empty:
        return None, None

    filtered_idx = filtered_matches.index[0]
    position = genre_filtered_df.index.get_loc(filtered_idx)

    # ── Cosine similarity within filtered pool ───────────
    show_vector = filtered_matrix[position]
    similarities = cosine_similarity(show_vector, filtered_matrix).flatten()

    similar_positions = similarities.argsort()[::-1][1:n+1]
    results = genre_filtered_df.iloc[similar_positions][
        ['title', 'type', 'listed_in', 'cluster']
    ].copy()
    results['similarity_score'] = similarities[similar_positions].round(3)

    return results, show_genres

# ── Streamlit UI ─────────────────────────────────────────
st.set_page_config(page_title="Netflix Show Recommender", page_icon="🎬", 
                   layout="wide")

st.title("🎬 Netflix Show Recommender")
st.markdown("Find similar shows and movies based on genre, cast, and content")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_input("Enter a Netflix show or movie title:", "")

with col2:
    genre_filter = st.selectbox(
        "Filter by specific genre (optional):",
        ["Auto (match input show's genres)"] + all_genres
    )

if user_input:
    with st.spinner("Finding similar shows..."):
        results, show_genres = get_recommendations(user_input, genre_filter)

    if results is None:
        st.error(f"'{user_input}' not found.")
        suggestions = df[df['title'].str.lower().str.contains(
            user_input.lower(), na=False)]['title'].head(5).tolist()
        if suggestions:
            st.write("Did you mean:")
            for s in suggestions:
                st.write(f"- {s}")
    else:
        st.success(f"Top 10 shows similar to **{user_input}**")
        if show_genres:
            st.caption(f"Matched genres: {', '.join(show_genres)}")
        st.dataframe(results.reset_index(drop=True), use_container_width=True)
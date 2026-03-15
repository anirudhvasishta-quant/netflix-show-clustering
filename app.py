import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ── Load data ────────────────────────────────────────────
df = pd.read_csv('netflix_clustered.csv')

# ── Build TF-IDF matrix ──────────────────────────────────
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# ── Recommendation function ──────────────────────────────
def get_recommendations(title, n=10):
    # Find the index of the show the user typed
    matches = df[df['title'].str.lower() == title.lower()]
    
    if matches.empty:
        return None
    
    idx = matches.index[0]
    
    # Compute cosine similarity between this show and all others
    show_vector = tfidf_matrix[idx]
    similarities = cosine_similarity(show_vector, tfidf_matrix).flatten()
    
    # Get top N most similar shows (excluding itself)
    similar_indices = similarities.argsort()[::-1][1:n+1]
    
    results = df.iloc[similar_indices][['title', 'type', 'listed_in', 'cluster']].copy()
    results['similarity'] = similarities[similar_indices].round(3)
    return results

# ── Streamlit UI ─────────────────────────────────────────
st.set_page_config(page_title="Netflix Show Recommender", page_icon="🎬")

st.title("🎬 Netflix Show Recommender")
st.markdown("Type a show or movie name to find similar content")

# Search box
user_input = st.text_input("Enter a Netflix show or movie title:", "")

if user_input:
    results = get_recommendations(user_input)
    
    if results is None:
        st.error(f"'{user_input}' not found. Try another title.")
        
        # Show suggestions
        suggestions = df[df['title'].str.lower().str.contains(
            user_input.lower(), na=False)]['title'].head(5).tolist()
        if suggestions:
            st.write("Did you mean:")
            for s in suggestions:
                st.write(f"- {s}")
    else:
        st.success(f"Top 10 shows similar to **{user_input}**")
        st.dataframe(results.reset_index(drop=True), use_container_width=True)
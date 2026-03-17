import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ── Load cleaned data ────────────────────────────────────
df = pd.read_csv('netflix_cleaned.csv')

# ── TF-IDF Vectorization ─────────────────────────────────
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

# ── Elbow Method (range extended to 25) ──────────────────
inertia = []
K_range = range(1, 26)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(tfidf_matrix)
    inertia.append(km.inertia_)
    print(f"K={k} done")

plt.figure(figsize=(12, 5))
plt.plot(K_range, inertia, marker='o', color='#E50914')
plt.title('Elbow Method — Finding Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(K_range)
plt.tight_layout()
plt.savefig('elbow_curve.png')
plt.show()

# ── Train Final KMeans with K=15 ─────────────────────────
k = 15
final_km = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = final_km.fit_predict(tfidf_matrix)

# ── See what's inside each cluster ───────────────────────
print("\n── Sample titles per cluster ──")
for i in range(k):
    cluster_titles = df[df['cluster'] == i]['title'].head(8).tolist()
    # Also show top genres in this cluster
    genres = df[df['cluster'] == i]['listed_in'].str.split(', ').explode()
    top_genres = genres.value_counts().head(3).index.tolist()
    print(f"\nCluster {i} — Top genres: {top_genres}")
    for title in cluster_titles:
        print(f"  - {title}")

# ── Save clustered data ──────────────────────────────────
df.to_csv('netflix_clustered.csv', index=False)
print("\nClustered data saved.")
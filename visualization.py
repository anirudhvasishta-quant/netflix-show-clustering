import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── Load data ────────────────────────────────────────────
df = pd.read_csv('netflix_clustered.csv')

# ── Rebuild TF-IDF ───────────────────────────────────────
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# ── PCA: 5000 → 50 dimensions ───────────────────────────
pca = PCA(n_components=50, random_state=42)
reduced_matrix = pca.fit_transform(tfidf_matrix.toarray())
print("PCA done. Shape:", reduced_matrix.shape)

# ── t-SNE: 50 → 2 dimensions ────────────────────────────
tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=500)
tsne_results = tsne.fit_transform(reduced_matrix)
print("t-SNE done.")

df['x'] = tsne_results[:, 0]
df['y'] = tsne_results[:, 1]

# ── Generate 15 distinct colors automatically ────────────
colors = cm.tab20(np.linspace(0, 1, 15))

# ── Build cluster labels from top genres ────────────────
cluster_labels = {}
for i in range(15):
    genres = df[df['cluster'] == i]['listed_in'].str.split(', ').explode()
    top_genre = genres.value_counts().index[0] if len(genres) > 0 else f"Cluster {i}"
    cluster_labels[i] = f"C{i}: {top_genre[:20]}"

# ── Plot ─────────────────────────────────────────────────
plt.figure(figsize=(16, 10))

for i in range(15):
    mask = df['cluster'] == i
    plt.scatter(
        df[mask]['x'],
        df[mask]['y'],
        c=[colors[i]],
        label=cluster_labels[i],
        alpha=0.5,
        s=8
    )

plt.title('Netflix Shows — t-SNE Cluster Visualization (K=15)', fontsize=14)
plt.legend(loc='upper right', fontsize=7, markerscale=2)
plt.tight_layout()
plt.savefig('tsne_clusters.png', dpi=150)
plt.show()
print("t-SNE plot saved.")
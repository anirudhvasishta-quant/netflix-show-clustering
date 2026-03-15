import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# ── Load clustered data ──────────────────────────────────
df = pd.read_csv('netflix_clustered.csv')

# ── Rebuild TF-IDF matrix ────────────────────────────────
# We need to recreate it since we can't save sparse matrices as CSV
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# ── Step 1: Reduce dimensions with PCA first ────────────
# t-SNE is slow on 5000 dimensions
# PCA reduces it to 50 dimensions first — much faster
pca = PCA(n_components=50, random_state=42)
reduced_matrix = pca.fit_transform(tfidf_matrix.toarray())
print("PCA done. Shape:", reduced_matrix.shape)

# ── Step 2: t-SNE reduces 50 → 2 dimensions ─────────────
# Now we can plot it on an X-Y chart
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
tsne_results = tsne.fit_transform(reduced_matrix)
print("t-SNE done.")

# ── Step 3: Add t-SNE coordinates back to dataframe ─────
df['x'] = tsne_results[:, 0]
df['y'] = tsne_results[:, 1]

# ── Step 4: Plot the clusters ────────────────────────────
cluster_colors = ['#E50914', '#00A8E0', '#00C853', '#FF6D00', '#9C27B0']
cluster_names  = ['Cluster 0', 'Cluster 1 (Docs)', 
                  'Cluster 2 (Comedy)', 'Cluster 3 (TV Series)', 
                  'Cluster 4 (International)']

plt.figure(figsize=(12, 8))

for i in range(5):
    mask = df['cluster'] == i
    plt.scatter(
        df[mask]['x'],
        df[mask]['y'],
        c=cluster_colors[i],
        label=cluster_names[i],
        alpha=0.5,
        s=10
    )

plt.title('Netflix Shows — t-SNE Cluster Visualization', fontsize=14)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('tsne_clusters.png', dpi=150)
plt.show()
print("t-SNE plot saved as tsne_clusters.png")
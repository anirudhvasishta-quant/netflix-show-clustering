import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ── Load cleaned data ────────────────────────────────────
df = pd.read_csv('netflix_cleaned.csv')

# ── Step 1: TF-IDF Vectorization ─────────────────────────
# Converts text into numbers that ML can understand
# max_features=5000 means we keep only top 5000 important words
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
# You'll see: (8807, 5000) → 8807 shows, each with 5000 word scores

# ── Step 2: Elbow Method ─────────────────────────────────
# Helps us find the best number of clusters (K)
# We try K from 1 to 15 and measure inertia (how tight clusters are)
inertia = []
K_range = range(1, 16)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(tfidf_matrix)
    inertia.append(km.inertia_)
    print(f"K={k} done")

# Plot the elbow curve
plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, marker='o', color='#E50914')
plt.title('Elbow Method — Finding Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(K_range)
plt.tight_layout()
plt.savefig('elbow_curve.png')
plt.show()

print("\nElbow curve saved. Look at the chart and find where the curve bends.")

# ── Step 3: Train Final KMeans with K=5 ─────────────────
# Now that we know K=5, we train the actual model
k = 5
final_km = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = final_km.fit_predict(tfidf_matrix)

# ── Step 4: See what's inside each cluster ───────────────
# Print 10 sample titles from each cluster
print("\n── Sample titles per cluster ──")
for i in range(k):
    cluster_titles = df[df['cluster'] == i]['title'].head(10).tolist()
    print(f"\nCluster {i}:")
    for title in cluster_titles:
        print(f"  - {title}")

# ── Step 5: Save the clustered data ─────────────────────
df.to_csv('netflix_clustered.csv', index=False)
print("\nClustered data saved as netflix_clustered.csv")
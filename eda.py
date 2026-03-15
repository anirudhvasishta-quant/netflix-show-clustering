import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('netflix_titles.csv')

# ── Basic info ──────────────────────────────────────────
print("Shape:", df.shape)           # rows x columns
print("\nColumns:\n", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())

# ── Missing values ──────────────────────────────────────
print("\nMissing values:\n", df.isnull().sum())

# ── Distribution: Movies vs TV Shows ───────────────────
plt.figure(figsize=(6,4))
df['type'].value_counts().plot(kind='bar', color=['#E50914','#221F1F'])
plt.title('Movies vs TV Shows on Netflix')
plt.xlabel('Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('type_distribution.png')
plt.show()

# ── Top 10 genres ───────────────────────────────────────
# listed_in column has genres like "Dramas, International Movies"
# We split and count each genre individually
from collections import Counter

genres = df['listed_in'].dropna().str.split(', ')
all_genres = [genre for sublist in genres for genre in sublist]
genre_counts = Counter(all_genres)

top_genres = pd.DataFrame(genre_counts.most_common(10), columns=['Genre', 'Count'])

plt.figure(figsize=(10,5))
sns.barplot(data=top_genres, x='Count', y='Genre', palette='Reds_r')
plt.title('Top 10 Genres on Netflix')
plt.tight_layout()
plt.savefig('top_genres.png')
plt.show()

# ── Content added by year ───────────────────────────────
df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
df['year_added'] = df['date_added'].dt.year

plt.figure(figsize=(10,4))
df['year_added'].value_counts().sort_index().plot(kind='bar', color='#E50914')
plt.title('Content Added to Netflix Per Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('yearly_content.png')
plt.show()

print("\nEDA complete. Charts saved.")
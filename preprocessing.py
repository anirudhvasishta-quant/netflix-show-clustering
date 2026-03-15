import pandas as pd

# ── Load data ────────────────────────────────────────────
df = pd.read_csv('netflix_titles.csv')
df = df.drop_duplicates()  

# ── Step 1: Drop columns we don't need for clustering ───
# director has too many missing values, show_id is just an ID
df = df.drop(columns=['show_id', 'director'])

# ── Step 2: Fill missing values ──────────────────────────
# For text columns, fill missing with empty string
df['cast']        = df['cast'].fillna('')
df['country']     = df['country'].fillna('')
df['rating']      = df['rating'].fillna('Not Rated')
df['duration']    = df['duration'].fillna('')
df['date_added']  = df['date_added'].fillna('')
df['description'] = df['description'].fillna('')
df['listed_in']   = df['listed_in'].fillna('')

# ── Step 3: Create a combined text feature ───────────────
# This is the key step — we merge important text columns
# into one single string per show. This becomes our input for ML.
df['combined_features'] = (
    df['cast'] + ' ' +
    df['listed_in'] + ' ' +
    df['description'] + ' ' +
    df['country'] + ' ' +
    df['type']
)

# ── Step 4: Clean the combined text ─────────────────────
# Lowercase everything, remove extra spaces
df['combined_features'] = df['combined_features'].str.lower().str.strip()

# ── Step 5: Verify ───────────────────────────────────────
print("Shape after cleaning:", df.shape)
print("\nMissing values now:\n", df.isnull().sum())
print("\nSample combined feature:\n", df['combined_features'][0])

# ── Save cleaned data ────────────────────────────────────
df.to_csv('netflix_cleaned.csv', index=False)
print("\nCleaned data saved as netflix_cleaned.csv")
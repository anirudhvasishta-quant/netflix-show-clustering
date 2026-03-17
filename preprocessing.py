import pandas as pd

# ── Load and merge both datasets ─────────────────────────
df_old = pd.read_csv('netflix_titles.csv')
df_new = pd.read_csv('netflix_titles_new.csv')

print(f"Old dataset: {df_old.shape[0]} shows")
print(f"New dataset: {df_new.shape[0]} shows")

# Combine both datasets
df = pd.concat([df_old, df_new], ignore_index=True)
print(f"Combined: {df.shape[0]} shows")

# ── Drop duplicates based on title ───────────────────────
df = df.drop_duplicates(subset=['title'])
print(f"After removing duplicates: {df.shape[0]} shows")

# ── Drop columns we don't need ───────────────────────────
df = df.drop(columns=['show_id', 'director'], errors='ignore')

# ── Fill missing values ──────────────────────────────────
df['cast']        = df['cast'].fillna('')
df['country']     = df['country'].fillna('')
df['rating']      = df['rating'].fillna('Not Rated')
df['duration']    = df['duration'].fillna('')
df['date_added']  = df['date_added'].fillna('')
df['description'] = df['description'].fillna('')
df['listed_in']   = df['listed_in'].fillna('')

# ── Create combined text feature ─────────────────────────
df['combined_features'] = (
    df['cast'] + ' ' +
    df['listed_in'] + ' ' +
    df['description'] + ' ' +
    df['country'] + ' ' +
    df['type']
)

df['combined_features'] = df['combined_features'].str.lower().str.strip()

# ── Verify ───────────────────────────────────────────────
print(f"\nFinal shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nYear range of content:")
df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
df['year_added'] = df['date_added'].dt.year
print(df['year_added'].value_counts().sort_index())

df.to_csv('netflix_cleaned.csv', index=False)
print("\nCleaned data saved as netflix_cleaned.csv")
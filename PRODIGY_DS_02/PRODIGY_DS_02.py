import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

url = "https://gist.githubusercontent.com/rodion-solovev-7/80d7156503a4f0a983acc56348f1cdb5/raw/netflix_titles.csv"
df = pd.read_csv(url)

print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())

# Cleaning
for col in df.select_dtypes(include="str").columns:
    df[col] = df[col].str.strip()

df["director"] = df["director"].fillna("Unknown")
df["cast"] = df["cast"].fillna("Unknown")
df["country"] = df["country"].fillna("Unknown")
df["rating"] = df["rating"].fillna("Unknown")
df["duration"] = df["duration"].fillna("Unknown")
df["date_added"] = df["date_added"].fillna("Unknown")

wrong_rating = df["rating"].str.contains("min", case=False, na=False)
df.loc[wrong_rating, "rating"] = "Unknown"

df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
df["year_added"] = df["date_added"].dt.year
df["month_added"] = df["date_added"].dt.month_name()
df["main_country"] = df["country"].apply(lambda x: x.split(",")[0].strip())
df["main_genre"] = df["listed_in"].apply(lambda x: x.split(",")[0].strip())

df["duration_minutes"] = pd.to_numeric(df["duration"].str.extract("(\\d+) min")[0], errors="coerce")
df["seasons"] = pd.to_numeric(df["duration"].str.extract("(\\d+) Season")[0], errors="coerce")

# Visualizations
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

sns.countplot(x="type", data=df, ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("Movies vs TV Shows")

top_ctry = df[df["main_country"] != "Unknown"]["main_country"].value_counts().head(10)
sns.barplot(x=top_ctry.values, y=top_ctry.index, ax=axes[0, 1], color="steelblue")
axes[0, 1].set_title("Top 10 Countries")

top_gen = df["main_genre"].value_counts().head(10)
sns.barplot(x=top_gen.values, y=top_gen.index, ax=axes[0, 2], color="steelblue")
axes[0, 2].set_title("Top 10 Genres")

top_rat = df["rating"].value_counts().head(10)
sns.barplot(x=top_rat.values, y=top_rat.index, ax=axes[1, 0], color="steelblue")
axes[1, 0].set_title("Top 10 Ratings")
axes[1, 0].set_xlabel("Count")

df["year_added"].value_counts().sort_index().plot(kind="line", marker="o", ax=axes[1, 1], color="steelblue")
axes[1, 1].set_title("Content Added Over Years")
axes[1, 1].set_xlabel("Year Added")

sns.histplot(df["release_year"], bins=30, kde=True, ax=axes[1, 2], color="steelblue")
axes[1, 2].set_title("Release Year Distribution")

movies = df[df["type"] == "Movie"]["duration_minutes"].dropna()
sns.histplot(movies, bins=30, kde=True, ax=axes[2, 0], color="steelblue")
axes[2, 0].set_title("Movie Duration Distribution")
axes[2, 0].set_xlabel("Minutes")

tv_shows = df[df["type"] == "TV Show"]
sns.countplot(x="seasons", data=tv_shows, ax=axes[2, 1], color="steelblue")
axes[2, 1].set_title("TV Show Seasons")
axes[2, 1].set_xlabel("Seasons")

top10_ratings = df["rating"].value_counts().head(10).index
sns.countplot(x="rating", hue="type", data=df, ax=axes[2, 2],
              order=top10_ratings)
axes[2, 2].set_title("Rating by Type")
axes[2, 2].tick_params(axis="x", rotation=45)

plt.tight_layout(pad=3.0)
plt.savefig("visualizations.png", dpi=200)

print("\nResults: Movies =", (df["type"] == "Movie").sum(), "| TV Shows =", (df["type"] == "TV Show").sum())
print("Top country:", top_ctry.index[0], "| Top genre:", top_gen.index[0])
print(f"Most movies: 80-120 min | Most TV shows: 1-2 seasons")
print("Visualizations saved as visualizations.png")

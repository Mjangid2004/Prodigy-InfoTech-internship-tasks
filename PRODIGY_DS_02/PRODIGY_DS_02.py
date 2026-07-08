"""
Task-02: Data Cleaning and Exploratory Data Analysis
Dataset: Netflix Movies and TV Shows Dataset


Steps covered:
1. Import dataset
2. View dataset rows and columns
3. Check missing values
4. Check duplicates
5. Handle missing and inconsistent values
6. Create simple new columns
7. Perform visualizations

Dataset file required: netflix_titles.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

url = "https://gist.githubusercontent.com/rodion-solovev-7/80d7156503a4f0a983acc56348f1cdb5/raw/netflix_titles.csv"
df = pd.read_csv(url)

print("Dataset loaded successfully")

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nShape of dataset:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe(include="all"))


print("\nMissing values before cleaning:")
print(df.isnull().sum())


print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()


print("\nUnique values in type column:")
print(df["type"].unique())

print("\nUnique values in rating column:")
print(df["rating"].unique())

print("\nUnique values in duration column:")
print(df["duration"].unique()[:20])


for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

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

# Create main_country column
# Some rows have multiple countries, so we take the first country
df["main_country"] = df["country"].apply(lambda x: x.split(",")[0])

# Create main_genre column
# Some rows have multiple genres, so we take the first genre
df["main_genre"] = df["listed_in"].apply(lambda x: x.split(",")[0])

# Extract movie duration in minutes
df["duration_minutes"] = df["duration"].str.extract("(\\d+) min")
df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce")

# Extract number of seasons for TV shows
df["seasons"] = df["duration"].str.extract("(\\d+) Season")
df["seasons"] = pd.to_numeric(df["seasons"], errors="coerce")

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDataset after cleaning:")
print(df.head())

df.to_csv("cleaned_netflix_dataset.csv", index=False)
print("\nCleaned dataset saved as cleaned_netflix_dataset.csv")



fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# 1. Count of Movies and TV Shows
sns.countplot(x="type", data=df, ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("Count of Movies and TV Shows")
axes[0, 0].set_xlabel("Type")
axes[0, 0].set_ylabel("Count")

# 2. Top 10 countries with most content
top_countries = df[df["main_country"] != "Unknown"]["main_country"].value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index, ax=axes[0, 1], color="steelblue")
axes[0, 1].set_title("Top 10 Countries with Most Netflix Content")
axes[0, 1].set_xlabel("Number of Titles")
axes[0, 1].set_ylabel("Country")

# 3. Top 10 genres
top_genres = df["main_genre"].value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index, ax=axes[0, 2], color="steelblue")
axes[0, 2].set_title("Top 10 Genres on Netflix")
axes[0, 2].set_xlabel("Number of Titles")
axes[0, 2].set_ylabel("Genre")

# 4. Rating distribution
top_ratings = df["rating"].value_counts().index[:10]
sns.countplot(x="rating", data=df, ax=axes[1, 0], order=top_ratings, color="steelblue")
axes[1, 0].set_title("Top 10 Content Ratings")
axes[1, 0].set_xlabel("Rating")
axes[1, 0].set_ylabel("Count")
axes[1, 0].tick_params(axis="x", rotation=45)

# 5. Content added over years
df["year_added"].value_counts().sort_index().plot(kind="line", marker="o", ax=axes[1, 1], color="steelblue")
axes[1, 1].set_title("Content Added on Netflix Over the Years")
axes[1, 1].set_xlabel("Year Added")
axes[1, 1].set_ylabel("Number of Titles")

# 6. Release year distribution
sns.histplot(df["release_year"], bins=30, kde=True, ax=axes[1, 2], color="steelblue")
axes[1, 2].set_title("Distribution of Release Years")
axes[1, 2].set_xlabel("Release Year")
axes[1, 2].set_ylabel("Count")

# 7. Movie duration distribution
movies = df[df["type"] == "Movie"]
sns.histplot(movies["duration_minutes"].dropna(), bins=30, kde=True, ax=axes[2, 0], color="steelblue")
axes[2, 0].set_title("Distribution of Movie Duration")
axes[2, 0].set_xlabel("Duration in Minutes")
axes[2, 0].set_ylabel("Number of Movies")

# 8. TV show seasons distribution
tv_shows = df[df["type"] == "TV Show"]
sns.countplot(x="seasons", data=tv_shows, ax=axes[2, 1], color="steelblue")
axes[2, 1].set_title("Number of Seasons in TV Shows")
axes[2, 1].set_xlabel("Seasons")
axes[2, 1].set_ylabel("Count")

# 9. Relationship between type and rating
sns.countplot(x="rating", hue="type", data=df, ax=axes[2, 2], order=df["rating"].value_counts().index[:10])
axes[2, 2].set_title("Relationship Between Content Type and Rating")
axes[2, 2].set_xlabel("Rating")
axes[2, 2].set_ylabel("Count")
axes[2, 2].tick_params(axis="x", rotation=45)

plt.tight_layout(pad=3.0)
plt.savefig("visualizations.png", dpi=200)
print("\nVisualizations saved as visualizations.png")


print("\nEDA Conclusions:")
print("1. Netflix has both Movies and TV Shows, but Movies are usually more in number.")
print("2. The United States and India are among the top content-producing countries.")
print("3. International Movies, Dramas, and Comedies are common genres.")
print("4. Most content belongs to popular ratings such as TV-MA, TV-14, and TV-PG.")
print("5. Netflix added more content in recent years compared to earlier years.")
print("6. Most movies have a duration between 80 to 120 minutes.")
print("7. Most TV shows have only 1 or 2 seasons.")

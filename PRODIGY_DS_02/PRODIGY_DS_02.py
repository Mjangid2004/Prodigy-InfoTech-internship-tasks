import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
sns.set_palette("viridis")
plt.rcParams["figure.figsize"] = (12, 6)

url = "https://raw.githubusercontent.com/sumitgirwal/google-play-store-data-analysis/master/googleplaystore.csv"
df = pd.read_csv(url)

print("Initial shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# Cleaning
df = df.drop_duplicates()
df = df[df["Category"] != "1.9"]
df.loc[df["Type"].isna(), "Type"] = "Free"
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df = df.dropna(subset=["Rating"])
df = df[df["Rating"] <= 5]
df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

def clean_size(s):
    if isinstance(s, str):
        s = s.strip()
        if s == "Varies with device": return np.nan
        if "M" in s: return float(s.replace("M", ""))
        if "k" in s: return float(s.replace("k", "")) / 1024
    return np.nan

df["Size_MB"] = df["Size"].apply(clean_size)

def clean_installs(i):
    if isinstance(i, str):
        i = i.replace("+", "").replace(",", "").strip()
        try: return int(i)
        except: return np.nan
    return np.nan

df["Installs"] = df["Installs"].apply(clean_installs)

def clean_price(p):
    if isinstance(p, str):
        p = p.replace("$", "").strip()
        try: return float(p)
        except: return 0.0
    return 0.0

df["Price"] = df["Price"].apply(clean_price)
df["Type"] = df["Type"].replace("0", "Free")

print("\nCleaned shape:", df.shape)
print(df[["Rating", "Reviews", "Size_MB", "Installs", "Price"]].describe())

# Visualizations
fig = plt.figure(figsize=(20, 24))

ax1 = fig.add_subplot(5, 2, 1)
ax1.hist(df["Rating"], bins=20, color="steelblue", edgecolor="white")
ax1.set_title("Distribution of App Ratings")
ax1.set_xlabel("Rating")
ax1.set_ylabel("Number of Apps")
ax1.axvline(df["Rating"].median(), color="red", ls="--", label=f'Median: {df["Rating"].median():.2f}')
ax1.legend()

ax2 = fig.add_subplot(5, 2, 2)
top_cats = df["Category"].value_counts().head(10)
top_cats.plot(kind="barh", color="steelblue", ax=ax2)
ax2.set_title("Top 10 App Categories")
ax2.set_xlabel("Number of Apps")

ax3 = fig.add_subplot(5, 2, 3)
type_counts = df["Type"].value_counts()
ax3.pie(type_counts.values, labels=type_counts.index, autopct="%1.1f%%",
        startangle=90, colors=["lightcoral", "lightskyblue"], explode=(0, 0.05))
ax3.set_title("Free vs Paid Apps")

ax4 = fig.add_subplot(5, 2, 4)
sns.boxplot(data=df[df["Category"].isin(top_cats.index)], x="Category", y="Rating", ax=ax4)
ax4.set_title("Rating by Category")
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha="right")

ax5 = fig.add_subplot(5, 2, 5)
df["Installs"].value_counts(bins=9).sort_index().plot(kind="bar", color="steelblue", ax=ax5)
ax5.set_title("Distribution of Installs")
ax5.set_xlabel("Installs Range")
ax5.set_ylabel("Number of Apps")
ax5.tick_params(axis="x", rotation=45)

ax6 = fig.add_subplot(5, 2, 6)
subset = df.dropna(subset=["Size_MB", "Rating"])
ax6.scatter(subset["Size_MB"], subset["Rating"], alpha=0.3, s=10, c="steelblue")
ax6.set_title("App Size vs Rating")
ax6.set_xlabel("Size (MB)")
ax6.set_ylabel("Rating")
ax6.set_xscale("log")

ax7 = fig.add_subplot(5, 2, 7)
paid = df[(df["Type"] == "Paid") & (df["Price"] > 0)]
ax7.hist(paid["Price"], bins=50, color="steelblue", edgecolor="white")
ax7.set_title("Price Distribution of Paid Apps")
ax7.set_xlabel("Price ($)")
ax7.set_ylabel("Number of Apps")

ax8 = fig.add_subplot(5, 2, 8)
df["Content Rating"].value_counts().plot(kind="barh", color="steelblue", ax=ax8)
ax8.set_title("Content Rating Distribution")
ax8.set_xlabel("Number of Apps")

ax9 = fig.add_subplot(5, 2, 9)
sns.heatmap(df[["Rating", "Reviews", "Size_MB", "Installs", "Price"]].corr(),
            annot=True, cmap="coolwarm", center=0, ax=ax9, square=True, fmt=".2f")
ax9.set_title("Correlation Heatmap")

ax10 = fig.add_subplot(5, 2, 10)
ax10.scatter(df["Reviews"], df["Rating"], alpha=0.3, s=10, c="steelblue")
ax10.set_title("Reviews vs Rating")
ax10.set_xlabel("Reviews")
ax10.set_ylabel("Rating")
ax10.set_xscale("symlog")

plt.tight_layout(pad=3.0)
plt.savefig("visualizations.png", dpi=200, bbox_inches="tight")
print("\nVisualizations saved as 'visualizations.png'")

# Insights
print("\nKey Insights:")
print(f"- Average rating: {df['Rating'].mean():.2f}")
print(f"- Top category: {top_cats.index[0]} ({top_cats.values[0]} apps)")
print(f"- Free apps: {type_counts['Free'] / type_counts.sum() * 100:.1f}%")
best = df.groupby("Category")["Rating"].mean()
print(f"- Best avg rating: {best.idxmax()} ({best.max():.2f})")
print(f"- Worst avg rating: {best.idxmin()} ({best.min():.2f})")
print(f"- Median app size: {df['Size_MB'].median():.1f} MB")
print(f"- Most expensive: ${df['Price'].max():.2f}")

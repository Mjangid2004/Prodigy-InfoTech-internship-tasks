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

print("=" * 60)
print("INITIAL DATA OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

df = df.drop_duplicates(keep="first")
print(f"After dropping duplicates: {df.shape}")

corrupt_idx = df[df["Category"] == "1.9"].index
if not corrupt_idx.empty:
    df = df.drop(corrupt_idx)
    print(f"Dropped corrupted row (index {corrupt_idx[0]}): misaligned fields")

df.loc[df["Type"].isna(), "Type"] = "Free"
print(f"Filled {df['Type'].isna().sum()} remaining NaN in 'Type'")

df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df = df.dropna(subset=["Rating"])
print(f"Dropped {1474 - df['Rating'].isna().sum()} rows with missing Rating")

df = df[df["Rating"] <= 5]
print(f"Removed rows with Rating > 5")

df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

def clean_size(size):
    if isinstance(size, str):
        size = size.strip()
        if size == "Varies with device":
            return np.nan
        if "M" in size:
            return float(size.replace("M", ""))
        if "k" in size:
            return float(size.replace("k", "")) / 1024
    return np.nan

df["Size_MB"] = df["Size"].apply(clean_size)

def clean_installs(installs):
    if isinstance(installs, str):
        installs = installs.replace("+", "").replace(",", "").strip()
        try:
            return int(installs)
        except:
            return np.nan
    return np.nan

df["Installs"] = df["Installs"].apply(clean_installs)

def clean_price(price):
    if isinstance(price, str):
        price = price.replace("$", "").strip()
        try:
            return float(price)
        except:
            return 0.0
    return 0.0

df["Price"] = df["Price"].apply(clean_price)
df["Type"] = df["Type"].replace("0", "Free")

print(f"\nCleaned data shape: {df.shape}")
print(f"Missing values after cleaning:\n{df.isnull().sum()}")
print(f"\nData types after cleaning:\n{df.dtypes}")

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print(df[["Rating", "Reviews", "Size_MB", "Installs", "Price"]].describe())

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS - VISUALIZATIONS")
print("=" * 60)

fig = plt.figure(figsize=(20, 24))

ax1 = fig.add_subplot(5, 2, 1)
ax1.hist(df["Rating"], bins=20, color="steelblue", edgecolor="white")
ax1.set_title("Distribution of App Ratings", fontsize=14, fontweight="bold")
ax1.set_xlabel("Rating")
ax1.set_ylabel("Number of Apps")
ax1.axvline(df["Rating"].median(), color="red", ls="--", label=f'Median: {df["Rating"].median()}')
ax1.legend()

ax2 = fig.add_subplot(5, 2, 2)
top_cats = df["Category"].value_counts().head(10)
bars = ax2.barh(top_cats.index[::-1], top_cats.values[::-1], color="steelblue")
ax2.set_title("Top 10 App Categories", fontsize=14, fontweight="bold")
ax2.set_xlabel("Number of Apps")
for bar, val in zip(bars, top_cats.values[::-1]):
    ax2.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2, str(val),
             va="center", fontsize=9)

ax3 = fig.add_subplot(5, 2, 3)
type_counts = df["Type"].value_counts()
colors = ["lightcoral", "lightskyblue"]
ax3.pie(type_counts.values, labels=type_counts.index, autopct="%1.1f%%",
        startangle=90, colors=colors, explode=(0, 0.05))
ax3.set_title("Free vs Paid Apps", fontsize=14, fontweight="bold")

ax4 = fig.add_subplot(5, 2, 4)
top_cats_plot = df[df["Category"].isin(top_cats.index)]
sns.boxplot(data=top_cats_plot, x="Category", y="Rating", ax=ax4)
ax4.set_title("Rating Distribution by Top Categories", fontsize=14, fontweight="bold")
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha="right")

ax5 = fig.add_subplot(5, 2, 5)
installs_bins = [0, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]
labels = ["0-10", "10-100", "100-1K", "1K-10K", "10K-100K", "100K-1M", "1M-10M", "10M-100M", "100M-1B"]
df["Installs_Bin"] = pd.cut(df["Installs"], bins=installs_bins, labels=labels, right=False)
installs_dist = df["Installs_Bin"].value_counts().sort_index()
bars = ax5.bar(installs_dist.index, installs_dist.values, color="steelblue")
ax5.set_title("Distribution of Installs", fontsize=14, fontweight="bold")
ax5.set_xlabel("Number of Installs")
ax5.set_ylabel("Number of Apps")
ax5.tick_params(axis="x", rotation=45)

ax6 = fig.add_subplot(5, 2, 6)
subset = df.dropna(subset=["Size_MB", "Rating"])
ax6.scatter(subset["Size_MB"], subset["Rating"], alpha=0.3, s=10, c="steelblue")
if len(subset) > 0:
    z = np.polyfit(subset["Size_MB"], subset["Rating"], 1)
    p = np.poly1d(z)
    x_sorted = np.sort(subset["Size_MB"].dropna())
    ax6.plot(x_sorted, p(x_sorted), color="red", lw=2)
ax6.set_title("App Size vs Rating", fontsize=14, fontweight="bold")
ax6.set_xlabel("Size (MB)")
ax6.set_ylabel("Rating")
ax6.set_xscale("log")

ax7 = fig.add_subplot(5, 2, 7)
paid = df[df["Type"] == "Paid"].copy()
paid = paid[paid["Price"] > 0]
paid_sorted = paid.sort_values("Price", ascending=False)
ax7.hist(paid_sorted["Price"], bins=50, color="steelblue", edgecolor="white")
ax7.set_title("Price Distribution of Paid Apps", fontsize=14, fontweight="bold")
ax7.set_xlabel("Price ($)")
ax7.set_ylabel("Number of Apps")
ax7.set_xscale("symlog")

ax8 = fig.add_subplot(5, 2, 8)
content_counts = df["Content Rating"].value_counts()
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(content_counts)))
bars = ax8.barh(content_counts.index, content_counts.values, color=colors)
ax8.set_title("Distribution of Content Ratings", fontsize=14, fontweight="bold")
ax8.set_xlabel("Number of Apps")
for bar, val in zip(bars, content_counts.values):
    ax8.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, str(val),
             va="center", fontsize=9)

ax9 = fig.add_subplot(5, 2, 9)
numeric_df = df[["Rating", "Reviews", "Size_MB", "Installs", "Price"]].dropna()
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax9, square=True,
            linewidths=0.5, fmt=".2f")
ax9.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")

ax10 = fig.add_subplot(5, 2, 10)
subset_reviews = df.dropna(subset=["Reviews", "Rating"])
ax10.scatter(subset_reviews["Reviews"], subset_reviews["Rating"], alpha=0.3, s=10, c="steelblue")
if len(subset_reviews) > 0:
    z = np.polyfit(subset_reviews["Reviews"], subset_reviews["Rating"], 1)
    p = np.poly1d(z)
    x_sorted = np.sort(subset_reviews["Reviews"].dropna())
    ax10.plot(x_sorted, p(x_sorted), color="red", lw=2)
ax10.set_title("Number of Reviews vs Rating", fontsize=14, fontweight="bold")
ax10.set_xlabel("Number of Reviews")
ax10.set_ylabel("Rating")
ax10.set_xscale("symlog")

plt.tight_layout(pad=3.0)
plt.savefig("visualizations.png", dpi=200, bbox_inches="tight")
print("\nVisualizations saved as 'visualizations.png'")

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print(f"1. Average app rating: {df['Rating'].mean():.2f}")
print(f"2. Most common category: {top_cats.index[0]} ({top_cats.values[0]} apps)")
print(f"3. Free apps constitute {type_counts['Free'] / type_counts.sum() * 100:.1f}% of the store")
best_rated_cat = df.groupby("Category")["Rating"].mean().idxmax()
worst_rated_cat = df.groupby("Category")["Rating"].mean().idxmin()
print(f"4. Highest rated category (avg): {best_rated_cat} ({df.groupby('Category')['Rating'].mean().max():.2f})")
print(f"5. Lowest rated category (avg): {worst_rated_cat} ({df.groupby('Category')['Rating'].mean().min():.2f})")
print(f"6. Apps with Rating > 4.5: {len(df[df['Rating'] > 4.5])} ({(len(df[df['Rating'] > 4.5]) / len(df)) * 100:.1f}%)")
print(f"7. Median app size: {df['Size_MB'].median():.1f} MB")
print(f"8. Most expensive paid app: ${df['Price'].max():.2f}")
print(f"9. Apps with >1M installs: {len(df[df['Installs'] >= 1_000_000])}")
print(f"10. Correlation between Reviews and Rating: {df['Reviews'].corr(df['Rating']):.3f}")

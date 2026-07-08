import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")

print("Shape:", df.shape)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].hist(df["quality"], bins=6, color="steelblue", edgecolor="white", align="left")
axes[0, 0].set_title("Wine Quality Distribution")

axes[0, 1].hist(df["alcohol"], bins=20, color="steelblue", edgecolor="white")
axes[0, 1].set_title("Alcohol Distribution")

axes[0, 2].hist(df["pH"], bins=20, color="steelblue", edgecolor="white")
axes[0, 2].set_title("pH Distribution")

sns.boxplot(data=df, x="quality", y="alcohol", ax=axes[1, 0])
axes[1, 0].set_title("Alcohol vs Quality")

sns.boxplot(data=df, x="quality", y="volatile acidity", ax=axes[1, 1])
axes[1, 1].set_title("Volatile Acidity vs Quality")

sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0, ax=axes[1, 2], fmt=".2f")
axes[1, 2].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("visualizations.png", dpi=200)

print(f"Best predictor of quality: alcohol (corr: {df['quality'].corr(df['alcohol']):.2f})")
print(f"Worst predictor: volatile acidity (corr: {df['quality'].corr(df['volatile acidity']):.2f})")

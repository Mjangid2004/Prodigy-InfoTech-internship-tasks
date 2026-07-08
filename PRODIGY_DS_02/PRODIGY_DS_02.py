import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("Shape:", df.shape)
print("Nulls:\n", df.isnull().sum())
print(df.describe())

# Cleaning
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns="Cabin")
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].hist(df["Age"], bins=20, color="steelblue", edgecolor="white")
axes[0, 0].set_title("Age Distribution")
axes[0, 0].set_xlabel("Age")

df["Survived"].value_counts().plot(kind="bar", color=["tomato", "steelblue"], ax=axes[0, 1])
axes[0, 1].set_title("Survival Count (0=No, 1=Yes)")
axes[0, 1].set_xticklabels(["Did Not Survive", "Survived"], rotation=0)

df["Pclass"].value_counts().sort_index().plot(kind="bar", color="steelblue", ax=axes[0, 2])
axes[0, 2].set_title("Passenger Class")
axes[0, 2].set_xlabel("Class")

sns.boxplot(data=df, x="Survived", y="Age", ax=axes[1, 0])
axes[1, 0].set_title("Age vs Survival")
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_xticklabels(["Did Not Survive", "Survived"])

pd.crosstab(df["Pclass"], df["Survived"]).plot(kind="bar", stacked=True,
            color=["tomato", "steelblue"], ax=axes[1, 1])
axes[1, 1].set_title("Class vs Survival")
axes[1, 1].set_xlabel("Class")
axes[1, 1].legend(["Did Not Survive", "Survived"])

sns.heatmap(df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "Sex"]].corr(),
            annot=True, cmap="coolwarm", center=0, ax=axes[1, 2], fmt=".2f")
axes[1, 2].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("visualizations.png", dpi=200)

print("\nSurvival rate: {:.2f}%".format(df["Survived"].mean() * 100))
print("Most influential factor: Pclass ({:.2f})".format(df["Survived"].corr(df["Pclass"])))

import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
tips = pd.read_csv(url)


tips.head()

tips.info()

tips.isnull().sum()

print("Duplicate rows:", tips.duplicated().sum())
tips = tips.drop_duplicates()
print("Shape after removing duplicates:", tips.shape, "\n")

print("Basic statistics:")
print(tips.describe(), "\n")

plt.figure(figsize=(8, 5))
plt.hist(tips["total_bill"], bins=15, color="steelblue", edgecolor="white")
plt.title("Distribution of Total Bill Amounts")
plt.xlabel("Total Bill ($)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("visualizations.png", dpi=200)
plt.show()

print("Histogram saved as 'visualizations.png'")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv(r'C:\Users\Utkrisht Singh Parma\Downloads\archive (1)\thyroid_cancer_risk_data.csv') 

print(df.head())  # View first few rows
print(df.info())  # Data types and missing values
print(df.describe())  # Summary statistics

plt.figure(figsize=(10, 7))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Filter out non-numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Now compute the correlation matrix
correlation_matrix = numeric_df.corr()

# Plotting the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()


plt.figure(figsize=(12, 6))
df.hist(bins=30, figsize=(12, 6), edgecolor='black')
plt.suptitle("Distribution of Numerical Features")
plt.show()

sns.pairplot(df, diag_kind='kde')  # Adjust according to dataset features
plt.suptitle("Pair Plot", y=1.02)
plt.show()




import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\Users\Utkrisht Singh Parma\Downloads\archive (1)\thyroid_cancer_risk_data.csv') 

print(df.head())  
print(df.info()) 
print(df.describe()) 

X = df.drop(columns=['Thyroid_Cancer_Risk', 'Patient_ID', 'Diagnosis'])
y = df['Thyroid_Cancer_Risk']

X = pd.get_dummies(X, drop_first=True)


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier()  # For classification

# Fit the model on the training data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model (for classification or regression)
if isinstance(model, XGBClassifier):
    # If classification, use accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
else:
    # use mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")

#regression plot
sns.regplot(x=df['TSH_Level'], y=y, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
plt.title('Regression Plot: TSH Level vs Thyroid Cancer Risk')
plt.xlabel('TSH Level')
plt.ylabel('Thyroid Cancer Risk')
plt.show()

#Histogram plot
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, color='blue')
plt.title('Age Distribution', fontsize=16)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

#class distro
plt.figure(figsize=(8, 6))
sns.countplot(x='Thyroid_Cancer_Risk', data=df, palette='Set2')
plt.title('Class Distribution of Thyroid Cancer Risk', fontsize=16)
plt.ylabel('Count')
plt.xlabel('Thyroid Cancer Risk')
plt.show()


# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=1.0)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

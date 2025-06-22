import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load dataset
df = pd.read_csv("dataset/Sample_with_Salary.csv")

# Replace empty strings with NaN
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# Filter only placed candidates
df = df[df['Placement(Y/N)?'] == 'Placed']

# Strip spaces in string fields
columns_to_strip = ['Gender', 'Stream', 'Internships(Y/N)', 'Training(Y/N)', 'Technical Course(Y/N)']
for col in columns_to_strip:
    df[col] = df[col].astype(str).str.strip()

# Drop rows with missing values in required fields
required_columns = ['Gender', '10th marks', '12th marks', 'Cgpa', 'Communication level',
                    'Internships(Y/N)', 'Training(Y/N)', 'Technical Course(Y/N)', 'Stream', 'Salary']
df = df.dropna(subset=required_columns)

# Encode categorical columns
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

df['Internships(Y/N)'] = df['Internships(Y/N)'].map({'Yes': 1, 'No': 0})
df['Training(Y/N)'] = df['Training(Y/N)'].map({'Yes': 1, 'No': 0})
df['Technical Course(Y/N)'] = df['Technical Course(Y/N)'].map({'Yes': 1, 'No': 0})

le_stream = LabelEncoder()
df['Stream'] = le_stream.fit_transform(df['Stream'])

# Save the stream encoder
os.makedirs("model", exist_ok=True)
with open("model/stream_encoder.pkl", "wb") as f:
    pickle.dump(le_stream, f)

# Define features and target
X = df[['Gender', '10th marks', '12th marks', 'Cgpa', 'Communication level',
        'Internships(Y/N)', 'Training(Y/N)', 'Technical Course(Y/N)', 'Stream']]
y = df['Salary']

# 🔍 Final safety check
print("\n✅ Checking input X for any remaining NaNs before model training:")
print(X.isnull().sum())

# Drop any row that still has NaN in input or output
final_df = pd.concat([X, y], axis=1).dropna()
X = final_df.drop("Salary", axis=1)
y = final_df["Salary"]

# Train-test split and model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("model/salary_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to model/salary_model.pkl")

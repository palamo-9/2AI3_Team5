import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Member 1 : data understanding

# Step 1: Load the dataset
df = pd.read_csv("insurance_data_linear.csv")

# Step 2: Basic info
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nBasic statistics:\n", df.describe())

# Step 3: Define Input (X) and Output (y)
X = df.drop(columns=["charges"])  # Input features
y = df["charges"]                  # Output / Target

print("\nInput features (X):")
print(X.head())

print("\nOutput (y):")
print(y.head())
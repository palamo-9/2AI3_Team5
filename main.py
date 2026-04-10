import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
#  Member 2 → Encoding 
# ==============================

df = pd.get_dummies(df, drop_first=True)

print("\nAfter Encoding:")
print(df.head())

# Update X and y after encoding
X = df.drop("charges", axis=1)
y = df["charges"]
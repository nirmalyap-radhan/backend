import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("car_price.csv")

# Features and target
X = data[["CarName", "fueltype", "doornumber", "carbody",
          "drivewheel", "wheelbase", "cylindernumber", "highwaympg"]]
y = data["price"]

# Categorical and numeric columns
categorical_cols = ["CarName", "fueltype", "doornumber", "carbody",
                    "drivewheel", "cylindernumber"]
numeric_cols = ["wheelbase", "highwaympg"]

# Fit Encoder on categorical columns
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_cat = encoder.fit_transform(X[categorical_cols])

# Final feature matrix
X_final = np.hstack([X_cat, X[numeric_cols].values])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and encoder
joblib.dump(model, "car_price_model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(categorical_cols, "categorical_cols.pkl")
joblib.dump(numeric_cols, "numeric_cols.pkl")

print("Model & Encoder Saved Successfully!")

# model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("student_performance.csv")

# Separate features and target
X = df.drop("final_score", axis=1)
y = df["final_score"]

# Preprocess categorical columns
categorical_features = ["family_support", "extracurricular"]

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(drop="if_binary"), categorical_features)],
    remainder="passthrough"
)

# Create Linear Regression pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Save model
joblib.dump(pipeline, "model.joblib")
print("✅ Model saved as model.joblib")
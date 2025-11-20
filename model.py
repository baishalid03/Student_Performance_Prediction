import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("student_performance.csv")

# Use correct column names from your CSV
X = data[['study_hours', 'attendance_percent', 'prev_grade', 'family_support', 'extracurricular']]
y = data['final_score']

# Convert categorical columns ('yes'/'no') to numeric
X['family_support'] = X['family_support'].map({'no': 0, 'yes': 1})
X['extracurricular'] = X['extracurricular'].map({'no': 0, 'yes': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save trained model
pickle.dump(model, open('model.pkl', 'wb'))

print("✅ Model trained and saved as model.pkl")
# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# SAVE the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")

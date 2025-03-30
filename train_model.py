import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("health_insurance_dataset.csv")

# Encode categorical columns
label_encoders = {}
categorical_columns = ["Insurance Provider", "Policy Name", "Coverage Type", 
                       "Coverage Details", "Eligibility Criteria", "Waiting Period", 
                       "Exclusions", "Additional Benefits", "Category"]

df_processed = df.copy()

for col in categorical_columns:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le  # Store encoders for later use

# Save the label encoders
joblib.dump(label_encoders, "label_encoders.pkl")

# Select relevant features
features = ["Insurance Provider", "Coverage Type", "Premium Amount (Annual)", 
            "Coverage Details", "Eligibility Criteria", "Network Hospitals", 
            "Claim Settlement Ratio (%)", "Waiting Period", "Exclusions", "Additional Benefits"]

X = df_processed[features]

# Train KNN model
knn_model = NearestNeighbors(n_neighbors=5, metric="euclidean")
knn_model.fit(X)

# Save the trained model
joblib.dump(knn_model, "knn_model.pkl")

print("Model training complete. Saved as 'knn_model.pkl'")

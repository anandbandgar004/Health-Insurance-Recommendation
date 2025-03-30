import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("C:/Users/Anand Bandgar/Downloads/health_insurance_dataset.csv")

# Load trained model and encoders
knn_model = joblib.load("knn_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Streamlit UI
st.title("🏥 Health Insurance Recommendation System")

st.sidebar.header("Enter Your Preferences")

# User Inputs
insurance_provider = st.sidebar.selectbox("Preferred Insurance Provider", df["Insurance Provider"].unique())
coverage_type = st.sidebar.selectbox("Coverage Type", df["Coverage Type"].unique())
premium_amount = st.sidebar.slider("Max Annual Premium (₹)", min_value=10000, max_value=100000, step=500)
coverage_details = st.sidebar.selectbox("Coverage Details", df["Coverage Details"].unique())
eligibility_criteria = st.sidebar.selectbox("Eligibility Criteria", df["Eligibility Criteria"].unique())
network_hospitals = st.sidebar.slider("Min Network Hospitals", min_value=50, max_value=500, step=10)
claim_ratio = st.sidebar.slider("Min Claim Settlement Ratio (%)", min_value=50, max_value=100, step=1)
waiting_period = st.sidebar.selectbox("Maximum Waiting Period", df["Waiting Period"].unique())
exclusions = st.sidebar.selectbox("Preferred Exclusions", df["Exclusions"].unique())
additional_benefits = st.sidebar.selectbox("Additional Benefits", df["Additional Benefits"].unique())

# Convert user input to model format
user_input = {
    "Insurance Provider": label_encoders["Insurance Provider"].transform([insurance_provider])[0],
    "Coverage Type": label_encoders["Coverage Type"].transform([coverage_type])[0],
    "Premium Amount (Annual)": premium_amount,
    "Coverage Details": label_encoders["Coverage Details"].transform([coverage_details])[0],
    "Eligibility Criteria": label_encoders["Eligibility Criteria"].transform([eligibility_criteria])[0],
    "Network Hospitals": network_hospitals,
    "Claim Settlement Ratio (%)": claim_ratio,
    "Waiting Period": label_encoders["Waiting Period"].transform([waiting_period])[0],
    "Exclusions": label_encoders["Exclusions"].transform([exclusions])[0],
    "Additional Benefits": label_encoders["Additional Benefits"].transform([additional_benefits])[0]
}

# Recommend function
def recommend_insurance(user_input):
    user_df = pd.DataFrame([user_input])
    distances, indices = knn_model.kneighbors(user_df)
    recommendations = df.iloc[indices[0]][["Policy ID", "Insurance Provider", "Policy Name", "Premium Amount (Annual)", "Coverage Type", "Category"]]
    return recommendations

# Get recommendations when button is clicked
if st.sidebar.button("Find Best Plans"):
    recommendations = recommend_insurance(user_input)
    st.subheader("📋 Recommended Insurance Plans")
    st.dataframe(recommendations)

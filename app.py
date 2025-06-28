import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load files
df_severity = pd.read_csv("Symptom-severity.csv")
df_description = pd.read_csv("symptom_Description.csv")
df_precaution = pd.read_csv("symptom_precaution.csv")
df_main = pd.read_csv("dataset.csv")

# Load model and label encoder
model = pickle.load(open("decision_tree_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Severity dictionary
severity_dict = dict(zip(df_severity['Symptom'].str.strip(), df_severity['weight']))

# Get full list of symptoms
all_symptoms = sorted(list(severity_dict.keys()))

# Helper functions
def encode_input(symptoms, max_len=17):
    """Convert symptom names into severity-encoded list of length 17"""
    encoded = [severity_dict.get(sym.strip(), 0) for sym in symptoms]
    padded = encoded + [0]*(max_len - len(encoded))  # pad with 0s
    return padded[:max_len]

def get_description(disease_name):
    row = df_description[df_description['Disease'].str.lower() == disease_name.lower()]
    return row['Description'].values[0] if not row.empty else "No description available."

def get_precautions(disease_name):
    row = df_precaution[df_precaution['Disease'].str.lower() == disease_name.lower()]
    return row.iloc[0, 1:].dropna().tolist() if not row.empty else ["No precautions available."]

# -------------------------- Streamlit UI --------------------------

st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("🩺 Disease Prediction from Symptoms")
st.write("Select your symptoms from the dropdown to predict the possible disease.")

# Symptom selection
selected_symptoms = st.multiselect("Select up to 17 symptoms:", all_symptoms)

if st.button("Predict Disease"):

    if not selected_symptoms:
        st.error("Please select at least 1 symptom.")
    else:
        input_vector = encode_input(selected_symptoms)
        input_vector = np.array(input_vector).reshape(1, -1)

        # Predict
        predicted_label = model.predict(input_vector)[0]
        predicted_disease = label_encoder.inverse_transform([predicted_label])[0]

        # Fetch info
        description = get_description(predicted_disease)
        precautions = get_precautions(predicted_disease)

        # Display results
        st.success(f"🧬 Predicted Disease: **{predicted_disease}**")
        st.subheader("🧾 Description:")
        st.write(description)

        st.subheader("🩺 Recommended Precautions:")
        for p in precautions:
            st.markdown(f"- {p}")

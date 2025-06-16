import streamlit as st
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

# Load models and encoders
model = joblib.load("crime_type_classifier_weighted.pkl")
text_model = joblib.load("sentence_transformer_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
structured_encoders = joblib.load("structured_encoders.pkl")

# Structured input options
structured_cols = ['district', 'crime_location_type', 'victim_age',
                   'accused_age', 'season', 'reported_by', 'victim_gender', 
                   'accused_gender', 'weapon_used', 'vehicle_involved', 
                   'num_accused', 'relationship', 'day_of_week', 'hour_of_crime']

st.title("Crime Category Prediction (FIR Narrative Weighted)")

# FIR Narrative input
fir_text = st.text_area("Enter FIR Narrative", height=150)

# Structured Inputs
user_inputs = {}
for col in structured_cols:
    if col in ['victim_age', 'accused_age', 'num_accused', 'hour_of_crime']:
        user_inputs[col] = st.number_input(f"{col.replace('_', ' ').capitalize()}", min_value=0, max_value=100, value=30)
    else:
        le = structured_encoders[col]
        user_inputs[col] = st.selectbox(f"{col.replace('_', ' ').capitalize()}", le.classes_.tolist())
        user_inputs[col] = le.transform([user_inputs[col]])[0]

# Predict button
if st.button("Predict Crime Category"):
    if not fir_text.strip():
        st.warning("Please enter an FIR narrative.")
    else:
        # Encode narrative
        narrative_embed = text_model.encode([fir_text])
        narrative_weighted = np.hstack([narrative_embed] * 3)

        # Combine structured + text features
        structured_array = np.array([user_inputs[col] for col in structured_cols]).reshape(1, -1)
        final_input = np.hstack([structured_array, narrative_weighted])

        # Predict
        prediction = model.predict(final_input)[0]
        pred_label = label_encoder.inverse_transform([prediction])[0]

        st.success(f"📌 **Predicted Crime Category:** `{pred_label}`")

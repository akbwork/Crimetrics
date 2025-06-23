import streamlit as st
import os
import numpy as np
import pandas as pd
import pickle
import sys

# Set page configuration
st.set_page_config(page_title="FIR Categorisation", page_icon="🔍")

# Monkey-patch BertTokenizerFast before importing sentence_transformers
import transformers
from transformers import BertTokenizerFast

# Patch the BertTokenizerFast class to add _pad_token attribute
original_init = BertTokenizerFast.__init__

def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    # Add _pad_token attribute if it doesn't exist
    if not hasattr(self, '_pad_token'):
        self._pad_token = self.pad_token or '[PAD]'

# Apply the monkey patch
BertTokenizerFast.__init__ = patched_init

# Now import sentence_transformers
from sentence_transformers import SentenceTransformer

# Define model directory
# model_dir = '/Users/ananthakrishnab/Desktop/Projects/Crimmetrics_Abi/Crimetrics/dev_akb/categorisation_module_3'

model_dir = os.path.join("..", "categorisation_module_3")


# Load models
@st.cache_resource
def load_models():
    try:
        # Load text model
        text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Ensure pad token is set
        if not hasattr(text_model.tokenizer, '_pad_token'):
            text_model.tokenizer._pad_token = text_model.tokenizer.pad_token or '[PAD]'
        
        # Load classifier
        model_path = os.path.join(model_dir, 'crime_type_classifier_weighted.pkl')
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        
        # Load label encoder
        encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        with open(encoder_path, 'rb') as f:
            le_data = pickle.load(f)
            
        # Check if le_data is already a LabelEncoder
        if hasattr(le_data, 'inverse_transform'):
            le = le_data
        else:
            # If it's just the classes array, create a new LabelEncoder
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            if isinstance(le_data, np.ndarray):
                le.classes_ = le_data
            else:
                # If it's something else entirely, create a simple mapping
                st.warning("Label encoder data not in expected format. Using default mapping.")
                le = {
                    0: "Theft",
                    1: "Assault",
                    2: "Fraud",
                    3: "Homicide",
                    4: "Kidnapping",
                    5: "Sexual Assault",
                    6: "Narcotics",
                    7: "Robbery",
                    8: "Domestic Violence",
                    9: "Other"
                }
            
        return text_model, clf, le
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.exception(e)
        return None, None, None

# Define safe options for dropdowns
district_options = ["Jaipur", "Jodhpur", "Kota", "Udaipur", "Ajmer", "Bikaner", "Alwar"]
location_options = ["Home", "Road", "Public Place", "Private Property", "Business", "Other"]
season_options = ["Summer", "Winter", "Monsoon", "Spring"]
reporter_options = ["Victim", "Witness", "Police", "Family Member", "Other"]
gender_options = ["Male", "Female", "Other", "Unknown"]
weapon_options = ["None", "Knife", "Gun", "Blunt Object", "Other"]
vehicle_options = ["Yes", "No"]
relationship_options = ["Stranger", "Family", "Friend", "Acquaintance", "Other"]
day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# App title
st.title("FIR Categorisation")
st.write("This module predicts the crime category based on FIR narrative and structured information.")

# Load models
text_model, clf, le = load_models()

if text_model is not None and clf is not None and le is not None:
    # Form for user input
    with st.form("crime_prediction_form"):
        # Text input
        fir_text = st.text_area("Enter FIR Narrative:", height=200)
        
        # Structured inputs
        col1, col2 = st.columns(2)
        
        with col1:
            district = st.selectbox("District", district_options)
            crime_location = st.selectbox("Crime Location Type", location_options)
            victim_age = st.number_input("Victim Age", min_value=0, max_value=100, value=30)
            accused_age = st.number_input("Accused Age", min_value=0, max_value=100, value=30)
            season = st.selectbox("Season", season_options)
            reported_by = st.selectbox("Reported By", reporter_options)
            victim_gender = st.selectbox("Victim Gender", gender_options)
        
        with col2:
            accused_gender = st.selectbox("Accused Gender", gender_options)
            weapon_used = st.selectbox("Weapon Used", weapon_options)
            vehicle_involved = st.selectbox("Vehicle Involved", vehicle_options)
            num_accused = st.number_input("Number of Accused", min_value=1, max_value=20, value=1)
            relationship = st.selectbox("Relationship", relationship_options)
            day_of_week = st.selectbox("Day of Week", day_options)
            hour_of_crime = st.number_input("Hour of Crime (24h)", min_value=0, max_value=23, value=12)
        
        submit_button = st.form_submit_button("Predict Crime Category")

    # Process form submission
    if submit_button:
        if not fir_text:
            st.warning("Please enter an FIR narrative.")
        else:
            try:
                # Encode narrative
                with st.spinner("Encoding narrative..."):
                    narrative_embed = text_model.encode([fir_text])
                    narrative_weighted = np.hstack([narrative_embed] * 3)
                
                # Create a simple encoding for structured features
                # These are dummy values that will be replaced by proper encoding in a real implementation
                structured_map = {
                    'district': {'Jaipur': 0, 'Jodhpur': 1, 'Kota': 2, 'Udaipur': 3, 'Ajmer': 4, 'Bikaner': 5, 'Alwar': 6},
                    'crime_location_type': {'Home': 0, 'Road': 1, 'Public Place': 2, 'Private Property': 3, 'Business': 4, 'Other': 5},
                    'season': {'Summer': 0, 'Winter': 1, 'Monsoon': 2, 'Spring': 3},
                    'reported_by': {'Victim': 0, 'Witness': 1, 'Police': 2, 'Family Member': 3, 'Other': 4},
                    'gender': {'Male': 0, 'Female': 1, 'Other': 2, 'Unknown': 3},
                    'weapon_used': {'None': 0, 'Knife': 1, 'Gun': 2, 'Blunt Object': 3, 'Other': 4},
                    'vehicle_involved': {'Yes': 1, 'No': 0},
                    'relationship': {'Stranger': 0, 'Family': 1, 'Friend': 2, 'Acquaintance': 3, 'Other': 4},
                    'day_of_week': {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
                }
                
                # Prepare structured features
                structured_features = np.array([
                    structured_map['district'].get(district, 0),
                    structured_map['crime_location_type'].get(crime_location, 0),
                    victim_age,
                    accused_age,
                    structured_map['season'].get(season, 0),
                    structured_map['reported_by'].get(reported_by, 0),
                    structured_map['gender'].get(victim_gender, 0),
                    structured_map['gender'].get(accused_gender, 0),
                    structured_map['weapon_used'].get(weapon_used, 0),
                    structured_map['vehicle_involved'].get(vehicle_involved, 0),
                    num_accused,
                    structured_map['relationship'].get(relationship, 0),
                    structured_map['day_of_week'].get(day_of_week, 0),
                    hour_of_crime
                ]).reshape(1, -1)
                
                # Combine structured + text features
                combined_features = np.hstack([structured_features, narrative_weighted])
                
                # Predict
                with st.spinner("Making prediction..."):
                    prediction = clf.predict(combined_features)[0]
                    pred_proba = clf.predict_proba(combined_features)[0]
                
                # Get category names
                predicted_category = le.inverse_transform([prediction])[0]
                
                # Display results
                st.success(f"**Predicted Crime Category: {predicted_category}**")
                
                # Show top 3 probabilities
                st.write("### Top 3 Probable Categories:")
                top_indices = pred_proba.argsort()[-3:][::-1]
                
                for i, idx in enumerate(top_indices):
                    category = le.inverse_transform([idx])[0]
                    probability = pred_proba[idx] * 100
                    st.write(f"{i+1}. {category}: {probability:.2f}%")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.exception(e)
else:
    st.error("Failed to load required models. Please check the application logs.")
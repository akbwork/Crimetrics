import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

st.set_page_config(page_title="FIR Quality Assessment", layout="wide")
st.title("📋 FIR Quality Assessment Score")

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Define scoring logic
required_keywords = {
    'victim': ['victim', 'complainant', 'girl', 'boy', 'woman', 'man'],
    'accused': ['accused', 'offender', 'culprit', 'suspect'],
    'date': ['date', 'on', 'around', 'time'],
    'location': ['at', 'in', 'near', 'location', 'district'],
    'crime': ['murder', 'rape', 'theft', 'assault', 'robbery', 'violence', 'molestation']
}

def get_completeness(text):
    score = 0
    for key in required_keywords:
        if any(word in text.lower() for word in required_keywords[key]):
            score += 0.6
    return round(min(score, 3), 2)

def get_consistency(text):
    inconsistent_phrases = ['not clear', 'unknown', 'unable to', 'unclear', 'confused']
    score = 2
    for phrase in inconsistent_phrases:
        if phrase in text.lower():
            score -= 0.5
    return max(0, round(score, 2))

crime_type_templates = {
    "Murder": "The accused killed the victim.",
    "Rape": "The victim was sexually assaulted by the accused.",
    "Theft": "The accused stole valuables from the victim.",
    "Robbery": "The accused forcefully took possessions from the victim.",
    "Assault": "The accused physically attacked the victim.",
    "Kidnapping": "The victim was abducted by the accused.",
    "Molestation": "The victim was sexually harassed.",
    "Cyber Crime": "The accused committed an offense using digital means.",
    "Dowry Harassment": "The victim was harassed for dowry.",
}

def get_relevance(text, category):
    fir_embed = model.encode(text, convert_to_tensor=True)
    template = crime_type_templates.get(category, "The accused committed a crime.")
    template_embed = model.encode(template, convert_to_tensor=True)
    score = util.cos_sim(fir_embed, template_embed).item()
    return round(5 * max(0, score), 2)

# Upload CSV
uploaded_file = st.file_uploader("Upload FIR Dataset CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'fir_narrative' not in df.columns or 'crime_category' not in df.columns:
        st.error("CSV must have 'fir_narrative' and 'crime_category' columns")
    else:
        st.success("File uploaded. Scoring in progress...")

        tqdm.pandas()
        df["completeness_score"] = df["fir_narrative"].progress_apply(get_completeness)
        df["consistency_score"] = df["fir_narrative"].progress_apply(get_consistency)
        df["relevance_score"] = df.apply(lambda x: get_relevance(x["fir_narrative"], x["crime_category"]), axis=1)
        df["fir_quality_score"] = df["completeness_score"] + df["consistency_score"] + df["relevance_score"]

        # ⬇️ New: Calculate and show overall score
        overall_score = round(df["fir_quality_score"].mean(), 2)
        st.metric("📊 Overall FIR Dataset Quality Score", f"{overall_score} / 10")

        st.success("✅ Scoring completed!")

        st.dataframe(df[["fir_narrative", "crime_category", "completeness_score", "consistency_score", "relevance_score", "fir_quality_score"]])

        st.download_button("Download Scored CSV", data=df.to_csv(index=False), file_name="fir_scored.csv")

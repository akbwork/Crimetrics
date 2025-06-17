import streamlit as st
import os

st.set_page_config(
    page_title="Crimmetrics Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 24px;
        color: #475569;
        margin-top: 0px;
    }
    .module-header {
        font-size: 20px;
        font-weight: bold;
        color: #1E3A8A;
    }
    .module-desc {
        font-size: 15px;
        color: #4B5563;
    }
    .highlight {
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header" style="font-size:30px;">Crimmetrics Analytics Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header" style="font-size:30px;">Advanced Crime Data Analysis & Intelligence</p>', unsafe_allow_html=True)

# Brief intro with highlight
st.markdown("""
<div class="highlight">
Crimmetrics is an integrated suite of AI-powered tools designed to help law enforcement agencies 
digitize, analyze, and extract actionable insights from FIR (First Information Report) data. 
The platform leverages natural language processing, machine learning, and data visualization to 
transform unstructured crime reports into structured, searchable intelligence.
</div>
""", unsafe_allow_html=True)

# Platform statistics
st.markdown("### Platform Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("FIRs Processed", "1,243", "+12%")
with col2:
    st.metric("Average Quality Score", "7.8/10", "+2.1%")
with col3:
    st.metric("Categorization Accuracy", "94.7%", "+1.8%")
with col4:
    st.metric("Processing Time", "45 sec", "-15%")

# Module cards
st.markdown("## Key Modules")

# Create two rows of module cards
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

with row1_col1:
    st.markdown('<p class="module-header">📄 FIR Digitisation</p>', unsafe_allow_html=True)
    st.markdown('<p class="module-desc">Convert physical FIR documents into structured digital data using OCR and AI parsing. This module extracts key information from scanned FIRs including complainant details, accused information, and incident descriptions.</p>', unsafe_allow_html=True)
    if st.button("Open FIR Digitisation", key="btn_fir_dig"):
        st.switch_page("pages/1_FIR_Digitisation.py")

with row1_col2:
    st.markdown('<p class="module-header">✏️ Auto Complete</p>', unsafe_allow_html=True)
    st.markdown('<p class="module-desc">Intelligently predict and fill missing information in FIR data, focusing on the crucial relationship field between victims and accused. Improves data completeness and consistency for better analysis.</p>', unsafe_allow_html=True)
    if st.button("Open Auto Complete", key="btn_auto_complete"):
        st.switch_page("pages/2_Auto_Complete.py")

with row2_col1:
    st.markdown('<p class="module-header">📋 FIR Quality Assessment</p>', unsafe_allow_html=True)
    st.markdown('<p class="module-desc">Evaluate the quality and completeness of FIR narratives with an AI-powered scoring system. Identifies gaps in reporting and measures consistency, relevance, and detail level of crime reports.</p>', unsafe_allow_html=True)
    if st.button("Open FIR Quality Assessment", key="btn_fir_quality"):
        st.switch_page("pages/3_FIR_Quality.py")

with row2_col2:
    st.markdown('<p class="module-header">🔍 Crime Categorisation</p>', unsafe_allow_html=True)
    st.markdown('<p class="module-desc">Automatically classify crime reports into relevant categories using a hybrid model that combines structured data and narrative text analysis. Ensures consistent categorization across reports.</p>', unsafe_allow_html=True)
    if st.button("Open Crime Categorisation", key="btn_categorisation"):
        st.switch_page("pages/4_Categorisation.py")

# Modus Operandi clustering gets a full-width section to highlight it
st.markdown("---")
st.markdown('<p class="module-header">📊 Modus Operandi Clustering</p>', unsafe_allow_html=True)
st.markdown('<p class="module-desc">Identify patterns and clusters in crime data by analyzing FIR narratives using advanced NLP and clustering algorithms. Discover common criminal techniques, link similar cases, and visualize crime patterns across time and geography.</p>', unsafe_allow_html=True)

if st.button("Open Modus Operandi Clustering", key="btn_clustering"):
    st.switch_page("pages/5_Clustering.py")

# Footer with information
st.markdown("---")
st.markdown("""
### About Crimmetrics

Crimmetrics helps law enforcement agencies transform their crime data workflows through:

- **Digitization**: Convert paper documents to structured data
- **Analysis**: Extract patterns and insights from large volumes of FIR data
- **Intelligence**: Surface connections and trends to guide investigations
- **Efficiency**: Reduce manual processing time and increase data accuracy
            
Arca Ai Production
""")
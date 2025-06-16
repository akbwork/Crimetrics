import streamlit as st

st.set_page_config(
    page_title="Crimmetrics Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("Crimmetrics Dashboard")
st.write("Welcome to the Crimmetrics advanced analytics platform!")

st.markdown("""
## Available Modules

This platform offers the following key modules:

1. **FIR Digitisation** - Convert physical FIR documents into structured digital data
2. **Auto Complete** - Intelligent form auto-completion for faster data entry
3. **Categorisation** - Automatically categorize crime reports into relevant categories
4. **Clustering** - Identify patterns and clusters in crime data

Use the sidebar to navigate between modules.
""")

# Display some sample metrics or visualizations on the home page
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Documents Processed", "1,243", "+12%")
with col2:
    st.metric("Accuracy Score", "94.7%", "+2.1%")
with col3:
    st.metric("Processing Time", "45 sec", "-15%")

# Add an image or logo
st.image("https://via.placeholder.com/800x300?text=Crimmetrics+Analytics", 
         caption="Crimmetrics - Advanced Crime Analytics Platform")
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Categorisation | Crimmetrics",
    page_icon="🏷️",
    layout="wide"
)

st.title("Categorisation Module")
st.write("Automatically categorize crime reports into relevant categories.")

# Upload option for CSV
uploaded_file = st.file_uploader("Upload crime data CSV", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    try:
        data = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded data with {len(data)} records")
        
        # Display sample of the data
        st.subheader("Data Preview")
        st.dataframe(data.head())
        
        # Categorisation options
        st.subheader("Categorisation Options")
        
        col1, col2 = st.columns(2)
        with col1:
            category_column = st.selectbox(
                "Select text column to categorize",
                data.columns.tolist()
            )
        with col2:
            algorithm = st.selectbox(
                "Select categorisation algorithm",
                ["Rule-based", "Machine Learning", "NLP Classification"]
            )
        
        # Categorisation button
        if st.button("Categorize Data"):
            with st.spinner("Categorizing data..."):
                # Mock categorisation process
                st.info("Simulating categorisation (your actual ML code would go here)")
                
                # Create mock categorized data
                data['Category'] = np.random.choice(
                    ["Violent Crime", "Property Crime", "Cybercrime", "White Collar", "Drug-related"],
                    size=len(data)
                )
                data['Confidence'] = np.random.uniform(0.7, 0.99, len(data))
                
                # Display results
                st.subheader("Categorisation Results")
                st.dataframe(data)
                
                # Category distribution
                st.subheader("Category Distribution")
                category_counts = data['Category'].value_counts()
                st.bar_chart(category_counts)
                
                # Download button for categorized data
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Categorized Data",
                    data=csv,
                    file_name="categorized_crime_data.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Example and explanation
st.markdown("""
---
### How Categorisation Works

This module analyzes crime descriptions and other relevant fields to automatically categorize incidents.
It can help:

- Standardize crime categorization across districts
- Identify emerging crime patterns
- Generate consistent statistics for reporting
- Support resource allocation based on crime types

The module uses NLP techniques to extract key information from text descriptions and machine learning
to predict the most appropriate category.
""")
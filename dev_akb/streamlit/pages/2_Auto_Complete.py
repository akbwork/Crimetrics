import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

# Set page config first before any other Streamlit command
st.set_page_config(
    page_title="Auto Complete | Crimmetrics",
    page_icon="✏️",
    layout="wide"
)

# Add the auto_complete_module_1 directory to Python path
module_path = "/Users/ananthakrishnab/Desktop/Projects/Crimmetrics_Abi/Crimetrics/dev_akb/auto_complete_module_1"
if module_path not in sys.path:
    sys.path.append(module_path)

# Now try to import the module
try:
    from Missing_Data_Imputation import impute_relationship
    st.success("Successfully loaded Auto Complete module")
except ImportError as e:
    st.error(f"Error importing Missing_Data_Imputation module: {e}")
    
    # Define a mock function if import fails
    def impute_relationship(uploaded_file):
        """Mock function that returns sample data for demonstration purposes"""
        try:
            df = pd.read_csv(uploaded_file)
            st.info("Using placeholder imputation (module not properly loaded)")
            
            # Create mock report data
            report_dict = {
                'Self': {'precision': 0.92, 'recall': 0.89, 'f1-score': 0.90, 'support': 120},
                'Friend': {'precision': 0.85, 'recall': 0.82, 'f1-score': 0.83, 'support': 90},
                'Family': {'precision': 0.78, 'recall': 0.75, 'f1-score': 0.76, 'support': 80},
                'accuracy': 0.85,
                'macro avg': {'precision': 0.85, 'recall': 0.82, 'f1-score': 0.83, 'support': 290},
                'weighted avg': {'precision': 0.86, 'recall': 0.85, 'f1-score': 0.85, 'support': 290}
            }
            
            # Create mock confusion matrix
            cm = [[108, 8, 4], 
                  [12, 74, 4], 
                  [9, 11, 60]]
            
            class_labels = ['Self', 'Friend', 'Family']
            
            # Add mock "relationship" column if missing
            if 'relationship' not in df.columns:
                df['relationship'] = 'Self'
                df.loc[df.index % 3 == 1, 'relationship'] = 'Friend'
                df.loc[df.index % 3 == 2, 'relationship'] = 'Family'
                
            return df, report_dict, cm, class_labels, None
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return pd.DataFrame(), {}, [[0]], [''], None

st.title("Relationship Imputation from FIR Data")

uploaded_file = st.file_uploader("Upload your FIR dataset CSV", type=["csv"])

if uploaded_file is not None:
    with st.spinner("Running imputation, please wait..."):
        imputed_df, report_dict, cm, class_labels, output_path = impute_relationship(uploaded_file)

    if report_dict:
        st.success("Imputation complete and evaluated.")
        st.write("### Classification Report")
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='Blues'), height=300)

        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("⚠️ No missing 'relationship' values to impute or evaluation skipped.")

    st.write("### Preview of Imputed Data")
    st.dataframe(imputed_df.head(20))

    if output_path:
        with open(output_path, "rb") as f:
            st.download_button("Download Imputed CSV", f, file_name="imputed_relationship_output.csv")
    else:
        # Provide download option even without output_path
        csv = imputed_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Imputed CSV",
            csv,
            file_name="imputed_relationship_output.csv",
            mime="text/csv"
        )

# Add information about the module
st.markdown("""
---
### About Auto Complete: Relationship Imputation

This module uses machine learning to predict missing relationship values in FIR data. The system:

- Analyzes patterns in existing relationship data
- Uses features like age, gender, and crime type to make predictions
- Provides accuracy metrics to evaluate prediction quality
- Fills in missing values without altering existing data
""")

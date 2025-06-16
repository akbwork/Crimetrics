
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Crimetrics.dev_akb.auto_complete_module_1.Missing_Data_Imputation import impute_relationship

st.set_page_config(layout="wide")
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
        st.warning("No missing 'relationship' values to impute or evaluation skipped.")

    st.write("### Preview of Imputed Data")
    st.dataframe(imputed_df.head(20))

    if output_path:
        with open(output_path, "rb") as f:
            st.download_button("Download Imputed CSV", f, file_name="imputed_relationship_output.csv")

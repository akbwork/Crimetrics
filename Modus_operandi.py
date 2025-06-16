# modus_operandi_clustering.py

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

st.title("Modus Operandi Clustering for FIR Analysis")

# Load Data
uploaded_file = st.file_uploader("Upload FIR dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Ensure necessary fields exist
    required_columns = ['fir_narrative', 'district', 'date_of_crime', 'crime_category']
    if not all(col in df.columns for col in required_columns):
        st.error("Dataset must include 'fir_narrative', 'district', 'date_of_crime', and 'crime_category' columns.")
    else:
        # Drop rows with missing narratives
        df = df.dropna(subset=['fir_narrative'])

        st.header("Sample FIR Narratives")
        st.write(df['fir_narrative'].sample(3).values)

        # TF-IDF Vectorization
        st.header("Step 1: Convert Narratives to Numerical Features")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['fir_narrative'])

        st.write(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

        # KMeans Clustering
        st.header("Step 2: Grouping FIRs into Similar Crime Patterns")
        n_clusters = st.slider("Select number of pattern groups (clusters):", 2, 15, 5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(tfidf_matrix)

        df['Cluster'] = labels

        # Dynamic Naming
        st.subheader("Assign Meaningful Names to Pattern Groups")
        cluster_name_map = {}
        for i in range(n_clusters):
            name = st.text_input(f"Name for Cluster {i}:", value=f"Pattern {i}")
            cluster_name_map[i] = name

        df['Cluster_Label'] = df['Cluster'].map(cluster_name_map)

        st.success("FIRs have been grouped by Modus Operandi patterns.")

        # Cluster summary
        st.subheader("FIR Count by Cluster")
        cluster_counts = df['Cluster_Label'].value_counts().sort_index()
        st.dataframe(cluster_counts.rename_axis("Pattern Group").reset_index(name="FIR Count"))

        # Examples from a cluster
        st.subheader("Examples from a Pattern Group")
        selected_cluster = st.selectbox("Select a pattern group:", sorted(df['Cluster_Label'].unique()))
        st.write(df[df['Cluster_Label'] == selected_cluster][['fir_narrative']].sample(min(5, len(df[df['Cluster_Label'] == selected_cluster]))))

        # Word clouds
        st.subheader("Key Terms in Each Group")
        for label in sorted(df['Cluster_Label'].unique()):
            cluster_text = " ".join(df[df['Cluster_Label'] == label]['fir_narrative'].values)
            wc = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
            st.markdown(f"**{label} - Keywords**")
            st.image(wc.to_array(), use_container_width=True)

        # PCA Plot
        st.header("Visualize Patterns with PCA")
        pca = PCA(n_components=2, random_state=42)
        components = pca.fit_transform(tfidf_matrix.toarray())

        df['PCA1'] = components[:, 0]
        df['PCA2'] = components[:, 1]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster_Label', palette='tab10')
        plt.title("FIR Narrative Clusters")
        st.pyplot(fig)

        # Additional Insights
        st.header("Further Insights")

        # 1. Cluster distribution by district
        st.subheader("1. Pattern Distribution by District")
        dist_cluster = df.groupby(['district', 'Cluster_Label']).size().unstack(fill_value=0)
        st.dataframe(dist_cluster)

        # 2. Cluster trend over time
        st.subheader("2. Pattern Trend Over Time")
        df['date_of_crime'] = pd.to_datetime(df['date_of_crime'], errors='coerce')
        df['year_month'] = df['date_of_crime'].dt.to_period('M')
        trend = df.groupby(['year_month', 'Cluster_Label']).size().unstack(fill_value=0)
        st.line_chart(trend)

        # 3. Cluster vs Crime Category
        st.subheader("3. Pattern Group vs Crime Category")
        cluster_category = df.groupby(['Cluster_Label', 'crime_category']).size().unstack(fill_value=0)
        st.dataframe(cluster_category)

        # Download CSV
        st.subheader("Download Clustered FIRs")
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, "clustered_firs.csv", "text/csv")

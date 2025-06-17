import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from wordcloud import WordCloud

st.set_page_config(
    page_title="Clustering | Crimmetrics",
    page_icon="📊",
    layout="wide"
)

st.title("Modus Operandi Clustering")
st.write("Identify patterns and clusters in crime data based on behavioral factors for intelligence-led policing.")

# File upload for behavioral clustering
uploaded_file = st.file_uploader("Upload FIR dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load the data
    try:
        df = pd.read_csv(uploaded_file)
        
        # Ensure necessary fields exist
        required_columns = ['fir_narrative', 'district', 'date_of_crime', 'crime_category']
        if not all(col in df.columns for col in required_columns):
            st.error("Dataset must include 'fir_narrative', 'district', 'date_of_crime', and 'crime_category' columns.")
        else:
            # Drop rows with missing narratives
            df = df.dropna(subset=['fir_narrative'])
            
            # Display basic dataset info
            st.subheader("Dataset Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of FIRs", len(df))
            with col2:
                st.metric("Number of Districts", df['district'].nunique())
            with col3:
                st.metric("Crime Categories", df['crime_category'].nunique())
            
            # Sample narratives
            with st.expander("Sample FIR Narratives"):
                st.write(df['fir_narrative'].sample(3).values)
            
            # Clustering parameters
            st.subheader("Clustering Parameters")
            col1, col2 = st.columns(2)
            
            with col1:
                # TF-IDF parameters
                max_features = st.slider("Max Features for Text Vectorization", 100, 5000, 1000)
                use_stopwords = st.checkbox("Remove Common Words (Stopwords)", value=True)
                
            with col2:
                # Clustering parameters
                n_clusters = st.slider("Number of Pattern Groups (Clusters)", 2, 15, 5)
                clustering_method = st.selectbox(
                    "Clustering Method", 
                    ["K-means", "Agglomerative", "DBSCAN"]
                )
            
            # Process the data and perform clustering
            if st.button("Run Clustering Analysis"):
                with st.spinner("Processing narratives and generating clusters..."):
                    # Step 1: TF-IDF Vectorization
                    st.info("Step 1: Converting narratives to numerical features...")
                    stopwords = 'english' if use_stopwords else None
                    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stopwords)
                    tfidf_matrix = vectorizer.fit_transform(df['fir_narrative'])
                    
                    st.success(f"Created feature matrix with {tfidf_matrix.shape[1]} features from {len(df)} FIR narratives")
                    
                    # Step 2: Perform clustering based on selected method
                    st.info(f"Step 2: Grouping FIRs using {clustering_method}...")
                    
                    if clustering_method == "K-means":
                        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        labels = model.fit_predict(tfidf_matrix)
                    elif clustering_method == "Agglomerative":
                        # Convert sparse matrix to dense for hierarchical clustering
                        tfidf_dense = tfidf_matrix.toarray()
                        model = AgglomerativeClustering(n_clusters=n_clusters)
                        labels = model.fit_predict(tfidf_dense)
                    elif clustering_method == "DBSCAN":
                        # DBSCAN doesn't require n_clusters
                        model = DBSCAN(eps=0.5, min_samples=5)
                        labels = model.fit_predict(tfidf_matrix)
                        # Replace -1 (noise) with the next cluster number for visualization
                        if -1 in labels:
                            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                            labels[labels == -1] = n_clusters
                            n_clusters += 1
                    
                    df['Cluster'] = labels
                    
                    # Step 3: Dynamic naming of clusters
                    st.subheader("Assign Meaningful Names to Pattern Groups")
                    
                    # Initialize cluster_name_map
                    cluster_name_map = {}
                    
                    # For each cluster, find top words to suggest a name
                    for i in range(max(labels) + 1):
                        cluster_docs = df[df['Cluster'] == i]['fir_narrative'].values
                        if len(cluster_docs) > 0:
                            cluster_text = " ".join(cluster_docs)
                            
                            # Get top terms for this cluster
                            vectorizer_cluster = TfidfVectorizer(max_features=5, stop_words='english')
                            vectorizer_cluster.fit_transform([cluster_text])
                            top_terms = vectorizer_cluster.get_feature_names_out()
                            suggested_name = f"Pattern {i}: {', '.join(top_terms)}"
                            
                            # Let user modify the name
                            name = st.text_input(f"Name for Cluster {i}:", value=suggested_name)
                            cluster_name_map[i] = name
                    
                    df['Cluster_Label'] = df['Cluster'].map(cluster_name_map)
                    
                    st.success("FIRs have been grouped by Modus Operandi patterns.")
                    
                    # Display results
                    st.header("Clustering Results")
                    
                    # Cluster summary
                    st.subheader("FIR Count by Pattern Group")
                    cluster_counts = df['Cluster_Label'].value_counts().sort_index()
                    st.dataframe(cluster_counts.rename_axis("Pattern Group").reset_index(name="FIR Count"))
                    
                    # Examples from a cluster
                    st.subheader("Examples from a Pattern Group")
                    selected_cluster = st.selectbox("Select a pattern group:", sorted(df['Cluster_Label'].unique()))
                    st.write(df[df['Cluster_Label'] == selected_cluster][['fir_narrative']].sample(min(5, len(df[df['Cluster_Label'] == selected_cluster]))))
                    
                    # Word clouds
                    st.subheader("Key Terms in Each Group")
                    tab_labels = sorted(df['Cluster_Label'].unique())
                    word_cloud_tabs = st.tabs([f"Group {i+1}" for i in range(len(tab_labels))])
                    
                    for i, (tab, label) in enumerate(zip(word_cloud_tabs, tab_labels)):
                        with tab:
                            cluster_text = " ".join(df[df['Cluster_Label'] == label]['fir_narrative'].values)
                            try:
                                wc = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
                                st.markdown(f"**{label} - Keywords**")
                                st.image(wc.to_array(), use_container_width=True)
                            except Exception as e:
                                st.error(f"Could not generate word cloud: {str(e)}")
                    
                    # PCA Plot for visualization
                    st.header("Visualize Patterns with PCA")
                    try:
                        # Convert sparse matrix to dense for PCA
                        tfidf_dense = tfidf_matrix.toarray()
                        pca = PCA(n_components=2, random_state=42)
                        components = pca.fit_transform(tfidf_dense)
                        
                        df['PCA1'] = components[:, 0]
                        df['PCA2'] = components[:, 1]
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster_Label', palette='tab10')
                        plt.title("FIR Narrative Clusters")
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error generating PCA plot: {str(e)}")
                    
                    # Additional Insights
                    st.header("Additional Insights")
                    
                    insight_tabs = st.tabs(["District Distribution", "Time Trends", "Crime Categories"])
                    
                    with insight_tabs[0]:
                        # 1. Cluster distribution by district
                        st.subheader("Pattern Distribution by District")
                        try:
                            dist_cluster = df.groupby(['district', 'Cluster_Label']).size().unstack(fill_value=0)
                            st.dataframe(dist_cluster)
                            
                            # Bar chart of top districts for each cluster
                            fig, ax = plt.subplots(figsize=(12, 8))
                            dist_cluster.plot(kind='bar', stacked=True, ax=ax)
                            plt.title("Pattern Distribution Across Districts")
                            plt.xlabel("District")
                            plt.ylabel("Number of FIRs")
                            plt.xticks(rotation=45)
                            plt.legend(title="Pattern Group", bbox_to_anchor=(1.05, 1), loc='upper left')
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error generating district distribution: {str(e)}")
                    
                    with insight_tabs[1]:
                        # 2. Cluster trend over time
                        st.subheader("Pattern Trend Over Time")
                        try:
                            df['date_of_crime'] = pd.to_datetime(df['date_of_crime'], errors='coerce')
                            df['year_month'] = df['date_of_crime'].dt.to_period('M')
                            trend = df.groupby(['year_month', 'Cluster_Label']).size().unstack(fill_value=0)
                            
                            # Convert Period index to datetime for plotting
                            trend.index = trend.index.to_timestamp()
                            
                            st.line_chart(trend)
                        except Exception as e:
                            st.error(f"Error generating time trends: {str(e)}")
                    
                    with insight_tabs[2]:
                        # 3. Cluster vs Crime Category
                        st.subheader("Pattern Group vs Crime Category")
                        try:
                            cluster_category = df.groupby(['Cluster_Label', 'crime_category']).size().unstack(fill_value=0)
                            st.dataframe(cluster_category)
                            
                            # Heatmap of cluster vs crime category
                            fig, ax = plt.subplots(figsize=(12, 8))
                            sns.heatmap(cluster_category, annot=True, cmap="YlGnBu", fmt="d", ax=ax)
                            plt.title("Distribution of Crime Categories Within Pattern Groups")
                            plt.xlabel("Crime Category")
                            plt.ylabel("Pattern Group")
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error generating crime category analysis: {str(e)}")
                    
                    # Download CSV
                    st.subheader("Download Clustered FIRs")
                    csv = df.to_csv(index=False)
                    st.download_button("Download CSV", csv, "clustered_firs.csv", "text/csv")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a CSV file containing FIR data to begin analysis.")

# # Information about the module
# st.markdown("""
# ---
# ### About Clustering Analysis

# This module applies advanced clustering algorithms to identify patterns in crime data, which can:

# - Reveal previously unknown connections between criminal activities
# - Support predictive policing initiatives
# - Optimize resource allocation and patrol strategies
# - Identify serial offenders through behavioral patterns

# Both spatial and behavioral clustering approaches can provide valuable intelligence for law enforcement.

# #### Modus Operandi Clustering

# The Modus Operandi Clustering feature uses natural language processing to analyze FIR narratives and group similar crime patterns together. This can help in:

# - Identifying crime series committed by the same perpetrator(s)
# - Discovering common criminal techniques and strategies
# - Linking seemingly unrelated cases based on subtle behavioral patterns
# - Supporting crime linkage analysis for investigative purposes
# """)
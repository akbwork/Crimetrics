
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix
import os

def impute_relationship(uploaded_file_path):
    df = pd.read_csv(uploaded_file_path)

    # Drop rows with no FIR narrative or no accused data (for minimal usable sample)
    df = df.dropna(subset=['fir_narrative', 'accused_gender', 'accused_age'])

    # Backup for output
    df_output = df.copy()

    # Columns for modeling
    structured_features = ['crime_location_type', 'accused_gender', 'accused_age', 'district']
    target_column = 'relationship'

    # Fit TF-IDF only on FIR narrative (non-null)
    tfidf = TfidfVectorizer(max_features=100)
    tfidf_matrix = tfidf.fit_transform(df['fir_narrative'].fillna(""))

    # One-hot encode categorical + numeric features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    df_encoded = encoder.fit_transform(df[structured_features])

    # Combine all features
    X = hstack([df_encoded, tfidf_matrix])
    y = df[target_column]

    # Drop rows where y is null (for training)
    mask_notnull = y.notna()
    X_train = X[mask_notnull]
    y_train = y[mask_notnull]

    # Fit model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Rows where relationship is missing
    mask_null = y.isna()
    if mask_null.sum() == 0:
        return df_output, {}, [], [], None  # No imputation to do

    # Prepare features for missing rows
    tfidf_missing = tfidf.transform(df.loc[mask_null, 'fir_narrative'].fillna(""))
    encoded_missing = encoder.transform(df.loc[mask_null, structured_features])
    X_missing = hstack([encoded_missing, tfidf_missing])

    # Predict
    predicted = model.predict(X_missing)

    # Save predictions
    df_output.loc[mask_null, target_column] = predicted

    # Evaluate using true labels (from imputed_relationship_fir.csv)
    # Assumes you created and uploaded a parallel file with ground truth
    try:
        original_df = pd.read_csv('imputed_relationship_fir.csv')
        eval_mask = df_output.index[mask_null]
        actual = original_df.loc[eval_mask, target_column]
        predicted_series = pd.Series(predicted, index=actual.index)

        actual = actual.dropna()
        predicted_series = predicted_series.loc[actual.index]

        report_dict = classification_report(actual, predicted_series, output_dict=True, zero_division=0)
        cm = confusion_matrix(actual, predicted_series, labels=model.classes_)
        labels = model.classes_.tolist()
    except Exception as e:
        report_dict = {}
        cm = []
        labels = []

    # Save final file
    output_path = 'imputed_relationship_output.csv'
    df_output.to_csv(output_path, index=False)

    return df_output, report_dict, cm, labels, output_path

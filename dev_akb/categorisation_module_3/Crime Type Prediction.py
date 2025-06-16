import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import joblib

# Load dataset
df = pd.read_csv("crime_type_prediction_dataset_rajasthan.csv")
df.dropna(subset=["fir_narrative", "crime_category"], inplace=True)

# Encode categorical fields
categorical_cols = ['district', 'crime_location_type', 'season',
                    'reported_by', 'victim_gender', 'accused_gender', 'weapon_used',
                    'vehicle_involved', 'relationship', 'day_of_week']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    if col == 'day_of_week':
        correct_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        le.classes_ = np.array(correct_order)
        df[col] = le.transform(df[col])
    else:
        df[col] = le.fit_transform(df[col])
    encoders[col] = le

# DateTime processing
df['hour_of_crime'] = pd.to_datetime(df['time_of_crime'], errors='coerce').dt.hour.fillna(0).astype(int)

# FIR Text Embedding - WITH TOKENIZER FIX
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# Fix the pad token issue
if not hasattr(text_model.tokenizer, '_pad_token'):
    # First check if this is a BertTokenizerFast object
    if hasattr(text_model.tokenizer, 'pad_token'):
        # Set the pad token properly
        text_model.tokenizer.pad_token = '[PAD]'
        # Also set the internal attributes
        text_model.tokenizer._pad_token = '[PAD]'
    else:
        # If it's not a standard tokenizer, use eos_token as pad_token
        text_model.tokenizer.pad_token = text_model.tokenizer.eos_token
        text_model.tokenizer._pad_token = text_model.tokenizer.eos_token
        
    # Make sure these are also set
    if hasattr(text_model.tokenizer, 'pad_token_id'):
        if text_model.tokenizer.pad_token_id is None:
            text_model.tokenizer.pad_token_id = text_model.tokenizer.eos_token_id
    
    # Set padding side if it's not set
    if hasattr(text_model.tokenizer, 'padding_side'):
        text_model.tokenizer.padding_side = 'right'

# Now encode the text after fixing the tokenizer
X_text = text_model.encode(df['fir_narrative'].tolist(), show_progress_bar=True)

# Structured Features
structured_cols = ['district', 'crime_location_type', 'victim_age',
                   'accused_age', 'season', 'reported_by', 'victim_gender', 
                   'accused_gender', 'weapon_used', 'vehicle_involved', 
                   'num_accused', 'relationship', 'day_of_week', 'hour_of_crime']
X_struct = df[structured_cols].values

# 💥 Weight FIR narrative 3x more
X_text_weighted = np.hstack([X_text] * 3)

# Combine structured and weighted text features
X_final = np.hstack([X_struct, X_text_weighted])

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['crime_category'])

# Model training
model = XGBClassifier(eval_metric='mlogloss')
print("5-Fold CV Accuracy:", cross_val_score(model, X_final, y, cv=5).mean())
model.fit(X_final, y)

# Save model with native XGBoost format (to avoid version warnings)
model.save_model("crime_type_classifier_weighted.json")

# Also save the model in pickle format for backward compatibility
joblib.dump(model, "crime_type_classifier_weighted.pkl")

# For the tokenizer fix, we'll create a wrapper class to ensure the pad token is set
class FixedSentenceTransformer(SentenceTransformer):
    """Wrapper for SentenceTransformer that ensures pad_token is set properly."""
    def __init__(self, model_name_or_path):
        super().__init__(model_name_or_path)
        # Fix the tokenizer
        if not hasattr(self.tokenizer, '_pad_token'):
            if hasattr(self.tokenizer, 'pad_token'):
                self.tokenizer.pad_token = '[PAD]'
                self.tokenizer._pad_token = '[PAD]'
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer._pad_token = self.tokenizer.eos_token
            
            if hasattr(self.tokenizer, 'pad_token_id'):
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            if hasattr(self.tokenizer, 'padding_side'):
                self.tokenizer.padding_side = 'right'

# Save the fixed model
fixed_model = FixedSentenceTransformer('all-MiniLM-L6-v2')
fixed_model.save('fixed_sentence_transformer')

# Save other resources
joblib.dump(text_model, "sentence_transformer_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(encoders, "structured_encoders.pkl")

print("Models and encoders saved successfully!")

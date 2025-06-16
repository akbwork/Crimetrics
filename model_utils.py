import spacy
from sentence_transformers import SentenceTransformer, util
import re

# Load NLP models
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Small & fast

def extract_entities(text):
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

def compute_completeness(entities):
    score = 0
    for label in ['PERSON', 'DATE', 'GPE', 'ORG', 'TIME']:
        if label in entities:
            score += 0.8  # Up to 4 points
    return min(score, 4.0)

def compute_consistency(entities, row):
    score = 0
    if 'DATE' in entities and str(row['date_of_crime']).split("-")[0] in entities['DATE']:
        score += 1
    if 'PERSON' in entities and row['victim_gender'].lower() in entities['PERSON'].lower():
        score += 1
    if 'GPE' in entities and row['district'].lower() in entities['GPE'].lower():
        score += 1
    return score

def compute_relevance(text, ipc_section):
    vec1 = embedder.encode(text, convert_to_tensor=True)
    vec2 = embedder.encode(ipc_section, convert_to_tensor=True)
    similarity = float(util.pytorch_cos_sim(vec1, vec2))
    return min(similarity * 3, 3.0)  # Normalize to 0–3

def assess_quality(row):
    text = row['fir_narrative']
    ipc = row['ipc_section']
    entities = extract_entities(text)

    completeness = compute_completeness(entities)
    consistency = compute_consistency(entities, row)
    relevance = compute_relevance(text, ipc)

    total = round(completeness + consistency + relevance, 2)
    return total, completeness, consistency, relevance

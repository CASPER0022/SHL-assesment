import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load catalog
with open('shl_catalogue.json', 'r', encoding='utf-8') as f:
    catalog = json.load(f)

# Prepare texts
texts = []
for item in catalog:
    name = item.get('name', '')
    desc = item.get('description', '')
    text = f"{name} {desc}"
    texts.append(text)

# Create TF-IDF index
print("Creating TF-IDF index...")
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)

# Save index
data_to_save = {
    'vectorizer': vectorizer,
    'tfidf_matrix': tfidf_matrix,
    'catalog': catalog
}

with open('catalog_index_tfidf.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("TF-IDF Indexing complete. Saved to catalog_index_tfidf.pkl")

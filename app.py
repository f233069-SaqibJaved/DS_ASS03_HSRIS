import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import re
import math
from collections import Counter
import pickle

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(page_title="HSRIS Support Search", page_icon="🔍", layout="wide")
st.title("🛠️ Hybrid Semantic Retrieval & Intelligence System")
st.markdown("Search past customer support tickets using a blend of Keyword (TF-IDF) and Semantic (GloVe) matching.")

# ==========================================
# 2. DATA & MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Loading data and building search indices... (This may take a minute)")
def load_data_and_models():
    # 1. Load Data
    # Make sure this matches the exact name of your downloaded CSV file!
    try:
        df = pd.read_csv('customer_support_tickets.csv') 
    except FileNotFoundError:
        st.error("⚠️ 'customer_support_tickets.csv' not found. Please ensure it is in the same folder as app.py.")
        st.stop()
        
    # Drop rows with missing descriptions to prevent errors
    df = df.dropna(subset=['Ticket Description']).reset_index(drop=True)

    # 2. Text Processing Helpers
    def tokenize(text): return re.findall(r'\b\w+\b', str(text).lower())
    def generate_ngrams(tokens, n): return ['_'.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    def process_text(text):
        tokens = tokenize(text)
        return tokens + generate_ngrams(tokens, 2) + generate_ngrams(tokens, 3)

    # 3. Build TF-IDF
    all_terms = []
    for desc in df['Ticket Description']:
        all_terms.extend(process_text(desc))
        
    term_counts = Counter(all_terms)
    vocab = {term: i for i, (term, _) in enumerate(term_counts.most_common(5000))}
    
    N = len(df)
    df_counts = Counter()
    for desc in df['Ticket Description']:
        terms = set(process_text(desc))
        for term in terms:
            if term in vocab:
                df_counts[term] += 1
                
    idf = {term: math.log(N / (1 + df_counts[term])) for term in vocab}
    
    indices, values = [], []
    for doc_idx, desc in enumerate(df['Ticket Description']):
        terms = process_text(desc)
        term_freqs = Counter([t for t in terms if t in vocab])
        total_terms = len(terms) if len(terms) > 0 else 1
        for term, count in term_freqs.items():
            indices.append([doc_idx, vocab[term]])
            values.append((count / total_terms) * idf[term])
            
    if not indices:
        tfidf_matrix = torch.zeros((N, len(vocab)))
    else:
        tfidf_matrix = torch.sparse_coo_tensor(torch.tensor(indices).t(), torch.tensor(values, dtype=torch.float32), (N, len(vocab))).to_dense()
    
    tfidf_matrix = F.normalize(tfidf_matrix, p=2, dim=1)

    # 4. Load GloVe
    word2idx = {'<UNK>': 0}
    embeddings = [np.zeros(300)]
    
    try:
        with open('glove.pkl', 'rb') as f:
            glove_dict = pickle.load(f)
            for word, vector in glove_dict.items():
                vec = np.asarray(vector, dtype='float32')
                if vec.shape == (300,):
                    word2idx[word] = len(embeddings)
                    embeddings.append(vec)
    except FileNotFoundError:
        st.error("⚠️ 'glove.pkl' not found. Please ensure it is in the same folder as app.py.")
        st.stop()
                
    embedding_layer = torch.nn.Embedding.from_pretrained(torch.FloatTensor(np.array(embeddings)))
    
    def get_sentence_embedding(text):
        tokens = tokenize(text)
        vecs, weights = [], []
        for token in tokens:
            idx = word2idx.get(token, 0)
            vecs.append(embedding_layer(torch.tensor(idx)))
            weights.append(idf.get(token, 1.0))
        if not vecs: return torch.zeros(300)
        vecs = torch.stack(vecs)
        weights = torch.tensor(weights).unsqueeze(1)
        return torch.sum(vecs * weights, dim=0) / torch.sum(weights)

    # Build GloVe database matrix
    glove_matrix = torch.stack([get_sentence_embedding(desc) for desc in df['Ticket Description']])
    glove_matrix = F.normalize(glove_matrix, p=2, dim=1)
    
    return df, vocab, idf, word2idx, embedding_layer, tfidf_matrix, glove_matrix, process_text, get_sentence_embedding

# Initialize
df, vocab, idf, word2idx, embedding_layer, tfidf_db, glove_db, process_text, get_sentence_embedding = load_data_and_models()

# ==========================================
# 3. UI LAYOUT & SEARCH LOGIC
# ==========================================
st.sidebar.header("Search Settings")
alpha = st.sidebar.slider(
    "Keyword vs Semantic (Alpha)", 
    min_value=0.0, max_value=1.0, value=0.4, step=0.1, 
    help="0.0 = GloVe Semantic Only | 1.0 = TF-IDF Keyword Only"
)

query = st.text_area("Enter a new ticket description here:", "My account is locked and I cannot reset my password.")

if st.button("Find Similar Tickets"):
    with st.spinner("Searching database..."):
        # 1. Process Query for TF-IDF
        terms = process_text(query)
        term_freqs = Counter([t for t in terms if t in vocab])
        total_terms = len(terms) if len(terms) > 0 else 1
        
        q_indices, q_values = [], []
        for term, count in term_freqs.items():
            q_indices.append([0, vocab[term]])
            q_values.append((count / total_terms) * idf[term])
            
        if not q_indices: 
            q_tfidf = torch.zeros((1, len(vocab)))
        else: 
            q_tfidf = torch.sparse_coo_tensor(torch.tensor(q_indices).t(), torch.tensor(q_values, dtype=torch.float32), (1, len(vocab))).to_dense()
        
        q_tfidf_norm = F.normalize(q_tfidf, p=2, dim=1)
        
        # 2. Process Query for GloVe
        q_glove_norm = F.normalize(get_sentence_embedding(query).unsqueeze(0), p=2, dim=1)
        
        # 3. Compute Similarities (Matrix Multiplication)
        tfidf_sim = torch.mm(q_tfidf_norm, tfidf_db.t())
        glove_sim = torch.mm(q_glove_norm, glove_db.t())
        
        # 4. Hybrid Score
        final_score = (alpha * tfidf_sim) + ((1 - alpha) * glove_sim)
        
        # 5. Get Top 3 Results
        top_k = 3
        top3_indices = torch.topk(final_score, top_k, dim=-1).indices[0].numpy()
        top3_scores = torch.topk(final_score, top_k, dim=-1).values[0].numpy()
        
        # 6. Predict Ticket Type (Majority Vote of Top 3)
        retrieved_types = df.iloc[top3_indices]['Ticket Type'].tolist()
        predicted_type = Counter(retrieved_types).most_common(1)[0][0]
        
        # 7. Display Results
        st.success(f"**Predicted Ticket Type:** {predicted_type}")
        st.markdown("### Top 3 Past Resolutions")
        
        for i, idx in enumerate(top3_indices):
            ticket = df.iloc[idx]
            score = top3_scores[i] * 100
            
            with st.expander(f"Match #{i+1} - {ticket['Ticket Type']} (Confidence: {score:.1f}%)", expanded=True):
                st.markdown(f"**Description:** {ticket['Ticket Description']}")
                # Adjust 'Resolution' to match whatever your actual column name is if needed
                resolution = ticket.get('Resolution', 'No resolution logged.')
                st.markdown(f"**Resolution/Response:** {resolution}")

import streamlit as st
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import spacy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import nltk

# Initialize NLP tools
nltk.download('punkt')
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("Please run: python -m spacy download en_core_web_sm")

# --- CONFIGURATION ---
# Using a model that aligns with Paper 5's finding that RoBERTa is a strong baseline
MODEL_NAME = "Hello-SimpleAI/chatgpt-detector-roberta" 

st.set_page_config(page_title="Research Prototype: Multi-Grain AI Detection", layout="wide")

# --- LOAD MODELS ---
@st.cache_resource
def load_deep_learning_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_deep_learning_model()

# --- SCIENTIFIC HELPER FUNCTIONS ---

def get_sentence_scores(text):
    """
    Implements the logic from Paper 1 (Lekkala et al.) & Paper 2 (Su et al.):
    Fine-grained, sentence-level segmentation to detect 'Hybrid' text.
    """
    sentences = nltk.sent_tokenize(text)
    sentence_scores = []
    
    for sent in sentences:
        if len(sent.split()) < 3: continue # Skip fragments
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).tolist()[0]
        # Assuming Index 1 is AI (Check model card)
        sentence_scores.append({"sentence": sent, "ai_prob": probs[1]})
        
    return sentence_scores

def extract_stylometric_features(text):
    """
    Implements the 'Lightweight Approach' from Paper 4 (Aityan et al.).
    Extracts explicit linguistic features rather than black-box embeddings.
    """
    doc = nlp(text)
    
    # 1. Syntactic Complexity (Mean Sentence Length)
    # Paper 3 notes this is crucial for detecting creative fiction
    sent_lens = [len(sent) for sent in doc.sents]
    avg_sent_len = np.mean(sent_lens) if sent_lens else 0
    
    # 2. Burstiness / Variance (Paper 4)
    # AI models tend to have low variance in sentence length
    len_variance = np.std(sent_lens) if sent_lens else 0
    
    # 3. Part-of-Speech Ratios (Paper 4)
    # AI often overuses determiners/nouns, humans use more diverse verbs
    pos_counts = doc.count_by(spacy.attrs.POS)
    total_words = len(doc)
    
    # Calculate ratios
    noun_ratio = pos_counts.get(92, 0) / total_words # 92 is NOUN
    verb_ratio = pos_counts.get(100, 0) / total_words # 100 is VERB
    adj_ratio = pos_counts.get(84, 0) / total_words  # 84 is ADJ
    
    return {
        "Avg Sentence Length": avg_sent_len,
        "Burstiness (Std Dev)": len_variance,
        "Noun Ratio": noun_ratio,
        "Verb Ratio": verb_ratio,
        "Adjective Ratio": adj_ratio
    }

# --- UI LAYOUT ---

st.title("Hybrid Fine-Grained AI Detection System")
st.markdown("""
**Research Objective:** Distinguish Human vs. Co-Authored vs. AI Text.
**References:** Implements concepts from *Lekkala et al. (Sentence Segmentation)* and *Aityan et al. (Stylometric Features)*.
""")

text_input = st.text_area("Input Text (Paste a mix of Human and AI text to test segmentation):", height=200)

if st.button("Run Hybrid Analysis"):
    if not text_input:
        st.warning("Please enter text.")
    else:
        with st.spinner("Performing Sentence-Level Segmentation & Stylometric Extraction..."):
            
            # 1. Document Level (Baseline)
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = model(**inputs).logits
            doc_prob = torch.softmax(logits, dim=1).tolist()[0][1]
            
            # 2. Fine-Grained Level (Paper 1 & 2)
            sent_data = get_sentence_scores(text_input)
            
            # 3. Stylometric Level (Paper 3 & 4)
            style_feats = extract_stylometric_features(text_input)
            
            # --- DASHBOARD ---
            
            # Global Verdict
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Global AI Probability", f"{doc_prob*100:.1f}%")
            with col2:
                # Logic from Paper 5: If variance of scores is high, it might be Hybrid
                score_std = np.std([x['ai_prob'] for x in sent_data])
                classification = "Uncertain"
                if doc_prob > 0.8: classification = "Mostly AI"
                elif doc_prob < 0.2: classification = "Mostly Human"
                elif score_std > 0.25: classification = "Hybrid / Co-Authored" # High variance = mixed sources
                
                st.metric("Classification Type", classification)
                st.caption("*Based on distribution of sentence scores*")
            with col3:
                st.metric("Burstiness Score", f"{style_feats['Burstiness (Std Dev)']:.2f}")
                st.caption("*Higher = More likely Human (Paper 4)*")

            st.divider()
            
            # VISUALIZATION 1: Fine-Grained Heatmap (Paper 1)
            st.subheader("1. Sentence-Level Segmentation (Heatmap)")
            st.markdown("Visualizing the transition points between Human and AI text.")
            
            # Construct colored HTML string
            html_string = ""
            for item in sent_data:
                score = item['ai_prob']
                # Red = AI, Green = Human. Opacity depends on confidence.
                if score > 0.5:
                    color = f"rgba(255, 0, 0, {score})"
                else:
                    color = f"rgba(0, 255, 0, {1-score})"
                
                html_string += f'<span style="background-color: {color}; padding: 3px; border-radius: 3px; margin-right: 2px;" title="AI Score: {score:.2f}">{item["sentence"]}</span> '
            
            st.markdown(f'<div style="line-height: 1.8; border:1px solid #ccc; padding:10px; border-radius:5px;">{html_string}</div>', unsafe_allow_html=True)
            st.caption("🔴 Red = AI Generated | 🟢 Green = Human Written")

            # VISUALIZATION 2: Stylometric Radar (Paper 4)
            st.subheader("2. Stylometric Fingerprint")
            st.markdown("Analyzing linguistic features described by *Aityan et al.*")
            
            # Normalize for radar chart (Approximate baselines)
            categories = list(style_feats.keys())
            values = list(style_feats.values())
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Input Text'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, max(values) + 5])
                ),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # VISUALIZATION 3: Distribution (Paper 5)
            st.subheader("3. Probability Distribution (Source Consistency)")
            scores = [x['ai_prob'] for x in sent_data]
            fig_hist = px.histogram(x=scores, nbins=10, labels={'x': 'AI Probability', 'y': 'Sentence Count'}, 
                                    title="Is the text consistent? (Paper 5: DETree Concept)")
            st.plotly_chart(fig_hist, use_container_width=True)
            if np.std(scores) > 0.2:
                st.info("ℹ️ **Observation:** The wide spread in this histogram suggests **Co-Authoring/Polishing** (Mix of High AI and Low AI sentences).")

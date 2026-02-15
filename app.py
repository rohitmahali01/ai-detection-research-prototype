import streamlit as st
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import spacy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import nltk
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_NAME = "Hello-SimpleAI/chatgpt-detector-roberta"

st.set_page_config(
    page_title="Research Prototype: Multi-Grain AI Detection",
    layout="wide"
)

# -----------------------------
# SAFE NLTK SETUP (Cloud Compatible)
# -----------------------------
@st.cache_resource
def setup_nltk():
    nltk.data.path.append("/tmp")
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", download_dir="/tmp", quiet=True)

setup_nltk()

# -----------------------------
# LOAD SPACY (Pre-installed via requirements.txt)
# -----------------------------
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# -----------------------------
# LOAD HUGGINGFACE MODEL (Stable)
# -----------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )
    model.eval()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# SENTENCE-LEVEL SEGMENTATION (Batched)
# -----------------------------
def get_sentence_scores(text):
    sentences = nltk.sent_tokenize(text)
    sentence_scores = []

    if not sentences:
        return sentence_scores

    # Batch inference for speed
    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)[:, 1].tolist()

    for sent, prob in zip(sentences, probs):
        if len(sent.split()) >= 3:
            sentence_scores.append({
                "sentence": sent,
                "ai_prob": prob
            })

    return sentence_scores

# -----------------------------
# STYLOMETRIC FEATURE EXTRACTION
# -----------------------------
def extract_stylometric_features(text):
    doc = nlp(text)

    sent_lens = [len(sent) for sent in doc.sents]
    avg_sent_len = np.mean(sent_lens) if sent_lens else 0
    len_variance = np.std(sent_lens) if sent_lens else 0

    pos_counts = doc.count_by(spacy.attrs.POS)
    total_words = max(len(doc), 1)

    noun_ratio = pos_counts.get(92, 0) / total_words
    verb_ratio = pos_counts.get(100, 0) / total_words
    adj_ratio = pos_counts.get(84, 0) / total_words

    return {
        "Avg Sentence Length": avg_sent_len,
        "Burstiness (Std Dev)": len_variance,
        "Noun Ratio": noun_ratio,
        "Verb Ratio": verb_ratio,
        "Adjective Ratio": adj_ratio
    }

# -----------------------------
# UI
# -----------------------------
st.title("Hybrid Fine-Grained AI Detection System")

st.markdown("""
**Research Objective:** Distinguish Human vs. Co-Authored vs. AI Text.  
Implements sentence-level segmentation and stylometric variance analysis.
""")

text_input = st.text_area(
    "Input Text (Paste Human, AI, or Mixed Text):",
    height=200
)

# -----------------------------
# ANALYSIS BUTTON
# -----------------------------
if st.button("Run Hybrid Analysis"):

    if not text_input.strip():
        st.warning("Please enter valid text.")
    else:
        with st.spinner("Running Deep + Stylometric Analysis..."):

            # -----------------------------
            # 1. Document-Level Inference
            # -----------------------------
            inputs = tokenizer(
                text_input,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            with torch.no_grad():
                logits = model(**inputs).logits

            doc_prob = torch.softmax(logits, dim=1)[0][1].item()

            # -----------------------------
            # 2. Sentence-Level
            # -----------------------------
            sent_data = get_sentence_scores(text_input)

            # -----------------------------
            # 3. Stylometric
            # -----------------------------
            style_feats = extract_stylometric_features(text_input)

            # -----------------------------
            # DASHBOARD METRICS
            # -----------------------------
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Global AI Probability", f"{doc_prob*100:.1f}%")

            with col2:
                if len(sent_data) > 0:
                    score_std = np.std([x["ai_prob"] for x in sent_data])
                else:
                    score_std = 0

                classification = "Uncertain"

                if doc_prob > 0.8:
                    classification = "Mostly AI"
                elif doc_prob < 0.2:
                    classification = "Mostly Human"
                elif score_std > 0.25:
                    classification = "Hybrid / Co-Authored"

                st.metric("Classification Type", classification)
                st.caption("Based on sentence score distribution")

            with col3:
                st.metric(
                    "Burstiness Score",
                    f"{style_feats['Burstiness (Std Dev)']:.2f}"
                )
                st.caption("Higher = More likely Human")

            st.divider()

            # -----------------------------
            # HEATMAP VISUALIZATION
            # -----------------------------
            st.subheader("1. Sentence-Level Segmentation (Heatmap)")

            html_string = ""

            for item in sent_data:
                score = item["ai_prob"]

                if score > 0.5:
                    color = f"rgba(255, 0, 0, {score})"
                else:
                    color = f"rgba(0, 255, 0, {1-score})"

                safe_sentence = item["sentence"].replace("<", "&lt;").replace(">", "&gt;")

                html_string += f"""
                <span style="background-color:{color};
                padding:4px;
                border-radius:4px;
                margin-right:3px;"
                title="AI Score: {score:.2f}">
                {safe_sentence}
                </span>
                """

            st.markdown(
                f'<div style="line-height:1.8; border:1px solid #ccc; padding:10px; border-radius:5px;">{html_string}</div>',
                unsafe_allow_html=True
            )

            st.caption("Red = AI | Green = Human")

            # -----------------------------
            # RADAR CHART
            # -----------------------------
            st.subheader("2. Stylometric Fingerprint")

            categories = list(style_feats.keys())
            values = list(style_feats.values())

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself"
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(values) + 1]
                    )
                ),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # -----------------------------
            # HISTOGRAM
            # -----------------------------
            st.subheader("3. Probability Distribution")

            scores = [x["ai_prob"] for x in sent_data]

            if scores:
                fig_hist = px.histogram(
                    x=scores,
                    nbins=10,
                    labels={"x": "AI Probability", "y": "Sentence Count"},
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                if np.std(scores) > 0.2:
                    st.info(
                        "Wide distribution suggests possible Hybrid / Co-Authoring."
                    )
            else:
                st.info("Not enough sentences for distribution analysis.")

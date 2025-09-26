# streamlit_demo.py

import streamlit as st
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import re
from bert_score import score
import textstat

# 1️⃣ Correct device assignment
if torch.cuda.is_available():
    device = 0  # GPU 0
else:
    device = -1  # CPU fallback

# 2️⃣ Load model & tokenizer (cached for fast reloads)
@st.cache_resource
def load_model(model_name="sshleifer/distilbart-cnn-12-6"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Correct device handling
    if torch.cuda.is_available():
        device = 0  # GPU 0
        model = model.half().cuda()
    else:
        device = None  # CPU
        model = model.float()  # keep on CPU

    # Pass correct device to pipeline
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device if device is not None else None
    )

    return tokenizer, summarizer

tokenizer, summarizer = load_model()

st.title("💊 Medical Guideline Summarizer Demo")

# 3️⃣ PDF Upload
uploaded_file = st.file_uploader("Upload a medical guideline PDF", type="pdf")

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        # Extract 1–2 paragraphs from page 4 (0-indexed: 3)
        page_index = 3 if len(pdf.pages) > 3 else 0
        page_text = pdf.pages[page_index].extract_text()
        paragraphs = page_text.split("\n\n")
        cleaned_text = " ".join(paragraphs[:2])

    st.subheader("📄 Original Text (1–2 paragraphs)")
    st.write(cleaned_text)

    # 4️⃣ Token chunking (max 512 for safety)
    tokens = tokenizer.encode(cleaned_text, add_special_tokens=False)
    if len(tokens) > 512:
        tokens = tokens[:512]
    chunk_text = tokenizer.decode(tokens, skip_special_tokens=True)

    # 5️⃣ Summarization
    summary = summarizer(chunk_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']

    st.subheader("✏️ Summary")
    st.write(summary)

    # 7️⃣ Executive summary (optional second-pass)
    executive_summary = summarizer(summary, max_length=100, min_length=40, do_sample=False)[0]['summary_text']
    st.subheader("🎯 Executive Summary")
    st.write(executive_summary)

    # 8️⃣ Evaluation metrics
    st.subheader("📊 Evaluation Metrics")

    # BERTScore (requires reference summary)
    reference_summary = st.text_area("Optional: paste reference summary for BERTScore")
    if reference_summary.strip():
        P, R, F1 = score([summary], [reference_summary], lang="en")
        st.write(f"BERTScore F1: {F1.mean().item():.4f}")
    else:
        st.write("BERTScore skipped: no reference provided.")

    # Readability metrics
    st.write("Flesch Reading Ease:", textstat.flesch_reading_ease(summary))
    st.write("Gunning Fog Index:", textstat.gunning_fog(summary))

# Medical Guideline Summarizer

## Project Overview

The Medical Guideline Summarizer is an interactive NLP pipeline that extracts and summarizes content from medical PDFs. It leverages state-of-the-art transformer models (`distilbart-cnn-12-6`) and PDF processing tools (`pdfplumber`) to provide concise summaries. The project includes evaluation metrics for semantic similarity and readability.

## Features

- Extracts 1–2 paragraphs from PDF pages for summarization.
- Token-based chunking and Seq2Seq summarization optimized for CPU/GPU.
- Highlights medical terms (e.g., drug names, dosages) in the summary.
- Provides evaluation metrics:
  - **BERTScore** for semantic similarity.
  - **Flesch Reading Ease** and **Gunning Fog Index** for readability.
- Interactive Streamlit demo for PDF upload, summarization, and metric visualization.

## Tech Stack

- Python 3.10+
- Hugging Face Transformers
- PyTorch
- Streamlit
- pdfplumber
- BERTScore
- textstat

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-folder>
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run streamlit_demo.py
```

2. Upload a medical PDF in the app.
3. View the summarized text, bullet points with highlighted medical terms, executive summary, and evaluation metrics.

## Evaluation Metrics

- **BERTScore:** Measures semantic similarity between generated summary and reference summary.
- **Flesch Reading Ease:** Indicates text readability; lower scores are more complex.
- **Gunning Fog Index:** Estimates the years of education required to understand the summary.

## Project Outcome

- Demonstrates ability to handle NLP pipelines end-to-end.
- Optimized for low-memory GPUs and CPU usage.
- Useful for healthcare NLP applications and data science portfolios.

## Author

Ling Chin Ung
Email: jasonling23@yahoo.com
LinkedIn/GitHub: <your-link>

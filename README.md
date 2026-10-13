# GenAI Summarizer: Advanced NLP & Text Intelligence Assistant
### Generative AI with NLP Training Project
**License:** MIT (Free & Open Source)  
**API Cost:** $0.00 (No Paid API Keys Required)  

---

## 📌 Project Overview

**GenAI Summarizer: Advanced NLP & Text Intelligence Assistant** is a complete, beginner-friendly, and educational Generative AI and Natural Language Processing web application. Designed as a practical *Generative AI with NLP* project, this platform bridges deep learning Sequence-to-Sequence neural networks and classical statistical NLP algorithms.

The application allows users to input articles, essays, research excerpts, or news reports and generate concise summaries using Hugging Face's `google/flan-t5-small` model, extract salient keyphrases using Scikit-Learn's TF-IDF vectorizer, analyze quantitative document statistics, compare abstractive vs. extractive summarization side-by-side, and download UTF-8 reports.

---

## ✨ Main Features

1. **Text Ingestion & Validation:**
   - Large interactive text area with live character and word counters.
   - Pre-loaded domain sample buttons (*AI in Healthcare*, *Clean Energy*, *Quantum Security*).
   - Validation boundaries ensuring non-trivial input (minimum 15 words, maximum 8,000 characters).
   - Clear input and reset controls.

2. **Generative AI Summarization (Hugging Face Flan-T5):**
   - Employs `google/flan-t5-small` (~80M parameters, ~308MB footprint).
   - Runs locally on standard consumer CPU hardware.
   - Provides **Short**, **Medium**, and **Long** summary length presets.
   - Utilizes beam search decoding with repetition suppression.
   - Transparent disclaimer reminding users that AI-generated summaries should be reviewed.

3. **Documented Extractive NLP Fallback & Comparison Lab:**
   - Pure-Python / Scikit-Learn sentence salience ranker combining TF-IDF word relevance and positional bias.
   - Automatically triggered if neural weights cannot be initialized (e.g. offline first run or memory limits).
   - Clearly labeled with informative badges (`Extractive NLP Fallback Engine`).
   - Dedicated educational lab tab to observe abstractive paraphrasing vs. extractive sentence selection side-by-side.

4. **TF-IDF Keyword Extraction:**
   - Extracts top $K$ keyphrases and individual words with normalized relevance scores ($0.0$ to $1.0$).
   - English stopword suppression and 1-to-2 gram support.
   - Fallback frequency analyzer for short texts or edge cases.

5. **Quantitative Document Metrics:**
   - Live metrics: Original Words, Summary Words, Sentence Counts, Character Counts.
   - Compression calculation: Word reduction percentage ($\%$ reduction) safe against division-by-zero errors.

6. **Export & Download Options:**
   - Download Summary as `.txt`
   - Download Complete Analysis Report as `.txt` (with metadata, metrics, keywords, and text)
   - Download Side-by-Side Source + Summary as `.txt`

---

## 🛠️ Technology Stack

- **Python:** 3.10 or newer (Verified on Python 3.14)
- **Web Interface:** Streamlit (>= 1.30.0)
- **Generative AI:** Hugging Face Transformers (`transformers>=4.38.0`)
- **Neural Backend:** PyTorch (`torch>=2.1.0`, CPU mode)
- **Model:** `google/flan-t5-small` (Seq2Seq Transformer)
- **NLP & Keyword Extraction:** Scikit-Learn (`scikit-learn>=1.3.0`, `TfidfVectorizer`)
- **Testing:** Pytest (`pytest>=7.4.0`)
- **Data & Numerical:** NumPy, SciPy

---

## 📂 Project Folder Structure

```text
ai_text_summarizer_nlp/
│
├── app.py                      # Main Streamlit web application
├── summarizer.py               # Generative model manager & Extractive fallback engine
├── keyword_extractor.py        # TF-IDF & frequency-based keyword extraction
├── text_utils.py               # Sentence splitting, word counts, validation & export formatters
├── requirements.txt            # Python dependency manifest
├── README.md                   # Complete project documentation
├── .gitignore                  # Git exclusions (caches, virtual environments)
│
├── tests/                      # Automated pytest test suite
│   ├── test_text_utils.py          # Tests for counters, splitting, validation & export
│   ├── test_keyword_extractor.py   # Tests for TF-IDF, edge cases & fallback
│   └── test_summarizer.py          # Tests for presets, fallbacks, mocks & status
│
└── docs/                       # Comprehensive documentation
    ├── PROJECT_REPORT.md           # 21-section academic project report with placeholders
    └── TESTING_GUIDE.md            # Step-by-step testing and verification manual
```

---

## 🚀 Quick Setup & Installation Guide (Windows)

### 1. Open PowerShell / Command Prompt
Navigate to the directory where you want to run the project:
```powershell
cd "d:\generative ai with npl\ai_text_summarizer_nlp"
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Upgrade Pip & Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧪 Running Automated Tests

To run the automated test suite without downloading models:
```powershell
python -m pytest tests -v
```

All 21 unit and mock tests will execute in approximately 2 to 3 seconds.

---

## 🌐 Launching the Application

Start the Streamlit development server:
```powershell
python -m streamlit run app.py
```

Streamlit will print the local URL:
```text
  Local URL: http://localhost:8501
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📥 Model Download Requirements & Offline Execution

- **First Run:** When the application runs inference with `google/flan-t5-small` for the first time, it downloads approximately **308 MB** of model files from Hugging Face. An active internet connection is required for this initial step.
- **Local Caching:** After download, model weights are stored permanently in the Hugging Face cache (`~/.cache/huggingface/hub`).
- **Offline Inference:** Subsequent inferences run entirely offline using the local cached weights.
- **Extractive Fallback:** If internet access is unavailable during the first run or if system memory is constrained, the application automatically triggers the **Extractive NLP Fallback Engine**, ensuring the application remains functional even without connectivity.

---

## 🔒 Privacy Considerations

- **100% Local Inference:** All text submitted to the application is processed locally on your computer's CPU.
- **Zero Telemetry:** No user text, prompts, or generated summaries are sent to third-party cloud APIs, external logging servers, or tracking services.
- **No Secret Keys:** The project does not require or store paid API credentials.

---

## ⚠️ Known Limitations

1. **Context Window:** The `flan-t5-small` model has an input limit of 512 tokens. Extremely long documents are truncated to the first 512 tokens.
2. **CPU Inference Latency:** Beam search on older dual-core CPUs may require 3 to 7 seconds per summary.
3. **Language Scope:** The application is tuned and evaluated for English text.

---

## ❓ Common Errors & Solutions

| Error | Cause | Solution |
|---|---|---|
| `'streamlit' is not recognized` | Virtual environment is inactive or script path missing. | Run `python -m streamlit run app.py` |
| `First-time model download failed` | No internet connection during initial download. | Connect to internet for initial download, or use the built-in **Extractive NLP** mode. |
| `Address already in use: 8501` | Another Streamlit instance is active. | Launch with a different port: `python -m streamlit run app.py --server.port 8502` |

---

## 🎓 Academic & Training Information

This project was built for the **Generative AI with NLP** training and academic curriculum.  
For detailed technical documentation, review [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) and [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md).

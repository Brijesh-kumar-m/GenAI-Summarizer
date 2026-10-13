# TESTING GUIDE & VERIFICATION MANUAL
## GenAI Summarizer: Advanced NLP & Text Intelligence Assistant
### Generative AI with NLP Project

This guide provides step-by-step instructions for installing dependencies, running automated test suites, launching the Streamlit web application, executing manual test cases, capturing screenshots for project reports, and resolving common errors.

---

## 1. Environment & Installation

### A. Prerequisites
- **Operating System:** Windows 10 or 11 (64-bit recommended)
- **Python:** Python 3.10 or newer (Verified on Python 3.14)
- **Terminal:** PowerShell or Command Prompt

### B. Verification of Python Version
Open PowerShell and verify Python is installed:
```powershell
python --version
```
*Expected Output:* `Python 3.10.x` or higher (e.g. `Python 3.14.x`).

### C. Creating and Activating Virtual Environment (Recommended)
Navigate to the project root directory and set up a virtual environment:
```powershell
cd "d:\generative ai with npl\ai_text_summarizer_nlp"
python -m venv .venv
.venv\Scripts\activate
```
*(Notice: The terminal prompt should now display `(.venv)`).*

### D. Upgrading Pip and Installing Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
*(All dependencies including `streamlit`, `transformers`, `torch`, `scikit-learn`, `numpy`, and `pytest` will be installed).*

---

## 2. Running Automated Tests

The project includes an automated test suite located in the `tests/` folder covering input validation, word/char counters, sentence segmentation, TF-IDF keyword extraction, length presets, and mock error handling.

To execute all tests with detailed verbosity:
```powershell
python -m pytest tests -v
```

### Expected Automated Test Output
```text
tests/test_text_utils.py::test_empty_input_handling PASSED
tests/test_text_utils.py::test_word_count_calculation PASSED
tests/test_text_utils.py::test_sentence_splitting_and_counting PASSED
tests/test_text_utils.py::test_input_length_validation PASSED
tests/test_text_utils.py::test_reduction_percentage_calculation PASSED
tests/test_text_utils.py::test_text_statistics_dictionary PASSED
tests/test_text_utils.py::test_output_formatting PASSED
tests/test_keyword_extractor.py::test_keyword_extraction_meaningful_text PASSED
tests/test_keyword_extractor.py::test_keyword_extraction_very_short_text PASSED
tests/test_keyword_extractor.py::test_keyword_extraction_empty_text PASSED
tests/test_keyword_extractor.py::test_keyword_count_configuration PASSED
tests/test_keyword_extractor.py::test_stopwords_only_text PASSED
tests/test_keyword_extractor.py::test_fallback_keyword_extractor PASSED
tests/test_summarizer.py::test_summary_length_configuration_validation PASSED
tests/test_summarizer.py::test_summarize_empty_and_short_input PASSED
tests/test_summarizer.py::test_extractive_summarization PASSED
tests/test_summarizer.py::test_extractive_preserves_order PASSED
tests/test_summarizer.py::test_summarize_force_extractive PASSED
tests/test_summarizer.py::test_error_handling_model_initialization_failure PASSED
tests/test_summarizer.py::test_error_handling_model_inference_failure PASSED
tests/test_summarizer.py::test_get_model_status PASSED

======================== 21 passed in 2.30s ========================
```

---

## 3. Launching the Streamlit Application

To start the local Streamlit web server, run:
```powershell
python -m streamlit run app.py
```
*(Or if running from workspace root: `python -m streamlit run ai_text_summarizer_nlp\app.py`)*

### Terminal Output
```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```
The application will automatically launch in your default browser window. If it does not open automatically, navigate to `http://localhost:8501` manually.

---

## 4. First-Time Model Download Notice

- **First Inference Run:** When you click **"Generate Summary & Run NLP Analysis"** for the very first time, the application contacts the Hugging Face Model Hub to download the weights for `google/flan-t5-small` (~308 MB).
- **Download Time:** Depending on your internet speed, this download may take between 30 to 90 seconds. A loading spinner will indicate that the model is downloading and loading.
- **Subsequent Runs:** Once downloaded, the model files are cached locally in your Hugging Face cache directory (`~/.cache/huggingface/hub`). Subsequent runs will load locally without requiring an internet connection.
- **Offline Fallback:** If internet access is not available during the first run, the application will automatically invoke the documented **Extractive NLP Fallback** engine and present an informative banner explaining that extractive sentence scoring was utilized.

---

## 5. Manual Testing Checklist & Test Walkthrough

Follow this step-by-step checklist to test every feature of the application:

### Test Case 1: Empty Text Input
1. Ensure the text area is empty.
2. Click **"🚀 Generate Summary & Run NLP Analysis"**.
3. **Expected Behavior:** An error banner appears stating: *"Input text is empty. Please enter or paste some text to summarize."* No crash occurs.

### Test Case 2: Very Short Input (< 15 Words)
1. Type a short sentence: *"AI is interesting."* (3 words).
2. Look at the live status card next to the character counter.
3. **Expected Behavior:** An indicator displays *"⚠️ Need min 15 words"*. If you click summarize, a validation error warns that meaningful summarization requires at least 15 words.

### Test Case 3: Sample Text Load
1. Click the **"📋 Sample: AI Healthcare"** button above the input box.
2. **Expected Behavior:** The text area instantly fills with an educational paragraph discussing machine learning, medical imaging, drug discovery, and physician supervision. Live word counter updates to ~130+ words.

### Test Case 4: Generative AI Summarization
1. With the sample text loaded, keep the engine setting on **"Generative AI (Flan-T5)"** and select **"Medium"** length.
2. Click **"🚀 Generate Summary & Run NLP Analysis"**.
3. Observe the loading spinner: *"⏳ Running inference & extracting linguistic features..."*
4. **Expected Output:**
   - A blue badge appears: `✨ Generative AI Engine (Hugging Face Flan-T5)`.
   - A concise abstractive summary appears in the info box.
   - The disclaimer banner is visible underneath.

### Test Case 5: Text Statistics Verification
1. Inspect the 5 high-level metric cards above the tabs.
2. **Expected Metrics:**
   - **Original Words:** ~130–150 words.
   - **Summary Words:** ~30–70 words.
   - **Word Reduction Rate:** Displays an approximate reduction between 45% and 75%.
   - Verify no divide-by-zero errors occur.

### Test Case 6: TF-IDF Keyword Extraction
1. Switch to the **"🔑 Keyphrase Extraction (TF-IDF)"** tab.
2. **Expected Output:**
   - Progress bars display top extracted keywords (e.g., *healthcare*, *medical*, *learning*, *diagnostic*, *models*).
   - Numerical TF-IDF scores between 0.0 and 1.0 are shown next to each term.

### Test Case 7: Side-by-Side Comparative Lab
1. Switch to the **"🧪 Generative vs Extractive Lab"** tab.
2. **Expected Output:**
   - Left column displays the neural **Generative AI** abstractive rewriting.
   - Right column displays the statistical **Extractive NLP** sentence salience selection.
   - This illustrates to students how abstractive AI rephrases sentences while extractive NLP extracts exact source sentences.

### Test Case 8: File Download Verification
1. Scroll down to Section 3: **"💾 3. Download & Export Analysis"**.
2. Click **"📥 Download Summary (.txt)"** -> Verify downloaded file opens in Notepad and contains only the summary with metadata header.
3. Click **"📊 Download Complete Report (.txt)"** -> Verify downloaded file contains the original text, summary, statistics table, and TF-IDF keywords list.
4. Click **"📑 Download Source + Summary (.txt)"** -> Verify downloaded file contains both original and generated texts with UTF-8 encoding.

### Test Case 9: Length Option Variations
1. In the sidebar, change the summary length from **"Medium"** to **"Short"**.
2. Click summarize. Verify the resulting summary is shorter (~25–45 words).
3. Change to **"Long"**. Verify the resulting summary is longer and covers more detail.

### Test Case 10: Clear All Data
1. Click **"🗑️ Clear Input"** or sidebar **"🧹 Clear All Data"**.
2. **Expected Behavior:** All input boxes and generated results reset to their initial clean state.

---

## 6. How to Capture Authentic Screenshots for Your Report

To add authentic figures to your `docs/PROJECT_REPORT.md`:
1. Launch the application in Google Chrome or Microsoft Edge.
2. Press **Windows Key + Shift + S** to open the Windows Snipping Tool.
3. Capture the following recommended views:
   - Full screen of the application header and input area with Sample 1 loaded.
   - The Generated Summary tab showing the `✨ Generative AI Engine` badge.
   - The Keyword Extraction tab showing the TF-IDF salience progress bars.
   - The Quantitative Document Metrics tab with original vs. summary statistics.
   - The Comparative Laboratory tab showing Generative vs. Extractive side-by-side.
   - The exported report text file opened in Windows Notepad.
4. Save the images in an `images/` or `assets/` folder and embed them in Section 16 of `docs/PROJECT_REPORT.md`.

---

## 7. How to Stop the Streamlit Server

To gracefully terminate the running Streamlit server:
1. Click on the PowerShell or Command Prompt terminal running Streamlit.
2. Press `Ctrl + C` on your keyboard.
3. If prompted with `Terminate batch job (Y/N)?`, type `Y` and press `Enter`.

---

## 8. Troubleshooting & Common Issues

| Issue / Error | Root Cause | Solution |
|---|---|---|
| `'streamlit' is not recognized as an internal or external command` | The virtual environment is not active or Streamlit scripts are not on system PATH. | Run Streamlit using Python module syntax: `python -m streamlit run app.py` |
| `First-time model download failed for 'google/flan-t5-small'` | No active internet connection during initial run or Hugging Face hub timeout. | Verify your internet connection. Alternatively, the app automatically switches to the built-in **Extractive NLP Fallback**, which works 100% offline. |
| `Torch not found or ImportError` | PyTorch wheel not installed in active environment. | Run `pip install torch` or check requirements installation. |
| `Port 8501 is already in use` | A previous Streamlit process is still running in another terminal. | Start Streamlit on a different port: `python -m streamlit run app.py --server.port 8502` |
| Model inference takes 10+ seconds on CPU | Normal behavior on older dual-core CPUs with beam search. | Select **"Short"** length preset or use **"Extractive NLP"** mode for sub-second responses. |

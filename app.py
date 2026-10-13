"""
app.py - GenAI Summarizer: Advanced NLP & Text Intelligence Assistant
=====================================================================
Course: Generative AI with NLP Training Project

A professional, student-friendly web application integrating:
1. Generative AI Text Summarization (Hugging Face Flan-T5 Seq2Seq).
2. Extractive NLP Fallback & Comparative Sentence Salience.
3. TF-IDF Keyword and Keyphrase Extraction with Relevance Scoring.
4. Comprehensive Text Analytics, Reading Time Saved & Readability Grading.
5. Exportable UTF-8 Reports and Standalone Summaries.
"""

import sys
import os
import streamlit as st

# Ensure local modules can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from text_utils import (
    count_words,
    count_characters,
    count_sentences,
    calculate_reduction,
    validate_input_text,
    get_text_statistics,
    format_summary_txt,
    format_complete_report_txt,
    format_as_bullet_points,
)
from keyword_extractor import extract_keywords
from summarizer import (
    summarize_text,
    extractive_summarize,
    get_model_status,
    LENGTH_PRESETS,
    DEFAULT_MODEL_NAME,
)

# -----------------------------------------------------------------------------
# Pre-defined Sample Texts for Students and Evaluators
# -----------------------------------------------------------------------------
SAMPLE_TEXTS = {
    "AI in Healthcare": (
        "Artificial intelligence and machine learning technologies are fundamentally revolutionizing "
        "the modern healthcare landscape. Hospitals and clinical research institutions worldwide are "
        "deploying deep learning models to accelerate disease diagnosis, analyze medical imaging scans "
        "such as MRIs and CT scans, and detect malignant tumors with remarkable accuracy. Furthermore, "
        "generative AI systems are assisting pharmacologists in discovering novel drug molecules by "
        "simulating biochemical interactions in silico, reducing development timelines from decades to months. "
        "Natural language processing models parse unstructured electronic health records, enabling physicians "
        "to synthesize comprehensive patient histories instantly and avoid dangerous drug interactions. "
        "Despite these revolutionary advancements, medical practitioners emphasize that automated AI systems "
        "must serve as assistive diagnostic aids rather than autonomous substitutes for clinical judgment, "
        "underscoring the enduring necessity of human supervision, ethical data handling, and rigorous "
        "validation in safety-critical medical contexts."
    ),
    "Renewable Energy & Climate": (
        "The transition toward renewable energy represents humanity's primary strategy to mitigate catastrophic "
        "global climate change. Over the past decade, solar photovoltaic efficiency has surged while production "
        "costs have plummeted by more than eighty percent, making solar energy cheaper than fossil fuels in most "
        "parts of the world. Simultaneously, next-generation offshore wind turbines now generate massive megawatts "
        "of clean electricity with minimal terrestrial footprint. However, the intermittent nature of solar and "
        "wind resources creates substantial challenges for legacy power grids. To overcome this intermittency, "
        "governments and energy companies are investing heavily in utility-scale lithium iron phosphate batteries, "
        "pumped hydro storage, and green hydrogen synthesis. Smart grids equipped with predictive machine learning "
        "algorithms forecast hourly energy demand, dynamically routing clean power where it is needed most. "
        "A successful clean energy transition requires sustained policy incentives, cross-border grid integration, "
        "and resilient supply chains for critical minerals."
    ),
    "Quantum Computing & Security": (
        "Quantum computing leverages the non-classical phenomena of quantum mechanics, specifically superposition "
        "and quantum entanglement, to process complex information in fundamentally new ways. Unlike classical "
        "computers that store information as binary bits representing zeros or ones, quantum processors utilize "
        "qubits capable of existing in multidimensional superpositions. This architectural paradigm allows quantum "
        "systems to execute parallel calculations at speeds unimaginable with silicon semiconductors. Prominent "
        "applications include simulating molecular chemistry for room-temperature superconductors, optimizing "
        "global financial portfolios, and cracking standard asymmetric cryptographic protocols such as RSA. "
        "Because future fault-tolerant quantum computers could compromise modern internet banking security, "
        "cryptographers and security agencies have initiated the global transition to Post-Quantum Cryptography "
        "(PQC). Developing physical quantum hardware remains exceptionally challenging due to quantum decoherence, "
        "which requires cryogenic temperatures close to absolute zero to maintain qubit stability."
    ),
}

# -----------------------------------------------------------------------------
# Page Configuration & Custom CSS Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GenAI Summarizer: Advanced NLP & Text Intelligence Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Custom Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Container polish */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
        max-width: 1250px;
    }

    /* Modern Hero Header Banner */
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 45%, #0e7490 100%);
        color: #ffffff;
        padding: 2.2rem 2.4rem;
        border-radius: 18px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px -5px rgba(15, 23, 42, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.1) inset;
        position: relative;
        overflow: hidden;
    }
    
    .hero-card::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.18) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .hero-title {
        color: #ffffff !important;
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.05rem;
        line-height: 1.5;
        max-width: 850px;
        margin-bottom: 1rem;
        font-weight: 400;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(52, 211, 153, 0.4);
        color: #6ee7b7;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
        display: inline-block;
    }

    .tag-badges-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 0.8rem;
    }

    .hero-tag {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #f1f5f9;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 500;
    }

    /* Metric Glass Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 14px;
        margin-bottom: 1.5rem;
    }

    .metric-card-glass {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card-glass:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
    }

    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }

    .metric-sub {
        font-size: 0.8rem;
        color: #059669;
        font-weight: 600;
        margin-top: 0.25rem;
    }

    /* Summary Presentation Box */
    .summary-box {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #2563eb;
        border-radius: 12px;
        padding: 1.6rem;
        font-size: 1.06rem;
        line-height: 1.7;
        color: #1e293b;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        margin: 1rem 0;
    }

    /* Engine Badges */
    .badge-neural {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        padding: 5px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    .badge-extractive {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #fefce8;
        border: 1px solid #fef08a;
        color: #a16207;
        padding: 5px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* Keyword Tags */
    .keyword-pill {
        display: inline-flex;
        align-items: center;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 8px 14px;
        margin: 4px;
        font-size: 0.9rem;
        font-weight: 600;
        color: #1e293b;
        transition: all 0.2s ease;
    }

    .keyword-pill:hover {
        background: #f1f5f9;
        border-color: #cbd5e1;
    }

    .keyword-score {
        background: #e0e7ff;
        color: #3730a3;
        font-size: 0.76rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 6px;
        margin-left: 8px;
    }

    /* Modern Footer */
    .modern-footer {
        text-align: center;
        padding: 2.2rem 1rem 1rem 1rem;
        margin-top: 3.5rem;
        border-top: 1px solid #e2e8f0;
        color: #64748b;
        font-size: 0.88rem;
    }
    
    .modern-footer strong {
        color: #1e293b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "text_input_area" not in st.session_state:
    st.session_state["text_input_area"] = ""
if "summary_output" not in st.session_state:
    st.session_state.summary_output = None
if "keywords_output" not in st.session_state:
    st.session_state.keywords_output = None
if "stats_output" not in st.session_state:
    st.session_state.stats_output = None
if "extractive_comparison" not in st.session_state:
    st.session_state.extractive_comparison = None
if "summary_view_format" not in st.session_state:
    st.session_state.summary_view_format = "Paragraph"


def set_sample_text(text: str):
    """Callback to load sample text directly into the text area state."""
    st.session_state["text_input_area"] = text


def clear_input_text():
    """Callback to clear the text area."""
    st.session_state["text_input_area"] = ""


def clear_all():
    """Callback to reset all inputs and results."""
    st.session_state["text_input_area"] = ""
    st.session_state.summary_output = None
    st.session_state.keywords_output = None
    st.session_state.stats_output = None
    st.session_state.extractive_comparison = None


# -----------------------------------------------------------------------------
# Sidebar Configuration & Architecture Details
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")

    summary_length_key = st.selectbox(
        "Summary Length Preset",
        options=["short", "medium", "long"],
        index=1,
        format_func=lambda x: LENGTH_PRESETS[x]["label"],
        help="Controls the target length budget and beam search limits.",
    )

    num_keywords = st.slider(
        "Number of Keywords",
        min_value=3,
        max_value=15,
        value=7,
        step=1,
        help="Select the number of salient TF-IDF key terms to extract.",
    )

    st.markdown("---")
    st.markdown("### 🔬 Inference Engine")
    engine_mode = st.radio(
        "Processing Pipeline",
        options=["Generative AI (Flan-T5)", "Extractive NLP (Sentence Salience)"],
        index=0,
        help=(
            "Generative AI synthesizes novel phrasing via Transformer Seq2Seq. "
            "Extractive NLP extracts exact high-salience sentences from source."
        ),
    )
    force_extractive = "Extractive" in engine_mode

    st.markdown("---")
    st.markdown("### ℹ️ Architecture Specs")
    model_status = get_model_status()
    st.markdown(f"**Model:** `{DEFAULT_MODEL_NAME}`")
    st.markdown("**Architecture:** Encoder-Decoder Seq2Seq")
    st.markdown("**Parameters:** ~80 Million (Float32)")
    st.markdown(f"**Compute Device:** `{model_status['device'].upper()}`")
    st.markdown("**Local Weight Size:** ~308 MB (Cached)")
    st.markdown("**API Cost:** **$0.00** (Free, Open Source)")

    st.info(
        "💡 **Offline Ready:** Runs 100% locally on your CPU. "
        "No text or data is ever sent to external cloud APIs or third parties."
    )

    st.markdown("---")
    st.button("🧹 Clear All Data & Reset", on_click=clear_all, use_container_width=True)

# -----------------------------------------------------------------------------
# Top Hero Header Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="status-pill">
            <span class="status-dot"></span>
            <span>Local Neural Engine Active • CPU Mode • 100% Private</span>
        </div>
        <h1 class="hero-title">🧠 GenAI Summarizer</h1>
        <p class="hero-subtitle">
            Advanced NLP & Text Intelligence Assistant — A dual-engine Natural Language Processing platform combining 
            Generative AI deep learning text synthesis with statistical TF-IDF keyphrase semantics and readability analytics.
        </p>
        <div class="tag-badges-container">
            <span class="hero-tag">🎓 Generative AI with NLP</span>
            <span class="hero-tag">🤖 Google Flan-T5 Seq2Seq</span>
            <span class="hero-tag">📊 Scikit-Learn TF-IDF</span>
            <span class="hero-tag">⚡ Zero Paid API Cost</span>
            <span class="hero-tag">🔒 100% Client-Side Privacy</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Section 1: Text Input, File Upload & Sample Pickers
# -----------------------------------------------------------------------------
st.markdown("### 📝 1. Input Document or Article")

# Sample text buttons toolbar
sample_col1, sample_col2, sample_col3, sample_col4 = st.columns([1, 1, 1, 0.7])
with sample_col1:
    st.button(
        "📋 AI in Healthcare",
        on_click=set_sample_text,
        args=(SAMPLE_TEXTS["AI in Healthcare"],),
        use_container_width=True,
        help="Load pre-built healthcare sample text.",
    )
with sample_col2:
    st.button(
        "🌱 Clean Energy",
        on_click=set_sample_text,
        args=(SAMPLE_TEXTS["Renewable Energy & Climate"],),
        use_container_width=True,
        help="Load pre-built clean energy sample text.",
    )
with sample_col3:
    st.button(
        "⚛️ Quantum Security",
        on_click=set_sample_text,
        args=(SAMPLE_TEXTS["Quantum Computing & Security"],),
        use_container_width=True,
        help="Load pre-built quantum cryptography sample text.",
    )
with sample_col4:
    st.button("🗑️ Clear", on_click=clear_input_text, use_container_width=True, help="Clear input text.")

# Optional File Upload Expander
with st.expander("📂 Or upload a text file (.txt, .md)", expanded=False):
    uploaded_file = st.file_uploader(
        "Upload a document from your computer",
        type=["txt", "md"],
        help="Select a plain text file to load into the summarizer.",
    )
    if uploaded_file is not None:
        try:
            file_text = uploaded_file.read().decode("utf-8")
            if file_text.strip():
                st.session_state["text_input_area"] = file_text
                st.success(f"Loaded {uploaded_file.name} ({count_words(file_text):,} words)")
        except Exception as e:
            st.error(f"Error reading file: {e}")

input_text = st.text_area(
    "Enter or paste the text you want to analyze and summarize below:",
    height=220,
    placeholder="Paste an article, research excerpt, essay, or news story here (minimum 15 words)...",
    key="text_input_area",
    max_chars=8000,
)

# Real-time Counter Metrics
live_words = count_words(input_text)
live_chars = count_characters(input_text)
live_sents = count_sentences(input_text)

counter_col1, counter_col2, counter_col3, counter_col4 = st.columns(4)
counter_col1.metric("Words", f"{live_words:,}")
counter_col2.metric("Characters", f"{live_chars:,} / 8,000")
counter_col3.metric("Sentences", f"{live_sents:,}")
if live_words < 15 and live_words > 0:
    counter_col4.warning("⚠️ Need min 15 words")
elif live_words >= 15:
    counter_col4.success("✅ Ready to summarize")
else:
    counter_col4.info("ℹ️ Enter text to start")

# Extract actual text entered by user (from state or text area)
actual_user_text = (
    st.session_state.get("text_input_area", "").strip()
    if ("text_input_area" in st.session_state and st.session_state["text_input_area"])
    else input_text.strip()
)

# Validation Warning
is_valid_input, val_msg = validate_input_text(actual_user_text)

# Primary Execution Button
execute_btn = st.button("🚀 Generate Summary & Run NLP Analysis", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Section 2: Execution Pipeline
# -----------------------------------------------------------------------------
if execute_btn:
    if not is_valid_input:
        st.error(f"❌ {val_msg}")
    else:
        with st.spinner("⏳ Running Transformer inference & computing semantic metrics..."):
            # 1. Summarization with exact user text and explicit summary length
            summary_res = summarize_text(
                actual_user_text,
                length_mode=summary_length_key,
                force_extractive=force_extractive,
            )
            st.session_state.summary_output = summary_res

            # 2. Comparative Extractive summary (for educational lab tab)
            extractive_res = extractive_summarize(
                actual_user_text,
                length_mode=summary_length_key
            )
            st.session_state.extractive_comparison = extractive_res

            # 3. Keyword Extraction
            kw_res, kw_msg = extract_keywords(actual_user_text, top_k=num_keywords)
            st.session_state.keywords_output = (kw_res, kw_msg)

            # 4. Text Statistics & Readability
            stats = get_text_statistics(actual_user_text, summary_res.get("summary", ""))
            st.session_state.stats_output = stats

            st.toast("Summary and NLP analysis generated successfully!", icon="🎉")

# -----------------------------------------------------------------------------
# Section 3: Results Section
# -----------------------------------------------------------------------------
if st.session_state.summary_output is not None and st.session_state.stats_output is not None:
    summary_data = st.session_state.summary_output
    stats = st.session_state.stats_output
    keywords_list, kw_status = st.session_state.keywords_output

    st.markdown("---")
    st.markdown("### 📊 2. Analysis Results & Summary")

    # High-level Statistics Metric Cards (5 Cards)
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.markdown(
            f"""
            <div class="metric-card-glass">
                <div class="metric-label">⏱️ Time Saved</div>
                <div class="metric-value">{stats.get('time_saved_str', 'N/A')}</div>
                <div class="metric-sub">Read in {stats.get('summ_reading_time_str', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card-glass">
                <div class="metric-label">📉 Word Reduction</div>
                <div class="metric-value">{stats['reduction_percentage']}%</div>
                <div class="metric-sub">{stats['summary_words']} of {stats['original_words']} words</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card-glass">
                <div class="metric-label">📖 Readability</div>
                <div class="metric-value">{stats.get('summ_flesch_score', 0)}</div>
                <div class="metric-sub">{stats.get('summ_flesch_grade', 'Standard')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card-glass">
                <div class="metric-label">📝 Sentences</div>
                <div class="metric-value">{stats['summary_sentences']} <span style="font-size: 1rem; color:#64748b;">/ {stats['original_sentences']}</span></div>
                <div class="metric-sub">Condensed {stats['original_sentences']} sents</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c5:
        st.markdown(
            f"""
            <div class="metric-card-glass">
                <div class="metric-label">🤖 Architecture</div>
                <div class="metric-value" style="font-size: 1.25rem; font-weight: 700; padding-top: 5px;">
                    {'Extractive' if summary_data.get('is_fallback') else 'Flan-T5'}
                </div>
                <div class="metric-sub">{'TF-IDF Salience' if summary_data.get('is_fallback') else 'Neural Seq2Seq'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Result Tabs
    tab_summary, tab_keywords, tab_stats, tab_lab = st.tabs([
        "📄 Generated Summary",
        "🔑 Keyphrase Extraction (TF-IDF)",
        "📈 Detailed Statistics & Readability",
        "🧪 Generative vs Extractive Lab",
    ])

    # -------------------------------------------------------------------------
    # Tab 1: Summary Output
    # -------------------------------------------------------------------------
    with tab_summary:
        format_col1, format_col2 = st.columns([2, 1])
        with format_col1:
            if summary_data.get("is_fallback"):
                st.markdown(
                    """
                    <span class="badge-extractive">⚠️ Extractive NLP Fallback Engine</span>
                    <span style="color: #64748b; font-size: 0.85rem; margin-left: 8px;">
                        Algorithm: Statistical TF-IDF Sentence Salience & Position Ranker
                    </span>
                    """,
                    unsafe_allow_html=True,
                )
                if summary_data.get("fallback_reason"):
                    st.warning(f"Notice: {summary_data['fallback_reason']}")
            else:
                st.markdown(
                    """
                    <span class="badge-neural">✨ Generative AI Engine (Hugging Face Flan-T5)</span>
                    <span style="color: #64748b; font-size: 0.85rem; margin-left: 8px;">
                        Architecture: Encoder-Decoder Transformer (google/flan-t5-small)
                    </span>
                    """,
                    unsafe_allow_html=True,
                )

        with format_col2:
            view_mode = st.radio(
                "Display Format",
                ["Paragraph", "Key Takeaways (Bullets)"],
                horizontal=True,
                label_visibility="collapsed",
            )

        # Content based on view mode
        raw_summary = summary_data["summary"]
        if view_mode == "Key Takeaways (Bullets)":
            display_summary = format_as_bullet_points(raw_summary)
        else:
            display_summary = raw_summary

        st.markdown(
            f"""
            <div class="summary-box">
                {display_summary.replace(chr(10), '<br>')}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Copy Codeblock for immediate 1-click clipboard copy
        with st.expander("📋 One-Click Copy Container", expanded=False):
            st.code(display_summary, language="text")

        # Educational Disclaimer
        st.caption(
            "⚠️ **Disclaimer:** " + summary_data.get(
                "disclaimer",
                "Model-generated summaries are probabilistic approximations. Always verify critical facts."
            )
        )

    # -------------------------------------------------------------------------
    # Tab 2: Keyword Extraction
    # -------------------------------------------------------------------------
    with tab_keywords:
        st.markdown("#### Extracted Keywords & Semantic Salience")
        st.caption("Extracted using Term Frequency-Inverse Document Frequency (TF-IDF) with stopword suppression and 1-2 gram n-grams.")

        if keywords_list:
            # Modern pill cloud visualization
            pill_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.5rem;">'
            for kw, score in keywords_list:
                score_pct = int(score * 100)
                pill_html += f'<span class="keyword-pill">🏷️ {kw} <span class="keyword-score">{score_pct}%</span></span>'
            pill_html += '</div>'
            st.markdown(pill_html, unsafe_allow_html=True)

            # Detailed score progress breakdown
            st.markdown("##### Detailed Salience Breakdown")
            kw_cols = st.columns(2)
            for idx, (kw, score) in enumerate(keywords_list):
                target_col = kw_cols[idx % 2]
                with target_col:
                    st.progress(float(score), text=f"**{kw}** — Salience: {score:.4f}")
        else:
            st.warning(kw_status or "No distinctive keywords could be extracted from this text.")

    # -------------------------------------------------------------------------
    # Tab 3: Detailed Statistics & Readability
    # -------------------------------------------------------------------------
    with tab_stats:
        st.markdown("#### Quantitative Document Metrics & Readability")
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown("##### 📄 Source Document")
            st.write(f"- **Total Words:** {stats['original_words']}")
            st.write(f"- **Characters (with spaces):** {stats['original_characters']}")
            st.write(f"- **Characters (no spaces):** {stats['original_chars_nospace']}")
            st.write(f"- **Sentence Count:** {stats['original_sentences']}")
            avg_orig = stats['original_words'] / max(stats['original_sentences'], 1)
            st.write(f"- **Average Words per Sentence:** {avg_orig:.1f}")
            st.write(f"- **Estimated Reading Time:** {stats.get('orig_reading_time_str', 'N/A')}")
            st.write(f"- **Flesch Reading Ease:** {stats.get('orig_flesch_score', 'N/A')} ({stats.get('orig_flesch_grade', '')})")

        with col_s2:
            st.markdown("##### ⚡ Summary Document")
            st.write(f"- **Summary Words:** {stats['summary_words']}")
            st.write(f"- **Summary Characters:** {stats['summary_characters']}")
            st.write(f"- **Summary Sentences:** {stats['summary_sentences']}")
            avg_summ = stats['summary_words'] / max(stats['summary_sentences'], 1)
            st.write(f"- **Average Words per Sentence:** {avg_summ:.1f}")
            st.write(f"- **Estimated Reading Time:** {stats.get('summ_reading_time_str', 'N/A')}")
            st.write(f"- **Flesch Reading Ease:** {stats.get('summ_flesch_score', 'N/A')} ({stats.get('summ_flesch_grade', '')})")
            st.write(f"- **Overall Compression:** **{stats['reduction_percentage']}% reduction**")

    # -------------------------------------------------------------------------
    # Tab 4: Generative vs Extractive Comparison Lab (Educational)
    # -------------------------------------------------------------------------
    with tab_lab:
        st.markdown("#### 🧪 Educational Comparative Lab")
        st.caption(
            "Compare how Generative AI rewrites and synthesizes concepts vs how traditional "
            "Extractive NLP extracts exact sentences using statistical TF-IDF salience."
        )

        col_comp1, col_comp2 = st.columns(2)
        with col_comp1:
            st.markdown("##### 🤖 Generative AI (Flan-T5)")
            st.markdown(
                "_Abstractive synthesis: The model reads the input and creates concise new phrasing._"
            )
            st.text_area(
                "Generative AI Output",
                value=summary_data["summary"],
                height=160,
                disabled=True,
                key="gen_comp_area",
            )

        with col_comp2:
            st.markdown("##### 📐 Extractive NLP (TF-IDF Salience)")
            st.markdown(
                "_Statistical extraction: Selects top existing sentences based on word importance._"
            )
            ext_text = (
                st.session_state.extractive_comparison["summary"]
                if st.session_state.extractive_comparison
                else "N/A"
            )
            st.text_area(
                "Extractive NLP Output",
                value=ext_text,
                height=160,
                disabled=True,
                key="ext_comp_area",
            )

    # -------------------------------------------------------------------------
    # Section 4: Download Functionality
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 💾 3. Download & Export Analysis")

    down_col1, down_col2, down_col3 = st.columns(3)

    # 1. Summary Only TXT
    summary_txt_content = format_summary_txt(
        summary_data["summary"],
        model_name=summary_data.get("model_name", DEFAULT_MODEL_NAME),
        is_fallback=summary_data.get("is_fallback", False),
    )
    with down_col1:
        st.download_button(
            label="📥 Download Summary (.txt)",
            data=summary_txt_content.encode("utf-8"),
            file_name="ai_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # 2. Complete Analysis Report TXT
    complete_report_content = format_complete_report_txt(
        original_text=input_text,
        summary_text=summary_data["summary"],
        stats=stats,
        keywords=keywords_list,
        model_name=summary_data.get("model_name", DEFAULT_MODEL_NAME),
        is_fallback=summary_data.get("is_fallback", False),
    )
    with down_col2:
        st.download_button(
            label="📊 Download Complete Report (.txt)",
            data=complete_report_content.encode("utf-8"),
            file_name="complete_nlp_analysis_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # 3. Source + Summary Combined TXT
    combined_content = (
        "==========================================================\n"
        "GENAI SUMMARIZER: ADVANCED NLP & TEXT INTELLIGENCE\n"
        "Original Source Text & Generated Summary\n"
        "==========================================================\n\n"
        "--- ORIGINAL SOURCE TEXT ---\n"
        f"{input_text.strip()}\n\n"
        "--- GENERATED SUMMARY ---\n"
        f"{summary_data['summary'].strip()}\n\n"
        f"Compression Rate: {stats['reduction_percentage']}%\n"
        f"Time Saved: {stats.get('time_saved_str', 'N/A')}\n"
    )
    with down_col3:
        st.download_button(
            label="📑 Download Source + Summary (.txt)",
            data=combined_content.encode("utf-8"),
            file_name="source_and_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

# -----------------------------------------------------------------------------
# Section 5: Educational Project Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="modern-footer">
        <strong>GenAI Summarizer: Advanced NLP & Text Intelligence Assistant</strong><br>
        Developed for students & practitioners in <strong>Generative AI & NLP</strong><br>
        Built with Streamlit • Hugging Face Transformers (Flan-T5) • Scikit-Learn TF-IDF • PyTorch<br>
        <em>No paid API keys required • 100% Free, Open Source & Offline Compatible</em>
    </div>
    """,
    unsafe_allow_html=True,
)

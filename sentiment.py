# --- START OF FINAL, CORRECTED sentiment.py ---

import streamlit as st
import pandas as pd
from typing import List

# Scikit-learn for NMF Topic Modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Transformers for Free, Open-Source AI Model
from transformers import pipeline

# --- PAGE CONFIGURATION ---
# Set the page configuration as the first Streamlit command
st.set_page_config(
    page_title="Customer Review & Topic Analysis",
    page_icon="📊",
    layout="wide"
)

# --- MODEL AND STATE MANAGEMENT ---

def initialize_session_state():
    """Initialize all necessary session state variables for a multi-page app experience."""
    if 'screen' not in st.session_state:
        st.session_state.screen = 'upload'
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None

@st.cache_resource
def get_summarizer():
    """
    Load and cache a powerful, free Hugging Face AI model.
    This function runs only once, making subsequent runs much faster.
    """
    st.info("Please wait, loading AI model for the first time...")
    # Using a more capable model for better, more accurate summarization
    return pipeline("summarization", model="Falconsai/text_summarization")

def get_ai_interpretation(keywords: List[str]) -> str:
    """Get AI interpretation of topic keywords using a better, local model."""
    try:
        summarizer = get_summarizer()
        keywords_str = ", ".join(keywords)
        
        # A more detailed prompt for higher quality results
        prompt = f"""
        As an expert product analyst, you are analyzing negative customer reviews.
        The keywords from one of the main topics of complaints are: "{keywords_str}".
        What is the specific problem or issue these customers are facing?
        Summarize the core problem in one clear and actionable sentence.
        """
        
        result = summarizer(prompt, max_length=60, min_length=10, do_sample=False)
        return result[0]['summary_text'].strip()
    except Exception as e:
        return f"❌ Error during AI interpretation: {str(e)}"

# --- UI SCREEN RENDERING ---

def render_upload_screen():
    """Render the data upload and configuration screen."""
    st.title("📊 Customer Review & Topic Analysis")
    st.header("📁 Step 1: Upload Your Review Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(2))
            st.write(f"**Columns found:** {', '.join(df.columns.tolist())} | **Total rows:** {len(df):,}")
            
            st.subheader("🎯 Step 2: Assign Your Columns")
            col1, col2 = st.columns(2)
            rating_column = col1.selectbox("A: Column with review ratings (1-5)", options=df.columns, key='rating_select')
            text_column = col2.selectbox("B: Column with the review text", options=df.columns, key='text_select')
            
            if rating_column == text_column:
                st.error("⚠️ Please select different columns for ratings and text.")
                return

            if st.button("🔄 Process and Clean Data", type="primary"):
                with st.spinner("Processing data..."):
                    processed_df = df[[rating_column, text_column]].copy()
                    processed_df.columns = ['rating', 'review_text']
                    initial_rows = len(processed_df)
                    
                    # --- THIS IS THE BUG FIX ---
                    # 1. Drop rows with any missing values first
                    processed_df.dropna(inplace=True)
                    # 2. Convert rating to numeric, turning errors into NaT (Not a Number)
                    processed_df['rating'] = pd.to_numeric(processed_df['rating'], errors='coerce')
                    # 3. Drop any rows where the conversion to numeric failed
                    processed_df.dropna(subset=['rating'], inplace=True)
                    # 4. Ensure review text is a string
                    processed_df['review_text'] = processed_df['review_text'].astype(str)
                    
                    st.session_state.cleaned_df = processed_df
                    final_rows = len(processed_df)

                    if final_rows == 0:
                        st.error("❌ No valid data remained after cleaning. Please check your columns and data.")
                        return

                    st.success(f"✅ Data cleaned successfully! Retained {final_rows:,} of {initial_rows:,} rows.")
                    st.session_state.screen = 'analysis'
                    st.rerun()

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

def render_analysis_screen():
    """Render the analysis and results screen."""
    st.title("🔍 Step 3: Analyze Topics from Reviews")
    
    if st.button("← Start Over"):
        initialize_session_state() # Reset state
        st.rerun()

    df = st.session_state.cleaned_df
    st.info(f"📊 **Data Summary:** {len(df):,} reviews ready for analysis.")
    st.divider()

    col1, col2 = st.columns(2)
    analysis_type = col1.selectbox(
        "Choose which reviews to analyze:",
        options=["Problems & Challenges (Ratings 1 & 2)", "Strengths & Positives (Ratings 4 & 5)"]
    )
    top_n = col2.number_input("How many topics to find?", min_value=3, max_value=15, value=5)

    if st.button("🔍 Find Topics", type="primary"):
        with st.spinner("Analyzing topics with NMF model... This may take a moment."):
            if "Problems" in analysis_type:
                filtered_df = df[df['rating'].isin([1, 2])]
                analysis_title = "Main Problems & Challenges"
            else:
                filtered_df = df[df['rating'].isin([4, 5])]
                analysis_title = "Strengths & Positive Feedback"

            if len(filtered_df) < top_n:
                st.warning(f"⚠️ Not enough reviews ({len(filtered_df)}) to find {top_n} topics.")
                return

            docs = filtered_df['review_text'].tolist()
            try:
                vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
                tfidf = vectorizer.fit_transform(docs)
                nmf_model = NMF(n_components=top_n, random_state=42, max_iter=1000, l1_ratio=0.0)
                
                W = nmf_model.fit_transform(tfidf)
                topic_assignments = W.argmax(axis=1)
                topic_counts = pd.Series(topic_assignments).value_counts().sort_index()

                st.session_state.model_results = {
                    'analysis_title': analysis_title,
                    'nmf_model': nmf_model,
                    'vectorizer': vectorizer,
                    'topic_counts': topic_counts
                }
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error during topic modeling: {str(e)}")
    
    if st.session_state.model_results is not None:
        render_results()

def render_results():
    """Render the analysis results."""
    results = st.session_state.model_results
    st.divider()
    st.subheader(f"📈 Results for: {results['analysis_title']}")
    
    nmf_model = results['nmf_model']
    vectorizer = results['vectorizer']
    feature_names = vectorizer.get_feature_names_out()

    for topic_idx, topic in enumerate(nmf_model.components_):
        top_keywords_indices = topic.argsort()[:-11:-1]
        keywords = [feature_names[i] for i in top_keywords_indices]
        
        review_count = results['topic_counts'].get(topic_idx, 0)
        
        st.markdown(f"### 🎯 Topic #{topic_idx + 1}")
        col1, col2 = st.columns([3, 1])
        col1.write(f"**Keywords:** `{', '.join(keywords)}`")
        col2.metric("Reviews in Topic", f"{review_count:,}")
        
        with st.expander("🤖 Get AI Interpretation"):
            interpretation = get_ai_interpretation(keywords)
            st.info(f"💡 **AI Analysis:** {interpretation}")

# --- MAIN APP LOGIC ---

def main():
    """Main application function to control screen flow."""
    initialize_session_state()
    if st.session_state.screen == 'upload':
        render_upload_screen()
    else: # Catches 'analysis' screen
        render_analysis_screen()

if __name__ == '__main__':
    main()

# --- END OF FILE ---

# --- START OF FINAL sentiment.py FILE ---

import streamlit as st
import pandas as pd
from typing import List

# Scikit-learn for NMF Topic Modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Transformers for Free, Open-Source AI Model (no API key needed)
from transformers import pipeline

# Set page configuration
st.set_page_config(
    page_title="Customer Review Sentiment & Topic Analysis",
    page_icon="📊",
    layout="wide"
)

def initialize_session_state():
    """Initialize all necessary session state variables"""
    if 'screen' not in st.session_state:
        st.session_state.screen = 'upload'
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None

@st.cache_resource
def get_summarizer():
    """
    Load and cache a powerful, free Hugging Face AI model.
    This function runs only once to avoid reloading the model on every run.
    """
    st.info("Please wait, loading a powerful AI model for the first time...")
    # Using a more capable model for better summarization
    return pipeline("summarization", model="Falconsai/text_summarization")

def get_ai_interpretation(keywords: List[str]) -> str:
    """
    Get AI interpretation of topic keywords using a better, local model.
    
    Args:
        keywords: List of topic keywords
        
    Returns:
        AI-generated interpretation string
    """
    try:
        # Load the cached AI model
        summarizer = get_summarizer()
        
        # An improved prompt that gives the model a clear role and task
        keywords_str = ", ".join(keywords)
        prompt = f"""
        You are an expert product manager analyzing customer feedback.
        The following keywords were extracted from a set of negative customer reviews:
        
        KEYWORDS: "{keywords_str}"
        
        Based on these keywords, what is the specific problem or complaint the customers are reporting?
        Summarize the core issue in one clear and concise sentence.
        """

        # Make API call to the local model
        result = summarizer(prompt, max_length=60, min_length=10, do_sample=False)
        return result[0]['summary_text'].strip()
        
    except Exception as e:
        return f"❌ Error getting AI interpretation: {str(e)}"

def reset_session_state():
    """Reset all session state variables to start over"""
    st.session_state.screen = 'upload'
    st.session_state.df = None
    st.session_state.cleaned_df = None
    st.session_state.model_results = None

def render_upload_screen():
    """Render the data upload and configuration screen"""
    st.title("📊 Customer Review Sentiment & Topic Analysis App")
    st.header("📁 Step 1: Upload Your CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing customer reviews",
        type=['csv'],
        help="Upload a CSV file with customer review ratings and text"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(2))
            st.write(f"**Columns found:** {', '.join(df.columns.tolist())} | **Total rows:** {len(df):,}")
            
            st.subheader("🎯 Step 2: Assign Your Columns")
            col1, col2 = st.columns(2)
            
            rating_column = col1.selectbox(
                "A: Column with review ratings (1-5)",
                options=df.columns.tolist(), key='rating_select'
            )
            text_column = col2.selectbox(
                "B: Column with the review text",
                options=df.columns.tolist(), key='text_select'
            )
            
            if rating_column == text_column:
                st.error("⚠️ Please select different columns for ratings and review text.")
                return
            
            if st.button("🔄 Process and Clean Data", type="primary"):
                with st.spinner("Processing data..."):
                    processed_df = df[[rating_column, text_column]].copy()
                    processed_df.columns = ['rating', 'review_text']
                    
                    initial_rows = len(processed_df)
                    processed_df = processed_df.dropna()
                    processed__df = processed_df[pd.to_numeric(processed_df['rating'], errors='coerce').notna()]
                    processed_df['review_text'] = processed_df['review_text'].astype(str)
                    
                    st.session_state.cleaned_df = processed_df
                    final_rows = len(processed_df)
                    
                    if final_rows == 0:
                        st.error("❌ No valid data remaining after cleaning.")
                        return
                    
                    st.success(f"✅ Data cleaned successfully! {initial_rows:,} → {final_rows:,} rows")
                    st.session_state.screen = 'analysis'
                    st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")

def render_analysis_screen():
    """Render the analysis and results screen"""
    st.title("🔍 Step 3: Analyze Topics from Reviews")
    
    if st.button("← Start Over"):
        reset_session_state()
        st.rerun()
    
    df = st.session_state.cleaned_df
    st.info(f"📊 **Data Summary:** {len(df):,} reviews available for analysis.")
    st.divider()
    
    col1, col2 = st.columns(2)
    analysis_type = col1.selectbox(
        "Choose which reviews to analyze:",
        options=[
            "1: Main Problems & Challenges (Ratings 1 & 2)",
            "2: Strengths & Positive Feedback (Ratings 4 & 5)"
        ]
    )
    top_n = col2.number_input(
        "How many topics do you want to find?",
        min_value=3, max_value=15, value=5
    )
    
    if st.button("🔍 Find Topics", type="primary"):
        with st.spinner("Analyzing topics with NMF model..."):
            if "Problems" in analysis_type:
                filtered_df = df[df['rating'].isin([1, 2])]
                analysis_title = "Main Problems & Challenges"
            else:
                filtered_df = df[df['rating'].isin([4, 5])]
                analysis_title = "Strengths & Positive Feedback"
            
            if len(filtered_df) < top_n:
                st.warning(f"⚠️ Not enough reviews found ({len(filtered_df)}).")
                return
            
            docs = filtered_df['review_text'].tolist()
            
            try:
                vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
                tfidf = vectorizer.fit_transform(docs)
                
                nmf_model = NMF(n_components=top_n, random_state=42, max_iter=1000)
                nmf_model.fit(tfidf)
                
                topic_assignments = nmf_model.transform(tfidf).argmax(axis=1)
                topic_counts = pd.Series(topic_assignments).value_counts().sort_index()

                st.session_state.model_results = {
                    'analysis_title': analysis_title,
                    'nmf_model': nmf_model,
                    'vectorizer': vectorizer,
                    'topic_counts': topic_counts
                }
                
                st.success(f"✅ Analysis complete!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during topic modeling: {str(e)}")
    
    if st.session_state.model_results is not None:
        render_results()

def render_results():
    """Render the analysis results"""
    results = st.session_state.model_results
    st.divider()
    st.subheader(f"📈 Results for: {results['analysis_title']}")
    
    nmf_model = results['nmf_model']
    vectorizer = results['vectorizer']
    feature_names = vectorizer.get_feature_names_out()
    
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_keywords_indices = topic.argsort()[:-10 - 1:-1]
        keywords = [feature_names[i] for i in top_keywords_indices]
        keywords_str = ', '.join(keywords)
        
        review_count = results['topic_counts'].get(topic_idx, 0)
        
        st.markdown(f"### 🎯 **Topic #{topic_idx + 1}**")
        
        col1, col2 = st.columns([3, 1])
        col1.write(f"**Keywords:** `{keywords_str}`")
        col2.metric("Reviews in Topic", f"{review_count:,}")
        
        with st.expander("🤖 Get AI Interpretation"):
            interpretation = get_ai_interpretation(keywords)
            st.info(f"💡 **AI Analysis:** {interpretation}")
        
        st.divider()

def main():
    """Main application function"""
    initialize_session_state()
    if st.session_state.screen == 'upload':
        render_upload_screen()
    elif st.session_state.screen == 'analysis':
        render_analysis_screen()

if __name__ == '__main__':
    main()

# --- END OF FINAL sentiment.py FILE ---
import streamlit as st
import PyPDF2
from transformers import pipeline


st.set_page_config(page_title="📘 All-in-One PDF NLP App", layout="wide")
st.title("📘 All-in-One PDF NLP App")
st.write("Perform Summarization, Question Answering, and Text Classification on your PDFs using Hugging Face models.")

#pdf extraction
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


#loading models
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    return summarizer, qa_pipeline, classifier

summarizer, qa_pipeline, classifier = load_models()


#upload PDF
uploaded_file = st.file_uploader("📂 Upload your PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("✅ PDF text extracted successfully!")
    
    st.subheader("📄 PDF Text Preview:")
    st.text_area("Extracted Text", pdf_text[:2000] + "..." if len(pdf_text) > 2000 else pdf_text, height=200)
    
    st.divider()

    tab1, tab2, tab3 = st.tabs(["📝 Summarization", "❓ Q&A", "🔍 Classification"])

# ---------- TAB 1: SUMMARIZATION ----------
    with tab1:
        st.subheader("📝 Summarize PDF Content")
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = summarizer(pdf_text[:1000], max_length=250, min_length=80, do_sample=False)[0]['summary_text']
            st.success("✅ Summary Generated!")
            st.write(summary)

    # ---------- TAB 2: QUESTION ANSWERING ----------
    with tab2:
        st.subheader("❓ Ask a Question About the PDF")
        question = st.text_input("Enter your question:")
        if question:
            if st.button("Get Answer"):
                with st.spinner("Finding answer..."):
                    answer = qa_pipeline({'context': pdf_text, 'question': question})
                st.success("✅ Answer Found!")
                st.write(f"**Answer:** {answer['answer']}")
                st.write(f"**Confidence:** {answer['score']:.2f}")

    # ---------- TAB 3: TEXT CLASSIFICATION ----------
    with tab3:
        st.subheader("🔍 Classify PDF Text (Sentiment)")
        if st.button("Classify Text"):
            with st.spinner("Classifying..."):
                classification = classifier(pdf_text[:512])[0]  # Limit to 512 tokens
            st.success("✅ Classification Complete!")
            st.write(f"**Label:** {classification['label']}")
            st.write(f"**Confidence:** {classification['score']:.2f}")
else:

    st.info("👆 Please upload a PDF to begin.")

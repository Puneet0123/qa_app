import streamlit as st
from transformers import pipeline

# Title 
st.set_page_config(page_title="Mini QA App")
st.title("Question Answering App with Transfomers")
st.markdown("Ask Questions from any paragraph using BERT-based models.")

# Sidebar: Model Selection 
model_choice = st.sidebar.selectbox("Choose a QA model", ["distilbert-base-uncased-distilled-squad",
                                                         "bert-large-uncased-whole-word-masking-finetuned-squad",
                                                         "deepset/roberta-base-squad2"])
# Load pipeline
@st.cache_resource
def load_qa_model(model_name):
    return pipeline("question-answering", model = model_name)

qa_pipeline = load_qa_model(model_choice)

#User inputs 
context = st.text_area("Enter Context Paragraph", height=200, placeholder="Paste your article or context here..")
question = st.text_input("Ask a Question", placeholder="What is this text about?")

# show result 
if st.button("Get Answer") and context and question:
    with st.spinner("Thinking..."):
        result = qa_pipeline(question = question, context = context)
        st.success(f"Answer: {result['answer']}")
        st.caption(f"Confidence: {result['score']}")